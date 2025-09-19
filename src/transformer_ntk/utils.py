import csv
import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F


# ------------------------- Repro & simple logging -------------------------


def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CSVLogger:
    """Minimal CSV logger: creates header on first write."""

    def __init__(self, path: str):
        self.path = path
        self._header_written = False

    def log(self, row: dict):
        mode = "a" if self._header_written else "w"
        with open(self.path, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row)


@contextmanager
def evaluating(module: torch.nn.Module):
    """Temporarily switch to eval() (no grad), then restore training mode."""
    was_training = module.training
    try:
        module.eval()
        with torch.no_grad():
            yield
    finally:
        if was_training:
            module.train()


# ------------------------- Data helpers -------------------------


def _normalize_columns(X: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Normalize each column (token) to have L2 <= 1. X: (..., d, T)
    """
    norms = torch.linalg.vector_norm(X, ord=2, dim=-2, keepdim=True).clamp_min(
        eps
    )  # (..., 1, T)
    scale = torch.clamp(1.0 / norms, max=1.0)  # do not blow up small vectors
    return X * scale


def make_inputs(
    n: int,
    d: int,
    T: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Sample X_j ~ N(0,1) entrywise then column-normalize: shape (n, d, T).
    """
    X = torch.randn(n, d, T, device=device, dtype=dtype)
    X = _normalize_columns(X)
    return X


# ------------------------- Activation & derivatives -------------------------


def get_activation(
    name: str,
) -> Tuple[
    Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]
]:
    """
    Return (sigma, sigma') with bounded derivatives (tanh by default).
    """
    name = name.lower()
    if name in {"tanh", "gelu-tanh"}:

        def sigma(x):
            return torch.tanh(x)

        def sigma_p(x):
            return 1.0 - torch.tanh(x).pow(2)

        return sigma, sigma_p
    elif name in {"erf"}:
        # erf activation (common in NTK literature); derivative is 2/sqrt(pi) * exp(-x^2)
        sqrt_pi = math.sqrt(math.pi)

        def sigma(x):
            return torch.erf(x / math.sqrt(2.0))  # scaled variant; bounded

        def sigma_p(x):
            return (2.0 / sqrt_pi) * torch.exp(-0.5 * x.pow(2))

        return sigma, sigma_p
    else:
        raise ValueError(f"Unsupported activation for theory: {name}")


# ------------------------- Limit-model (teacher) machinery -------------------------


@dataclass
class VBound:
    """Upper bounds for v maps: sup |v_c|, sup ||v_u||_2, sup ||v_w||_F."""

    nu_c: float = 1.0
    nu_u: float = 1.0
    nu_w: float = 1.0


def sample_phi(num: int, d: int, device=None, dtype=torch.float32):
    """
    Sample φ=(c,U,W) from the init distribution (NO pairing for teacher).
    c ~ Rad(±1), U ~ N(0, I_d), W_{kℓ} ~ N(0,1) i.i.d.
    """
    c = (torch.randint(0, 2, (num,), device=device) * 2 - 1).to(dtype)
    U = torch.randn(num, d, device=device, dtype=dtype)
    W = torch.randn(num, d, d, device=device, dtype=dtype)
    return c, U, W


def default_v_maps(c, U, W, X_p, sigma, sigma_p, bounds, mode="rich", e=None):
    """
    Nonzero all components while keeping:
      sup|v_c| <= nu_c,  sup||v_u||_2 <= nu_u,  sup||v_w||_F <= nu_w,
    and v depends only on φ=(c,U,W).  Uses:
      v_c: odd in U  (pairs with odd σ)
      v_u: even in U (pairs with even σ')
      v_w: ~ U e^T   (drives quadratic U^T M U term)
    """
    assert mode in {"rich"}

    # v_c
    WX_T = torch.einsum("mdD,D->md", W, X_p[:, -1])  # (B,k,d)
    # z_{b,m,t} = X_{b,:,t}^T (W_m X_T^{(b)}) => (B,k,T)
    z = torch.einsum("td,md->mt", X_p.T, WX_T)  # (B,k,T)
    alpha = F.softmax(z, dim=-1)  # (B,k,T)

    # a = X * alpha over tokens: (B,d,T) x (B,k,T) -> (B,k,d)
    a = torch.einsum("dt,mt->md", X_p, alpha)

    # s = U^T a: (k,d) x (B,k,d) -> (B,k)
    s = torch.einsum("md,md->m", U, a)
    v_c = sigma(s)

    # v_u
    v_u = sigma_p(s).unsqueeze(-1) * a

    # v_w: Frobenius-normalized U e^T (even contribution downstream)
    first = torch.einsum("dt,mt,Dt->mdD", X_p, alpha, X_p)
    cov = first - torch.einsum("md,mD->mdD", a, a)

    MU = torch.einsum("mdD, mD -> md", cov, U)  # (B,k,d)
    MUX = torch.einsum("md, D -> mdD", MU, X_p[:, -1])  # (B,k,d)
    v_w = sigma_p(s).unsqueeze(-1).unsqueeze(-1) * MUX

    return v_c, v_u, v_w


def teacher_predict_tilde_f(
    X: torch.Tensor,
    X_p: torch.Tensor,
    *,
    num_mc: int,
    activation: str = "tanh",
    bounds: VBound = VBound(),
    chunk: int = 1024,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Monte Carlo approximation of the limit model:
    \tilde f(X; v) = E[ φ_c(X) v_c + <φ_u(X), v_u> + <φ_w(X), v_w>_F ].

    Args:
      X: (B, d, T), columns already L2-bounded (we don't rescale here).
      num_mc: number of φ samples for the expectation.
      activation: 'tanh' or 'erf' (bounded derivatives).
      bounds: VBound for sup norms of v maps.
      chunk: process φ in chunks to limit memory.
    Returns:
      y: (B,) teacher outputs.
    """
    if X.dim() == 2:
        X = X.unsqueeze(0)
    B, d, T = X.shape
    dev = device or X.device
    dt = dtype or X.dtype

    sigma, sigma_p = get_activation(activation)

    # last tokens X_T: (B,d)
    X_T = X[:, :, -1]

    total = torch.zeros(B, device=dev, dtype=dt)
    done = 0
    while done < num_mc:
        k = min(chunk, num_mc - done)
        done += k

        # Sample φ batch and construct bounded v(φ)
        c, U, W = sample_phi(k, d, device=dev, dtype=dt)
        v_c, v_u, v_w = default_v_maps(c, U, W, X_p, sigma, sigma_p, bounds)

        # Compute a_i(X) for all (b, m) in this chunk
        # WX_T: (k,d,d) @ (B,d) -> (B,k,d)
        WX_T = torch.einsum("mdk,bk->bmd", W, X_T)  # (B,k,d)
        # z_{b,m,t} = X_{b,:,t}^T (W_m X_T^{(b)}) => (B,k,T)
        z = torch.einsum("btd,bmd->bmt", X.transpose(1, 2), WX_T)  # (B,k,T)
        alpha = F.softmax(z, dim=-1)  # (B,k,T)

        # a = X * alpha over tokens: (B,d,T) x (B,k,T) -> (B,k,d)
        a = torch.einsum("bdt,bmt->bmd", X, alpha)

        # s = U^T a: (k,d) x (B,k,d) -> (B,k)
        s = torch.einsum("md,bmd->bm", U, a)

        # φ_c term: E[σ(s) v_c]
        term_c = (sigma(s) * v_c.unsqueeze(0)).mean(dim=1)  # (B,)

        # φ_u term: E[σ'(s) <a, v_u>]
        inner_u = torch.einsum("bmd,md->bm", a, v_u)  # (B,k)
        term_u = (sigma_p(s) * inner_u).mean(dim=1)  # (B,)

        # φ_w term:
        # J_s = diag(alpha) - alpha alpha^T (we never build full T×T; use identity:
        # M = X J_s X^T = sum_t α_t X_t X_t^T - (X α)(X α)^T
        Xa = a  # (B,k,d) = X α
        # First moment: sum_t α_t X_t X_t^T  -> (B,k,d,d)
        first = torch.einsum("bdt,bmt,bDt->bmdD", X, alpha, X)
        # Covariance: first - (Xa)(Xa)^T
        cov = first - torch.einsum("bmd,bmD->bmdD", Xa, Xa)

        MU = torch.einsum("bmdD, mD -> bmd", cov, U)  # (B,k,d)
        vec = sigma_p(s).unsqueeze(-1) * MU  # (B,k,d)

        # φ_w = vec ⊗ X_T  (rank-1 d×d), Frobenius with v_w:
        # <vec ⊗ X_T, v_w>_F = vec^T (v_w X_T)
        v_w_XT = torch.einsum("mdD, bD -> bm d", v_w, X_T)  # (B,k,d)
        inner_w = torch.einsum("bmd,bmd->bm", vec, v_w_XT)  # (B,k)
        term_w = inner_w.mean(dim=1)  # (B,)

        total += term_c + term_u + term_w

    y = total / float(num_mc)
    return y


def make_dataset(
    n: int,
    d: int,
    T: int,
    *,
    num_mc_teacher: int = 2048,
    activation: str = "tanh",
    bounds: VBound = VBound(),
    noise_std: float = 0.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build (X, y) using the limit-model teacher \tilde f with Monte Carlo.
    """
    X = make_inputs(n + 1, d, T, device=device, dtype=dtype)
    y = teacher_predict_tilde_f(
        X[:-1, :, :],
        X[-1, :, :],
        num_mc=num_mc_teacher,
        activation=activation,
        bounds=bounds,
        device=device,
        dtype=dtype,
    )
    if noise_std > 0:
        y = y + noise_std * torch.randn_like(y)

    # y = (y - torch.mean(y)) / (torch.std(y) + 1e-8)
    return X, y


# ------------------------- Evaluation & simple training step -------------------------


def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def evaluate(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    with evaluating(model):
        pred = model(X)
        return float(mse(pred, y).item())


def gd_step_with_projection(
    model: torch.nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    lr: float,
    rho_c: Optional[float] = None,
    rho_u: Optional[float] = None,
    rho_w: Optional[float] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> float:
    """
    One full-batch GD step on MSE, then (optional) PGD-style projection via model.project_.
    Returns the scalar loss value.
    """
    if optimizer is None:
        # basic SGD without momentum; pass in your own optimizer if you prefer
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    optimizer.zero_grad(set_to_none=True)
    pred = model(X)
    loss = F.mse_loss(pred, y)
    loss.backward()
    optimizer.step()

    # Optional projection if model provides project_ (your TransformerNet does)
    if rho_c is not None and rho_u is not None and rho_w is not None:
        project = getattr(model, "project_", None)
        if callable(project):
            project(rho_c=rho_c, rho_u=rho_u, rho_w=rho_w)

    return float(loss.item())
