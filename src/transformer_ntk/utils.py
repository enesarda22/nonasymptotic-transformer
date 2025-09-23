import csv
import math
import os
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

        if os.path.exists(path):
            self._header_written = True
        else:
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


def _ar1_cov_T(T: int, rho: float, device=None, dtype=torch.float32) -> torch.Tensor:
    """AR(1) Toeplitz covariance across tokens: C[i,j] = rho**|i-j|."""
    idx = torch.arange(T, device=device)
    C = rho ** (idx[:, None] - idx[None, :]).abs()
    return C.to(dtype)


def make_inputs(
    n: int,
    d: int,
    T: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    *,
    token_corr: str | None = None,  # None | "ar1" | "custom"
    rho: float = 0.8,  # used when token_corr=="ar1"
    cov_T: torch.Tensor | None = None,  # (T,T) used when token_corr=="custom"
    normalize: bool = True,
) -> torch.Tensor:
    """
    Sample X ~ N(0, I) and (optionally) correlate tokens (columns) via a T×T covariance.
    Then enforce column-wise L2 <= 1.
      token_corr=None: i.i.d. tokens
      token_corr="ar1": AR(1) with parameter rho in [0,1)
      token_corr="custom": use provided cov_T (T×T, PSD)
    Returns: X ∈ R^{n×d×T}
    """
    X = torch.randn(n, d, T, device=device, dtype=dtype)

    if token_corr is not None:
        if token_corr == "ar1":
            C = _ar1_cov_T(T, rho=rho, device=device, dtype=dtype)
        elif token_corr == "custom":
            assert cov_T is not None and cov_T.shape == (T, T), "cov_T must be (T,T)"
            C = cov_T.to(device=device, dtype=dtype)
        else:
            raise ValueError(f"unknown token_corr={token_corr}")

        # Cholesky (jitter for numerical stability)
        C_jit = C + 1e-6 * torch.eye(T, device=device, dtype=dtype)
        L = torch.linalg.cholesky(C_jit)  # C = L L^T

        # Correlate along the last (token) dimension: (...,T) @ (T,T) -> (...,T)
        X = torch.matmul(X, L.T)

    if normalize:
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


def make_dataset(
    n: int,
    d: int,
    T: int,
    teacher_model=None,
    R=16,
    num_mc=4096,
    nu=3.0,
    noise_std: float = 0.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    method="teacher",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build (X, y) using the limit-model teacher \tilde f with Monte Carlo.
    """
    X = make_inputs(n, d, T, device=device, dtype=dtype)

    if method == "teacher":
        with torch.no_grad():
            y = teacher_model(X)
    elif method == "kernel":
        idx = torch.randperm(n)[:R]
        anchors = X[idx].detach().clone()  # (R,d,T)

        # Gram on anchors
        K_RR = _kernel_section_mc(anchors, anchors, num_mc=num_mc, activation="tanh")

        # Sample alpha with target RKHS norm ||f*||_H = nu
        alpha0 = torch.randn(R)
        norm2 = alpha0 @ (K_RR @ alpha0)
        alpha = (nu / math.sqrt(float(norm2))) * alpha0

        # Labels for all X
        K_XR = _kernel_section_mc(X, anchors, num_mc=num_mc, activation="tanh")
        y = K_XR @ alpha
    else:
        raise ValueError(f"Unsupported method: {method}")

    if noise_std > 0:
        y = y + noise_std * torch.randn_like(y)

    return X, y


# ------------------------- Evaluation & simple training step -------------------------
def evaluate(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    with evaluating(model):
        pred = model(X)
        return float(F.mse_loss(pred, y).item())


def _kernel_section_mc(
    X: torch.Tensor,  # (B,d,T)
    X_anchors: torch.Tensor,  # (R,d,T)
    *,
    num_mc: int,
    activation: str = "tanh",
    temperature: float = 1.0,
    chunk: int = 1024,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Monte-Carlo estimate of the NTK section(s):
      K(X, X') = E[ φ_c(X)φ_c(X') + <φ_u(X),φ_u(X')> + <φ_w(X),φ_w(X')>_F ].
    Returns K ∈ R^{B×R}. No clipping; intended as a teacher primitive.
    """
    if X.dim() == 2:
        X = X.unsqueeze(0)
    if X_anchors.dim() == 2:
        X_anchors = X_anchors.unsqueeze(0)
    B, d, T = X.shape
    R = X_anchors.shape[0]
    dev = device or X.device
    dt = dtype or X.dtype
    inv_tau = 1.0 / max(1e-12, float(temperature))
    X = X.to(dev, dt)
    Xa = X_anchors.to(dev, dt)
    X_T = X[:, :, -1]  # (B,d)
    Xa_T = Xa[:, :, -1]  # (R,d)
    xT_dot = torch.einsum("bd,Rd->bR", X_T, Xa_T)  # (B,R)
    sigma, sigma_p = get_activation(activation)
    K_sum = torch.zeros(B, R, device=dev, dtype=dt)
    done = 0
    while done < num_mc:
        k = min(chunk, num_mc - done)
        done += k
        # φ samples
        _, U, W = sample_phi(k, d, device=dev, dtype=dt)  # c not needed
        # ----- Features at X -----
        WX_T = torch.einsum("mdk,bk->bmd", W, X_T)  # (B,k,d)
        z = torch.einsum("btd,bmd->bmt", X.transpose(1, 2), WX_T)  # (B,k,T)
        z = z * inv_tau
        alpha = torch.softmax(z, dim=-1)  # (B,k,T)
        a = torch.einsum("bdt,bmt->bmd", X, alpha)  # (B,k,d)
        s = torch.einsum("md,bmd->bm", U, a)  # (B,k)
        phi_cX = sigma(s)  # (B,k)
        sigpX = sigma_p(s)  # (B,k)
        first = torch.einsum("bdt,bmt,bDt->bmdD", X, alpha, X)  # (B,k,d,d)
        cov = first - torch.einsum("bmd,bmD->bmdD", a, a)  # (B,k,d,d)
        MU = torch.einsum("bmdD,mD->bmd", cov, U)  # (B,k,d)
        vecX = sigpX.unsqueeze(-1) * MU  # (B,k,d)
        # ----- Features at anchors -----
        WXa_T = torch.einsum("mdk,Rk->Rmd", W, Xa_T)  # (R,k,d)
        za = torch.einsum("Rtd,Rmd->Rmt", Xa.transpose(1, 2), WXa_T)  # (R,k,T)
        za = za * inv_tau
        alphaa = torch.softmax(za, dim=-1)  # (R,k,T)
        aa = torch.einsum("Rdt,Rmt->Rmd", Xa, alphaa)  # (R,k,d)
        sa = torch.einsum("md,Rmd->Rm", U, aa)  # (R,k)
        phi_cA = sigma(sa)  # (R,k)
        sigpA = sigma_p(sa)  # (R,k)
        firsta = torch.einsum("Rdt,Rmt,RDt->RmdD", Xa, alphaa, Xa)  # (R,k,d,d)
        cova = firsta - torch.einsum("Rmd,RmD->RmdD", aa, aa)  # (R,k,d,d)
        MUA = torch.einsum("RmdD,mD->Rmd", cova, U)  # (R,k,d)
        vecA = sigpA.unsqueeze(-1) * MUA  # (R,k,d)
        # ----- Accumulate terms -----
        # c-term:   mean_k [ φ_c(X) φ_c(X') ]
        term_c = torch.einsum("bk,rk->br", phi_cX, phi_cA)
        # u-term:   mean_k [ < σ'(s) a , σ'(s') a' > ]
        term_u = torch.einsum(
            "bkd,rkd->br", sigpX.unsqueeze(-1) * a, sigpA.unsqueeze(-1) * aa
        )
        # w-term:   mean_k [ (vecX · vecA) * (X_T · X_T') ]
        inner_vec = torch.einsum("bkd,rkd->br", vecX, vecA)  # (B,R)
        term_w = inner_vec * xT_dot
        K_sum += term_c + term_u + term_w
    K = K_sum / float(num_mc)
    return K
