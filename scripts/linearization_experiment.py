import argparse
import math
from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.nn.functional as F

from transformer_ntk.data import make_inputs
from transformer_ntk.model import Transformer
from transformer_ntk.utils import get_activation, CSVLogger, parse_int_list


@torch.no_grad()
def _clone_params(model):
    """Return detached copies (W,U,c) from a TransformerNet-like model."""
    return model.W.detach().clone(), model.U.detach().clone(), model.c.detach().clone()


@torch.no_grad()
def in_omega_rho(
    model: Transformer, rho_c: float, rho_u: float, rho_w: float, tol: float = 1e-6
) -> bool:
    m = model.m
    scale = 1.0 / math.sqrt(m)
    W_diff = (model.W - model.W0).reshape(m, -1).norm(dim=1)
    U_diff = (model.U - model.U0).norm(dim=1)
    c_diff = (model.c - model.c0).abs()
    return bool(
        torch.all(W_diff <= rho_w * scale + tol)
        and torch.all(U_diff <= rho_u * scale + tol)
        and torch.all(c_diff <= rho_c * scale + tol)
    )


@torch.no_grad()
def _sample_deltas_in_omega(
    m: int,
    d: int,
    *,
    rho_c: float,
    rho_u: float,
    rho_w: float,
    device,
    dtype,
    dist: str = "uniform_ball",
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample (ΔW, ΔU, Δc) with per-head bounds:
      |Δc_i| <= rho_c/√m,  ||ΔU_i||_2 <= rho_u/√m,  ||ΔW_i||_F <= rho_w/√m.
    """
    if seed is not None:
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)

        def _randn(shape):
            return torch.randn(shape, generator=g, device="cpu", dtype=dtype)

        def _rand(shape):
            return torch.rand(shape, generator=g, device="cpu", dtype=dtype)
    else:

        def _randn(shape):
            return torch.randn(shape, device="cpu", dtype=dtype)

        def _rand(shape):
            return torch.rand(shape, device="cpu", dtype=dtype)

    scale = 1.0 / math.sqrt(m)
    rc, ru, rw = rho_c * scale, rho_u * scale, rho_w * scale

    if dist == "uniform_ball":
        # c: uniform in [-rc, rc]
        dc = (_rand((m,)) * 2 - 1) * rc

        # U: sample direction on sphere, radius ~ U^{1/d}
        z = _randn((m, d))
        z = z / z.norm(dim=1, keepdim=True).clamp_min(1e-12)
        r = _rand((m, 1)) ** (1.0 / d)
        dU = z * (ru * r)

        # W: same idea in Frobenius space of dim d^2
        z = _randn((m, d * d))
        z = z / z.norm(dim=1, keepdim=True).clamp_min(1e-12)
        r = _rand((m, 1)) ** (1.0 / (d * d))
        dW = (z * (rw * r)).view(m, d, d)

    elif dist == "gaussian_clip":
        # Gaussian then rescale to boundary if needed
        dc = _randn((m,)) * rc
        dU = _randn((m, d))
        nU = dU.norm(dim=1, keepdim=True).clamp_min(1e-12)
        dU = dU * torch.clamp(ru / nU, max=1.0)
        dW = _randn((m, d, d))
        nW = dW.view(m, -1).norm(dim=1, keepdim=True).clamp_min(1e-12)
        dW = (dW.view(m, -1) * torch.clamp(rw / nW, max=1.0)).view(m, d, d)

    else:
        raise ValueError("dist must be 'uniform_ball' or 'gaussian_clip'")

    return dW.to(device), dU.to(device), dc.to(device)


def _adversarial_deltas_in_omega(
    model,
    X: torch.Tensor,  # (B,d,T) or (d,T)
    *,
    rho_c: float,
    rho_u: float,
    rho_w: float,
    device=None,
    dtype=None,
    batch_chunk: int = 0,  # chunk X if very large
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Construct adversarial (ΔW, ΔU, Δc) ∈ Ω_ρ that maximize the 'white' term:
        (1/√m) Σ_i Δc_i [ h_i(X; θ_i) - h_i(X; θ_i^(0)) ].
    Strategy: for each head i, align ΔU_i and ΔW_i with ∇_{U_i,W_i} h_i at the paired init,
    aggregated over the provided batch X. Set Δc_i on the boundary with a sign that makes
    the contribution positive (given that Δh_i ≈ ∇h_i · Δθ_i ≥ 0 by alignment).

    Assumes `model` is currently at the paired/symmetric initialization θ^(0).
    Returns tensors shaped like model.W (m,d,d), model.U (m,d), model.c (m,).
    """
    # Ensure batch
    if X.dim() == 2:
        X = X.unsqueeze(0)
    if device is None:
        device = next(model.parameters()).device
    if dtype is None:
        dtype = next(model.parameters()).dtype

    X = X.to(device=device, dtype=dtype)
    B = X.shape[0]
    m = model.m
    d = model.d

    # Per-head radii (Ω_ρ)
    scale = 1.0 / math.sqrt(m)
    ru, rw, rc = rho_u * scale, rho_w * scale, rho_c * scale

    # Outputs
    dU = torch.zeros((m, d), device=device, dtype=dtype)
    dW = torch.zeros((m, d, d), device=device, dtype=dtype)
    # Choose Δc_i at the boundary with positive sign (since Δh_i >= 0 by construction)
    dc = torch.full((m,), rc, device=device, dtype=dtype)

    # Save & restore c around per-head gradient extraction
    with torch.no_grad():
        c_backup = model.c.detach().clone()

    def _sum_over_batch_with_head(i: int) -> torch.Tensor:
        """
        Temporarily set c to one-hot with c_i = √m so that f(X) == h_i(X),
        then return sum_b h_i(X_b) (a scalar) for grad extraction.
        """
        with torch.no_grad():
            model.c.zero_()
            model.c[i] = math.sqrt(m)
        # optional chunking over batch to avoid OOM while keeping the same graph per head
        if batch_chunk and batch_chunk > 0 and B > batch_chunk:
            s = 0.0
            for b0 in range(0, B, batch_chunk):
                b1 = min(B, b0 + batch_chunk)
                s = s + model(X[b0:b1]).sum()
            return s
        else:
            return model(X).sum()

    # Build gradients per head and align Δθ_i with them
    for i in range(m):
        model.zero_grad(set_to_none=True)
        # h_i sum over batch (scalar)
        with torch.enable_grad():
            s_i = _sum_over_batch_with_head(i)
            # grads wrt ALL heads; we'll slice head i
            gW_all, gU_all = torch.autograd.grad(
                s_i,
                (model.W, model.U),
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )
        gU_i = gU_all[i]  # (d,)
        gW_i = gW_all[i]  # (d,d)

        # Normalize; if gradient nearly zero, fall back to a fixed axis
        gU_norm = gU_i.norm().clamp_min(1e-12)
        gW_norm = gW_i.view(-1).norm().clamp_min(1e-12)
        if float(gU_norm) == 0.0:
            e0 = torch.zeros_like(gU_i)
            e0[0] = 1.0
            dU[i] = ru * e0
        else:
            dU[i] = ru * (gU_i / gU_norm)

        if float(gW_norm) == 0.0:
            E = torch.zeros_like(gW_i)
            E[0, 0] = 1.0
            dW[i] = rw * E
        else:
            dW[i] = rw * (gW_i / gW_norm)

    # Restore original c
    with torch.no_grad():
        model.c.copy_(c_backup)

    return dW, dU, dc


def perturb_and_eval_linearization(
    model: Transformer,
    X: torch.Tensor,  # (B,d,T) or (d,T)
    rho_c: float,
    rho_u: float,
    rho_w: float,
    dist: str = "uniform_ball",
    seed: Optional[int] = None,
    batch_chunk: int = 0,
    restore: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Assumes `model` is at the symmetric init (paired). Steps:
      1) sample Δφ ∈ Ω_ρ around the current (paired) params,
      2) compute f_lin(X) = <∇_φ f(X; φ^(0)), Δφ>  (since f(X; φ^(0)) = 0),
      3) set φ := φ^(0) + Δφ and compute f(X; φ),
      4) optionally restore model to φ^(0).

    Returns: (f_lin, f_new, (ΔW, ΔU, Δc)), each of shape (B,), (B,), and tensors for deltas.
    """
    # Ensure X is a batch, and pick device/dtype
    if X.dim() == 2:
        X = X.unsqueeze(0)
    B, d, T = X.shape
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    X = X.to(device=device, dtype=dtype)

    # Center (paired) params are the current ones
    with torch.no_grad():
        W0 = model.W.detach().clone()
        U0 = model.U.detach().clone()
        c0 = model.c.detach().clone()

    # Sample Δ inside Ω_ρ
    # dW, dU, dc = _sample_deltas_in_omega(
    #     m=model.m,
    #     d=d,
    #     rho_c=rho_c,
    #     rho_u=rho_u,
    #     rho_w=rho_w,
    #     device=device,
    #     dtype=dtype,
    #     dist=dist,
    #     seed=seed,
    # )

    dW, dU, dc = _adversarial_deltas_in_omega(
        model=model,
        X=X,
        rho_c=rho_c,
        rho_u=rho_u,
        rho_w=rho_w,
        device=device,
        dtype=dtype,
        batch_chunk=batch_chunk,
    )

    # --- 1) Linearized outputs at φ^(0) along Δφ ---
    def _chunk_lin(Xchunk: torch.Tensor) -> torch.Tensor:
        vals = []
        for i in range(Xchunk.shape[0]):
            xi = Xchunk[i : i + 1]
            model.zero_grad(set_to_none=True)
            # Forward at φ^(0); paired init ⇒ output ≈ 0, but we still need the grad
            fi = model(xi).sum()
            gW, gU, gc = torch.autograd.grad(
                fi,
                (model.W, model.U, model.c),
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )
            val = (gW * dW).sum() + (gU * dU).sum() + (gc * dc).sum()
            vals.append(val.detach())
        return torch.stack(vals, dim=0)

    if batch_chunk and batch_chunk > 0:
        chunks = []
        for s in range(0, B, batch_chunk):
            e = min(B, s + batch_chunk)
            chunks.append(_chunk_lin(X[s:e]))
        y_lin = torch.cat(chunks, dim=0)
    else:
        y_lin = _chunk_lin(X)

    # --- 2) Nonlinear outputs at φ = φ^(0) + Δφ ---
    with torch.no_grad():
        model.W.copy_(W0 + dW)
        model.U.copy_(U0 + dU)
        model.c.copy_(c0 + dc)
        assert in_omega_rho(model, rho_c, rho_u, rho_w)
        y = model(X).detach().clone()

    # --- 3) Optional restore to φ^(0) ---
    if restore:
        with torch.no_grad():
            model.W.copy_(W0)
            model.U.copy_(U0)
            model.c.copy_(c0)

    return y_lin, y, (dW, dU, dc)


def main():
    parser = argparse.ArgumentParser(description="Linearization error experiment.")

    # Data / teacher
    parser.add_argument("--d", type=int, default=8, help="feature dim")
    parser.add_argument("--T", type=int, default=16, help="sequence length")
    parser.add_argument("--n", type=int, default=5000, help="dataset size (total)")
    parser.add_argument(
        "--widths", type=str, nargs="+", default=["8", "16", "32", "64", "128", "256"]
    )
    parser.add_argument(
        "--activation", type=str, default="tanh", choices=["tanh", "erf"]
    )
    parser.add_argument("--runs-dir", type=str, default="runs")
    parser.add_argument("--exp-name", type=str, default="width_sweep")
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"]
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",  # allows one or more values
        default=[123],
    )
    parser.add_argument(
        "--pgd-radii",
        type=float,
        nargs=3,
        default=[3.0, 3.0, 3.0],
        metavar=("rho_c", "rho_u", "rho_w"),
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    sweep_csv = runs_dir / f"{args.exp_name}.csv"
    logger = CSVLogger(str(sweep_csv))

    widths = parse_int_list(args.widths)
    n, d, T = args.n, args.d, args.T
    rho_c, rho_u, rho_w = args.pgd_radii
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    for seed in args.seeds:
        for m in widths:
            # initialize teacher model
            sigma, _ = get_activation(args.activation)
            teacher_model = Transformer(
                d=d,
                T=T,
                m=m,
                activation=sigma,
                symmetric_init=True,
                device=device,
            ).to(device)
            teacher_model.eval()

            X = make_inputs(n, d, T, device=device)
            y_lin, y, _ = perturb_and_eval_linearization(
                model=teacher_model,
                X=X,
                rho_c=rho_c,
                rho_u=rho_u,
                rho_w=rho_w,
                seed=seed,
                batch_chunk=512,
                restore=False,
            )

            l1 = float(F.l1_loss(y_lin, y).item())
            print(f"[m={m:>4}] linearization_err={l1:.6f}  ")
            logger.log(
                {
                    "seed": seed,
                    "d": d,
                    "T": T,
                    "n": n,
                    "activation": args.activation,
                    "rho_c": rho_c,
                    "rho_u": rho_u,
                    "rho_w": rho_w,
                    "m": m,
                    "linearization_l1": l1,
                    "device": str(device),
                }
            )


if __name__ == "__main__":
    main()
