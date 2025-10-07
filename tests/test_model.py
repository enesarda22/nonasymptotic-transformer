from transformer_ntk.model import Transformer

import math
import pytest
import torch
import torch.nn.functional as F


def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@pytest.mark.parametrize("d,T,m,B", [(5, 7, 6, 3), (8, 1, 4, 2), (3, 10, 8, 4)])
def test_zero_output_at_init_even_m(d, T, m, B):
    """Paired init (even m) ⇒ f(X;ϕ)=0 for any X."""
    set_seed(123)
    model = Transformer(d=d, T=T, m=m, symmetric_init=True)
    X = torch.randn(B, d, T)
    with torch.no_grad():
        out = model(X)
    assert torch.allclose(out, torch.zeros(B, dtype=out.dtype), atol=1e-6)


def manual_forward(model: Transformer, X: torch.Tensor) -> torch.Tensor:
    """Compute f(X) per the paper definition (for cross-check)."""
    if X.dim() == 2:
        X = X.unsqueeze(0)
    X_T = X[:, :, -1]  # (B,d)
    WX_T = torch.einsum("mdk,bk->bmd", model.W, X_T)  # (B,m,d)
    z = torch.einsum("btd,bmd->bmt", X.transpose(1, 2), WX_T)  # (B,m,T)
    alpha = torch.softmax(z, dim=-1)  # (B,m,T)
    a = torch.einsum("bdt,bmt->bmd", X, alpha)  # (B,m,d)
    s = torch.einsum("md,bmd->bm", model.U, a)  # (B,m)
    h = model.activation(s)  # (B,m)
    out = torch.einsum("m,bm->b", model.c, h) / math.sqrt(model.m)  # (B,)
    return out


@pytest.mark.parametrize("d,T,m,B", [(6, 9, 4, 5), (7, 3, 8, 2)])
def test_forward_matches_manual(d, T, m, B):
    """Numerical check: module forward == manual formula."""
    set_seed(42)
    model = Transformer(d=d, T=T, m=m, symmetric_init=False)
    X = torch.randn(B, d, T)
    with torch.no_grad():
        out1 = model(X)
        out2 = manual_forward(model, X)
    assert torch.allclose(out1, out2, atol=1e-6)


@pytest.mark.parametrize("d,T,m,B", [(4, 5, 6, 3)])
def test_nonzero_output_if_no_symmetric_init(d, T, m, B):
    """Without paired init, output is generally nonzero."""
    set_seed(7)
    model = Transformer(d=d, T=T, m=m, symmetric_init=False)
    X = torch.randn(B, d, T)
    with torch.no_grad():
        out = model(X)
    assert not torch.allclose(out, torch.zeros_like(out), atol=1e-7)


@pytest.mark.parametrize("d,T,m", [(5, 7, 6)])
def test_projection_keeps_within_radii(d, T, m):
    """PGD projection respects per-parameter radii around init."""
    set_seed(11)
    model = Transformer(d=d, T=T, m=m, symmetric_init=True)
    # Nudge params away from init
    with torch.no_grad():
        model.W.add_(torch.randn_like(model.W))
        model.U.add_(torch.randn_like(model.U))
        model.c.add_(torch.randn_like(model.c) * 0.5)

    rho_c, rho_u, rho_w = 0.3, 0.7, 1.1
    model.project_(rho_c=rho_c, rho_u=rho_u, rho_w=rho_w)

    scale = 1.0 / math.sqrt(m)
    # Check constraints
    W_diff = (model.W - model.W0).reshape(m, -1).norm(dim=1)
    U_diff = (model.U - model.U0).norm(dim=1)
    c_diff = (model.c - model.c0).abs()

    assert torch.all(W_diff <= rho_w * scale + 1e-7)
    assert torch.all(U_diff <= rho_u * scale + 1e-7)
    assert torch.all(c_diff <= rho_c * scale + 1e-7)


@pytest.mark.parametrize("d,T,m,B", [(3, 4, 6, 8)])
def test_backward_runs_and_grad_shapes(d, T, m, B):
    """Backprop produces finite grads with correct shapes."""
    set_seed(99)
    model = Transformer(d=d, T=T, m=m, symmetric_init=False)
    X = torch.randn(B, d, T, requires_grad=False)
    y = torch.randn(B)

    pred = model(X)
    loss = F.mse_loss(pred, y)
    loss.backward()

    assert model.W.grad is not None and model.W.grad.shape == model.W.shape
    assert model.U.grad is not None and model.U.grad.shape == model.U.shape
    assert model.c.grad is not None and model.c.grad.shape == model.c.shape
    # Finite values
    assert torch.isfinite(model.W.grad).all()
    assert torch.isfinite(model.U.grad).all()
    assert torch.isfinite(model.c.grad).all()


@pytest.mark.parametrize("d,T,m,B", [(4, 1, 6, 5)])  # T=1 edge case
def test_T_equals_one_edge_case(d, T, m, B):
    """When T=1, softmax=1, so a(X;W)=X and forward reduces accordingly."""
    set_seed(21)
    model = Transformer(d=d, T=T, m=m, symmetric_init=False)
    X = torch.randn(B, d, T)  # last col is the only col
    with torch.no_grad():
        out_model = model(X)
        # a = X directly; check reduction
        X_flat = X.squeeze(-1)  # (B,d)
        s = torch.einsum("md,bd->bm", model.U, X_flat)
        h = model.activation(s)
        out_manual = torch.einsum("m,bm->b", model.c, h) / math.sqrt(model.m)
    assert torch.allclose(out_model, out_manual, atol=1e-6)
