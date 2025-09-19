from transformer_ntk.utils import (
    set_seed,
    CSVLogger,
    evaluating,
    _normalize_columns,
    make_inputs,
    get_activation,
    VBound,
    sample_phi,
    default_v_maps,
    teacher_predict_tilde_f,
    make_dataset,
    gd_step_with_projection,
)
from transformer_ntk.model import TransformerNet

import csv
import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


def test_set_seed_repro():
    set_seed(123)
    a1 = torch.randn(3, 3)
    set_seed(123)
    a2 = torch.randn(3, 3)
    assert torch.allclose(a1, a2)


@pytest.mark.parametrize("d,T", [(5, 7), (8, 1), (3, 10)])
def test_normalize_columns_bounds(d, T):
    X = torch.randn(d, T)
    Xn = _normalize_columns(X)
    # each column norm <= 1
    norms = torch.linalg.vector_norm(Xn, ord=2, dim=0)
    assert torch.all(norms <= 1.0 + 1e-7)


@pytest.mark.parametrize("n,d,T", [(10, 6, 9)])
def test_make_inputs_shape_and_bounds(n, d, T):
    X = make_inputs(n, d, T)
    assert X.shape == (n, d, T)
    norms = torch.linalg.vector_norm(X, ord=2, dim=1)  # (n, T)
    assert torch.all(norms <= 1.0 + 1e-7)


@pytest.mark.parametrize("name", ["erf", "tanh"])
def test_get_activation_bounded(name):
    sigma, sigma_p = get_activation(name)
    x = torch.linspace(-5, 5, steps=1001)
    y = sigma(x)
    yp = sigma_p(x)
    # |sigma| <= 1 and 0 <= sigma' <= 1.15 (loose bounds)
    assert torch.all(y.abs() <= 1.0 + 1e-6)
    assert torch.all(yp >= -1e-6)
    assert torch.all(yp <= 1.15 + 1e-6)


def test_sample_phi_and_default_v_maps_bounds():
    d, k = 7, 128
    c, U, W = sample_phi(k, d)
    bounds = VBound(nu_c=0.7, nu_u=1.3, nu_w=2.0)
    v_c, v_u, v_w = default_v_maps(c, U, W, bounds)
    assert v_c.shape == (k,)
    assert v_u.shape == (k, d)
    assert v_w.shape == (k, d, d)
    # Sup-norm bounds
    assert torch.all(v_c.abs() <= bounds.nu_c + 1e-7)
    assert torch.all(torch.linalg.vector_norm(v_u, dim=1) <= bounds.nu_u + 1e-7)
    flat_w = v_w.view(k, -1)
    assert torch.all(torch.linalg.vector_norm(flat_w, dim=1) <= bounds.nu_w + 1e-6)


@pytest.mark.parametrize("B,d,T,num_mc", [(4, 5, 6, 256), (2, 3, 1, 256)])
def test_teacher_predict_shapes_and_finiteness(B, d, T, num_mc):
    set_seed(0)
    X = make_inputs(B, d, T)
    y = teacher_predict_tilde_f(
        X, num_mc=num_mc, activation="tanh", bounds=VBound(1.0, 1.0, 1.0), chunk=64
    )
    assert y.shape == (B,)
    assert torch.isfinite(y).all()


def test_make_dataset_shapes_and_nonconstant():
    set_seed(1)
    n, d, T = 64, 6, 7
    X, y = make_dataset(n, d, T, num_mc_teacher=256, activation="tanh", noise_std=0.0)
    assert X.shape == (n, d, T)
    assert y.shape == (n,)
    # Should not be a constant vector (very unlikely)
    assert y.std() > 0.0


def test_gd_step_with_projection_runs(tmp_path):
    set_seed(5)
    d, T, m, n = 6, 7, 8, 128
    model = TransformerNet(d=d, T=T, m=m, symmetric_init=True)  # f=0 at init
    X, y = make_dataset(n, d, T, num_mc_teacher=256, activation="tanh", noise_std=0.0)
    Xb = X  # full batch
    yb = y
    # One step GD + projection should run and keep within radii
    loss0 = F.mse_loss(model(Xb), yb).item()
    val = gd_step_with_projection(
        model, Xb, yb, lr=1e-2, rho_c=0.5, rho_u=1.0, rho_w=1.0
    )
    assert isinstance(val, float)
    # Check projection bounds
    scale = 1.0 / math.sqrt(m)
    W_diff = (model.W - model.W0).reshape(m, -1).norm(dim=1)
    U_diff = (model.U - model.U0).norm(dim=1)
    c_diff = (model.c - model.c0).abs()
    assert torch.all(W_diff <= 1.0 * scale + 1e-7)
    assert torch.all(U_diff <= 1.0 * scale + 1e-7)
    assert torch.all(c_diff <= 0.5 * scale + 1e-7)
    # Loss should be finite (may or may not decrease in a single step)
    assert math.isfinite(loss0) and math.isfinite(val)


def test_evaluating_context_restores_mode():
    net = nn.Sequential(nn.Linear(3, 2), nn.ReLU(), nn.Linear(2, 1))
    assert net.training
    with evaluating(net):
        assert not net.training
        _ = net(torch.randn(1, 3))
    assert net.training  # restored


def test_csvlogger_writes(tmp_path):
    path = tmp_path / "log.csv"
    logger = CSVLogger(str(path))
    logger.log({"m": 8, "loss": 1.23})
    logger.log({"m": 16, "loss": 0.87})
    # Read back
    with open(path, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["m"] == "8" and rows[0]["loss"] == "1.23"
    assert rows[1]["m"] == "16" and rows[1]["loss"] == "0.87"
