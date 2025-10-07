import csv
import math
import os
from contextlib import contextmanager
from typing import Callable, Tuple, List

import torch
import torch.nn.functional as F


def parse_int_list(xs: List[str]) -> List[int]:
    out = []
    for x in xs:
        if "," in x:
            out.extend(int(t) for t in x.split(",") if t.strip())
        else:
            out.append(int(x))
    # unique & sorted (optional)
    return sorted(set(out))


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


# ------------------------- Evaluation & simple training step -------------------------
def evaluate(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    with evaluating(model):
        pred = model(X)
        return float(F.mse_loss(pred, y).item())


def train_pgd_fullbatch(
    model: torch.nn.Module,
    X_tr: torch.Tensor,
    y_tr: torch.Tensor,
    *,
    steps: int,
    lr: float,
    rho_c: float,
    rho_u: float,
    rho_w: float,
    X_val: torch.Tensor = None,
    y_val: torch.Tensor = None,
    log_curve_to: str = None,
    # --- early stopping knobs ---
    early_stop: bool = False,
    patience: int = 500,  # number of *steps* with no sufficient improvement
    rel_min_delta: float = 1e-3,  # relative improvement needed to reset patience (0.1% by default)
    abs_min_delta: float = 0.0,  # or demand an absolute improvement (kept 0; rel usually better)
    restore_best: bool = True,
) -> Tuple[float, float, int]:
    """
    Full-batch PGD training with per-step projection and optional early stopping.
    Returns: (min_train_loss, min_val_loss, best_step)
    """
    device = next(model.parameters()).device
    X_tr, y_tr = X_tr.to(device), y_tr.to(device)
    if X_val is not None:
        X_val, y_val = X_val.to(device), y_val.to(device)

    opt = torch.optim.SGD(model.parameters(), lr=lr)
    curve_logger = CSVLogger(log_curve_to) if log_curve_to else None

    min_train = math.inf
    min_val = math.inf if X_val is not None else float("nan")
    best_step = -1
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # helper to decide if improvement is "significant"
    def improved(new: float, ref: float) -> bool:
        if not math.isfinite(ref):
            return True
        rel_ok = (ref - new) >= rel_min_delta * max(ref, 1e-12)
        abs_ok = (ref - new) >= abs_min_delta
        return rel_ok or abs_ok

    no_improve = (
        0  # steps since last significant improvement (on val if present, else train)
    )

    for s in range(steps):
        model.train()
        opt.zero_grad(set_to_none=True)
        pred = model(X_tr)

        # --- ADD: function (output) gradient norms; does NOT touch .grad ---
        # (pred.mean()).backward(retain_graph=True)  # fills .grad with ∂(mean f)/∂θ
        # fgn = grad_norms_by_block(model)  # your existing helper on .grad
        # opt.zero_grad(set_to_none=True)  # IMPORTANT: clear before loss backward

        loss = F.mse_loss(pred, y_tr)
        loss.backward()

        # ---- ADD: gradient-norm logging (after backward, before step) ------------
        gn = grad_norms_by_block(model)
        opt.step()

        # Project onto product balls around init (per-parameter)
        if rho_c and rho_u and rho_w:
            model.project_(rho_c=rho_c, rho_u=rho_u, rho_w=rho_w)

        # Train loss
        train_loss = float(loss.item())
        if train_loss < min_train:
            min_train = train_loss

        # Val loss (if provided)
        metric = train_loss
        if X_val is not None:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val)
                val_loss = float(F.mse_loss(val_pred, y_val).item())
            metric = val_loss
            if val_loss < min_val:
                min_val = val_loss

        # Early-stopping bookkeeping uses "metric" (val if exists, else train)
        if improved(metric, (min_val if X_val is not None else min_train)):
            best_step = s
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if curve_logger:
            row = {"step": s, "train_loss": train_loss}
            if X_val is not None:
                row["val_loss"] = val_loss

            row["grad_total"] = float(gn["total"])
            row["grad_c"] = float(gn["c"])
            row["grad_u"] = float(gn["u"])
            row["grad_w"] = float(gn["w"])
            curve_logger.log(row)

        if early_stop and no_improve >= patience:
            # stop early; best_state already captured the best
            break

    # Restore best weights if requested
    if restore_best:
        with torch.no_grad():
            for k, v in model.state_dict().items():
                v.copy_(best_state[k])

    if X_val is None:
        min_val = float("nan")
    return min_train, min_val, best_step


def grad_norms_by_block(model) -> dict:
    """
    Returns Frobenius/ℓ2 gradient norms grouped by parameter blocks and in total.
    Keys: total, c, u, w
      - Transformer: c, U, W
      - IndiRNN:     c, u, W_in  (mapped to c,u,w for consistency)
    """
    tot_sq = 0.0
    block_sq = {"c": 0.0, "u": 0.0, "w": 0.0}

    # robust grouping by parameter name
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        g2 = g.float().pow(2).sum().item()
        tot_sq += g2

        # bucket by semantic name
        if name == "c":
            block_sq["c"] += g2
        elif name in ("U", "u"):  # Transformer.U  or IndiRNN.u
            block_sq["u"] += g2
        elif name in ("W", "W_in"):  # Transformer.W or IndiRNN.W_in
            block_sq["w"] += g2
        else:
            # fallback by substring (in case names differ)
            lname = name.lower()
            if "c" == lname or lname.endswith(".c"):
                block_sq["c"] += g2
            elif "w_in" in lname or lname.endswith("w_in") or lname == "w":
                block_sq["w"] += g2
            elif lname == "u" or lname.endswith(".u") or lname == "u0":
                block_sq["u"] += g2

    return {
        "total": math.sqrt(tot_sq),
        "c": math.sqrt(block_sq["c"]),
        "u": math.sqrt(block_sq["u"]),
        "w": math.sqrt(block_sq["w"]),
    }


def func_grad_norms_by_block(model, outputs, hutch: int = 1) -> dict:
    """
    Frobenius norm of the Jacobian ∂f/∂θ estimated with Hutchinson.
    Returns block-wise norms for {total, c, u, w}. Does not modify .grad.
    """
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    names = [n for n, _ in params]
    plist = [p for _, p in params]

    tot_sq = c_sq = u_sq = w_sq = 0.0
    for _ in range(hutch):
        # Rademacher vector v ~ {±1}^B so E[v v^T] = I  ⇒  E||J^T v||^2 = ||J||_F^2
        v = torch.randint_like(outputs, low=0, high=2).float() * 2 - 1
        grads = torch.autograd.grad(
            outputs=outputs,
            inputs=plist,
            grad_outputs=v,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        for name, g in zip(names, grads):
            if g is None:
                continue
            s = g.detach().float().pow(2).sum().item()
            tot_sq += s
            if name == "c" or name.endswith(".c"):
                c_sq += s
            elif name in ("U", "u") or name.endswith(".U") or name.endswith(".u"):
                u_sq += s
            elif name in ("W", "W_in") or ("w_in" in name.lower()):
                w_sq += s

    scale = 1.0 / max(1, hutch)
    return {
        "f_total": math.sqrt(tot_sq * scale),
        "f_c": math.sqrt(c_sq * scale),
        "f_u": math.sqrt(u_sq * scale),
        "f_w": math.sqrt(w_sq * scale),
    }
