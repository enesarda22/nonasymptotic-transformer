import argparse
import math
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from transformer_ntk.data import make_ar_forecasting_data
from transformer_ntk.model import IndiRNN, Transformer
from transformer_ntk.utils import (
    set_seed,
    CSVLogger,
    get_activation,
)


def float_or_none(x):
    if x is None:
        return None
    if isinstance(x, str) and x.lower() == "none":
        return None
    return float(x)


@torch.no_grad()
def evaluate(model, X, y) -> float:
    pred = model(X)
    return torch.mean((pred - y) ** 2).item()


def grad_block_norms(model) -> Dict[str, float]:
    """Return L2 gradient norms per block and total, assuming .grad is populated."""
    norms = {}
    if isinstance(model, Transformer):
        blocks = {"W": model.W, "U": model.U, "c": model.c}
    elif isinstance(model, IndiRNN):
        blocks = {"W_in": model.W_in, "u": model.u, "c": model.c}
    else:
        raise TypeError("Unknown model type")

    total_sq = 0.0
    for name, p in blocks.items():
        g = p.grad if p.grad is not None else None
        val = float(g.norm().item()) if g is not None else 0.0
        norms[name] = val
        total_sq += val * val
    norms["total"] = math.sqrt(total_sq)
    return norms


def backward_fullbatch(model, X, y) -> Tuple[float, Dict[str, float]]:
    """Compute full-batch MSE and gradients; return (loss, grad_norms)."""
    model.zero_grad(set_to_none=True)
    pred = model(X)
    loss = F.mse_loss(pred, y)
    loss.backward()
    norms = grad_block_norms(model)
    return float(loss.item()), norms


@torch.no_grad()
def sgd_step(model, lr: float):
    for p in model.parameters():
        if p.grad is not None:
            p.add_(p.grad, alpha=-lr)


def project_if_needed(model, rho):
    rho_c, rho_u, rho_w = rho
    if rho_c and rho_u and rho_w:
        if isinstance(model, Transformer):
            model.project_(rho_c, rho_u, rho_w)
        elif isinstance(model, IndiRNN):
            model.project_(rho_c, rho_u, rho_w)
    else:
        return


def main():
    ap = argparse.ArgumentParser(
        description="Gradient-norm study on AR(L): IndiRNN vs Transformer"
    )
    # Data
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument(
        "--T-list", type=int, nargs="+", default=[128], help="sequence lengths to sweep"
    )
    ap.add_argument(
        "--L-list",
        type=int,
        nargs="+",
        default=[4, 8, 16, 32, 64, 96],
        help="AR lag(s); ensure L <= T",
    )
    ap.add_argument(
        "--alpha", type=float, default=0.999, help="AR coefficient |alpha|<1"
    )
    ap.add_argument("--noise-std", type=float, default=math.sqrt(0.1))
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--standardize-y", action="store_true", default=False)
    ap.add_argument("--normalize-columns", action="store_true", default=False)

    # Models / training
    ap.add_argument("--m", type=int, default=64, help="width for both models")
    ap.add_argument("--activation", type=str, default="tanh", choices=["tanh", "erf"])
    ap.add_argument(
        "--steps", type=int, default=400, help="GD steps (0 logs init only)"
    )
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument(
        "--rho",
        type=float_or_none,
        nargs=3,
        default=[3.0, 3.0, 3.0],
        help="Proj radii (rho_c, rho_u, rho_w); use 'None None None' for vanilla GD",
    )
    ap.add_argument("--log-every", type=int, default=10, help="log grads every k steps")

    # IO
    ap.add_argument("--runs-dir", type=str, default="runs")
    ap.add_argument("--exp-name", type=str, default="ar_gradnorms")
    ap.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"]
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=[123])

    args = ap.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    runs = Path(args.runs_dir)
    runs.mkdir(parents=True, exist_ok=True)
    out_csv = runs / f"{args.exp_name}.csv"
    logger = CSVLogger(str(out_csv))

    sigma, _ = get_activation(args.activation)

    for seed in args.seeds:
        set_seed(seed)

        for T in args.T_list:
            for L in args.L_list:
                if L > T:
                    print(f"Skip: L={L} > T={T}")
                    continue

                # --------------------------- data ---------------------------
                X_all, y_all = make_ar_forecasting_data(
                    n=args.n,
                    T=T,
                    L=L,
                    alpha=args.alpha,
                    sigma_eps=args.noise_std,
                    # add_pos=True,
                    # K_pos=8,
                    # pos_scale=1.0,
                    normalize_columns=args.normalize_columns,
                    standardize_y=args.standardize_y,
                    device=device,
                )
                d = X_all.shape[1]

                # train/val split
                n_val = int(args.val_frac * args.n)
                idx = torch.randperm(args.n, device=device)
                val_idx, tr_idx = idx[:n_val], idx[n_val:]
                X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
                X_val, y_val = X_all[val_idx], y_all[val_idx]

                # ------------------------ Transformer -----------------------
                trf = Transformer(
                    d=d,
                    T=T,
                    m=args.m,
                    activation=sigma,
                    symmetric_init=True,
                    device=device,
                    mask_last_key=False,  # forecasting: don't attend to self
                ).to(device)

                # NOTE: ensure scaled dot-product in your model:
                # in Transformer.forward, use: z = z / math.sqrt(self.d)

                rho_tuple = tuple(
                    None if (v is None or str(v).lower() == "none") else float(v)
                    for v in args.rho
                )

                # step 0 (init): gradients
                loss0, norms0 = backward_fullbatch(trf, X_tr, y_tr)
                eval0 = evaluate(trf, X_val, y_val)
                logger.log(
                    dict(
                        seed=seed,
                        model="Transformer",
                        m=args.m,
                        T=T,
                        L=L,
                        step=0,
                        train_mse=loss0,
                        val_mse=eval0,
                        grad_total=norms0["total"],
                        grad_W=norms0.get("W", float("nan")),
                        grad_U=norms0.get("U", float("nan")),
                        grad_c=norms0.get("c", float("nan")),
                        device=str(device),
                    )
                )

                # train and log
                for step in range(1, args.steps + 1):
                    # use same grads we measure for the step
                    loss, norms = backward_fullbatch(trf, X_tr, y_tr)
                    sgd_step(trf, args.lr)
                    project_if_needed(trf, rho_tuple)

                    if (step % args.log_every == 0) or (step == args.steps):
                        val = evaluate(trf, X_val, y_val)
                        logger.log(
                            dict(
                                seed=seed,
                                model="Transformer",
                                m=args.m,
                                T=T,
                                L=L,
                                step=step,
                                train_mse=loss,
                                val_mse=val,
                                grad_total=norms["total"],
                                grad_W=norms.get("W", float("nan")),
                                grad_U=norms.get("U", float("nan")),
                                grad_c=norms.get("c", float("nan")),
                                device=str(device),
                            )
                        )

                # --------------------------- IndiRNN -------------------------
                rnn = IndiRNN(
                    d=d,
                    T=T,
                    m=args.m,
                    activation=sigma,
                    symmetric_init=True,
                    device=device,
                ).to(device)

                loss0, norms0 = backward_fullbatch(rnn, X_tr, y_tr)
                eval0 = evaluate(rnn, X_val, y_val)
                logger.log(
                    dict(
                        seed=seed,
                        model="IndiRNN",
                        m=args.m,
                        T=T,
                        L=L,
                        step=0,
                        train_mse=loss0,
                        val_mse=eval0,
                        grad_total=norms0["total"],
                        grad_W_in=norms0.get("W_in", float("nan")),
                        grad_u=norms0.get("u", float("nan")),
                        grad_c=norms0.get("c", float("nan")),
                        device=str(device),
                    )
                )

                for step in range(1, args.steps + 1):
                    loss, norms = backward_fullbatch(rnn, X_tr, y_tr)
                    sgd_step(rnn, args.lr)
                    project_if_needed(rnn, rho_tuple)

                    if (step % args.log_every == 0) or (step == args.steps):
                        val = evaluate(rnn, X_val, y_val)
                        logger.log(
                            dict(
                                seed=seed,
                                model="IndiRNN",
                                m=args.m,
                                T=T,
                                L=L,
                                step=step,
                                train_mse=loss,
                                val_mse=val,
                                grad_total=norms["total"],
                                grad_W_in=norms.get("W_in", float("nan")),
                                grad_u=norms.get("u", float("nan")),
                                grad_c=norms.get("c", float("nan")),
                                device=str(device),
                            )
                        )

                print(f"✓ Done seed={seed} T={T} L={L}")

    print(f"CSV written to: {out_csv}")


if __name__ == "__main__":
    main()
