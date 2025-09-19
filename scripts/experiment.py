import argparse
import math
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F

from transformer_ntk.model import TransformerNet
from transformer_ntk.utils import (
    set_seed,
    make_dataset,
    get_activation,
    VBound,
    evaluate,
    CSVLogger,
)


def train_pgd_fullbatch(
    model: TransformerNet,
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
) -> Tuple[float, float, int]:
    """
    Full-batch PGD training loop.
    Returns: (min_train_loss, min_val_loss, best_step)
    """
    device = next(model.parameters()).device
    X_tr = X_tr.to(device)
    y_tr = y_tr.to(device)
    if X_val is not None:
        X_val = X_val.to(device)
        y_val = y_val.to(device)

    opt = torch.optim.SGD(model.parameters(), lr=lr)

    min_train = math.inf
    min_val = math.inf
    best_step = -1

    curve_logger = CSVLogger(log_curve_to) if log_curve_to else None

    for s in range(steps):
        opt.zero_grad(set_to_none=True)
        pred = model(X_tr)
        loss = F.mse_loss(pred, y_tr)
        loss.backward()
        opt.step()

        # Project onto product balls around init (per-parameter)
        model.project_(rho_c=rho_c, rho_u=rho_u, rho_w=rho_w)

        train_loss = float(loss.item())
        if train_loss < min_train:
            min_train = train_loss
            best_step = s

        if X_val is not None:
            with torch.no_grad():
                val_pred = model(X_val)
                val_loss = float(F.mse_loss(val_pred, y_val).item())
                if val_loss < min_val:
                    min_val = val_loss

        if curve_logger:
            row = {"step": s, "train_loss": train_loss}
            if X_val is not None:
                row["val_loss"] = val_loss
            curve_logger.log(row)

    if X_val is None:
        min_val = float("nan")
    return min_train, min_val, best_step


def parse_int_list(xs: List[str]) -> List[int]:
    out = []
    for x in xs:
        if "," in x:
            out.extend(int(t) for t in x.split(",") if t.strip())
        else:
            out.append(int(x))
    # unique & sorted (optional)
    return sorted(set(out))


def main():
    parser = argparse.ArgumentParser(
        description="Projected GD width sweep (teacher–student)."
    )

    # Data / teacher
    parser.add_argument("--d", type=int, default=8, help="feature dim")
    parser.add_argument("--T", type=int, default=16, help="sequence length")
    parser.add_argument("--n", type=int, default=5000, help="dataset size (total)")
    parser.add_argument(
        "--val_frac", type=float, default=0.2, help="validation fraction"
    )
    parser.add_argument(
        "--teacher-mc", type=int, default=2048, help="num MC samples for teacher"
    )
    parser.add_argument(
        "--teacher-activation", type=str, default="tanh", choices=["tanh", "erf"]
    )
    parser.add_argument(
        "--teacher-nu",
        type=float,
        nargs=3,
        default=[1.0, 1.0, 1.0],
        metavar=("nu_c", "nu_u", "nu_w"),
    )
    parser.add_argument("--noise-std", type=float, default=0.0)

    # Students
    parser.add_argument(
        "--widths", type=str, nargs="+", default=["8", "16", "32", "64", "128", "256"]
    )
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--activation", type=str, default="tanh", choices=["tanh", "erf"]
    )

    # Projection radii (rho >= \bar\nu is recommended)
    parser.add_argument(
        "--pgd-radii",
        type=float,
        nargs=3,
        default=[1.0, 1.0, 1.0],
        metavar=("rho_c", "rho_u", "rho_w"),
    )

    # Misc
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"]
    )
    parser.add_argument("--runs-dir", type=str, default="runs")
    parser.add_argument("--exp-name", type=str, default="width_sweep")
    parser.add_argument("--log-curves", action="store_true", help="write per-step CSVs")

    args = parser.parse_args()
    set_seed(args.seed)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    widths = parse_int_list(args.widths)
    d, T, n = args.d, args.T, args.n

    # Prepare output dirs
    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    sweep_csv = runs_dir / f"{args.exp_name}.csv"
    logger = CSVLogger(str(sweep_csv))

    # Build dataset via limit-model teacher
    teacher_bounds = VBound(*args.teacher_nu)
    X_all, y_all = make_dataset(
        n=n,
        d=d,
        T=T,
        num_mc_teacher=args.teacher_mc,
        activation=args.teacher_activation,
        bounds=teacher_bounds,
        noise_std=args.noise_std,
        device=device,
    )

    # Train/val split
    n_val = int(args.val_frac * n)
    idx = torch.randperm(n, device=device)
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
    X_val, y_val = X_all[val_idx], y_all[val_idx]

    # Student activation
    sigma, _ = get_activation(args.activation)

    # Sweep over widths
    for m in widths:
        if m % 2 != 0:
            raise ValueError(
                f"Width m={m} must be even for paired initialization (theory)."
            )

        model = TransformerNet(
            d=d, T=T, m=m, activation=sigma, symmetric_init=True, device=device
        ).to(device)

        # Optional per-step logging
        curve_path = None
        if args.log_curves:
            curve_path = str(runs_dir / f"{args.exp_name}_curve_m{m}.csv")

        min_tr, min_val, best_step = train_pgd_fullbatch(
            model,
            X_tr,
            y_tr,
            steps=args.steps,
            lr=args.lr,
            rho_c=args.pgd_radii[0],
            rho_u=args.pgd_radii[1],
            rho_w=args.pgd_radii[2],
            X_val=X_val,
            y_val=y_val,
            log_curve_to=curve_path,
        )

        # Final evaluation (optional)
        final_tr = evaluate(model, X_tr, y_tr)
        final_val = evaluate(model, X_val, y_val)

        logger.log(
            {
                "seed": args.seed,
                "d": d,
                "T": T,
                "n": n,
                "val_frac": args.val_frac,
                "teacher_mc": args.teacher_mc,
                "teacher_activation": args.teacher_activation,
                "teacher_nu_c": args.teacher_nu[0],
                "teacher_nu_u": args.teacher_nu[1],
                "teacher_nu_w": args.teacher_nu[2],
                "student_activation": args.activation,
                "rho_c": args.pgd_radii[0],
                "rho_u": args.pgd_radii[1],
                "rho_w": args.pgd_radii[2],
                "lr": args.lr,
                "steps": args.steps,
                "m": m,
                "min_train_loss": min_tr,
                "min_val_loss": min_val,
                "final_train_loss": final_tr,
                "final_val_loss": final_val,
                "best_step": best_step,
                "device": str(device),
            }
        )

        print(
            f"[m={m:>4}] min_train={min_tr:.6f}  min_val={min_val:.6f}  "
            f"final_train={final_tr:.6f} final_val={final_val:.6f}  best_step={best_step}"
        )

    print(f"✓ Sweep complete. Results at: {sweep_csv}")


if __name__ == "__main__":
    main()
