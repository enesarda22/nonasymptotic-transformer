import argparse
import math
import time
from pathlib import Path

import torch

from transformer_ntk.data import make_dataset
from transformer_ntk.model import Transformer
from transformer_ntk.utils import (
    set_seed,
    evaluate,
    CSVLogger,
    parse_int_list,
    get_activation,
    train_pgd_fullbatch,
)


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
    parser.add_argument("--teacher-m", type=int, default=4096, help="Teacher width")
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
    parser.add_argument(
        "--activation", type=str, default="tanh", choices=["tanh", "erf"]
    )

    # Projection radii (rho >= \bar\nu is recommended)
    parser.add_argument(
        "--pgd-radii",
        type=float,
        nargs=3,
        default=[None, None, None],
        metavar=("rho_c", "rho_u", "rho_w"),
    )

    # Misc
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",  # allows one or more values
        default=[123],
    )
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"]
    )
    parser.add_argument("--runs-dir", type=str, default="runs")
    parser.add_argument("--exp-name", type=str, default="width_sweep")
    parser.add_argument("--log-curves", action="store_true", help="write per-step CSVs")

    args = parser.parse_args()

    # Device
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    widths = parse_int_list(args.widths)
    d, T, n = args.d, args.T, args.n

    # Prepare output dirs
    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    sweep_csv = runs_dir / f"{args.exp_name}.csv"
    logger = CSVLogger(str(sweep_csv))
    sigma, _ = get_activation(args.activation)

    for seed in args.seeds:
        set_seed(seed)

        # Build dataset via teacher model
        # teacher_bounds = VBound(*args.teacher_nu)
        X_all, y_all = make_dataset(
            n=n,
            d=d,
            T=T,
            activation=args.activation,
            R=16,
            num_mc=args.teacher_m,
            nu=3.0,
            method="teacher",
            device=device,
            seed=seed,
        )

        # Train/val split
        n_val = int(args.val_frac * n)
        idx = torch.randperm(n, device=device)
        val_idx = idx[:n_val]
        tr_idx = idx[n_val:]

        X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
        X_val, y_val = X_all[val_idx], y_all[val_idx]

        # Sweep over widths
        for m in widths:
            t0 = time.perf_counter()
            if m % 2 != 0:
                raise ValueError(
                    f"Width m={m} must be even for paired initialization (theory)."
                )

            model = Transformer(
                d=d, T=T, m=m, activation=sigma, symmetric_init=True, device=device
            ).to(device)

            # Optional per-step logging
            curve_path = None
            if args.log_curves:
                curve_path = str(runs_dir / f"{args.exp_name}_curve_m{m}.csv")

            lr = 1.75 / math.sqrt(args.steps)
            min_tr, min_val, best_step = train_pgd_fullbatch(
                model,
                X_tr,
                y_tr,
                steps=args.steps,
                lr=lr,
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
                    "seed": seed,
                    "d": d,
                    "T": T,
                    "n": n,
                    "val_frac": args.val_frac,
                    "teacher_m": args.teacher_m,
                    "teacher_nu_c": args.teacher_nu[0],
                    "teacher_nu_u": args.teacher_nu[1],
                    "teacher_nu_w": args.teacher_nu[2],
                    "activation": args.activation,
                    "rho_c": args.pgd_radii[0],
                    "rho_u": args.pgd_radii[1],
                    "rho_w": args.pgd_radii[2],
                    "lr": lr,
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
            dt = time.perf_counter() - t0

            print(
                f"[m={m:>4}] min_train={min_tr:.6f}  min_val={min_val:.6f}  "
                f"final_train={final_tr:.6f} final_val={final_val:.6f}  best_step={best_step}  "
                f"time_passed:{dt:.6f}s    "
            )

    print(f"✓ Sweep complete. Results at: {sweep_csv}")


if __name__ == "__main__":
    main()
