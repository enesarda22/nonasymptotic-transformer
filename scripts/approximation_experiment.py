import argparse
from pathlib import Path

import torch

from transformer_ntk.data import lin_and_tilde_outputs_multianchor, make_inputs
from transformer_ntk.utils import (
    CSVLogger,
    parse_int_list,
)


def main():  #
    parser = argparse.ArgumentParser(description="Approximation error experiment.")

    # Data / teacher
    parser.add_argument("--d", type=int, default=8, help="feature dim")
    parser.add_argument("--T", type=int, default=16, help="sequence length")
    parser.add_argument("--n", type=int, default=5000, help="dataset size (total)")
    parser.add_argument("--teacher-m", type=int, default=8196, help="teacher width")
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
        "--teacher-nu",
        type=float,
        nargs=3,
        default=[3.0, 3.0, 3.0],
        metavar=("nu_c", "nu_u", "nu_w"),
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    sweep_csv = runs_dir / f"{args.exp_name}.csv"
    logger = CSVLogger(str(sweep_csv))

    widths = parse_int_list(args.widths)
    n, d, T, teacher_m = args.n, args.d, args.T, args.teacher_m
    nu_c, nu_u, nu_w = args.teacher_nu
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    for seed in args.seeds:
        for m in widths:
            X = make_inputs(n, d, T, device=device)
            y_lin, y, _, _ = lin_and_tilde_outputs_multianchor(
                X=X,
                teacher_m=teacher_m,
                m=m,
                num_anchors=16,
                nu_c=nu_c,
                nu_u=nu_u,
                nu_w=nu_w,
                activation=args.activation,
                seed=seed,
                device=device,
            )
            max_abs_dist = (y_lin - y).abs().max().item()
            print(f"[m={m:>4}] approximation={max_abs_dist:.6f}  ")
            logger.log(
                {
                    "seed": seed,
                    "d": d,
                    "T": T,
                    "n": n,
                    "activation": args.activation,
                    "nu_c": nu_c,
                    "nu_u": nu_u,
                    "nu_w": nu_w,
                    "teacher_m": teacher_m,
                    "m": m,
                    "max_absolute_dist": max_abs_dist,
                    "device": str(device),
                }
            )


if __name__ == "__main__":
    main()
