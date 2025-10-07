# experiments/exp_ar_sweep_r.py
import argparse
import math
from pathlib import Path

import torch

from transformer_ntk.data import make_ar_forecasting_data

# Your modules (assumed available in your repo)
from transformer_ntk.model import IndiRNN, Transformer
from transformer_ntk.utils import (
    set_seed,
    CSVLogger,
    get_activation,
    evaluate,
    train_pgd_fullbatch,
)


def float_or_none(x):
    if x.lower() == "none":
        return None
    return float(x)


def context_len_for_r(r: float, base_T: int, policy: str, c: float) -> int:
    """
    Choose effective sequence length given r.
    - "fixed": keep T=base_T
    - "matched": L(r) = min(base_T, ceil(c / (1 - r)))
    """
    if policy == "fixed":
        return base_T
    elif policy == "matched":
        if r >= 1.0:
            return base_T
        L = math.ceil(c / max(1e-6, 1.0 - r))
        return min(base_T, max(1, L))
    else:
        raise ValueError("context_policy must be 'fixed' or 'matched'.")


def main():
    ap = argparse.ArgumentParser(
        description="AR(1) sweep over r: IndiRNN vs Transformer"
    )
    # Data
    ap.add_argument("--n", type=int, default=5000, help="dataset size")
    ap.add_argument("--T", type=int, default=128, help="max context length")
    ap.add_argument(
        "--l-list",
        type=int,
        nargs="+",
        default=[96],
        help="AR(1) coefficients",
    )
    ap.add_argument("--noise-std", type=float, default=0.0, help="label noise on y")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--standardize-y", action="store_true", default=True)

    # Models / training
    ap.add_argument("--m", type=int, default=64, help="width (same for both models)")
    ap.add_argument("--activation", type=str, default="tanh", choices=["tanh", "erf"])
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument(
        "--rho",
        type=float_or_none,
        nargs=3,
        default=[None, None, None],
    )
    ap.add_argument(
        "--context-policy",
        type=str,
        default="fixed",
        choices=["fixed", "matched"],
        help="fixed: T constant; matched: L(r)=ceil(c/(1-r)) capped by T",
    )
    ap.add_argument(
        "--context-mult", type=float, default=2.0, help="c in L(r)=ceil(c/(1-r))"
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=[123])

    # IO
    ap.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"]
    )
    ap.add_argument("--runs-dir", type=str, default="runs")
    ap.add_argument("--exp-name", type=str, default="ar1_sweep_r")
    ap.add_argument("--log-curves", action="store_true", help="write per-step CSVs")
    ap.add_argument("--mask-last-key", action="store_true")

    args = ap.parse_args()

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

    # Setup logging
    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    out_csv = runs_dir / f"{args.exp_name}.csv"
    logger = CSVLogger(str(out_csv))

    sigma, _ = get_activation(args.activation)

    for seed in args.seeds:
        set_seed(seed)

        for L in args.l_list:
            T = L + 1
            tf_curve_path = None
            if args.log_curves:
                tf_curve_path = str(runs_dir / f"{args.exp_name}_tf_curve_L{L}.csv")

            # Generate dataset (d=1), standardized by default
            X_all, y_all = make_ar_forecasting_data(
                n=args.n,
                T=T,
                L=L,
                alpha=0.9,
                sigma_eps=math.sqrt(0.1),
                device=device,
                normalize_columns=False,
                standardize_y=True,
            )
            d = X_all.shape[1]

            # Train/val split
            n_val = int(args.val_frac * args.n)
            idx = torch.randperm(args.n, device=device)
            val_idx, tr_idx = idx[:n_val], idx[n_val:]
            X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
            X_val, y_val = X_all[val_idx], y_all[val_idx]

            # ---- Transformer
            trf = Transformer(
                d=d,
                T=T,
                m=args.m,
                activation=sigma,
                symmetric_init=True,
                device=device,
                mask_last_key=args.mask_last_key,
            ).to(device)

            rho_tuple = tuple(
                None if (v is None or str(v).lower() == "none") else float(v)
                for v in args.rho
            )

            tr_min_tr, tr_min_val, tr_best = train_pgd_fullbatch(
                model=trf,
                X_tr=X_tr,
                y_tr=y_tr,
                steps=args.steps,
                lr=args.lr,
                rho_c=rho_tuple[0],
                rho_u=rho_tuple[1],
                rho_w=rho_tuple[2],
                X_val=X_val,
                y_val=y_val,
                log_curve_to=tf_curve_path,
                early_stop=False,
                patience=400,
            )
            tr_final_tr = evaluate(trf, X_tr, y_tr)
            tr_final_val = evaluate(trf, X_val, y_val)

            logger.log(
                {
                    "seed": seed,
                    "model": "Transformer",
                    "activation": args.activation,
                    "m": args.m,
                    "L": L,
                    "T": T,
                    "steps": args.steps,
                    "lr": args.lr,
                    "rho_c": rho_tuple[0],
                    "rho_u": rho_tuple[1],
                    "rho_w": rho_tuple[2],
                    "min_train_mse": tr_min_tr,
                    "min_val_mse": tr_min_val,
                    "final_train_mse": tr_final_tr,
                    "final_val_mse": tr_final_val,
                    "best_step": tr_best,
                    "device": str(device),
                    "context_policy": args.context_policy,
                    "context_mult": args.context_mult,
                    "n": args.n,
                    "val_frac": args.val_frac,
                    "noise_std": args.noise_std,
                }
            )

            print(
                f"[Transformer] L={L} "
                f"min_val={tr_min_val:.6f} final_val={tr_final_val:.6f} best_step={tr_best}"
            )

            rnn_curve_path = None
            if args.log_curves:
                rnn_curve_path = str(runs_dir / f"{args.exp_name}_rnn_curve_L{L}.csv")

            # ---- IndiRNN
            rnn = IndiRNN(
                d=d,
                T=T,
                m=args.m,
                activation=sigma,
                symmetric_init=True,
                device=device,
            ).to(device)

            rnn_min_tr, rnn_min_val, rnn_best = train_pgd_fullbatch(
                model=rnn,
                X_tr=X_tr,
                y_tr=y_tr,
                steps=args.steps,
                lr=args.lr,
                rho_c=rho_tuple[0],
                rho_u=rho_tuple[1],
                rho_w=rho_tuple[2],
                X_val=X_val,
                y_val=y_val,
                log_curve_to=rnn_curve_path,
                early_stop=False,
                patience=400,
            )
            rnn_final_tr = evaluate(rnn, X_tr, y_tr)
            rnn_final_val = evaluate(rnn, X_val, y_val)

            logger.log(
                {
                    "seed": seed,
                    "model": "IndiRNN",
                    "activation": args.activation,
                    "m": args.m,
                    "L": L,
                    "T": T,
                    "steps": args.steps,
                    "lr": args.lr,
                    "rho_c": rho_tuple[0],
                    "rho_u": rho_tuple[1],
                    "rho_w": rho_tuple[2],
                    "min_train_mse": rnn_min_tr,
                    "min_val_mse": rnn_min_val,
                    "final_train_mse": rnn_final_tr,
                    "final_val_mse": rnn_final_val,
                    "best_step": rnn_best,
                    "device": str(device),
                    "context_policy": args.context_policy,
                    "context_mult": args.context_mult,
                    "n": args.n,
                    "val_frac": args.val_frac,
                    "noise_std": args.noise_std,
                }
            )

            print(
                f"[  IndiRNN L={L} "
                f"min_val={rnn_min_val:.6f} final_val={rnn_final_val:.6f} best_step={rnn_best}"
            )

    print(f"✓ Sweep complete. CSV at: {out_csv}")


if __name__ == "__main__":
    main()
