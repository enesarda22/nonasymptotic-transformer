"""
Plot gradient norms from exp_ar_gradnorms.py.

Outputs:
  - grad_total_vs_L_[init|earlyK].png
  - grad_total_vs_T_[init|earlyK].png
  - blocks_[model]_vs_L_[init|earlyK].png     (optional)
  - blocks_[model]_vs_T_[init|earlyK].png     (optional)

Usage examples:
  # L-sweep at fixed T, plot initialization norms and early-avg, plus per-block panels
  python scripts/plot_gradnorms.py --csv runs/gradnorms_Lsweep.csv \
      --plot L --T 128 --out runs/figs --agg init early-avg --early-K 50 --blocks

  # T-sweep at fixed L
  python scripts/plot_gradnorms.py --csv runs/gradnorms_Tsweep.csv \
      --plot T --L 32 --out runs/figs --agg init --blocks
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _ci95(xs: np.ndarray) -> float:
    """95% CI half-width for the mean."""
    xs = np.asarray(xs, dtype=float)
    if xs.size <= 1:
        return 0.0
    return 1.96 * xs.std(ddof=1) / np.sqrt(xs.size)


def _agg_frame(df: pd.DataFrame, group_cols: List[str], value_col: str) -> pd.DataFrame:
    """
    Return mean +/- 95% CI for value_col grouped by group_cols.
    Adds columns: 'mean', 'ci95', 'n'.
    """
    g = df.groupby(group_cols, dropna=False)[value_col]
    out = g.agg(n="count", mean="mean", std="std").reset_index()
    out["ci95"] = 1.96 * out["std"].fillna(0.0) / np.sqrt(out["n"].clip(lower=1))
    return out.drop(columns=["std"])


def _filter_agg(df: pd.DataFrame, agg: str, early_K: int) -> pd.DataFrame:
    """
    Select rows used for aggregation:

    - 'init': step==0
    - 'early-avg': average over steps in [1..K] (or up to max if fewer)
    """
    if agg == "init":
        return df[df["step"] == 0].copy()

    if agg == "early-avg":
        # Keep steps 1..K (if present), average per (seed, model, T, L)
        use = df[(df["step"] >= 1) & (df["step"] <= early_K)].copy()
        grp = ["seed", "model", "T", "L"]
        # Average grad columns over the early window:
        grad_cols = [c for c in df.columns if c.startswith("grad_")]
        val_cols = ["train_mse", "val_mse"] + grad_cols
        out = use.groupby(grp, dropna=False)[val_cols].mean().reset_index()
        out["step"] = early_K  # annotate
        return out

    raise ValueError(f"Unknown agg: {agg}")


def _ensure_out_dir(p: str) -> Path:
    out = Path(p)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _plot_lines(
    ax: plt.Axes,
    x: np.ndarray,
    ys: Tuple[np.ndarray, np.ndarray],
    cis: Tuple[np.ndarray, np.ndarray],
    labels: Tuple[str, str],
):
    """
    Plot two lines with error bars (Transformer vs IndiRNN).
    No explicit colors (let matplotlib choose).
    """
    for y, ci, label in zip(ys, cis, labels):
        ax.errorbar(x, y, yerr=ci, marker="o", linestyle="-", label=label, capsize=3)


def plot_total_vs_L(
    df: pd.DataFrame, T_fixed: int, out_dir: Path, agg_label: str, exp_name
):
    dfx = df[df["T"] == T_fixed]
    if dfx.empty:
        print(f"[warn] No rows for T={T_fixed}. Skipping total-vs-L.")
        return

    # Aggregate per (model, L)
    g_tf = _agg_frame(dfx[dfx["model"] == "Transformer"], ["L"], "grad_total")
    g_rn = _agg_frame(dfx[dfx["model"] == "IndiRNN"], ["L"], "grad_total")

    Ls = sorted(set(g_tf["L"]).union(set(g_rn["L"])))
    # Align
    tf_mean = g_tf.set_index("L").reindex(Ls)["mean"].to_numpy()
    tf_ci = g_tf.set_index("L").reindex(Ls)["ci95"].fillna(0.0).to_numpy()
    rn_mean = g_rn.set_index("L").reindex(Ls)["mean"].to_numpy()
    rn_ci = g_rn.set_index("L").reindex(Ls)["ci95"].fillna(0.0).to_numpy()

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    _plot_lines(
        ax, np.array(Ls), (tf_mean, rn_mean), (tf_ci, rn_ci), ("Transformer", "IndiRNN")
    )
    ax.set_xlabel("AR lag L")
    ax.set_ylabel("total grad norm")
    ax.set_title(f"Gradient norms vs L @ T={T_fixed}  ({agg_label})")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path = out_dir / f"grad_total_vs_L_{agg_label.replace(' ', '')}_{exp_name}.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[saved] {out_path}")


def plot_total_vs_T(
    df: pd.DataFrame, L_fixed: int, out_dir: Path, agg_label: str, exp_name
):
    dfx = df[df["L"] == L_fixed]
    if dfx.empty:
        print(f"[warn] No rows for L={L_fixed}. Skipping total-vs-T.")
        return

    g_tf = _agg_frame(dfx[dfx["model"] == "Transformer"], ["T"], "grad_total")
    g_rn = _agg_frame(dfx[dfx["model"] == "IndiRNN"], ["T"], "grad_total")

    Ts = sorted(set(g_tf["T"]).union(set(g_rn["T"])))
    tf_mean = g_tf.set_index("T").reindex(Ts)["mean"].to_numpy()
    tf_ci = g_tf.set_index("T").reindex(Ts)["ci95"].fillna(0.0).to_numpy()
    rn_mean = g_rn.set_index("T").reindex(Ts)["mean"].to_numpy()
    rn_ci = g_rn.set_index("T").reindex(Ts)["ci95"].fillna(0.0).to_numpy()

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    _plot_lines(
        ax, np.array(Ts), (tf_mean, rn_mean), (tf_ci, rn_ci), ("Transformer", "IndiRNN")
    )
    ax.set_xlabel("sequence length T")
    ax.set_ylabel("total grad norm")
    ax.set_title(f"Gradient norms vs T @ L={L_fixed}  ({agg_label})")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path = out_dir / f"grad_total_vs_T_{agg_label.replace(' ', '')}_{exp_name}.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[saved] {out_path}")


def plot_blocks(
    df: pd.DataFrame,
    varying: str,  # 'L' or 'T'
    fixed_val: int,  # T if varying=='L', else L
    out_dir: Path,
    agg_label: str,
):
    """
    Per-parameter-block grad norms. Creates one figure for each model.
    """
    assert varying in ("L", "T")
    if varying == "L":
        dfx = df[df["T"] == fixed_val]
        xname = "L"
        title_suffix = f"@ T={fixed_val}"
        fname_suffix = f"vs_L_{agg_label.replace(' ', '')}"
    else:
        dfx = df[df["L"] == fixed_val]
        xname = "T"
        title_suffix = f"@ L={fixed_val}"
        fname_suffix = f"vs_T_{agg_label.replace(' ', '')}"

    # Transformer blocks
    tf = dfx[dfx["model"] == "Transformer"].copy()
    if not tf.empty:
        cols = [("grad_W", "W"), ("grad_U", "U"), ("grad_c", "c")]
        fig, ax = plt.subplots(figsize=(6.2, 4.4))
        for col, lab in cols:
            if col in tf.columns:
                stats = _agg_frame(tf, [xname], col)
                xs = stats[xname].to_numpy()
                ax.errorbar(
                    xs,
                    stats["mean"].to_numpy(),
                    yerr=stats["ci95"].to_numpy(),
                    marker="o",
                    linestyle="-",
                    capsize=3,
                    label=lab,
                )
        ax.set_xlabel(xname)
        ax.set_ylabel("grad norm")
        ax.set_title(f"Transformer blocks {title_suffix}  ({agg_label})")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(title="block")
        fig.tight_layout()
        out = out_dir / f"blocks_transformer_{fname_suffix}.png"
        fig.savefig(out, dpi=180)
        plt.close(fig)
        print(f"[saved] {out}")

    # IndiRNN blocks
    rn = dfx[dfx["model"] == "IndiRNN"].copy()
    if not rn.empty:
        cols = [("grad_W_in", "W_in"), ("grad_u", "u"), ("grad_c", "c")]
        fig, ax = plt.subplots(figsize=(6.2, 4.4))
        for col, lab in cols:
            if col in rn.columns:
                stats = _agg_frame(rn, [xname], col)
                xs = stats[xname].to_numpy()
                ax.errorbar(
                    xs,
                    stats["mean"].to_numpy(),
                    yerr=stats["ci95"].to_numpy(),
                    marker="o",
                    linestyle="-",
                    capsize=3,
                    label=lab,
                )
        ax.set_xlabel(xname)
        ax.set_ylabel("grad norm")
        ax.set_title(f"IndiRNN blocks {title_suffix}  ({agg_label})")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(title="block")
        fig.tight_layout()
        out = out_dir / f"blocks_indirnn_{fname_suffix}.png"
        fig.savefig(out, dpi=180)
        plt.close(fig)
        print(f"[saved] {out}")


def main():
    exp_name = "gradnorms_Lsweep_high_alpha_high_noise_normalize_cols_no_masking"
    p = argparse.ArgumentParser()
    p.add_argument(
        "--csv",
        required=False,
        default=f"../runs/{exp_name}.csv",
        type=str,
        help="CSV from exp_ar_gradnorms.py",
    )
    p.add_argument(
        "--plot",
        required=False,
        default="L",
        choices=["L", "T"],
        help="Sweep axis: 'L' (vary L at fixed T) or 'T' (vary T at fixed L)",
    )
    p.add_argument("--T", type=int, default=128, help="Fix this T when plot='L'")
    p.add_argument("--L", type=int, default=None, help="Fix this L when plot='T'")
    p.add_argument(
        "--agg",
        nargs="+",
        default=["init"],
        choices=["init", "early-avg"],
        help="Aggregate at init and/or early-avg",
    )
    p.add_argument(
        "--early-K",
        type=int,
        default=50,
        help="Number of steps to average for 'early-avg'",
    )
    p.add_argument("--blocks", action="store_true", help="Also plot per-block figures")
    p.add_argument("--out", type=str, default="runs/figs")
    args = p.parse_args()

    out_dir = _ensure_out_dir(args.out)
    df = pd.read_csv(args.csv)

    # Sanity
    need_fixed = "T" if args.plot == "L" else "L"
    fixed_val = getattr(args, need_fixed)
    if fixed_val is None:
        raise SystemExit(f"--{need_fixed} must be provided when --plot={args.plot}")

    for agg in args.agg:
        dfa = _filter_agg(df, agg, args.early_K)
        label = "init" if agg == "init" else f"earlyK={args.early_K}"

        if args.plot == "L":
            plot_total_vs_L(
                dfa,
                T_fixed=fixed_val,
                out_dir=out_dir,
                agg_label=label,
                exp_name=exp_name,
            )
            if args.blocks:
                plot_blocks(
                    dfa,
                    varying="L",
                    fixed_val=fixed_val,
                    out_dir=out_dir,
                    agg_label=label,
                )
        else:
            plot_total_vs_T(
                dfa,
                L_fixed=fixed_val,
                out_dir=out_dir,
                agg_label=label,
                exp_name=exp_name,
            )
            if args.blocks:
                plot_blocks(
                    dfa,
                    varying="T",
                    fixed_val=fixed_val,
                    out_dir=out_dir,
                    agg_label=label,
                )


if __name__ == "__main__":
    main()
