#!/usr/bin/env python3
# scripts/plot_width_sweep.py
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ci95(series: pd.Series) -> float:
    n = series.count()
    if n <= 1:
        return np.nan
    return 1.96 * series.std(ddof=1) / math.sqrt(n)


def fit_power_law(m_vals: np.ndarray, y_vals: np.ndarray):
    # Fit y ≈ A * m^alpha  => log y = alpha * log m + log A
    mask = (m_vals > 0) & (y_vals > 0) & np.isfinite(y_vals)
    if mask.sum() < 2:
        return np.nan, np.nan
    x = np.log(m_vals[mask])
    y = np.log(y_vals[mask])
    alpha, logA = np.polyfit(x, y, 1)
    return alpha, np.exp(logA)


def plot_width_sweep(
    df: pd.DataFrame, metric: str, outdir: Path, title_suffix: str = ""
):
    # Expect columns: m, seed, <metric>
    assert metric in df.columns, f"Metric '{metric}' not found in CSV."
    grouped = (
        df.groupby("m")[metric].agg(["mean", "median", ci95, "count"]).reset_index()
    )
    grouped = grouped.sort_values("m")

    # Slope fit on medians (more robust), fallback to mean if needed
    y_for_fit = grouped["median"].to_numpy()
    m_for_fit = grouped["m"].to_numpy().astype(float)
    alpha, A = fit_power_law(m_for_fit, y_for_fit)

    fig = plt.figure(figsize=(6, 4))
    ax = fig.gca()
    # Error bars around mean (95% CI); dots for medians
    ax.errorbar(
        grouped["m"],
        grouped["mean"],
        yerr=grouped["ci95"],
        fmt="o",
        capsize=3,
        label=f"{metric} (mean ±95% CI)",
    )
    ax.plot(grouped["m"], grouped["median"], "s", label=f"{metric} (median)")

    # Reference fitted line (through median curve)
    if np.isfinite(alpha) and np.isfinite(A):
        ref_x = np.array([grouped["m"].min(), grouped["m"].max()], dtype=float)
        ref_y = A * (ref_x**alpha)
        ax.plot(ref_x, ref_y, "--", label=f"fit: ~ m^{alpha:.2f}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("width m (log)")
    ax.set_ylabel(metric.replace("_", " ") + " (log)")
    title = f"Width sweep: {metric}"
    if title_suffix:
        title += f" — {title_suffix}"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", ls=":", alpha=0.5)

    outdir.mkdir(parents=True, exist_ok=True)
    stem = f"width_sweep_{metric}"
    fig.tight_layout()
    fig.savefig(outdir / f"{stem}.png", dpi=200)
    fig.savefig(outdir / f"{stem}.pdf")
    plt.close(fig)

    # Print a tiny summary
    print(f"[{metric}] fitted slope alpha ≈ {alpha:.3f} (expect ~ -0.5), A ≈ {A:.3g}")
    print(f"Saved: {outdir / f'{stem}.png'} and .pdf")


def main():
    parser = argparse.ArgumentParser(description="Plot width sweep results.")
    parser.add_argument(
        "--csv",
        type=str,
        default="../runs/width_sweep_kernel_proj.csv",
        help="Path to runs/width_sweep.csv (from experiment.py).",
    )
    parser.add_argument(
        "--outdir", type=str, default="../plots", help="Directory to save plots."
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="min_train_loss",
        choices=[
            "min_val_loss",
            "min_train_loss",
            "final_val_loss",
            "final_train_loss",
        ],
        help="Which metric to plot on y-axis.",
    )
    parser.add_argument(
        "--filter",
        type=str,
        nargs="*",
        default=[],
        help="Optional key=val filters, e.g. teacher_activation=tanh lr=0.001",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    df = pd.read_csv(args.csv)

    # Optional filtering: only rows matching all key=val
    for kv in args.filter:
        if "=" not in kv:
            raise ValueError(f"Bad filter '{kv}', use key=val.")
        k, v = kv.split("=", 1)
        if k not in df.columns:
            raise ValueError(f"Filter key '{k}' not in CSV columns.")
        # Try numeric compare if possible
        try:
            v_num = float(v)
            df = df[np.isclose(df[k].astype(float), v_num)]
        except Exception:
            df = df[df[k].astype(str) == v]

    # Keep only needed columns
    needed = {"m", args.metric}
    if "seed" in df.columns:
        needed.add("seed")
    df = df[list(needed)].copy()

    # If there is no seed column, add a dummy one (so groupby semantics are consistent)
    if "seed" not in df.columns:
        df["seed"] = 0

    # Plot
    title_suffix = " ".join(args.filter)
    plot_width_sweep(df, args.metric, outdir, title_suffix=title_suffix)


if __name__ == "__main__":
    main()
