#!/usr/bin/env python3
# scripts/plot_width_sweep.py
from __future__ import annotations
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Callable, Optional, Tuple
import warnings

from matplotlib.ticker import LogLocator, LogFormatter
from matplotlib.offsetbox import AnchoredText


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


@dataclass(frozen=True)
class PlotStyle:
    """Visual defaults tuned for two-column conference papers (e.g., AISTATS)."""

    width_in: float = 3.25  # single-column width
    height_in: float = 2.5
    font_size: float = 8.5
    dpi: int = 300
    legend_outside: bool = False  # put legend outside the axes on the right


def plot_width_sweep(
    df: pd.DataFrame,
    metric: str,
    outdir: Path,
    title_suffix: str = "",
    exp_name: str = "temp",
    *,
    ci_fn: Callable[[pd.Series], float] = ci95,
    fit_fn: Callable[[np.ndarray, np.ndarray], Tuple[float, float]] = fit_power_law,
    expected_alpha: Optional[float] = None,
    style: PlotStyle = PlotStyle(),
    save_svg: bool = False,
) -> Tuple[float, float, Path]:
    """
    Plot mean ±95% CI band and median of `metric` as a function of model width `m`
    on log–log axes, with a power-law fit drawn as a reference.

    Parameters
    ----------
    df : DataFrame with columns: 'm', 'seed', and `metric`
    metric : str
        Column to aggregate and plot.
    outdir : Path
        Directory where figures are saved.
    title_suffix : str
        Extra string appended to the title (e.g., dataset name).
    exp_name : str
        Stem for output filenames.
    ci_fn : callable
        Aggregator returning the half-width of a 95% CI for the mean.
    fit_fn : callable
        Returns (alpha, A) for y ≈ A * m**alpha using (m, median(metric)).
    expected_alpha : float or None
        If provided, printed/annotated for quick comparison (e.g., -0.5).
    style : PlotStyle
        Visual tuning (size, fonts, legend placement, DPI).
    save_svg : bool
        Also save a vector .svg (optional).

    Returns
    -------
    alpha, A, pdf_path
    """
    if metric not in df.columns:
        raise KeyError(f"Metric '{metric}' not found. Columns: {list(df.columns)}")

    # Clean + guard for log-scale plotting
    work = df[["m", metric]].copy()
    work["m"] = pd.to_numeric(work["m"], errors="coerce")
    work[metric] = pd.to_numeric(work[metric], errors="coerce")
    work = work.dropna()

    # Log scale requires positive values
    nonpos = int((work[metric] <= 0).sum())
    if nonpos:
        warnings.warn(
            f"Dropping {nonpos} non-positive '{metric}' rows; log-scale requires y > 0."
        )
        work = work[work[metric] > 0]

    grouped = (
        work.groupby("m", as_index=False)[metric]
        .agg(mean="mean", median="median", ci95=ci_fn, count="count")
        .sort_values("m")
    )
    if grouped.empty:
        raise ValueError("No data left to plot after cleaning.")

    # Fit power law to medians (robust)
    m_for_fit = grouped["m"].astype(float).to_numpy()
    y_for_fit = grouped["median"].to_numpy()
    alpha, A = fit_fn(m_for_fit, y_for_fit)

    # Local, paper-friendly style (does not pollute global rcParams)
    rc = {
        "font.size": style.font_size,
        "axes.labelsize": style.font_size,
        "axes.titlesize": style.font_size,
        "xtick.labelsize": style.font_size - 1,
        "ytick.labelsize": style.font_size - 1,
        "legend.fontsize": style.font_size - 1,
        "axes.linewidth": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        # Embed vector text in PDFs for crisp results
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }

    with plt.rc_context(rc):
        fig, ax = plt.subplots(
            figsize=(style.width_in, style.height_in),
            constrained_layout=True,
            dpi=style.dpi,
        )

        # Mean + 95% CI as band (more legible than capped error bars on log scale)
        (mean_line,) = ax.plot(
            grouped["m"].to_numpy(),
            grouped["mean"].to_numpy(),
            marker="o",
            linestyle="-",
            label="mean ±95% CI",
            zorder=3,
        )
        col = mean_line.get_color()
        y_mean = grouped["mean"].to_numpy()
        y_ci = grouped["ci95"].to_numpy()

        # Clip CI lower bound away from zero for log scale
        tiny = max(1e-12, float(np.nanmin(y_mean)) * 1e-6)
        lower = np.maximum(y_mean - y_ci, tiny)
        upper = y_mean + y_ci
        ax.fill_between(
            grouped["m"].to_numpy(),
            lower,
            upper,
            alpha=0.20,
            linewidth=0,
            zorder=2,
            color=col,
        )

        # Medians as dashed squares
        ax.plot(
            grouped["m"].to_numpy(),
            grouped["median"].to_numpy(),
            marker="s",
            linestyle="--",
            label="median",
            zorder=4,
        )

        # Reference power-law fit line
        # if np.isfinite(alpha) and np.isfinite(A):
        #     ref_x = np.geomspace(grouped["m"].min(), grouped["m"].max(), 200)
        #     ref_y = A * (ref_x**alpha)
        #     ax.plot(
        #         ref_x,
        #         ref_y,
        #         linestyle=":",
        #         label=rf"fit: $\sim m^{{{alpha:.2f}}}$",
        #         zorder=1,
        #         color="0.4",
        #     )

        # Axes: log–log with clean ticks and subtle grids
        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.xaxis.set_major_locator(LogLocator(base=10))
        ax.xaxis.set_minor_locator(
            LogLocator(base=10, subs=tuple(np.arange(2, 10) * 0.1))
        )
        ax.yaxis.set_major_locator(LogLocator(base=10))
        ax.yaxis.set_minor_locator(
            LogLocator(base=10, subs=tuple(np.arange(2, 10) * 0.1))
        )
        ax.yaxis.set_major_formatter(LogFormatter(base=10))

        ax.grid(which="major", alpha=0.30)
        ax.grid(which="minor", alpha=0.15, linewidth=0.5)

        # Frames and margins
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.margins(x=0.03, y=0.05)

        # Labels & title
        ax.set_xlabel(r"width $m$")
        ax.set_ylabel("min training loss")
        # title = f"Width sweep: {clean_metric}"
        # if title_suffix:
        #     title += f" — {title_suffix}"
        # ax.set_title(title)

        # Legend placement
        if style.legend_outside:
            ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
        else:
            ax.legend(frameon=False, loc="best")

        # Small anchored slope summary
        # txt = rf"$\alpha$ = {alpha:.3f}"
        txt = rf"fit: $\sim m^{{{alpha:.2f}}}$"
        if expected_alpha is not None:
            txt += rf" (exp. {expected_alpha:+.2f})"
        ax.add_artist(AnchoredText(txt, loc="lower left", frameon=False))

        # Save
        outdir.mkdir(parents=True, exist_ok=True)
        stem = f"{exp_name}_{metric}"
        pdf_path = outdir / f"{stem}.pdf"
        png_path = outdir / f"{stem}.png"

        fig.savefig(pdf_path, bbox_inches="tight")
        fig.savefig(png_path, dpi=style.dpi, bbox_inches="tight")
        if save_svg:
            fig.savefig(outdir / f"{stem}.svg", bbox_inches="tight")
        plt.close(fig)

    # Console summary
    msg = f"[{metric}] fitted slope α ≈ {alpha:.3f}, A ≈ {A:.3g}"
    if expected_alpha is not None:
        msg += f" (expected {expected_alpha:+.3f})"
    print(msg)
    print(f"Saved: {png_path} and {pdf_path}")

    return alpha, A, pdf_path


def main():
    exp_name = "width_sweep_teacher_stepsizex15"

    parser = argparse.ArgumentParser(description="Plot width sweep results.")
    parser.add_argument(
        "--csv",
        type=str,
        default=f"../runs/{exp_name}.csv",
        help="Path to runs/width_sweep.csv (from training_experiment.py).",
    )
    parser.add_argument(
        "--outdir", type=str, default="../plots", help="Directory to save plots."
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="min_train_loss",
        choices=[
            "max_absolute_dist",
            "approximation_l1",
            "linearization_l1",
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
    plot_width_sweep(
        df,
        args.metric,
        outdir,
        title_suffix=title_suffix,
        exp_name=exp_name,
    )


if __name__ == "__main__":
    main()
