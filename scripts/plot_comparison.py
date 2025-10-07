#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot Transformer vs IndiRNN over context length L with AISTATS-style visuals.

Example:
    python plot_vs_L_aistats.py \
        --csv ../runs/arL-sincos-larger-var.csv \
        --metric min_val_mse \
        --agg mean \
        --err sem \
        --yscale log \
        --filter activation=tanh context_policy=fixed m=64 \
        --save ../plots

The figure shows per-model center (mean/median) over seeds with a soft band
representing uncertainty (95% CI from SEM, or ±1 SD if --err std).
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatter, MaxNLocator
from matplotlib.offsetbox import AnchoredText


# ----------------------- Styling (AISTATS) ---------------------------------
@dataclass(frozen=True)
class PlotStyle:
    """Visual defaults tuned for two-column conference papers (e.g., AISTATS)."""

    width_in: float = 3.25  # single-column width
    height_in: float = 2.5
    font_size: float = 8.5
    dpi: int = 300
    legend_outside: bool = False  # put legend outside the axes on the right


# ----------------------- Helpers -------------------------------------------
def _parse_value(val: str):
    """Best-effort cast of filter values: int -> float -> bool -> str."""
    # Try int
    try:
        ival = int(val)
        return ival
    except Exception:
        pass
    # Try float
    try:
        fval = float(val)
        return fval
    except Exception:
        pass
    # Try bool
    low = val.strip().lower()
    if low in {"true", "t", "yes", "y"}:
        return True
    if low in {"false", "f", "no", "n"}:
        return False
    return val


def _apply_filters(df: pd.DataFrame, filters: Iterable[str]) -> pd.DataFrame:
    """Apply simple key=value filters to DataFrame."""
    out = df.copy()
    for f in filters:
        if "=" not in f:
            warnings.warn(f"Ignoring filter without '=': {f!r}")
            continue
        key, raw = f.split("=", 1)
        key = key.strip()
        val = _parse_value(raw.strip())
        if key not in out.columns:
            warnings.warn(f"Filter key {key!r} not in columns; skipping.")
            continue
        out = out[out[key] == val]
    return out


def _fit_power_law(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Fit y ≈ A * x^alpha on positive x,y; returns (alpha, A).
    Used only for tiny lower-left annotation; safe to skip if insufficient data.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return np.nan, np.nan
    alpha, logA = np.polyfit(np.log(x[m]), np.log(y[m]), 1)
    return float(alpha), float(np.exp(logA))


# ----------------------- Core plotting -------------------------------------
def plot_vs_L(
    agg_df: pd.DataFrame,
    *,
    metric: str,
    yscale: str = "linear",
    err_df: Optional[pd.DataFrame] = None,
    outdir: Path = Path("../plots"),
    exp_name: str = "arL-sincos-larger-var",
    style: PlotStyle = PlotStyle(),
    annotate_alpha: bool = True,
    save_svg: bool = False,
) -> Tuple[Path, Path]:
    """
    Draw model-wise curves of center (mean/median) ± uncertainty over L.

    Parameters
    ----------
    agg_df : DataFrame with columns ['model', 'L', 'center'] for the chosen metric
    err_df : Optional DataFrame with columns ['model', 'L', 'half'] giving
             the half-width of the band (e.g., 1.96*SEM or SD). If None -> no band.
    yscale : 'linear' or 'log'.
    """
    # Local, paper-friendly style
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

        # One marker per model (colors from mpl defaults)
        markers = ["o", "s", "D", "^", "v", "P", "X"]
        models = list(agg_df["model"].unique())

        for i, model in enumerate(models):
            sub = agg_df[agg_df["model"] == model].sort_values("L")
            x = sub["L"].to_numpy()
            y = sub["center"].to_numpy()

            (line,) = ax.plot(
                x,
                y,
                marker=markers[i % len(markers)],
                linestyle="-",
                label=model,
                zorder=3,
            )
            col = line.get_color()

            if err_df is not None:
                e = err_df[err_df["model"] == model].set_index("L").reindex(sub["L"])
                half = e["half"].to_numpy()
                if yscale == "log":
                    # Guard lower away from zero for log scale
                    tiny = (
                        max(1e-12, float(np.nanmin(y[y > 0])) * 1e-6)
                        if np.any(y > 0)
                        else 1e-12
                    )
                    lower = np.maximum(y - half, tiny)
                else:
                    lower = y - half
                upper = y + half

                ax.fill_between(
                    x, lower, upper, alpha=0.20, linewidth=0, color=col, zorder=2
                )

        # Axes + scales
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        xticks = np.unique(agg_df["L"].to_numpy())
        try:
            ax.set_xticks(xticks)
        except Exception:
            pass  # fallback to auto

        if yscale == "log":
            ax.set_yscale("log")
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

        # Labels (minimal title for camera-ready)
        ax.set_xlabel(r"context length $L$")
        ax.set_ylabel(metric.replace("_", " "))

        # Legend
        if style.legend_outside:
            ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
        else:
            ax.legend(frameon=False, loc="best")

        # Optional tiny anchored power-law slopes per model
        if annotate_alpha:
            lines = []
            for model in models:
                sub = agg_df[agg_df["model"] == model]
                alpha, _A = _fit_power_law(
                    sub["L"].to_numpy(), sub["center"].to_numpy()
                )
                if np.isfinite(alpha):
                    lines.append(f"{model}: $\\alpha$={alpha:+.2f}")
            if lines:
                ax.add_artist(
                    AnchoredText("; ".join(lines), loc="lower left", frameon=False)
                )

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

    return png_path, pdf_path


# ----------------------- CLI / main ----------------------------------------
def main():
    exp_name = "arL-sincos-with-grads-dynamic-long-rad"

    ap = argparse.ArgumentParser(
        description="Plot Transformer vs IndiRNN over L (AISTATS style)."
    )
    ap.add_argument(
        "--csv",
        type=str,
        default=f"../runs/{exp_name}.csv",
        help="results CSV (from sweep)",
    )
    ap.add_argument(
        "--metric",
        type=str,
        default="min_val_mse",
        choices=["final_val_mse", "min_val_mse", "final_train_mse", "min_train_mse"],
        help="which metric to plot",
    )
    ap.add_argument(
        "--agg",
        type=str,
        default="mean",
        choices=["mean", "median"],
        help="aggregation over seeds",
    )
    ap.add_argument(
        "--err",
        type=str,
        default="sem",
        choices=["none", "sem", "std"],
        help="uncertainty across seeds: 'sem' -> 95%% CI band; 'std' -> ±1 SD; 'none' -> no band",
    )
    ap.add_argument(
        "--filter",
        type=str,
        nargs="*",
        default=[],
        help="filters like key=value (e.g. activation=tanh context_policy=fixed m=64)",
    )
    ap.add_argument("--yscale", type=str, default="linear", choices=["linear", "log"])
    ap.add_argument(
        "--save",
        type=str,
        default="../plots",
        help="directory to save (PNG+PDF; SVG optional). Use empty string to show only.",
    )
    ap.add_argument(
        "--legend-outside",
        action="store_true",
        help="place legend outside the axes on the right",
    )
    ap.add_argument("--svg", action="store_true", help="also save a vector .svg")
    ap.add_argument(
        "--no-alpha",
        action="store_true",
        help="disable small slope annotation",
        default=True,
    )
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Apply simple filters key=value
    df = _apply_filters(df, args.filter)

    # Keep only needed columns
    metric = args.metric
    needed = {"model", "L", metric}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    # Group by (model, L), aggregate across seeds/trials
    gb = df.groupby(["model", "L"])[metric]
    if args.agg == "mean":
        center = gb.mean()
    else:
        center = gb.median()

    # Compute spread as half-width for bands
    # - 'sem': we show 95% CI = 1.96 * SEM
    # - 'std': we show ±1 SD
    if args.err == "none":
        spread_half = None
    else:
        count = gb.count()
        sd = gb.std(ddof=1)
        if args.err == "sem":
            sem = sd / np.sqrt(count.clip(lower=1))
            spread_half = 1.96 * sem
        elif args.err == "std":
            spread_half = sd
        else:
            raise ValueError(f"Unknown err option: {args.err}")

    # Long-form frames for plotting
    cent_df = (
        center.reset_index()
        .rename(columns={metric: "center"})
        .sort_values(["model", "L"])
        .reset_index(drop=True)
    )

    if spread_half is not None:
        err_df = (
            spread_half.reset_index()
            .rename(columns={metric: "half"})
            .sort_values(["model", "L"])
            .reset_index(drop=True)
        )
    else:
        err_df = None

    # Output directory
    outdir = Path(args.save) if args.save else Path(".")
    style = PlotStyle(legend_outside=args.legend_outside)

    png_path, pdf_path = plot_vs_L(
        cent_df,
        metric=metric,
        yscale=args.yscale,
        err_df=err_df,
        outdir=outdir,
        exp_name=exp_name,
        style=style,
        annotate_alpha=not args.no_alpha,
        save_svg=args.svg,
    )

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    if args.svg:
        print(f"Saved: {outdir / (png_path.stem + '.svg')}")


if __name__ == "__main__":
    main()
