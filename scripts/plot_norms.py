# --- Max grad norm plotting (AISTATS-style) -------------------------------
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatter, MaxNLocator
from matplotlib.offsetbox import AnchoredText


# -------------------------------------------------------------------------
# If you already have this in your codebase, you can reuse your existing version.
@dataclass(frozen=True)
class PlotStyle:
    """Visual defaults tuned for two-column conference papers (e.g., AISTATS)."""

    width_in: float = 3.25  # single-column width
    height_in: float = 2.5
    font_size: float = 8.5
    dpi: int = 300
    legend_outside: bool = False  # put legend outside the axes on the right


# -------------------------------------------------------------------------


def _fit_power_law(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Fit y ≈ A * x^alpha on positive x,y; returns (alpha, A)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return np.nan, np.nan
    alpha, logA = np.polyfit(np.log(x[m]), np.log(y[m]), 1)
    return float(alpha), float(np.exp(logA))


def _prep_summary_for_model(
    summary_df: pd.DataFrame,
    model_label: str,
    L_values: List[int],
) -> pd.DataFrame:
    """
    Normalize a summary table with columns:
    'df', 'n_blocks', 'mean_block_max', 'lower_95', 'upper_95', 'sd_block_max'
    into a long form with columns: ['model','L','mean','lower','upper'].
    """
    req = {"df", "n_blocks", "mean_block_max", "sd_block_max"}
    missing = req - set(summary_df.columns)
    if missing:
        raise KeyError(f"{model_label}: missing columns {sorted(missing)}")

    work = summary_df.copy()
    # Order by the integer suffix in 'df' (df_0, df_1, ...)
    order = (
        work["df"].astype(str).str.extract(r"(\d+)")[0].astype(int)
        if "df" in work.columns
        else pd.Series(np.arange(len(work)))
    )
    work = work.assign(_order=order).sort_values("_order").reset_index(drop=True)

    # Map to L
    if len(L_values) < len(work):
        raise ValueError(
            f"{model_label}: got {len(work)} rows but only {len(L_values)} L values."
        )
    work["L"] = np.array(L_values[: len(work)], dtype=float)

    # Prefer provided CI bounds; otherwise compute from sd and n (z=1.96)
    have_bounds = {"lower_95", "upper_95"}.issubset(work.columns)
    if (
        have_bounds
        and work["lower_95"].notna().all()
        and work["upper_95"].notna().all()
    ):
        lower = work["lower_95"].to_numpy(dtype=float)
        upper = work["upper_95"].to_numpy(dtype=float)
    else:
        z = 1.96
        n = work["n_blocks"].to_numpy(dtype=float)
        sd = work["sd_block_max"].to_numpy(dtype=float)
        half = z * sd / np.sqrt(np.maximum(n, 1.0))
        mean = work["mean_block_max"].to_numpy(dtype=float)
        lower = mean - half
        upper = mean + half

    out = pd.DataFrame(
        {
            "model": model_label,
            "L": work["L"].to_numpy(dtype=float),
            "mean": work["mean_block_max"].to_numpy(dtype=float),
            "lower": lower.astype(float),
            "upper": upper.astype(float),
        }
    )

    # Guard for log-scale plotting
    tiny = max(1e-12, float(np.nanmin(out["mean"])) * 1e-6)
    out["lower"] = np.maximum(out["lower"], tiny)

    return out[["model", "L", "mean", "lower", "upper"]]


def plot_max_grad_norms(
    summaries: Dict[str, pd.DataFrame],
    *,
    L_values: List[int] = (4, 16, 72, 96),
    outdir: Path = Path("figs"),
    exp_name: str = "gradnorms",
    title_suffix: str = "",
    style: PlotStyle = PlotStyle(),
    logx: bool = False,
    logy: bool = True,
    annotate_alpha: bool = True,
    save_svg: bool = False,
) -> Tuple[Path, Path]:
    """
    Plot mean ±95% CI for max gradient norms vs. context length L
    for multiple models (e.g., {'RNN': summary_rnn, 'Transformer': summary_tf}).

    Returns
    -------
    (png_path, pdf_path)
    """
    # Prepare combined long-form table
    pieces = []
    for label, df in summaries.items():
        pieces.append(_prep_summary_for_model(df, label, list(L_values)))
    plot_df = pd.concat(pieces, ignore_index=True)

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

        # Nice, distinct markers; colors come from Matplotlib defaults
        markers = ["o", "s", "D", "^", "v", "P", "X"]
        for i, (label, _) in enumerate(summaries.items()):
            sub = plot_df[plot_df["model"] == label].sort_values("L")
            (line,) = ax.plot(
                sub["L"].to_numpy(),
                sub["mean"].to_numpy(),
                marker=markers[i % len(markers)],
                linestyle="-",
                label=label,
                zorder=3,
            )
            col = line.get_color()
            ax.fill_between(
                sub["L"].to_numpy(),
                sub["lower"].to_numpy(),
                sub["upper"].to_numpy(),
                alpha=0.20,
                linewidth=0,
                color=col,
                zorder=2,
            )

        # Axes + scales
        if logx:
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(LogLocator(base=10))
            ax.xaxis.set_minor_locator(
                LogLocator(base=10, subs=tuple(np.arange(2, 10) * 0.1))
            )
        else:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_xticks(list(L_values))

        if logy:
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
        ax.set_xlabel(r"lag $L$")
        ax.set_ylabel("max gradient norm")
        # Optional tiny title, often omitted in papers:
        # title = "Max gradient norms vs. context length"
        # if title_suffix:
        #     title += f" — {title_suffix}"
        # ax.set_title(title)

        # Legend
        if style.legend_outside:
            ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
        else:
            ax.legend(frameon=False, loc="best")

        # Optional: small anchored power-law slope fits (per model)
        if annotate_alpha:
            lines = []
            for label, _ in summaries.items():
                sub = plot_df[plot_df["model"] == label].sort_values("L")
                alpha, _A = _fit_power_law(sub["L"].to_numpy(), sub["mean"].to_numpy())
                if np.isfinite(alpha):
                    lines.append(f"{label}: $\\alpha$={alpha:+.2f}")
            if lines:
                ax.add_artist(
                    AnchoredText("; ".join(lines), loc="lower left", frameon=False)
                )

        # Save
        outdir.mkdir(parents=True, exist_ok=True)
        stem = f"{exp_name}_max_grad_norms"
        pdf_path = outdir / f"{stem}.pdf"
        png_path = outdir / f"{stem}.png"
        fig.savefig(pdf_path, bbox_inches="tight")
        fig.savefig(png_path, dpi=style.dpi, bbox_inches="tight")
        if save_svg:
            fig.savefig(outdir / f"{stem}.svg", bbox_inches="tight")
        plt.close(fig)

    return png_path, pdf_path


def block_max(series: pd.Series, block_size: int = 4000) -> pd.Series:
    """
    Compute block-wise maximum for a Series (ignores NaNs).
    Final (possibly partial) block is included.
    Returns a Series indexed by block number starting at 0.
    """
    n = len(series)
    if n == 0:
        return pd.Series(dtype=float)
    block_ids = np.floor_divide(np.arange(n), block_size)
    return (
        series.reset_index(drop=True)
        .groupby(block_ids)
        .agg(lambda x: x.max(skipna=True))
    )


def per_df_blockmax_mean_ci(
    dfs: Union[List[pd.DataFrame], Dict[str, pd.DataFrame]],
    column: str,
    block_size: int = 4000,
    labels: List[str] = None,
    z: float = 1.96,  # normal approx for 95% CI
) -> pd.DataFrame:
    """
    For each DF:
      - take block-wise maxima of `column` with block_size,
      - compute the mean across blocks,
      - compute 95% CI for the mean (mean ± z * sd/sqrt(n_blocks)).
    Returns a tidy DataFrame with one row per DF.
    """
    if isinstance(dfs, dict):
        labels_ = list(dfs.keys())
        df_list = list(dfs.values())
    else:
        df_list = list(dfs)
        labels_ = (
            labels if labels is not None else [f"df_{i}" for i in range(len(df_list))]
        )

    rows = []
    for name, df in zip(labels_, df_list):
        if column not in df.columns:
            raise KeyError(f"DataFrame '{name}' is missing column '{column}'.")
        bm = block_max(df[column], block_size)
        n = int(bm.count())
        mean = float(bm.mean()) if n > 0 else np.nan
        std = float(bm.std(ddof=1)) if n > 1 else np.nan
        se = std / np.sqrt(n) if (n > 1 and np.isfinite(std)) else np.nan
        lower = mean - z * se if np.isfinite(se) else np.nan
        upper = mean + z * se if np.isfinite(se) else np.nan
        rows.append(
            {
                "df": name,
                "n_blocks": n,
                "mean_block_max": mean,
                "lower_95": lower,
                "upper_95": upper,
                "sd_block_max": std,
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    Ls = [4, 16, 72, 96]
    df_rnns = []
    df_tfs = []
    for L in Ls:
        df_rnn = pd.read_csv(
            f"/Users/enesarda/repos/transformer-ntk/runs/arL-sincos-with-grads-dynamic-short-rad_rnn_curve_L{L}.csv"
        )
        df_rnns.append(df_rnn)

        df_tf = pd.read_csv(
            f"/Users/enesarda/repos/transformer-ntk/runs/arL-sincos-with-grads-dynamic-short-rad_tf_curve_L{L}.csv"
        )
        df_tfs.append(df_tf)

    summary_rnn = per_df_blockmax_mean_ci(df_rnns, column="grad_u", block_size=500)
    summary_tf = per_df_blockmax_mean_ci(df_tfs, column="grad_w", block_size=500)
    summaries = {"IndRNN": summary_rnn, "Transformer": summary_tf}

    plot_max_grad_norms(
        summaries,
        L_values=[4, 16, 72, 96],
        outdir=Path("figs"),
        exp_name="aistats",
        title_suffix="YourDataset",
        style=PlotStyle(legend_outside=False),
        logx=False,  # keep x linear (only 4 points); set True if you prefer log x
        logy=True,  # recommended for norms spanning orders of magnitude
        annotate_alpha=False,
        save_svg=False,
    )
