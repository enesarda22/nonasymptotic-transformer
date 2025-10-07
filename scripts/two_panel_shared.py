# two_panel_shared_x.py
from __future__ import annotations
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatter, MaxNLocator
from matplotlib.offsetbox import AnchoredText

from scripts.plot_norms import per_df_blockmax_mean_ci


# ----------------------- Shared styling ------------------------------------
@dataclass(frozen=True)
class PlotStyle:
    """Visual defaults tuned for two-column conference papers (e.g., AISTATS)."""

    width_in: float = 3.25  # we'll set height==width to make a square figure below
    height_in: float = 3.25
    font_size: float = 8.5
    dpi: int = 300
    legend_outside: bool = False  # not used for 2-panel (we keep legend inside)


# ----------------------- Small helpers -------------------------------------
def _fit_power_law(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Fit y ≈ A * x^alpha on positive x,y; returns (alpha, A)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return np.nan, np.nan
    alpha, logA = np.polyfit(np.log(x[m]), np.log(y[m]), 1)
    return float(alpha), float(np.exp(logA))


def _parse_value(val: str):
    """Best-effort cast for filter values."""
    try:
        ival = int(val)
        return ival
    except Exception:
        pass
    try:
        fval = float(val)
        return fval
    except Exception:
        pass
    low = val.strip().lower()
    if low in {"true", "t", "yes", "y"}:
        return True
    if low in {"false", "f", "no", "n"}:
        return False
    return val


def _apply_filters(df: pd.DataFrame, filters: Iterable[str]) -> pd.DataFrame:
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


# ----------------------- Grad norms prep (from your earlier code) ----------
def _prep_summary_for_model(
    summary_df: pd.DataFrame,
    model_label: str,
    L_values: List[int],
) -> pd.DataFrame:
    """
    Input columns:
      'df', 'n_blocks', 'mean_block_max', 'lower_95', 'upper_95', 'sd_block_max'
    Output columns:
      ['model', 'L', 'mean', 'lower', 'upper']
    """
    req = {"df", "n_blocks", "mean_block_max", "sd_block_max"}
    missing = req - set(summary_df.columns)
    if missing:
        raise KeyError(f"{model_label}: missing columns {sorted(missing)}")

    work = summary_df.copy()
    order = (
        work["df"].astype(str).str.extract(r"(\d+)")[0].astype(int)
        if "df" in work.columns
        else pd.Series(np.arange(len(work)))
    )
    work = work.assign(_order=order).sort_values("_order").reset_index(drop=True)

    if len(L_values) < len(work):
        raise ValueError(
            f"{model_label}: got {len(work)} rows but only {len(L_values)} L values."
        )
    work["L"] = np.array(L_values[: len(work)], dtype=float)

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

    tiny = max(1e-12, float(np.nanmin(out["mean"])) * 1e-6)
    out["lower"] = np.maximum(out["lower"], tiny)

    return out[["model", "L", "mean", "lower", "upper"]]


# ----------------------- DRAW on a given Axes: metric vs L ------------------
def draw_metric_vs_L_on_ax(
    ax: plt.Axes,
    sweep_df: pd.DataFrame,
    *,
    metric: str,
    agg: str = "mean",
    err: str = "sem",  # "none" | "sem" | "std"
    filters: Iterable[str] = (),
    yscale: str = "linear",  # "linear" | "log"
    annotate_alpha: bool = True,
) -> None:
    """
    Draw model-wise center (mean/median) ± band (95% CI from SEM or ±1 SD) over L.
    Does NOT create its own figure; draws onto `ax`.
    """
    df = _apply_filters(sweep_df, filters)
    needed = {"model", "L", metric}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    gb = df.groupby(["model", "L"])[metric]
    center = gb.mean() if agg == "mean" else gb.median()

    if err == "none":
        half = None
    else:
        count = gb.count()
        sd = gb.std(ddof=1)
        if err == "sem":
            sem = sd / np.sqrt(count.clip(lower=1))
            half = 1.96 * sem
        elif err == "std":
            half = sd
        else:
            raise ValueError(f"Unknown err option: {err}")

    cent_df = (
        center.reset_index()
        .rename(columns={metric: "center"})
        .sort_values(["model", "L"])
    )
    err_df = (
        half.reset_index().rename(columns={metric: "half"}).sort_values(["model", "L"])
        if half is not None
        else None
    )

    markers = ["o", "s", "D", "^", "v", "P", "X"]
    models = list(cent_df["model"].unique())

    for i, model in enumerate(models):
        sub = cent_df[cent_df["model"] == model].sort_values("L")
        x = sub["L"].to_numpy()
        y = sub["center"].to_numpy()
        (line,) = ax.plot(
            x, y, marker=markers[i % len(markers)], linestyle="-", label=model, zorder=3
        )
        col = line.get_color()

        if err_df is not None:
            e = err_df[err_df["model"] == model].set_index("L").reindex(sub["L"])
            halfvals = e["half"].to_numpy()
            if yscale == "log":
                tiny = (
                    max(1e-12, float(np.nanmin(y[y > 0])) * 1e-6)
                    if np.any(y > 0)
                    else 1e-12
                )
                lower = np.maximum(y - halfvals, tiny)
            else:
                lower = y - halfvals
            upper = y + halfvals
            ax.fill_between(
                x, lower, upper, alpha=0.20, linewidth=0, color=col, zorder=2
            )

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    xticks = np.unique(cent_df["L"].to_numpy())
    try:
        ax.set_xticks(xticks)
    except Exception:
        pass

    if yscale == "log":
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10))
        ax.yaxis.set_minor_locator(
            LogLocator(base=10, subs=tuple(np.arange(2, 10) * 0.1))
        )
        ax.yaxis.set_major_formatter(LogFormatter(base=10))

    ax.grid(which="major", alpha=0.30)
    ax.grid(which="minor", alpha=0.15, linewidth=0.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.margins(x=0.03, y=0.05)

    # ax.set_ylabel(metric.replace("_", " "))
    ax.set_ylabel("min val loss")

    if annotate_alpha:
        lines = []
        for model in models:
            sub = cent_df[cent_df["model"] == model]
            alpha, _A = _fit_power_law(sub["L"].to_numpy(), sub["center"].to_numpy())
            if np.isfinite(alpha):
                lines.append(f"{model}: $\\alpha$={alpha:+.2f}")
        if lines:
            ax.add_artist(
                AnchoredText("; ".join(lines), loc="lower left", frameon=False)
            )


# ---------------- DRAW on a given Axes: grad norms vs L --------------------
def draw_max_grad_norms_on_ax(
    ax: plt.Axes,
    summaries: Dict[
        str, pd.DataFrame
    ],  # e.g., {"RNN": summary_rnn, "Transformer": summary_tf}
    *,
    L_values: List[int] = (4, 16, 72, 96),
    logy: bool = True,
    annotate_alpha: bool = True,
) -> None:
    """
    Plot mean ±95% CI bands for max gradient norms vs L for each model into `ax`.
    """
    pieces = []
    for label, df in summaries.items():
        pieces.append(_prep_summary_for_model(df, label, list(L_values)))
    plot_df = pd.concat(pieces, ignore_index=True)

    markers = ["o", "s", "D", "^", "v", "P", "X"]
    models = list(plot_df["model"].unique())

    for i, model in enumerate(models):
        sub = plot_df[plot_df["model"] == model].sort_values("L")
        (line,) = ax.plot(
            sub["L"].to_numpy(),
            sub["mean"].to_numpy(),
            marker=markers[i % len(markers)],
            linestyle="-",
            label=model,
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
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.margins(x=0.03, y=0.05)

    ax.set_ylabel("max Jacobian norm")

    if annotate_alpha:
        lines = []
        for model in models:
            sub = plot_df[plot_df["model"] == model].sort_values("L")
            alpha, _A = _fit_power_law(sub["L"].to_numpy(), sub["mean"].to_numpy())
            if np.isfinite(alpha):
                lines.append(f"{model}: $\\alpha$={alpha:+.2f}")
        if lines:
            ax.add_artist(
                AnchoredText("; ".join(lines), loc="lower left", frameon=False)
            )


# ----------------------- Orchestrator: square 2-panel ----------------------
def plot_two_panel_square_shared_x(
    *,
    summaries: Dict[str, pd.DataFrame],  # for grad norms (RNN/TF tables)
    sweep_df: pd.DataFrame,  # CSV-loaded sweep for metric panel
    metric: str = "min_val_mse",
    agg: str = "mean",
    err: str = "sem",
    filters: Iterable[str] = (),
    L_values: List[int] = (4, 16, 72, 96),
    yscale_metric: str = "linear",  # 'linear' or 'log'
    yscale_gradlog: bool = True,  # log y for grad norms
    outdir: Path = Path("../plots"),
    exp_name: str = "aistats_2panel",
    style: PlotStyle = PlotStyle(width_in=3.25, height_in=3.25),
    save_svg: bool = False,
    panel_titles: Tuple[str, str] = ("validation MSE", "max gradient norms"),
) -> Tuple[Path, Path]:
    """
    Build a square figure with two rows (shared x), top = metric vs L, bottom = grad norms vs L.
    Returns (png_path, pdf_path).
    """
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
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }

    with plt.rc_context(rc):
        fig, (ax_bot, ax_top) = plt.subplots(
            nrows=2,
            ncols=1,
            sharex=True,
            figsize=(style.width_in, style.height_in),  # square
            constrained_layout=True,
            dpi=style.dpi,
        )

        # Top: metric vs L (no legend here; we’ll place it on the bottom axis)
        draw_metric_vs_L_on_ax(
            ax_top,
            sweep_df,
            metric=metric,
            agg=agg,
            err=err,
            filters=filters,
            yscale=yscale_metric,
            annotate_alpha=False,
        )
        # ax_top.set_title(panel_titles[0], loc="left", pad=2.0)
        # Hide x label on top; ticks will be shared
        ax_bot.set_xlabel("")
        plt.setp(ax_bot.get_xticklabels(), visible=False)

        # Bottom: grad norms vs L (legend here)
        draw_max_grad_norms_on_ax(
            ax_bot,
            summaries,
            L_values=L_values,
            logy=yscale_gradlog,
            annotate_alpha=False,
        )
        # ax_bot.set_title(panel_titles[1], loc="left", pad=2.0)
        ax_top.set_xlabel(r"lag $L$")

        # Single legend on the bottom axis (models are the same across panels)
        handles, labels = ax_bot.get_legend_handles_labels()
        ax_top.legend(handles, labels, frameon=False, loc="best")
        ax_bot.legend(handles, labels, frameon=False, loc="best")

        fig.align_ylabels([ax_top, ax_bot])

        # Save
        outdir.mkdir(parents=True, exist_ok=True)
        stem = f"{exp_name}_{metric}_and_gradnorms"
        pdf_path = outdir / f"{stem}.pdf"
        png_path = outdir / f"{stem}.png"
        fig.savefig(pdf_path, bbox_inches="tight")
        fig.savefig(png_path, dpi=style.dpi, bbox_inches="tight")
        if save_svg:
            fig.savefig(outdir / f"{stem}.svg", bbox_inches="tight")
        plt.close(fig)

    return png_path, pdf_path


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

    sweep_df = pd.read_csv("../runs/arL-sincos-with-grads-dynamic-long-rad.csv")

    png_path, pdf_path = plot_two_panel_square_shared_x(
        summaries=summaries,
        sweep_df=sweep_df,
        metric="min_val_mse",
        agg="mean",
        err="sem",  # 95% CI band; use "std" for ±1 SD or "none"
        filters=["activation=tanh", "context_policy=fixed", "m=64"],
        L_values=[4, 16, 72, 96],
        yscale_metric="log",  # or "linear" if you prefer
        yscale_gradlog=True,  # log y for grad norms
        outdir=Path("../plots"),
        exp_name="aistats_square",
        style=PlotStyle(width_in=3.25, height_in=3.0),  # square figure (single-column)
        save_svg=True,
        # panel_titles=("validation MSE", "max gradient norms"),
    )

    print("Saved:", png_path, pdf_path)
