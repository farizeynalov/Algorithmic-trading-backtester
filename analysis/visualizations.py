"""
Reusable plotting library for backtest analysis.

All functions return ``(fig, ax)`` or ``(fig, axes)`` — callers decide when to
render or save. No ``plt.show()`` or ``plt.savefig()`` calls live inside this
module; those are the caller's responsibility.

Typical usage::

    from analysis.visualizations import (
        plot_equity_curves, plot_drawdowns, plot_metrics_comparison,
        plot_rolling_metrics, plot_correlation_matrix,
        plot_monthly_returns_heatmap, plot_trade_analysis,
        plot_position_concentration, save_figure, PALETTE,
    )
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import math
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import seaborn as sns

from backtester import drawdown_series
from backtester.engine import BacktestResult
from backtester.metrics import TRADING_DAYS_PER_YEAR


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

PALETTE: dict[str, str] = {
    "momentum":       "#6C63FF",   # purple
    "mean_reversion": "#00D4AA",   # teal
    "ml_signal":      "#FF6B6B",   # coral
    "spy":            "#2C2C2A",   # near-black
    "cash":           "#B4B2A9",   # gray
}

_METRIC_LABELS: dict[str, str] = {
    "annualized_return_pct":  "Ann. Return (%)",
    "sharpe_ratio":           "Sharpe Ratio",
    "sortino_ratio":          "Sortino Ratio",
    "calmar_ratio":           "Calmar Ratio",
    "max_drawdown_pct":       "Max Drawdown (%)",
    "alpha_pct":              "Alpha (%)",
    "total_cost_pct":         "Total Cost (%)",
    "turnover_annual_pct":    "Annual Turnover (%)",
}

_DEFAULT_METRICS: list[str] = list(_METRIC_LABELS.keys())


# ---------------------------------------------------------------------------
# Style setup — called once at module import
# ---------------------------------------------------------------------------

def _setup_style() -> None:
    sns.set_theme(style="whitegrid", font_scale=1.05)
    plt.rcParams.update({
        "figure.dpi":         150,
        "savefig.dpi":        150,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.15,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "font.family":        "sans-serif",
    })


_setup_style()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _get_color(strategy_name: str, fallback: str = "#888888") -> str:
    """
    Resolve a PALETTE colour by stripping underscores from both key and name
    before matching. Handles "MeanReversion" → key "mean_reversion", etc.
    """
    name_clean = strategy_name.lower().replace("_", "")
    return next(
        (v for k, v in PALETTE.items() if k.replace("_", "") in name_clean),
        fallback,
    )


def _resolve_color(strategy_name: str, fallback_idx: int) -> str:
    """Return palette colour or fall back to the default prop-cycle colour."""
    c = _get_color(strategy_name)
    if c == "#888888":
        cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        return cycle[fallback_idx % len(cycle)]
    return c


def _label(name: str, maxlen: int = 25) -> str:
    return name[:maxlen]


def _is_benchmark(result: BacktestResult) -> bool:
    n = result.strategy_name.lower()
    return "buyandhold" in n or ("spy" in n and "momentum" not in n
                                 and "reversion" not in n and "signal" not in n)


# ---------------------------------------------------------------------------
# 1. plot_equity_curves
# ---------------------------------------------------------------------------

def plot_equity_curves(
    results: list[BacktestResult],
    benchmark: BacktestResult,
    figsize: tuple[int, int] = (14, 6),
    title: str = "Strategy equity curves",
) -> tuple[plt.Figure, plt.Axes]:
    """Normalize equity curves to 100 at the common start date and plot.

    Common start = latest first-date across all results and benchmark.
    Strategies are colored via PALETTE; benchmark is PALETTE["spy"]
    dashed; cash is a dotted flat line at 100. Each strategy is
    filled vs benchmark (green where above, red where below, alpha 0.08).

    Args:
        results: Strategy backtest results to plot.
        benchmark: Benchmark result (buy-and-hold SPY).
        figsize: Figure dimensions in inches.
        title: Axes title text.

    Returns:
        ``(fig, ax)`` — a ``(plt.Figure, plt.Axes)`` tuple. The caller
        controls rendering. Does not call ``plt.show()``. Call
        ``plt.close(fig)`` after ``st.pyplot(fig)`` in Streamlit to
        prevent figure accumulation.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Determine common start date
    all_starts = (
        [r.equity_curve.index[0] for r in results]
        + [benchmark.equity_curve.index[0]]
    )
    start_date = max(all_starts)

    # Normalize benchmark
    bench_ec = benchmark.equity_curve
    bench_ec = bench_ec.loc[bench_ec.index >= start_date]
    bench_norm = bench_ec / bench_ec.iloc[0] * 100.0

    # Normalize strategies
    strategy_data: list[tuple] = []
    for i, r in enumerate(results):
        ec = r.equity_curve.loc[r.equity_curve.index >= start_date]
        norm = ec / ec.iloc[0] * 100.0
        color = _resolve_color(r.strategy_name, i)
        ret = norm.iloc[-1] - 100.0
        sign = "+" if ret >= 0 else ""
        lbl = f"{_label(r.strategy_name)} ({sign}{ret:.1f}%)"
        strategy_data.append((norm, lbl, color))

    # Fill between each strategy and benchmark
    for norm, _lbl, color in strategy_data:
        common = norm.index.intersection(bench_norm.index)
        s_al = norm.reindex(common)
        b_al = bench_norm.reindex(common)
        ax.fill_between(
            common, s_al, b_al,
            where=(s_al >= b_al), color="#2ecc71", alpha=0.08,
        )
        ax.fill_between(
            common, s_al, b_al,
            where=(s_al < b_al), color="#e74c3c", alpha=0.08,
        )

    # Cash line (flat at 100)
    ax.plot(
        bench_norm.index,
        np.full(len(bench_norm), 100.0),
        color=PALETTE["cash"],
        linewidth=0.8,
        linestyle=":",
        label="Cash (100)",
        zorder=1,
    )

    # Benchmark line
    bench_ret = bench_norm.iloc[-1] - 100.0
    bench_sign = "+" if bench_ret >= 0 else ""
    bench_lbl = f"{_label(benchmark.strategy_name)} ({bench_sign}{bench_ret:.1f}%)"
    ax.plot(
        bench_norm.index,
        bench_norm.values,
        color=PALETTE["spy"],
        linewidth=2,
        linestyle="--",
        label=bench_lbl,
        zorder=3,
    )

    # Strategy lines
    for norm, lbl, color in strategy_data:
        ax.plot(norm.index, norm.values, color=color, linewidth=1.8,
                label=lbl, zorder=4)

    # Axis formatting
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate(rotation=0, ha="center")
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Portfolio value (base = 100)", fontsize=10)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.7)

    # Backtest period text annotation (bottom-right)
    period_start = start_date.strftime("%Y-%m-%d")
    period_end = bench_norm.index[-1].strftime("%Y-%m-%d")
    ax.text(
        0.98, 0.03,
        f"{period_start} \u2192 {period_end}",
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=8, color="#555555",
    )

    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# 2. plot_drawdowns
# ---------------------------------------------------------------------------

def plot_drawdowns(
    results: list[BacktestResult],
    benchmark: BacktestResult,
    figsize: tuple[int, int] = (14, 4),
) -> tuple[plt.Figure, plt.Axes]:
    """Drawdown series as filled areas for all results and benchmark.

    Each strategy's max-drawdown trough is annotated with its depth
    value. The benchmark is rendered semi-transparently.

    Args:
        results: Strategy backtest results to plot.
        benchmark: Benchmark result (buy-and-hold SPY).
        figsize: Figure dimensions in inches.

    Returns:
        ``(fig, ax)`` — a ``(plt.Figure, plt.Axes)`` tuple. The caller
        controls rendering. Does not call ``plt.show()``. Call
        ``plt.close(fig)`` after ``st.pyplot(fig)`` in Streamlit to
        prevent figure accumulation.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Benchmark
    bench_dd = drawdown_series(benchmark.equity_curve)
    ax.fill_between(
        bench_dd.index, bench_dd.values, 0,
        color=PALETTE["spy"], alpha=0.20,
        label=_label(benchmark.strategy_name),
    )
    ax.plot(bench_dd.index, bench_dd.values,
            color=PALETTE["spy"], linewidth=0.8, alpha=0.55)

    # Strategies
    for i, r in enumerate(results):
        dd = drawdown_series(r.equity_curve)
        color = _resolve_color(r.strategy_name, i)

        ax.fill_between(
            dd.index, dd.values, 0,
            color=color, alpha=0.35,
            label=_label(r.strategy_name),
        )
        ax.plot(dd.index, dd.values, color=color, linewidth=0.9, alpha=0.85)

        # Annotate max drawdown trough
        trough_idx = dd.idxmin()
        trough_val = float(dd.min())
        ax.annotate(
            f"{trough_val:.1f}%",
            xy=(trough_idx, trough_val),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=color,
            fontweight="bold",
        )

    # Zero reference line
    ax.axhline(0, color="black", linewidth=0.5, zorder=5)

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate(rotation=0, ha="center")
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Drawdown (%)", fontsize=10)
    ax.set_title("Drawdown comparison", fontsize=13, fontweight="bold")
    ax.legend(loc="lower left", fontsize=9, framealpha=0.7)

    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# 3. plot_metrics_comparison
# ---------------------------------------------------------------------------

def plot_metrics_comparison(
    results: list[BacktestResult],
    benchmark: BacktestResult,
    metrics_to_plot: list[str] | None = None,
    figsize: tuple[int, int] = (14, 8),
) -> tuple[plt.Figure, np.ndarray]:
    """2×4 grid of horizontal bar charts for 8 performance metrics.

    For ``max_drawdown_pct``, bars extend left (negative direction).
    Each bar is annotated with its numeric value.

    Args:
        results: Strategy backtest results to compare.
        benchmark: Benchmark result appended as the last row.
        metrics_to_plot: List of up to 8 metric keys to plot. Defaults
            to ``_DEFAULT_METRICS`` (the 8 standard metrics).
        figsize: Figure dimensions in inches.

    Returns:
        ``(fig, axes)`` where ``axes`` is ``np.ndarray`` of shape
        ``(2, 4)``. Does not call ``plt.show()``. Call
        ``plt.close(fig)`` after ``st.pyplot(fig)`` in Streamlit to
        prevent figure accumulation.
    """
    if metrics_to_plot is None:
        metrics_to_plot = _DEFAULT_METRICS

    all_results = list(results) + [benchmark]
    n = len(all_results)
    colors = [_resolve_color(r.strategy_name, i) for i, r in enumerate(all_results)]
    labels = [_label(r.strategy_name) for r in all_results]

    fig, axes = plt.subplots(2, 4, figsize=figsize)
    y_pos = np.arange(n)
    bar_h = 0.6

    for idx, metric in enumerate(metrics_to_plot[:8]):
        row, col = divmod(idx, 4)
        ax = axes[row, col]

        raw_vals = [r.metrics.get(metric) or 0.0 for r in all_results]
        is_neg = (metric == "max_drawdown_pct")

        # Draw bars
        for i, (val, color) in enumerate(zip(raw_vals, colors)):
            bar_len = -abs(val) if is_neg else val
            ax.barh(y_pos[i], bar_len, height=bar_h, color=color, alpha=0.85)

        # Compute 2% offset from current x-range for annotation placement
        xlim = ax.get_xlim()
        x_range = (xlim[1] - xlim[0]) or 1.0
        offset = x_range * 0.02

        for i, val in enumerate(raw_vals):
            if is_neg:
                bar_end = -abs(val)
                ax.text(bar_end - offset, y_pos[i], f"{val:.2f}",
                        ha="right", va="center", fontsize=9)
            else:
                ax.text(max(val, 0.0) + offset, y_pos[i], f"{val:.2f}",
                        ha="left", va="center", fontsize=9)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(_METRIC_LABELS.get(metric, metric), fontsize=9, fontweight="bold")
        ax.axvline(0, color="black", linewidth=0.5)
        ax.tick_params(axis="x", labelsize=8)

    fig.suptitle("Strategy performance comparison", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# 4. plot_rolling_metrics
# ---------------------------------------------------------------------------

def plot_rolling_metrics(
    results: list[BacktestResult],
    benchmark: BacktestResult,
    window: int = 63,
    figsize: tuple[int, int] = (14, 10),
) -> tuple[plt.Figure, np.ndarray]:
    """Three stacked subplots: rolling return, Sharpe, and max drawdown.

    Subplots share the x-axis. Regime shading marks the COVID-19
    period (2020) and the rate-hike period (2022).

    Args:
        results: Strategy backtest results to plot.
        benchmark: Benchmark result (buy-and-hold SPY).
        window: Rolling window in trading days. Default 63 (≈1 quarter).
        figsize: Figure dimensions in inches.

    Returns:
        ``(fig, axes)`` where ``axes`` is ``np.ndarray`` of shape
        ``(3,)``. Does not call ``plt.show()``. Call
        ``plt.close(fig)`` after ``st.pyplot(fig)`` in Streamlit to
        prevent figure accumulation.
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    ax0, ax1, ax2 = axes

    all_r = list(results) + [benchmark]
    colors = [_resolve_color(r.strategy_name, i) for i, r in enumerate(all_r)]

    def _ls(r: BacktestResult) -> tuple:
        return (2.0, "--") if _is_benchmark(r) else (1.5, "-")

    # ── Row 1: Rolling annualized return ─────────────────────────────────────
    for r, color in zip(all_r, colors):
        lw, ls = _ls(r)
        roll_ret = r.returns.rolling(window).mean() * TRADING_DAYS_PER_YEAR * 100
        ax0.plot(roll_ret.index, roll_ret.values,
                 color=color, linewidth=lw, linestyle=ls,
                 label=_label(r.strategy_name))
    ax0.axhline(0, color="black", linewidth=0.5)
    ax0.set_ylabel(f"{window}-day rolling ann. return (%)", fontsize=10)
    ax0.legend(fontsize=8, loc="upper left", ncol=2)
    ax0.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

    # ── Row 2: Rolling Sharpe ────────────────────────────────────────────────
    for r, color in zip(all_r, colors):
        lw, ls = _ls(r)
        roll_mean = r.returns.rolling(window).mean()
        roll_std = r.returns.rolling(window).std()
        roll_std = roll_std.replace(0, np.nan)
        roll_sharpe = (roll_mean / roll_std) * math.sqrt(TRADING_DAYS_PER_YEAR)
        ax1.plot(roll_sharpe.index, roll_sharpe.values,
                 color=color, linewidth=lw, linestyle=ls)
    ax1.axhline(0, color="gray", linewidth=0.8)
    ax1.axhline(1, color="#2ecc71", linewidth=0.8, linestyle="--", alpha=0.8)
    ax1.set_ylabel(f"{window}-day rolling Sharpe", fontsize=10)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    # ── Row 3: Rolling max drawdown ──────────────────────────────────────────
    for r, color in zip(all_r, colors):
        lw, ls = _ls(r)
        roll_dd = r.equity_curve.rolling(window).apply(
            lambda x: (x / x.cummax() - 1).min() * 100,
            raw=False,
        )
        ax2.plot(roll_dd.index, roll_dd.values,
                 color=color, linewidth=lw, linestyle=ls)
    ax2.set_ylabel(f"{window}-day rolling max drawdown (%)", fontsize=10)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

    # ── Regime shading ───────────────────────────────────────────────────────
    regimes = [
        ("2020-01-01", "2020-12-31", "#e74c3c", 0.12, "COVID-19"),
        ("2022-01-01", "2022-12-31", "#f39c12", 0.12, "Rate hikes"),
    ]
    for ax_ in axes:
        for start, end, clr, alpha, _ in regimes:
            ax_.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                        color=clr, alpha=alpha, zorder=0)

    # Regime labels on top subplot only (using xaxis transform: data-x, axes-y)
    for start, _, clr, _, label in regimes:
        ax0.text(
            pd.Timestamp(start), 1.0,
            f" {label}",
            transform=ax0.get_xaxis_transform(),
            fontsize=8, color=clr, va="top",
        )

    # ── X-axis (bottom subplot only — sharex) ────────────────────────────────
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate(rotation=0, ha="center")
    ax2.set_xlabel("Date", fontsize=10)

    fig.suptitle(f"Rolling {window}-day performance metrics",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# 5. plot_correlation_matrix
# ---------------------------------------------------------------------------

def plot_correlation_matrix(
    results: list[BacktestResult],
    benchmark: BacktestResult,
    figsize: tuple[int, int] = (8, 7),
) -> tuple[plt.Figure, plt.Axes]:
    """Pearson return-correlation heatmap for all strategies and benchmark.

    Adds an interpretation note below the figure noting whether low
    correlation (< 0.3 off-diagonal) implies diversification potential.

    Args:
        results: Strategy backtest results to include.
        benchmark: Benchmark result included as the final column/row.
        figsize: Figure dimensions in inches.

    Returns:
        ``(fig, ax)`` — a ``(plt.Figure, plt.Axes)`` tuple. Does not
        call ``plt.show()``. Call ``plt.close(fig)`` after
        ``st.pyplot(fig)`` in Streamlit to prevent figure accumulation.
    """
    all_r = list(results) + [benchmark]
    series_dict = {r.strategy_name[:20]: r.returns for r in all_r}
    returns_df = pd.concat(series_dict, axis=1).dropna()
    corr = returns_df.corr()

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        square=True,
        ax=ax,
        annot_kws={"fontsize": 9},
    )
    ax.set_title("Return correlation matrix", fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", labelrotation=30, labelsize=9)
    ax.tick_params(axis="y", labelrotation=0, labelsize=9)

    # Interpretation text
    n = len(corr)
    off_vals = [corr.iloc[i, j] for i in range(n) for j in range(n) if i != j]
    min_corr = min(off_vals) if off_vals else 1.0
    interp = (
        "Low correlation detected \u2014 diversification potential exists"
        if min_corr < 0.3
        else "Strategies are highly correlated \u2014 limited diversification"
    )
    fig.text(0.5, -0.02, interp,
             ha="center", fontsize=9, color="#555555", style="italic")

    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# 6. plot_monthly_returns_heatmap
# ---------------------------------------------------------------------------

def plot_monthly_returns_heatmap(
    result: BacktestResult,
    figsize: tuple[int, int] = (14, 6),
) -> tuple[plt.Figure, plt.Axes]:
    """Monthly return calendar heatmap: rows = years, cols = Jan–Dec + Full Year.

    Cells show the compounded monthly return in %. Green = positive,
    red = negative. The Full Year column shows the annual compounded
    return.

    Args:
        result: Single strategy backtest result to visualise.
        figsize: Figure dimensions in inches.

    Returns:
        ``(fig, ax)`` — a ``(plt.Figure, plt.Axes)`` tuple. Does not
        call ``plt.show()``. Call ``plt.close(fig)`` after
        ``st.pyplot(fig)`` in Streamlit to prevent figure accumulation.
    """
    monthly = result.returns.resample("ME").apply(
        lambda x: (1 + x).prod() - 1
    ) * 100

    # Build year × month pivot
    pivot_years = monthly.index.year
    pivot_months = monthly.index.month
    pivot = pd.DataFrame(
        monthly.values,
        index=pd.MultiIndex.from_arrays([pivot_years, pivot_months]),
    ).unstack(level=1)
    pivot.columns = pivot.columns.droplevel(0)
    pivot.index.name = "Year"
    pivot.columns.name = "Month"

    month_abbr = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
        5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
        9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
    }
    pivot.columns = [month_abbr.get(int(m), str(m)) for m in pivot.columns]

    # Annual return column
    annual = result.returns.resample("YE").apply(
        lambda x: (1 + x).prod() - 1
    ) * 100
    annual.index = annual.index.year
    annual.name = "Full Year"
    pivot = pivot.join(annual, how="left")

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        pivot,
        cmap="RdYlGn",
        center=0,
        annot=True,
        fmt=".1f",
        linewidths=0.5,
        ax=ax,
        annot_kws={"fontsize": 8},
    )
    ax.set_title(
        f"Monthly returns (%) \u2014 {result.strategy_name}",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Month", fontsize=10)
    ax.set_ylabel("Year", fontsize=10)
    ax.tick_params(axis="both", labelsize=9)

    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# 7. plot_trade_analysis
# ---------------------------------------------------------------------------

def plot_trade_analysis(
    result: BacktestResult,
    figsize: tuple[int, int] = (14, 8),
) -> tuple[plt.Figure, np.ndarray]:
    """2×2 panel: trade directions, monthly costs, size histogram, cost drag.

    Panels:
        - Top-left: donut chart of trade directions (BUY/SELL/SHORT/COVER).
        - Top-right: monthly commission and slippage stacked bars.
        - Bottom-left: KDE histogram of trade notional sizes.
        - Bottom-right: cumulative cost drag as % of initial capital.

    Args:
        result: Single strategy backtest result.
        figsize: Figure dimensions in inches.

    Returns:
        ``(fig, axes)`` where ``axes`` is ``np.ndarray`` of shape
        ``(2, 2)``. Does not call ``plt.show()``. Call
        ``plt.close(fig)`` after ``st.pyplot(fig)`` in Streamlit to
        prevent figure accumulation.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    trades = result.trades
    strat_color = _resolve_color(result.strategy_name, 0)
    initial_cap = float(result.config.get("initial_capital", 100_000))

    # ── Top-left: Trade direction pie ────────────────────────────────────────
    ax_pie = axes[0, 0]
    _no_trade_kw = dict(ha="center", va="center", fontsize=11,
                        transform=ax_pie.transAxes)
    if trades is not None and not trades.empty:
        dir_counts = trades["direction"].value_counts()
        dir_colors_map = {
            "BUY": "#2ecc71", "SELL": "#e74c3c",
            "SHORT": "#9b59b6", "COVER": "#3498db",
        }
        wcolors = [dir_colors_map.get(d, "#95a5a6") for d in dir_counts.index]
        _, _, autotexts = ax_pie.pie(
            dir_counts.values,
            labels=dir_counts.index.tolist(),
            colors=wcolors,
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops={"linewidth": 1.0, "edgecolor": "white"},
        )
        for at in autotexts:
            at.set_fontsize(9)
        ax_pie.add_patch(plt.Circle((0, 0), 0.5, color="white"))
    else:
        ax_pie.text(0.5, 0.5, "No trades", **_no_trade_kw)
    ax_pie.set_title("Trade directions", fontsize=11, fontweight="bold")

    # ── Top-right: Monthly costs stacked bar ─────────────────────────────────
    ax_cost = axes[0, 1]
    if trades is not None and not trades.empty and "date" in trades.columns:
        tc = trades.copy()
        tc["date"] = pd.to_datetime(tc["date"])
        tc = tc.set_index("date").sort_index()
        mc = tc["commission"].resample("ME").sum()
        ms = tc["slippage"].resample("ME").sum()
        cost_df = pd.DataFrame({"Commission": mc, "Slippage": ms}).fillna(0)
        bp = np.arange(len(cost_df))
        ax_cost.bar(bp, cost_df["Commission"].values,
                    label="Commission", color="#3498db", alpha=0.85)
        ax_cost.bar(bp, cost_df["Slippage"].values,
                    bottom=cost_df["Commission"].values,
                    label="Slippage", color="#e67e22", alpha=0.85)
        step = max(1, len(cost_df) // 10)
        ax_cost.set_xticks(bp[::step])
        ax_cost.set_xticklabels(
            cost_df.index[::step].strftime("%Y-%m"),
            rotation=30, ha="right", fontsize=8,
        )
        ax_cost.legend(fontsize=8)
    else:
        ax_cost.text(0.5, 0.5, "No trades", ha="center", va="center",
                     fontsize=11, transform=ax_cost.transAxes)
    ax_cost.set_ylabel("Cost ($)", fontsize=10)
    ax_cost.set_xlabel("Month", fontsize=10)
    ax_cost.set_title("Monthly transaction costs", fontsize=11, fontweight="bold")

    # ── Bottom-left: Trade size histogram ────────────────────────────────────
    ax_hist = axes[1, 0]
    if trades is not None and not trades.empty:
        notional = (trades["quantity"].abs() * trades["price"]).dropna()
        if len(notional) > 1:
            sns.histplot(
                notional,
                bins=min(40, len(notional)),
                kde=True,
                ax=ax_hist,
                color=strat_color,
            )
        else:
            ax_hist.text(0.5, 0.5, "Insufficient trades", ha="center",
                         va="center", fontsize=11, transform=ax_hist.transAxes)
    else:
        ax_hist.text(0.5, 0.5, "No trades", ha="center", va="center",
                     fontsize=11, transform=ax_hist.transAxes)
    ax_hist.set_xlabel("Trade notional ($)", fontsize=10)
    ax_hist.set_ylabel("Count", fontsize=10)
    ax_hist.set_title("Trade size distribution", fontsize=11, fontweight="bold")

    # ── Bottom-right: Cumulative cost drag ───────────────────────────────────
    ax_drag = axes[1, 1]
    if trades is not None and not trades.empty and "date" in trades.columns:
        td = trades.copy()
        td["date"] = pd.to_datetime(td["date"])
        td = td.sort_values("date")
        td["_total_cost"] = td["commission"] + td["slippage"]
        cum = td.groupby("date")["_total_cost"].sum().cumsum()
        cum_pct = cum / initial_cap * 100
        ax_drag.plot(cum_pct.index, cum_pct.values,
                     color=strat_color, linewidth=1.8)
        ax_drag.fill_between(cum_pct.index, cum_pct.values, 0,
                             color=strat_color, alpha=0.3)
        ax_drag.xaxis.set_major_locator(mdates.YearLocator())
        ax_drag.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        fig.autofmt_xdate(rotation=0, ha="center")
    else:
        ax_drag.text(0.5, 0.5, "No trades", ha="center", va="center",
                     fontsize=11, transform=ax_drag.transAxes)
    ax_drag.set_ylabel("Cumulative cost drag (% of capital)", fontsize=10)
    ax_drag.set_xlabel("Date", fontsize=10)
    ax_drag.set_title("Cumulative transaction cost drag",
                      fontsize=11, fontweight="bold")

    fig.suptitle(f"Trade analysis \u2014 {result.strategy_name}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# 8. plot_position_concentration
# ---------------------------------------------------------------------------

def plot_position_concentration(
    result: BacktestResult,
    figsize: tuple[int, int] = (14, 5),
) -> tuple[plt.Figure, np.ndarray]:
    """Portfolio weight heatmap and mean weight bar chart.

    Left panel: weekly-sampled weight heatmap over time (rows = tickers,
    cols = dates). Right panel: mean portfolio weight per ticker as a
    horizontal bar chart.

    Args:
        result: Single strategy backtest result.
        figsize: Figure dimensions in inches.

    Returns:
        ``(fig, axes)`` where ``axes`` is ``np.ndarray`` of shape
        ``(2,)`` containing ``[ax_heatmap, ax_bar]``. Does not call
        ``plt.show()``. Call ``plt.close(fig)`` after
        ``st.pyplot(fig)`` in Streamlit to prevent figure accumulation.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    ax_heat, ax_bar = axes
    positions = result.positions
    strat_color = _resolve_color(result.strategy_name, 0)

    if positions is not None and not positions.empty:
        # Weekly sample, drop all-zero tickers
        weekly = positions.resample("W").last().dropna(how="all")
        nonzero_cols = weekly.columns[(weekly != 0).any()]
        weekly = weekly[nonzero_cols]

        if not weekly.empty:
            heat_data = weekly.T
            sns.heatmap(
                heat_data,
                cmap="RdYlGn",
                center=0,
                vmin=-0.3,
                vmax=0.3,
                ax=ax_heat,
                linewidths=0.0,
                xticklabels=False,
                yticklabels=True,
            )
            # Sparse yearly x-ticks
            n_cols = heat_data.shape[1]
            step = max(1, n_cols // 8)
            tick_pos = list(range(0, n_cols, step))
            ax_heat.set_xticks([p + 0.5 for p in tick_pos])
            ax_heat.set_xticklabels(
                [heat_data.columns[p].strftime("%Y") for p in tick_pos],
                rotation=0, fontsize=8,
            )
        else:
            ax_heat.text(0.5, 0.5, "No positions", ha="center", va="center",
                         fontsize=11, transform=ax_heat.transAxes)

        ax_heat.set_xlabel("Date", fontsize=10)
        ax_heat.set_ylabel("Ticker", fontsize=10)
        ax_heat.tick_params(axis="y", labelsize=8)

        # Mean-weight bar chart
        mean_w = positions.mean().sort_values(ascending=False)
        mean_w = mean_w[mean_w != 0]
        if not mean_w.empty:
            bar_colors = [
                strat_color if w >= 0 else "#e74c3c"
                for w in mean_w.values
            ]
            ax_bar.barh(mean_w.index.tolist(), mean_w.values,
                        color=bar_colors, alpha=0.85)
            ax_bar.axvline(0, color="black", linewidth=0.5)
            ax_bar.tick_params(axis="y", labelsize=9)
        else:
            ax_bar.text(0.5, 0.5, "No positions", ha="center", va="center",
                        fontsize=11, transform=ax_bar.transAxes)
    else:
        for ax_ in [ax_heat, ax_bar]:
            ax_.text(0.5, 0.5, "No positions", ha="center", va="center",
                     fontsize=11, transform=ax_.transAxes)

    ax_heat.set_title("Portfolio weights over time", fontsize=11, fontweight="bold")
    ax_bar.set_title("Mean portfolio weight by ticker", fontsize=11, fontweight="bold")
    ax_bar.set_xlabel("Mean portfolio weight", fontsize=10)
    ax_bar.set_ylabel("Ticker", fontsize=10)
    fig.suptitle(f"Position concentration \u2014 {result.strategy_name}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# save_figure
# ---------------------------------------------------------------------------

def save_figure(
    fig: plt.Figure,
    name: str,
    output_dir: pathlib.Path,
    formats: list | None = None,
) -> list:
    """
    Save ``fig`` to one or more file formats under ``output_dir``.

    Parameters
    ----------
    fig        : plt.Figure
    name       : base filename, no extension (e.g. ``"01_equity_curves"``)
    output_dir : destination directory — created if it does not exist
    formats    : list of extension strings, default ``["png", "pdf"]``

    Returns
    -------
    list[pathlib.Path]  — paths of every file written
    """
    if formats is None:
        formats = ["png", "pdf"]
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[pathlib.Path] = []
    for fmt in formats:
        path = output_dir / f"{name}.{fmt}"
        fig.savefig(path)
        saved.append(path)
        print(f"  Saved: {path}")
    return saved
