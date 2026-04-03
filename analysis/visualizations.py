"""
Plotting and visualisation utilities for backtest analysis.

Implementation plan (Phase 4.1)
---------------------------------
Provides a consistent, publication-quality chart library for:

- Equity curves           : plot_equity_curve()
- Drawdown chart          : plot_drawdown()
- Rolling Sharpe ratio    : plot_rolling_sharpe()
- Return distribution     : plot_return_distribution()
- Strategy vs benchmark   : plot_vs_benchmark()
- Monthly returns heatmap : plot_monthly_heatmap()
- Correlation heatmap     : plot_correlation_matrix()

All functions accept a matplotlib Axes object so they can be embedded
in multi-panel figures or Streamlit layouts without tight coupling to
the display backend.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_equity_curve(
    equity_curve: pd.Series,
    benchmark: pd.Series | None = None,
    ax: plt.Axes | None = None,
    title: str = "Equity Curve",
) -> plt.Axes:
    """
    Plot portfolio equity curve, optionally overlaid with a benchmark.

    Parameters
    ----------
    equity_curve : pd.Series
        Daily portfolio value indexed by date, normalised to start at 1.0
        (or raw capital — the function will normalise internally).
    benchmark : pd.Series | None
        Optional benchmark series to overlay (e.g., SPY total return).
    ax : plt.Axes | None
        Axes to draw on.  If None, a new figure is created.
    title : str
        Chart title.

    Returns
    -------
    plt.Axes
        The populated Axes object.
    """
    raise NotImplementedError(
        "plot_equity_curve is not yet implemented. "
        "Scheduled for Phase 4.1."
    )


def plot_drawdown(
    equity_curve: pd.Series,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Plot the rolling drawdown from peak as a filled area chart.

    Parameters
    ----------
    equity_curve : pd.Series
        Daily portfolio value indexed by date.
    ax : plt.Axes | None
        Axes to draw on.  If None, a new figure is created.

    Returns
    -------
    plt.Axes
        The populated Axes object.
    """
    raise NotImplementedError(
        "plot_drawdown is not yet implemented. "
        "Scheduled for Phase 4.1."
    )


def plot_monthly_heatmap(
    returns: pd.Series,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Render a calendar heatmap of monthly returns (rows=years, cols=months).

    Parameters
    ----------
    returns : pd.Series
        Daily simple returns indexed by date.
    ax : plt.Axes | None
        Axes to draw on.  If None, a new figure is created.

    Returns
    -------
    plt.Axes
        The populated Axes object.
    """
    raise NotImplementedError(
        "plot_monthly_heatmap is not yet implemented. "
        "Scheduled for Phase 4.1."
    )
