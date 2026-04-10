"""
analysis package

Plotting utilities and post-backtest visualisation helpers.
"""

from analysis.visualizations import (
    plot_equity_curves,
    plot_drawdowns,
    plot_metrics_comparison,
    plot_rolling_metrics,
    plot_correlation_matrix,
    plot_monthly_returns_heatmap,
    plot_trade_analysis,
    plot_position_concentration,
    save_figure,
    PALETTE,
)

__all__ = [
    "plot_equity_curves",
    "plot_drawdowns",
    "plot_metrics_comparison",
    "plot_rolling_metrics",
    "plot_correlation_matrix",
    "plot_monthly_returns_heatmap",
    "plot_trade_analysis",
    "plot_position_concentration",
    "save_figure",
    "PALETTE",
]
