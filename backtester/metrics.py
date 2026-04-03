"""
Performance metrics for evaluating backtest results.

Implementation plan (Phase 3.2)
---------------------------------
All metrics are computed from a daily equity curve (pd.Series indexed by date).

Metrics to implement:
- Sharpe ratio          (annualised, risk-free rate adjustable)
- Sortino ratio         (downside deviation only)
- CAGR                  (compound annual growth rate)
- Maximum drawdown      (peak-to-trough percentage decline)
- Calmar ratio          (CAGR / max drawdown)
- Win rate              (% of days with positive P&L)
- Profit factor         (gross profit / gross loss)
- Value at Risk (VaR)   (95th and 99th percentile daily loss)
- Beta & Alpha          (vs benchmark equity curve)
- Annualised volatility
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR: int = 252


def compute_metrics(
    equity_curve: pd.Series,
    benchmark: pd.Series | None = None,
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    """
    Compute a full suite of performance metrics from an equity curve.

    Parameters
    ----------
    equity_curve : pd.Series
        Daily portfolio value indexed by date.
    benchmark : pd.Series | None
        Optional benchmark equity curve for beta/alpha calculation.
    risk_free_rate : float
        Annualised risk-free rate as a decimal (e.g., 0.05 for 5%).
        Default is 0.0.

    Returns
    -------
    dict[str, float]
        Mapping of metric name to value.
    """
    raise NotImplementedError(
        "compute_metrics is not yet implemented. "
        "Scheduled for Phase 3.2."
    )


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Compute the annualised Sharpe ratio.

    Parameters
    ----------
    returns : pd.Series
        Daily simple returns.
    risk_free_rate : float
        Annualised risk-free rate as a decimal.
    periods_per_year : int
        Number of return periods per year for annualisation.

    Returns
    -------
    float
        Annualised Sharpe ratio.
    """
    raise NotImplementedError("sharpe_ratio not yet implemented.")


def max_drawdown(equity_curve: pd.Series) -> float:
    """
    Compute the maximum peak-to-trough drawdown as a negative fraction.

    Parameters
    ----------
    equity_curve : pd.Series
        Daily portfolio value.

    Returns
    -------
    float
        Maximum drawdown as a negative decimal, e.g., -0.35 for a 35% drawdown.
    """
    raise NotImplementedError("max_drawdown not yet implemented.")


def cagr(equity_curve: pd.Series) -> float:
    """
    Compute the Compound Annual Growth Rate.

    Parameters
    ----------
    equity_curve : pd.Series
        Daily portfolio value indexed by date.

    Returns
    -------
    float
        CAGR as a decimal, e.g., 0.12 for 12% per year.
    """
    raise NotImplementedError("cagr not yet implemented.")
