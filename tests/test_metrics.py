"""
Unit tests for backtester/metrics.py.

Scheduled for full implementation in Phase 2.3, once the metric functions
are built.  Stubs are provided now so CI can discover and report them.
"""

import pytest


def test_sharpe_ratio_positive_returns() -> None:
    """
    Sharpe ratio must be strictly positive when all daily returns are positive.

    Setup (Phase 2.3):
    - Construct a pd.Series of small positive daily returns (e.g., 0.001 each day).
    - Call sharpe_ratio(returns, risk_free_rate=0.0).
    - Assert result > 0.
    """
    # TODO: implement in Phase 2.3
    pass


def test_max_drawdown_zero_for_monotonic_returns() -> None:
    """
    Max drawdown must be exactly 0.0 for a strictly monotonically increasing equity curve.

    Setup (Phase 2.3):
    - Construct a pd.Series starting at 100 and increasing by 1 each day (no dips).
    - Call max_drawdown(equity_curve).
    - Assert result == 0.0.
    """
    # TODO: implement in Phase 2.3
    pass
