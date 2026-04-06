"""
Tests for backtester/metrics.py.

All tests use synthetic data — no network calls or yfinance dependency.
Each test exercises a clearly identified property of compute_metrics()
or drawdown_series() so that failures pinpoint exactly which metric broke.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtester.engine import BacktestResult
from backtester.metrics import compute_metrics, drawdown_series


# ---------------------------------------------------------------------------
# Shared helper: build a BacktestResult with known properties
# ---------------------------------------------------------------------------

def _make_result(
    returns_array: list | np.ndarray,
    initial_capital: float = 100_000,
) -> BacktestResult:
    """
    Build a synthetic BacktestResult from a daily-returns array.

    Equity is computed as ``initial_capital * exp(cumsum(returns))``.
    Trades include one BUY at bar 10 and one SELL at bar 20, giving a
    deterministic but non-trivial trading-activity section.
    Positions hold AAPL=0.5 and SPY=0.5 throughout (constant weights).
    """
    dates = pd.bdate_range("2020-01-01", periods=len(returns_array))
    returns = pd.Series(returns_array, index=dates, dtype=float)
    equity = pd.Series(
        initial_capital * np.exp(returns.cumsum()),
        index=dates,
    )
    trades = pd.DataFrame({
        "date":      [dates[10], dates[20]],
        "ticker":    ["AAPL", "AAPL"],
        "direction": ["BUY", "SELL"],
        "quantity":  [10.0, 10.0],
        "price":     [150.0, 160.0],
        "commission":[7.5, 8.0],
        "slippage":  [3.0, 3.2],
        "net_cost":  [1507.5, -1588.8],
    })
    positions = pd.DataFrame(
        {"AAPL": [0.5] * len(dates), "SPY": [0.5] * len(dates)},
        index=dates,
    )
    return BacktestResult(
        strategy_name="test",
        equity_curve=equity,
        returns=returns,
        positions=positions,
        trades=trades,
        metrics={},
        config={
            "initial_capital": initial_capital,
            "benchmark":       "SPY",
            "risk_free_rate":  0.0,
        },
    )


# ---------------------------------------------------------------------------
# Test 1 — Sharpe is positive for all-positive returns
# ---------------------------------------------------------------------------

def test_sharpe_ratio_positive_returns() -> None:
    """
    When every daily return is strictly positive (0.1% per day), the annualised
    Sharpe ratio must be > 0.

    The Sortino ratio must be either greater than Sharpe (theoretically infinite
    with no downside days) or 0.0 (the graceful-degrade value when downside std
    is zero).  Both outcomes are acceptable per the spec.
    """
    result = _make_result([0.001] * 500)
    m = compute_metrics(result)

    assert m["sharpe_ratio"] > 0, (
        f"Expected positive Sharpe for all-positive returns, got {m['sharpe_ratio']}"
    )

    # Sortino: no downside days → std = 0 → degrade to 0.0, OR inflated > Sharpe
    assert m["sortino_ratio"] > m["sharpe_ratio"] or m["sortino_ratio"] == 0.0, (
        f"sortino={m['sortino_ratio']}, sharpe={m['sharpe_ratio']}: "
        "Sortino must exceed Sharpe (no downside) or gracefully degrade to 0.0"
    )


# ---------------------------------------------------------------------------
# Test 2 — Max drawdown is 0 for a strictly increasing equity curve
# ---------------------------------------------------------------------------

def test_max_drawdown_zero_for_monotonic_returns() -> None:
    """
    A strictly monotonically increasing equity curve must have:
      - max_drawdown_pct == 0.0 exactly
      - recovery_days is None (nothing to recover from)
    """
    result = _make_result(np.linspace(0.001, 0.003, 500).tolist())
    m = compute_metrics(result)

    assert m["max_drawdown_pct"] == 0.0, (
        f"Expected 0.0 max drawdown for monotonically increasing equity, "
        f"got {m['max_drawdown_pct']}"
    )
    assert m["recovery_days"] is None, (
        f"Expected recovery_days=None when there is no drawdown, "
        f"got {m['recovery_days']}"
    )


# ---------------------------------------------------------------------------
# Test 3 — Max drawdown is correct for a known crash-and-partial-recovery
# ---------------------------------------------------------------------------

def test_max_drawdown_correct_for_known_crash() -> None:
    """
    Equity rising to 150k, dropping to 90k, then partially recovering to 110k
    must produce:
      - max_drawdown_pct ≈ -40.0% (peak-to-trough: (90k-150k)/150k * 100)
      - max_drawdown_duration_days > 0
      - recovery_days is None (110k < 150k, so the peak is never recovered)
    """
    dates = pd.bdate_range("2020-01-01", periods=6)
    # Equity: flat at 100k → rise → peak 150k → crash 90k → partial 110k
    equity_values = [100_000.0, 120_000.0, 150_000.0, 120_000.0, 90_000.0, 110_000.0]
    equity_curve = pd.Series(equity_values, index=dates)
    returns = equity_curve.pct_change().fillna(0.0)

    result = BacktestResult(
        strategy_name="crash_test",
        equity_curve=equity_curve,
        returns=returns,
        positions=pd.DataFrame(
            {"AAPL": [0.5] * 6, "SPY": [0.5] * 6}, index=dates
        ),
        trades=pd.DataFrame(columns=[
            "date", "ticker", "direction", "quantity",
            "price", "commission", "slippage", "net_cost",
        ]),
        metrics={},
        config={"initial_capital": 100_000, "benchmark": "SPY", "risk_free_rate": 0.0},
    )

    m = compute_metrics(result)

    # Peak = 150k, trough = 90k  →  drawdown = -40.0%
    assert abs(m["max_drawdown_pct"] - (-40.0)) < 0.1, (
        f"Expected max drawdown ≈ -40.0%, got {m['max_drawdown_pct']:.4f}%"
    )
    assert m["max_drawdown_duration_days"] > 0, (
        f"Expected positive drawdown duration, got {m['max_drawdown_duration_days']}"
    )
    # Last equity value is 110k < 150k: strategy never recovers to the peak
    assert m["recovery_days"] is None, (
        f"Expected recovery_days=None (never reaches 150k peak), "
        f"got {m['recovery_days']}"
    )


# ---------------------------------------------------------------------------
# Test 4 — total_return_pct matches the equity curve exactly
# ---------------------------------------------------------------------------

def test_total_return_matches_equity_curve() -> None:
    """
    total_return_pct must equal (equity[-1] / equity[0] - 1) * 100 to
    floating-point precision (< 1e-9 difference).
    """
    returns_arr = [0.001] * 200 + [-0.0005] * 50 + [0.002] * 250
    result = _make_result(returns_arr)
    m = compute_metrics(result)

    expected = (
        float(result.equity_curve.iloc[-1] / result.equity_curve.iloc[0]) - 1.0
    ) * 100.0

    assert abs(m["total_return_pct"] - expected) < 1e-9, (
        f"total_return_pct={m['total_return_pct']:.10f} != expected={expected:.10f}"
    )


# ---------------------------------------------------------------------------
# Test 5 — cost metrics sum correctly
# ---------------------------------------------------------------------------

def test_cost_metrics_sum_correctly() -> None:
    """
    total_cost_pct must equal total_commission_pct + total_slippage_pct
    to floating-point precision.
    """
    result = _make_result([0.001] * 500)
    m = compute_metrics(result)

    expected_sum = m["total_commission_pct"] + m["total_slippage_pct"]
    assert abs(m["total_cost_pct"] - expected_sum) < 1e-9, (
        f"total_cost_pct={m['total_cost_pct']:.12f} != "
        f"commission+slippage={expected_sum:.12f}"
    )


# ---------------------------------------------------------------------------
# Test 6 — all 8 benchmark-relative keys present when benchmark is provided
# ---------------------------------------------------------------------------

def test_benchmark_metrics_present_when_provided() -> None:
    """
    When a benchmark BacktestResult is supplied, compute_metrics() must
    add all 8 benchmark-relative keys to the returned dict.

    Additionally, alpha_pct must equal
    annualized_return_pct - benchmark_annualized_return_pct to within 1e-6.
    """
    strategy_result = _make_result([0.001] * 500)
    benchmark_result = _make_result([0.0005] * 500)

    m = compute_metrics(strategy_result, benchmark=benchmark_result)

    required_keys = [
        "alpha_pct",
        "beta",
        "correlation_with_benchmark",
        "information_ratio",
        "benchmark_total_return_pct",
        "benchmark_annualized_return_pct",
        "benchmark_max_drawdown_pct",
        "benchmark_sharpe_ratio",
    ]
    for key in required_keys:
        assert key in m, (
            f"Expected benchmark key '{key}' in metrics dict, but it is absent.  "
            f"Present keys: {sorted(m.keys())}"
        )

    expected_alpha = m["annualized_return_pct"] - m["benchmark_annualized_return_pct"]
    assert abs(m["alpha_pct"] - expected_alpha) < 1e-6, (
        f"alpha_pct={m['alpha_pct']:.8f} != "
        f"ann_ret - bench_ann_ret={expected_alpha:.8f}"
    )


# ---------------------------------------------------------------------------
# Test 7 — benchmark-relative keys absent when benchmark=None
# ---------------------------------------------------------------------------

def test_benchmark_metrics_absent_when_not_provided() -> None:
    """
    When benchmark=None (the default), none of the benchmark-relative keys
    must appear in the returned metrics dict.
    """
    result = _make_result([0.001] * 500)
    m = compute_metrics(result, benchmark=None)

    absent_keys = [
        "alpha_pct",
        "beta",
        "information_ratio",
        "correlation_with_benchmark",
        "benchmark_total_return_pct",
        "benchmark_annualized_return_pct",
        "benchmark_max_drawdown_pct",
        "benchmark_sharpe_ratio",
    ]
    for key in absent_keys:
        assert key not in m, (
            f"Key '{key}' must be absent when benchmark=None, "
            f"but it was present with value {m.get(key)}"
        )


# ---------------------------------------------------------------------------
# Test 8 — drawdown_series is always <= 0 and starts at 0
# ---------------------------------------------------------------------------

def test_drawdown_series_never_positive() -> None:
    """
    drawdown_series() must never return a positive value (a drawdown cannot
    be a gain), and the first value must be exactly 0.0 (the portfolio starts
    at its own peak).
    """
    result = _make_result(([0.001, -0.002, 0.003, -0.001, 0.002] * 100))
    dd = drawdown_series(result.equity_curve)

    assert isinstance(dd, pd.Series), (
        f"drawdown_series must return a pd.Series, got {type(dd)}"
    )
    assert (dd > 0).sum() == 0, (
        f"drawdown_series contains {(dd > 0).sum()} positive value(s); "
        "drawdown must always be <= 0"
    )
    assert dd.iloc[0] == 0.0, (
        f"First drawdown must be 0.0 (portfolio is at peak on day 0), "
        f"got {dd.iloc[0]}"
    )
