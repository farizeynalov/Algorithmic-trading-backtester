"""
Tests for backtester/engine.py.

All tests use synthetic price data — no network calls or yfinance dependency.

The five tests cover the two non-negotiable engine invariants:
  1. No lookahead bias  (structural, via shift(1))
  2. No silent failures (assertions fire early with descriptive messages)

Plus correctness of position sizing, cost accounting, and benchmark runs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtester.base import BaseStrategy
from backtester.engine import Backtester, BacktestResult


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_data() -> pd.DataFrame:
    """756 business days of synthetic prices for 4 tickers (incl. SPY)."""
    dates = pd.bdate_range("2020-01-01", "2022-12-31")
    tickers = ["AAPL", "MSFT", "GOOGL", "SPY"]
    np.random.seed(42)
    prices = pd.DataFrame(
        100 * np.exp(
            np.random.randn(len(dates), len(tickers)).cumsum(axis=0) * 0.01
        ),
        index=dates,
        columns=tickers,
    )
    return prices


# ---------------------------------------------------------------------------
# Reusable mock strategies
# ---------------------------------------------------------------------------

class _AlwaysLongStrategy(BaseStrategy):
    """Returns signal = 1.0 for every ticker on every date."""

    def __init__(self) -> None:
        self.received_index: pd.DatetimeIndex | None = None

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        self.received_index = data.index.copy()
        return pd.DataFrame(1.0, index=data.index, columns=data.columns)

    def get_name(self) -> str:
        return "AlwaysLong"


class _FlatStrategy(BaseStrategy):
    """Returns signal = 0 for every ticker on every date (cash-only)."""

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(0.0, index=data.index, columns=data.columns)

    def get_name(self) -> str:
        return "FlatStrategy"


class _RandomSignalStrategy(BaseStrategy):
    """Uniform random signals in [-1, 1] with a fixed seed."""

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        rng = np.random.default_rng(seed=99)
        raw = rng.uniform(-1.0, 1.0, size=(len(data), len(data.columns)))
        return pd.DataFrame(raw, index=data.index, columns=data.columns)

    def get_name(self) -> str:
        return "RandomSignal"


# ---------------------------------------------------------------------------
# Test 1 — No lookahead bias
# ---------------------------------------------------------------------------

def test_no_lookahead_bias(synthetic_data: pd.DataFrame) -> None:
    """
    Structural proof that shift(1) prevents day-0 signals from affecting
    day-0 positions.

    Mechanism:
      - Strategy emits signal = 1.0 on every date (including date 0).
      - After _resolve_signals() applies shift(1), the effective signal on
        date 0 is 0 (NaN → 0), so no position is held on date 0.
      - The day-0 signal only takes effect on date 1, where we expect a
        non-zero position.
    """
    strategy = _AlwaysLongStrategy()
    engine = Backtester(synthetic_data)
    result = engine.run(strategy)

    # Strategy must have received the full dataset
    assert strategy.received_index is not None
    assert len(strategy.received_index) == len(synthetic_data), (
        "Strategy did not receive the complete price history"
    )

    # INVARIANT 1: first-bar positions must ALL be zero
    # (day-0 signal → shift(1) → NaN → 0 weight → no position)
    first_day_positions = result.positions.iloc[0]
    assert (first_day_positions == 0.0).all(), (
        f"Lookahead bias detected: day-0 positions are non-zero.\n"
        f"Non-zero positions: {first_day_positions[first_day_positions != 0].to_dict()}\n"
        "The shift(1) in _resolve_signals must zero out the first bar."
    )

    # Day 1 should have non-zero positions (day-0 signal executed on day 1)
    second_day_positions = result.positions.iloc[1]
    assert (second_day_positions > 0).any(), (
        "Expected non-zero positions on day 1: the day-0 signal should have "
        "executed here, but all weights are still zero."
    )


# ---------------------------------------------------------------------------
# Test 2 — Equity curve starts at initial capital
# ---------------------------------------------------------------------------

def test_equity_curve_starts_at_initial_capital(synthetic_data: pd.DataFrame) -> None:
    """
    A flat strategy (no positions, no trades) must produce an equity curve
    that starts at — and stays at — the initial capital of $100,000.

    This verifies:
      - The first equity value equals initial_capital exactly.
      - No phantom P&L or costs are created when nothing is traded.
      - The equity curve is non-increasing (in this case, perfectly flat).
    """
    engine = Backtester(synthetic_data)
    result = engine.run(_FlatStrategy())

    assert result.equity_curve.iloc[0] == 100_000.0, (
        f"First equity value should equal initial_capital=100,000.  "
        f"Got: {result.equity_curve.iloc[0]:.2f}"
    )

    # No positions → no P&L, no trades → no costs → perfectly flat
    assert (result.equity_curve == 100_000.0).all(), (
        "Equity changed with no positions held.  "
        f"Min: {result.equity_curve.min():.4f}, "
        f"Max: {result.equity_curve.max():.4f}"
    )

    # Monotonically non-increasing (trivially satisfied for flat curve)
    assert (result.equity_curve.diff().iloc[1:] <= 1e-9).all(), (
        "Equity curve is not monotonically non-increasing for the flat strategy"
    )


# ---------------------------------------------------------------------------
# Test 3 — Position size never exceeds max_position_size
# ---------------------------------------------------------------------------

def test_position_size_never_exceeds_max(synthetic_data: pd.DataFrame) -> None:
    """
    With max_position_size=0.25, no individual position weight may ever
    exceed 0.25 in absolute value, regardless of the raw signal magnitude.

    This tests that both the equal_weight normalisation and the per-position
    clip inside _compute_weights honour the configured cap.
    """
    config = {"max_position_size": 0.25}
    engine = Backtester(synthetic_data, config=config)
    result = engine.run(_RandomSignalStrategy())

    max_weight = float(result.positions.abs().max().max())

    assert max_weight <= 0.25 + 1e-9, (
        f"A position weight of {max_weight:.6f} exceeded max_position_size=0.25.  "
        "Check _compute_weights clipping logic."
    )


# ---------------------------------------------------------------------------
# Test 4 — Transaction costs are always non-negative and non-zero in total
# ---------------------------------------------------------------------------

def test_transaction_costs_always_positive(synthetic_data: pd.DataFrame) -> None:
    """
    Every individual commission and slippage charge must be >= 0, and the
    total friction cost across all trades must be strictly positive (i.e.
    at least one trade occurred).

    Negative costs would indicate a sign error in the cost model; zero total
    costs would indicate that no trades were ever executed.
    """
    engine = Backtester(synthetic_data)
    result = engine.run(_AlwaysLongStrategy())

    assert len(result.trades) > 0, (
        "No trades were recorded.  Expected at least the initial buy-in "
        "on day 1 when the day-0 long signal takes effect."
    )

    assert (result.trades["commission"] >= 0).all(), (
        "Found negative commission values — cost model has a sign error.\n"
        f"{result.trades[result.trades['commission'] < 0]}"
    )
    assert (result.trades["slippage"] >= 0).all(), (
        "Found negative slippage values — cost model has a sign error.\n"
        f"{result.trades[result.trades['slippage'] < 0]}"
    )

    total_costs = (
        result.trades["commission"].sum() + result.trades["slippage"].sum()
    )
    assert total_costs > 0, (
        f"Total friction costs are {total_costs:.6f}; expected > 0.  "
        "No trades appear to have been costed."
    )


# ---------------------------------------------------------------------------
# Test 5 — Benchmark run produces a valid BacktestResult
# ---------------------------------------------------------------------------

def test_benchmark_run_produces_valid_result(synthetic_data: pd.DataFrame) -> None:
    """
    run_benchmark() must return a well-formed BacktestResult whose equity
    curve spans the full data history and contains at least one trade
    (the initial buy of the benchmark ticker on day 1).
    """
    engine = Backtester(synthetic_data)
    result = engine.run_benchmark()

    assert isinstance(result, BacktestResult), (
        f"run_benchmark() must return a BacktestResult, got {type(result).__name__}"
    )

    # Equity curve must cover the same dates as the input data
    assert len(result.equity_curve) == len(synthetic_data), (
        f"equity_curve has {len(result.equity_curve)} rows but data has "
        f"{len(synthetic_data)} rows."
    )
    pd.testing.assert_index_equal(
        result.equity_curve.index,
        synthetic_data.index,
        check_names=False,
        obj="equity_curve.index vs synthetic_data.index",
    )

    # At least the initial benchmark buy must appear in the trade log
    assert len(result.trades) >= 1, (
        f"Expected at least 1 trade (initial benchmark buy), "
        f"got {len(result.trades)}.  "
        "Check that the rebalance_buffer is not suppressing the first trade."
    )

    # The benchmark ticker must appear in the trade log
    assert "SPY" in result.trades["ticker"].values, (
        "Benchmark ticker 'SPY' not found in trades.  "
        f"Tickers traded: {result.trades['ticker'].unique().tolist()}"
    )
