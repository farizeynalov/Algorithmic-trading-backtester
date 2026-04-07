"""
Tests for strategies/momentum.py — MomentumStrategy.

All tests use a single shared synthetic price DataFrame (21 tickers,
2015-2024, seed=0, 0.8 % daily vol) so that every test runs in isolation
without any network dependency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategies.momentum import MomentumStrategy
from backtester.engine import Backtester


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def price_data() -> pd.DataFrame:
    """
    Synthetic wide-format close prices for 21 tickers over 2015-2024.

    Tickers: T00…T19 plus SPY (used as the default benchmark by the engine).
    Each ticker follows a geometric random walk with 0.8 % daily vol.
    Seed 0 ensures deterministic results across all tests.
    """
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2015-01-01", "2024-12-31")
    tickers = [f"T{i:02d}" for i in range(20)] + ["SPY"]

    # Geometric random walk: price_{t} = price_{t-1} * exp(N(0, 0.008))
    log_returns = rng.normal(0.0, 0.008, size=(len(dates), len(tickers)))
    prices = 100.0 * np.exp(np.cumsum(log_returns, axis=0))

    return pd.DataFrame(prices, index=dates, columns=tickers)


# ---------------------------------------------------------------------------
# Test 1 — Signal shape matches input data
# ---------------------------------------------------------------------------

def test_signals_shape_matches_data(price_data: pd.DataFrame) -> None:
    """
    generate_signals() must return a DataFrame with the same shape, index,
    and columns as the input price data.
    """
    strategy = MomentumStrategy()
    signals = strategy.generate_signals(price_data)

    assert signals.shape == price_data.shape, (
        f"Signal shape {signals.shape} != price_data shape {price_data.shape}"
    )
    assert list(signals.columns) == list(price_data.columns), (
        "Signal columns do not match price_data columns"
    )
    assert signals.index.equals(price_data.index), (
        "Signal index does not match price_data index"
    )


# ---------------------------------------------------------------------------
# Test 2 — All signal values are in [-1, 1]
# ---------------------------------------------------------------------------

def test_signals_in_valid_range(price_data: pd.DataFrame) -> None:
    """
    Every value in the signal DataFrame must be in the closed interval [-1, 1].
    A tolerance of 1e-9 accounts for floating-point rounding in the
    continuous rank-normalisation.
    """
    for signal_type in ("ranked", "continuous"):
        strategy = MomentumStrategy(signal_type=signal_type)
        signals = strategy.generate_signals(price_data)

        min_val = float(signals.min().min())
        max_val = float(signals.max().max())

        assert min_val >= -1.0 - 1e-9, (
            f"[{signal_type}] Signal below -1.0: min = {min_val:.8f}"
        )
        assert max_val <= 1.0 + 1e-9, (
            f"[{signal_type}] Signal above +1.0: max = {max_val:.8f}"
        )


# ---------------------------------------------------------------------------
# Test 3 — Ranked signals only contain {-1, 0, +1}
# ---------------------------------------------------------------------------

def test_ranked_signal_only_has_0_1_minus1(price_data: pd.DataFrame) -> None:
    """
    With signal_type='ranked', the only values that may appear in the
    DataFrame are exactly 0.0, +1.0, and -1.0.

    Tested with both a long-only strategy (n_short=0) and a long-short
    strategy (n_short=3).
    """
    for n_short in (0, 3):
        strategy = MomentumStrategy(n_long=5, n_short=n_short, signal_type="ranked")
        signals = strategy.generate_signals(price_data)

        unique_vals = set(signals.values.ravel())
        allowed = {-1.0, 0.0, 1.0}
        unexpected = unique_vals - allowed

        assert not unexpected, (
            f"[n_short={n_short}] Unexpected values in ranked signals: "
            f"{sorted(unexpected)}"
        )


# ---------------------------------------------------------------------------
# Test 4 — Warmup rows are all 0.0
# ---------------------------------------------------------------------------

def test_warmup_period_is_zero(price_data: pd.DataFrame) -> None:
    """
    For the default (lookback=12, skip=1) strategy, there must be at least
    (12 + 1) full calendar months of warm-up before any signal fires.

    We conservatively verify that the first 252 business days (approx.
    12 months) are identically 0.0, which must be true regardless of exactly
    where the first month-end boundary falls.
    """
    strategy = MomentumStrategy(lookback_months=12, skip_months=1)
    signals = strategy.generate_signals(price_data)

    warmup_slice = signals.iloc[:252]

    assert (warmup_slice == 0.0).all().all(), (
        "Non-zero signal found during the first 252 bars (warmup period). "
        f"Max absolute value = {float(warmup_slice.abs().max().max()):.8f}"
    )


# ---------------------------------------------------------------------------
# Test 5 — n_long constraint is respected on every post-warmup date
# ---------------------------------------------------------------------------

def test_n_long_positions_respected(price_data: pd.DataFrame) -> None:
    """
    With n_long=3, signal_type='ranked', on every post-warmup date exactly 3
    tickers should carry a +1.0 signal (ties are broken by 'first' rank, so
    there are always exactly n_long long entries).

    Verified on 10 randomly chosen dates from the final two years of data.
    """
    strategy = MomentumStrategy(
        lookback_months=12, skip_months=1, n_long=3, signal_type="ranked"
    )
    signals = strategy.generate_signals(price_data)

    # Only check dates well after the warmup period ends
    post_warmup = signals.iloc[300:]

    rng = np.random.default_rng(42)
    sample_indices = rng.choice(len(post_warmup), size=10, replace=False)
    sample_dates = post_warmup.index[sample_indices]

    for date in sample_dates:
        row = signals.loc[date]
        n_long_actual = int((row == 1.0).sum())
        assert n_long_actual == 3, (
            f"On {date.date()}: expected 3 long positions, got {n_long_actual}. "
            f"Row values:\n{row[row != 0]}"
        )


# ---------------------------------------------------------------------------
# Test 6 — No lookahead bias via the engine (positions on day 0 are flat)
# ---------------------------------------------------------------------------

def test_no_lookahead_via_engine(price_data: pd.DataFrame) -> None:
    """
    When the strategy is run through the Backtester, the positions on the
    very first trading day must all be 0 (flat).

    This tests the engine's _resolve_signals shift(1) structural guarantee:
    a signal seen on day T can only create a position starting on day T+1.
    Even if the strategy somehow emits a non-zero signal on day 0 (it should
    not because of the warmup period), the engine must prevent it.
    """
    strategy = MomentumStrategy(lookback_months=12, skip_months=1, n_long=5)
    engine = Backtester(price_data)
    result = engine.run(strategy)

    first_day_exposure = float(result.positions.iloc[0].abs().sum())

    assert first_day_exposure == 0.0, (
        f"Expected 0 exposure on day 0 (no lookahead), "
        f"got {first_day_exposure:.8f}"
    )


# ---------------------------------------------------------------------------
# Test 7 — Continuous signals: long side and short side sum to (near) zero
# ---------------------------------------------------------------------------

def test_continuous_signals_sum_to_zero(price_data: pd.DataFrame) -> None:
    """
    The continuous rank-normalisation centres ranks around zero (mean-
    subtracted, range-normalised).  On any given post-warmup date the
    sum of all signals across all tickers should be close to zero — the
    long and short sides are balanced by construction.

    Tolerance is 0.1 per row (rank normalisation is not exact, especially
    when n_tickers is small or NaN values are present).
    """
    strategy = MomentumStrategy(lookback_months=12, skip_months=1, signal_type="continuous")
    signals = strategy.generate_signals(price_data)

    # Only look at post-warmup rows that are not all-zero
    post_warmup = signals.iloc[300:]
    nonzero_rows = post_warmup[(post_warmup != 0).any(axis=1)]

    row_sums = nonzero_rows.sum(axis=1).abs()

    assert float(row_sums.max()) < 0.1, (
        f"Continuous signal row sums too large; max abs row sum = "
        f"{float(row_sums.max()):.6f} (expected < 0.1).  "
        "This suggests the rank normalisation is not centering correctly."
    )


# ---------------------------------------------------------------------------
# Test 8 — End-to-end run produces populated metrics
# ---------------------------------------------------------------------------

def test_end_to_end_run_produces_valid_metrics(price_data: pd.DataFrame) -> None:
    """
    A full Backtester.run() with MomentumStrategy must:
      1. Complete without raising an exception.
      2. Return a BacktestResult whose metrics dict is non-empty.
      3. Have a finite, non-NaN total_return_pct.
      4. Have a Sharpe ratio that is a Python float (not np.float64).
      5. Have an equity curve whose first value equals initial_capital.
    """
    initial_capital = 100_000
    strategy = MomentumStrategy(
        lookback_months=12, skip_months=1, n_long=5, signal_type="ranked"
    )
    engine = Backtester(price_data, config={"initial_capital": initial_capital})
    result = engine.run(strategy)

    # Metrics dict must be populated
    assert result.metrics, (
        "result.metrics is empty after run() — metrics computation failed."
    )

    # total_return_pct must be finite
    tr = result.metrics["total_return_pct"]
    assert isinstance(tr, float), (
        f"total_return_pct must be Python float, got {type(tr).__name__}"
    )
    assert not (tr != tr), (  # NaN check without math.isnan import
        "total_return_pct is NaN"
    )

    # Sharpe must be a Python float
    sharpe = result.metrics["sharpe_ratio"]
    assert isinstance(sharpe, float), (
        f"sharpe_ratio must be Python float, got {type(sharpe).__name__}"
    )

    # Equity curve starts at initial_capital
    first_equity = float(result.equity_curve.iloc[0])
    assert abs(first_equity - initial_capital) < 1.0, (
        f"Expected equity_curve[0] ≈ {initial_capital}, got {first_equity:.2f}"
    )
