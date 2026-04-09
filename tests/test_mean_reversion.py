"""
Tests for strategies/mean_reversion.py — MeanReversionStrategy.

All tests use a single shared synthetic price DataFrame (21 tickers,
2015-2024, seed=0, 0.8% daily vol) so that every test runs in isolation
without any network dependency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategies.mean_reversion import MeanReversionStrategy
from backtester.engine import Backtester


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def price_data() -> pd.DataFrame:
    """
    Synthetic wide-format close prices for 21 tickers over 2015-2024.

    Tickers: T00…T19 plus SPY (used as the default benchmark by the engine).
    Each ticker follows a geometric random walk with 0.8% daily vol.
    Seed 0 ensures deterministic results across all tests.
    """
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2015-01-01", "2024-12-31")
    tickers = [f"T{i:02d}" for i in range(20)] + ["SPY"]

    log_returns = rng.normal(0.0, 0.008, size=(len(dates), len(tickers)))
    prices = 100.0 * np.exp(np.cumsum(log_returns, axis=0))

    return pd.DataFrame(prices, index=dates, columns=tickers)


# ---------------------------------------------------------------------------
# Test 1 — Signal shape matches input data
# ---------------------------------------------------------------------------

def test_signals_shape_matches_data(price_data: pd.DataFrame) -> None:
    """generate_signals() must return a DataFrame with the same shape, index,
    and columns as the input price data."""
    strategy = MeanReversionStrategy()
    signals = strategy.generate_signals(price_data)

    assert signals.shape == price_data.shape, (
        f"Signal shape {signals.shape} != price_data shape {price_data.shape}"
    )
    assert signals.columns.tolist() == price_data.columns.tolist(), (
        "Signal columns do not match price_data columns"
    )
    assert signals.index.equals(price_data.index), (
        "Signal index does not match price_data index"
    )


# ---------------------------------------------------------------------------
# Test 2 — All signal values are in [-1, 1]
# ---------------------------------------------------------------------------

def test_signals_in_valid_range(price_data: pd.DataFrame) -> None:
    """Every value in the signal DataFrame must be in [-1, 1] (±1e-9 tolerance)."""
    for signal_type in ("binary", "scaled"):
        strategy = MeanReversionStrategy(signal_type=signal_type)
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
# Test 3 — Binary signals only contain {-1, 0, +1}
# ---------------------------------------------------------------------------

def test_binary_signals_only_contain_valid_values(price_data: pd.DataFrame) -> None:
    """With signal_type='binary', the only values that may appear are
    exactly -1.0, 0.0, and +1.0."""
    strategy = MeanReversionStrategy(signal_type="binary")
    signals = strategy.generate_signals(price_data)

    unique_vals = set(np.round(signals.values.ravel(), 10))
    allowed = {-1.0, 0.0, 1.0}
    unexpected = unique_vals - allowed

    assert not unexpected, (
        f"Unexpected values in binary signals: {sorted(unexpected)}"
    )


# ---------------------------------------------------------------------------
# Test 4 — Warmup period is all zeros
# ---------------------------------------------------------------------------

def test_warmup_period_is_zero(price_data: pd.DataFrame) -> None:
    """With default bb_window=20 and rsi_window=14, the first 34 bars must be
    identically 0.0 (warmup = bb_window + rsi_window)."""
    strategy = MeanReversionStrategy(bb_window=20, rsi_window=14)
    signals = strategy.generate_signals(price_data)

    warmup = 20 + 14  # bb_window + rsi_window
    warmup_slice = signals.iloc[:warmup]

    assert float(warmup_slice.abs().max().max()) == 0.0, (
        f"Non-zero signal found during warmup period (first {warmup} bars). "
        f"Max abs value = {float(warmup_slice.abs().max().max()):.8f}"
    )


# ---------------------------------------------------------------------------
# Test 5 — RSI computed correctly
# ---------------------------------------------------------------------------

def test_rsi_computed_correctly(price_data: pd.DataFrame) -> None:
    """_compute_rsi() must return values in [0, 100] with no NaN after
    rsi_window bars, and the same index as the input."""
    strategy = MeanReversionStrategy()
    ticker = price_data.columns[0]
    rsi = strategy._compute_rsi(price_data[ticker])

    assert rsi.index.equals(price_data.index), (
        "RSI index must match input price series index"
    )
    assert float(rsi.min()) >= 0.0 - 1e-9, (
        f"RSI value below 0: min = {float(rsi.min()):.6f}"
    )
    assert float(rsi.max()) <= 100.0 + 1e-9, (
        f"RSI value above 100: max = {float(rsi.max()):.6f}"
    )

    post_warmup_rsi = rsi.iloc[strategy.rsi_window:]
    assert post_warmup_rsi.isna().sum() == 0, (
        f"NaN values found in RSI after warmup period: "
        f"{post_warmup_rsi.isna().sum()} NaNs"
    )


# ---------------------------------------------------------------------------
# Test 6 — Bollinger Bands ordering: upper >= middle >= lower
# ---------------------------------------------------------------------------

def test_bands_upper_ge_middle_ge_lower(price_data: pd.DataFrame) -> None:
    """After the warmup period, upper >= middle and middle >= lower must hold
    at every non-NaN position."""
    strategy = MeanReversionStrategy()
    ticker = price_data.columns[0]
    upper, middle, lower = strategy._compute_bands(price_data[ticker])

    # Drop NaN rows (warmup period)
    valid = ~(upper.isna() | middle.isna() | lower.isna())

    assert (upper[valid] >= middle[valid]).all(), (
        "Upper band must be >= middle band at all non-NaN positions"
    )
    assert (middle[valid] >= lower[valid]).all(), (
        "Middle band must be >= lower band at all non-NaN positions"
    )


# ---------------------------------------------------------------------------
# Test 7 — No lookahead via engine (first day positions are flat)
# ---------------------------------------------------------------------------

def test_no_lookahead_via_engine(price_data: pd.DataFrame) -> None:
    """When run through Backtester, positions on day 0 must be 0 (flat).
    The engine's shift(1) in _resolve_signals() enforces this structurally."""
    strategy = MeanReversionStrategy()
    engine = Backtester(price_data)
    result = engine.run(strategy)

    first_day_exposure = float(result.positions.iloc[0].abs().sum())

    assert first_day_exposure == 0.0, (
        f"Expected 0 exposure on day 0 (no lookahead), "
        f"got {first_day_exposure:.8f}"
    )


# ---------------------------------------------------------------------------
# Test 8 — exit_at_mean generates >= trades than exit_at_band
# ---------------------------------------------------------------------------

def test_exit_at_mean_has_more_trades_than_exit_at_band(
    price_data: pd.DataFrame,
) -> None:
    """Exiting at the mean is a tighter exit condition than waiting for the
    far band, producing at least as many round-trips."""
    engine = Backtester(price_data)

    result_mean = engine.run(MeanReversionStrategy(exit_at_mean=True))
    result_band = engine.run(MeanReversionStrategy(exit_at_mean=False))

    n_mean = result_mean.metrics["n_trades"]
    n_band = result_band.metrics["n_trades"]

    assert n_mean >= n_band, (
        f"exit_at_mean should produce >= trades than exit_at_band, "
        f"but got {n_mean} (mean) vs {n_band} (band)"
    )


# ---------------------------------------------------------------------------
# Test 9 — Scaled signal: intensity increases with distance from band
# ---------------------------------------------------------------------------

def test_scaled_signal_intensity_increases_with_distance(
    price_data: pd.DataFrame,
) -> None:
    """A price series further below the lower band should produce a higher
    absolute scaled signal than one closer to the band."""
    strategy = MeanReversionStrategy(signal_type="scaled", rsi_oversold=49.0)

    # Build a controlled synthetic series: 60-day flat period followed by
    # a sharp drop so that the last bar is below the lower band.
    rng = np.random.default_rng(42)
    n = 200
    dates = pd.bdate_range("2018-01-01", periods=n)

    # Baseline: small drop (price just below lower band)
    log_r_small = np.zeros(n)
    log_r_small[60:] = -0.005  # gentle decline
    prices_small = pd.Series(
        100.0 * np.exp(np.cumsum(log_r_small)), index=dates, name="X"
    )

    # Large drop: price far below lower band
    log_r_large = np.zeros(n)
    log_r_large[60:] = -0.03  # steep decline
    prices_large = pd.Series(
        100.0 * np.exp(np.cumsum(log_r_large)), index=dates, name="X"
    )

    sig_small = strategy._signals_for_ticker(prices_small)
    sig_large = strategy._signals_for_ticker(prices_large)

    # Find post-warmup dates where the large-drop series is long (signal > 0)
    warmup = strategy.bb_window + strategy.rsi_window
    post = sig_large.iloc[warmup:]
    long_days = post[post > 0]

    if len(long_days) > 0:
        # On the days where both series are in a long position, the
        # larger deviation should have a higher or equal signal
        common_long_days = long_days.index.intersection(
            sig_small[sig_small > 0].index
        )
        if len(common_long_days) > 0:
            assert float(sig_large[common_long_days].mean()) >= float(
                sig_small[common_long_days].mean()
            ), (
                "Scaled signal intensity should be higher for larger price "
                "deviation from the lower Bollinger Band"
            )


# ---------------------------------------------------------------------------
# Test 10 — End-to-end run produces valid metrics
# ---------------------------------------------------------------------------

def test_end_to_end_run_produces_valid_metrics(price_data: pd.DataFrame) -> None:
    """A full Backtester.run() with MeanReversionStrategy must complete and
    return a result with coherent metric values."""
    strategy = MeanReversionStrategy()
    engine = Backtester(price_data)
    result = engine.run(strategy)

    assert result.metrics, (
        "result.metrics is empty after run() — metrics computation failed"
    )

    sharpe = result.metrics["sharpe_ratio"]
    assert sharpe is not None, (
        "sharpe_ratio must not be None"
    )
    assert isinstance(result.metrics["total_return_pct"], float), (
        f"total_return_pct must be Python float, "
        f"got {type(result.metrics['total_return_pct']).__name__}"
    )
    assert result.metrics["n_trades"] >= 0, (
        f"n_trades must be >= 0, got {result.metrics['n_trades']}"
    )
    assert result.metrics["max_drawdown_pct"] <= 0, (
        f"max_drawdown_pct must be <= 0, got {result.metrics['max_drawdown_pct']}"
    )
    assert result.metrics["turnover_annual_pct"] >= 0, (
        f"turnover_annual_pct must be >= 0, "
        f"got {result.metrics['turnover_annual_pct']}"
    )
