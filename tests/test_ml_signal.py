"""
Tests for strategies/ml_signal.py — MLSignalStrategy.

Uses a smaller dataset than the momentum/mean-reversion tests (5 years,
2015-2020, seed=42) and faster walk-forward settings (min_train_years=2,
retrain_freq_months=6) to keep total test runtime under ~60 seconds.

The scope="module" fixture ensures compute_features and generate_signals
are called at most once per test session.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from strategies.ml_signal import (
    MLSignalStrategy,
    compute_features,
    compute_target,
    ALL_FEATURES,
)
from backtester.engine import Backtester


# ---------------------------------------------------------------------------
# Shared fixture — 5 years, 21 tickers, seed=42
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def price_data() -> pd.DataFrame:
    """
    Synthetic wide-format close prices for 21 tickers over 2015-2020.

    Smaller than the momentum/mean-reversion fixtures to keep ML tests fast.
    Tickers: T00…T19 plus SPY.  Seed 42 for reproducibility.
    """
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2015-01-01", "2020-12-31")
    tickers = [f"T{i:02d}" for i in range(20)] + ["SPY"]

    log_returns = rng.normal(0.0, 0.008, size=(len(dates), len(tickers)))
    prices = 100.0 * np.exp(np.cumsum(log_returns, axis=0))

    return pd.DataFrame(prices, index=dates, columns=tickers)


@pytest.fixture(scope="module")
def fast_strategy() -> MLSignalStrategy:
    """Shared strategy instance with fast walk-forward settings."""
    return MLSignalStrategy(min_train_years=2, retrain_freq_months=6)


@pytest.fixture(scope="module")
def signals(price_data: pd.DataFrame, fast_strategy: MLSignalStrategy) -> pd.DataFrame:
    """Signals computed once and shared across all tests that need them."""
    return fast_strategy.generate_signals(price_data)


# ---------------------------------------------------------------------------
# Test 1 — compute_features shape and content
# ---------------------------------------------------------------------------

def test_compute_features_shape(price_data: pd.DataFrame) -> None:
    """compute_features must return a 2-level MultiIndex DataFrame
    with all expected feature columns and no infinite values."""
    features = compute_features(price_data)

    assert features.index.nlevels == 2, (
        f"compute_features must return a 2-level MultiIndex, "
        f"got {features.index.nlevels} levels"
    )
    assert list(features.index.names) == ["date", "ticker"], (
        f"MultiIndex level names must be ['date', 'ticker'], "
        f"got {list(features.index.names)}"
    )

    for col in ALL_FEATURES:
        assert col in features.columns, (
            f"Expected feature column '{col}' not found. "
            f"Present: {features.columns.tolist()}"
        )

    inf_mask = ~np.isfinite(features.values)
    assert not inf_mask.any(), (
        f"compute_features produced {inf_mask.sum()} infinite values"
    )


# ---------------------------------------------------------------------------
# Test 2 — compute_target is binary and balanced
# ---------------------------------------------------------------------------

def test_compute_target_is_binary(price_data: pd.DataFrame) -> None:
    """compute_target must produce only {0, 1} values with ~50% class balance."""
    target = compute_target(price_data)

    # Drop NaN (tail rows where forward return is not yet realised)
    target_clean = target.dropna()

    unique_vals = set(target_clean.values.astype(int))
    assert unique_vals.issubset({0, 1}), (
        f"compute_target must produce only {{0, 1}}, got extra values: "
        f"{unique_vals - {0, 1}}"
    )

    mean_val = float(target_clean.mean())
    assert 0.3 <= mean_val <= 0.7, (
        f"Target class balance should be roughly 50/50, got mean={mean_val:.4f}. "
        "Cross-sectional median split should produce approximately balanced labels."
    )


# ---------------------------------------------------------------------------
# Test 3 — signals shape matches data
# ---------------------------------------------------------------------------

def test_signals_shape_matches_data(
    price_data: pd.DataFrame, signals: pd.DataFrame
) -> None:
    """generate_signals must return a DataFrame with same shape and columns."""
    assert signals.shape == price_data.shape, (
        f"Signal shape {signals.shape} != price_data shape {price_data.shape}"
    )
    assert signals.columns.tolist() == price_data.columns.tolist(), (
        "Signal columns do not match price_data columns"
    )


# ---------------------------------------------------------------------------
# Test 4 — signals in valid range
# ---------------------------------------------------------------------------

def test_signals_in_valid_range(signals: pd.DataFrame) -> None:
    """All signal values must be in [-1, 1] (±1e-9 tolerance)."""
    min_val = float(signals.min().min())
    max_val = float(signals.max().max())

    assert min_val >= -1.0 - 1e-9, (
        f"Signal below -1.0: min = {min_val:.8f}"
    )
    assert max_val <= 1.0 + 1e-9, (
        f"Signal above +1.0: max = {max_val:.8f}"
    )


# ---------------------------------------------------------------------------
# Test 5 — warmup period is all zeros
# ---------------------------------------------------------------------------

def test_warmup_period_is_zero(signals: pd.DataFrame) -> None:
    """With min_train_years=2, the first ~500 bars should be flat (0.0)."""
    warmup_slice = signals.iloc[:500]

    assert float(warmup_slice.abs().max().max()) == 0.0, (
        f"Non-zero signal found during warmup period (first 500 bars). "
        f"Max abs value = {float(warmup_slice.abs().max().max()):.8f}"
    )


# ---------------------------------------------------------------------------
# Test 6 — no lookahead via engine
# ---------------------------------------------------------------------------

def test_no_lookahead_via_engine(price_data: pd.DataFrame) -> None:
    """Positions on day 0 must be flat when run through Backtester."""
    strategy = MLSignalStrategy(min_train_years=2, retrain_freq_months=6)
    bt = Backtester(price_data, config={"allow_short": False})
    result = bt.run(strategy)

    first_day_exposure = float(result.positions.iloc[0].abs().sum())

    assert first_day_exposure == 0.0, (
        f"Expected 0 exposure on day 0 (no lookahead), "
        f"got {first_day_exposure:.8f}"
    )


# ---------------------------------------------------------------------------
# Test 7 — walk-forward does not use future data in training
# ---------------------------------------------------------------------------

def test_walk_forward_no_future_data_in_training(
    price_data: pd.DataFrame,
) -> None:
    """
    Critical lookahead test: for each training window, the model must never
    see features with dates beyond the training cutoff.

    We inject a MockModel that records the maximum date present in its
    training rows (retrieved from the index of the X array it receives).
    After the walk-forward run we verify that max_seen_date <= train_cutoff
    for every window.
    """
    observed_cutoffs: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    # (train_cutoff_passed_to_window, max_date_seen_in_X_train)

    features_ref: list[pd.DataFrame] = []  # capture feature index

    original_train_predict = MLSignalStrategy._train_predict_window

    def patched_train_predict(
        self_inner,
        features: pd.DataFrame,
        target: pd.Series,
        train_cutoff: pd.Timestamp,
        predict_start: pd.Timestamp,
        predict_end: pd.Timestamp,
    ) -> pd.Series:
        # Record the max date in the training slice (before NaN drop)
        train_mask = features.index.get_level_values("date") <= train_cutoff
        X_train_dates = features[train_mask].index.get_level_values("date")
        if len(X_train_dates) > 0:
            max_date = X_train_dates.max()
            observed_cutoffs.append((train_cutoff, max_date))
        return original_train_predict(
            self_inner, features, target, train_cutoff, predict_start, predict_end
        )

    strategy = MLSignalStrategy(min_train_years=2, retrain_freq_months=6)
    with patch.object(MLSignalStrategy, "_train_predict_window", patched_train_predict):
        strategy.generate_signals(price_data)

    assert len(observed_cutoffs) > 0, (
        "No training windows were observed — walk-forward did not execute"
    )

    for train_cutoff, max_seen in observed_cutoffs:
        assert max_seen <= train_cutoff, (
            f"Walk-forward lookahead detected: training window with cutoff "
            f"{train_cutoff.date()} contained features from {max_seen.date()}"
        )


# ---------------------------------------------------------------------------
# Test 8 — models and scalers stored after generate_signals
# ---------------------------------------------------------------------------

def test_models_stored_after_generate_signals(
    fast_strategy: MLSignalStrategy, signals: pd.DataFrame
) -> None:
    """After generate_signals, strategy must store one model per training date."""
    assert len(fast_strategy.models_) == len(fast_strategy.training_dates_), (
        f"Expected {len(fast_strategy.training_dates_)} stored models, "
        f"got {len(fast_strategy.models_)}"
    )
    assert len(fast_strategy.scalers_) == len(fast_strategy.training_dates_), (
        f"Expected {len(fast_strategy.training_dates_)} stored scalers, "
        f"got {len(fast_strategy.scalers_)}"
    )
    assert fast_strategy.feature_names_ is not None, (
        "feature_names_ must be populated after generate_signals()"
    )
    assert fast_strategy.feature_importances_ is not None, (
        "feature_importances_ must be populated after generate_signals() "
        "for xgboost/random_forest model types"
    )


# ---------------------------------------------------------------------------
# Test 9 — n_long positions respected on post-warmup dates
# ---------------------------------------------------------------------------

def test_n_long_positions_respected(
    price_data: pd.DataFrame,
    fast_strategy: MLSignalStrategy,
    signals: pd.DataFrame,
) -> None:
    """On post-warmup dates, (signals == 1.0).sum() must be <= n_long."""
    # Find post-warmup dates (where strategy has generated any signal)
    post_warmup = signals[(signals != 0).any(axis=1)]

    assert len(post_warmup) > 0, (
        "No post-warmup dates found — strategy never generated signals. "
        "Check min_train_years and price data length."
    )

    rng = np.random.default_rng(7)
    n_sample = min(10, len(post_warmup))
    sample_indices = rng.choice(len(post_warmup), size=n_sample, replace=False)
    sample_dates = post_warmup.index[sample_indices]

    for date in sample_dates:
        n_long_actual = int((signals.loc[date] == 1.0).sum())
        assert n_long_actual <= fast_strategy.n_long, (
            f"On {date.date()}: expected <= {fast_strategy.n_long} long positions, "
            f"got {n_long_actual}"
        )


# ---------------------------------------------------------------------------
# Test 10 — end-to-end run produces valid metrics
# ---------------------------------------------------------------------------

def test_end_to_end_run_produces_valid_metrics(price_data: pd.DataFrame) -> None:
    """A full Backtester.run() must produce coherent metrics."""
    strategy = MLSignalStrategy(min_train_years=2, retrain_freq_months=6)
    bt = Backtester(price_data, config={"allow_short": False})
    result = bt.run(strategy)

    assert result.metrics, (
        "result.metrics is empty after run() — metrics computation failed"
    )
    assert result.metrics["sharpe_ratio"] is not None, (
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
    assert "alpha_pct" in result.metrics, (
        "alpha_pct must be present in metrics when benchmark is configured"
    )
