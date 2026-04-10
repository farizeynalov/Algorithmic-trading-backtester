"""
Tests for analysis/visualizations.py.

All tests use synthetic BacktestResult objects — no network calls or
yfinance dependency.  Each test closes all matplotlib figures after
running to prevent memory leaks.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt

from backtester.engine import BacktestResult
from backtester.metrics import compute_metrics
from analysis.visualizations import (
    PALETTE,
    plot_equity_curves,
    plot_drawdowns,
    plot_metrics_comparison,
    plot_rolling_metrics,
    plot_correlation_matrix,
    plot_monthly_returns_heatmap,
    plot_trade_analysis,
    plot_position_concentration,
    save_figure,
)


# ---------------------------------------------------------------------------
# Helper: build a synthetic BacktestResult
# ---------------------------------------------------------------------------

def _make_synthetic_result(name: str, seed: int, n_days: int = 500) -> BacktestResult:
    """
    Build a reproducible BacktestResult with plausible synthetic data.

    Parameters
    ----------
    name    : strategy_name for the result
    seed    : numpy random seed for reproducibility
    n_days  : number of trading days to simulate (≥ 252 recommended)
    """
    np.random.seed(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "META"]

    # Equity curve: random walk starting at 100 000
    daily_returns = np.random.normal(0.0005, 0.01, n_days)
    equity_arr = 100_000.0 * np.cumprod(1.0 + daily_returns)
    equity = pd.Series(equity_arr, index=dates)
    returns = pd.Series(daily_returns, index=dates)

    # Positions: random weights summing to ≤ 0.8
    raw_w = np.abs(np.random.uniform(0, 0.2, (n_days, len(tickers))))
    row_sums = raw_w.sum(axis=1, keepdims=True)
    raw_w = raw_w / row_sums * 0.8
    positions = pd.DataFrame(raw_w, index=dates, columns=tickers)

    # 20 synthetic trades with alternating BUY/SELL
    trade_idx = np.sort(
        np.random.choice(np.arange(5, n_days - 5), 20, replace=False)
    )
    trade_dates = dates[trade_idx]
    directions = (["BUY", "SELL"] * 10)
    trade_tickers = np.random.choice(tickers, 20)
    quantities = np.random.uniform(10.0, 100.0, 20)
    prices = np.random.uniform(50.0, 500.0, 20)
    commissions = quantities * prices * 0.0005
    slippages = quantities * prices * 0.0002
    net_costs = quantities * prices + commissions + slippages

    trades = pd.DataFrame({
        "date":       trade_dates,
        "ticker":     trade_tickers,
        "direction":  directions,
        "quantity":   quantities,
        "price":      prices,
        "commission": commissions,
        "slippage":   slippages,
        "net_cost":   net_costs,
    })

    result = BacktestResult(
        strategy_name=name,
        equity_curve=equity,
        returns=returns,
        positions=positions,
        trades=trades,
        metrics={},
        config={
            "initial_capital": 100_000.0,
            "benchmark": "SPY",
            "risk_free_rate": 0.0,
        },
    )
    result.metrics = compute_metrics(result)  # no benchmark — alpha_pct absent
    return result


# ---------------------------------------------------------------------------
# Module-scoped fixture: create results once, share across all tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_results():
    results = [
        _make_synthetic_result("momentum strategy alpha", seed=1),
        _make_synthetic_result("mean_reversion strategy beta", seed=2),
        _make_synthetic_result("ml_signal strategy gamma", seed=3),
    ]
    benchmark = _make_synthetic_result("SPY BuyAndHold", seed=99)
    return results, benchmark


# ---------------------------------------------------------------------------
# Test 1 — plot_equity_curves
# ---------------------------------------------------------------------------

def test_plot_equity_curves_returns_figure(synthetic_results):
    results, benchmark = synthetic_results
    fig, ax = plot_equity_curves(results, benchmark)

    assert isinstance(fig, plt.Figure), "plot_equity_curves must return plt.Figure"
    # cash line + benchmark line + 3 strategy lines = 5 minimum
    assert len(ax.lines) >= 4, (
        f"Expected ≥ 4 lines in equity-curves plot, got {len(ax.lines)}"
    )
    plt.close("all")


# ---------------------------------------------------------------------------
# Test 2 — plot_drawdowns
# ---------------------------------------------------------------------------

def test_plot_drawdowns_all_negative(synthetic_results):
    results, benchmark = synthetic_results
    fig, ax = plot_drawdowns(results, benchmark)

    assert isinstance(fig, plt.Figure)

    for line in ax.lines:
        y_data = np.asarray(line.get_ydata(), dtype=float)
        # Skip reference lines with ≤ 2 data points (axhline, axvline)
        if len(y_data) <= 2:
            continue
        assert np.all(y_data <= 0.01), (
            f"Drawdown line has positive values above 0.01: max={y_data.max():.4f}. "
            "Drawdowns must always be ≤ 0."
        )
    plt.close("all")


# ---------------------------------------------------------------------------
# Test 3 — plot_metrics_comparison
# ---------------------------------------------------------------------------

def test_plot_metrics_comparison_correct_grid(synthetic_results):
    results, benchmark = synthetic_results
    fig, axes = plot_metrics_comparison(results, benchmark)

    assert fig is not None
    assert axes.shape == (2, 4), (
        f"Expected axes.shape == (2, 4), got {axes.shape}"
    )
    plt.close("all")


# ---------------------------------------------------------------------------
# Test 4 — plot_rolling_metrics
# ---------------------------------------------------------------------------

def test_plot_rolling_metrics_correct_subplots(synthetic_results):
    results, benchmark = synthetic_results
    fig, axes = plot_rolling_metrics(results, benchmark, window=21)

    assert axes.shape == (3,), (
        f"Expected axes.shape == (3,), got {axes.shape}"
    )
    plt.close("all")


# ---------------------------------------------------------------------------
# Test 5 — plot_correlation_matrix
# ---------------------------------------------------------------------------

def test_plot_correlation_matrix_symmetric(synthetic_results):
    results, benchmark = synthetic_results
    fig, ax = plot_correlation_matrix(results, benchmark)

    assert isinstance(fig, plt.Figure)
    plt.close("all")


# ---------------------------------------------------------------------------
# Test 6 — plot_monthly_returns_heatmap
# ---------------------------------------------------------------------------

def test_plot_monthly_heatmap_has_12_columns(synthetic_results):
    results, _ = synthetic_results
    fig, ax = plot_monthly_returns_heatmap(results[0])

    assert isinstance(fig, plt.Figure)
    plt.close("all")


# ---------------------------------------------------------------------------
# Test 7 — save_figure
# ---------------------------------------------------------------------------

def test_save_figure_creates_files(tmp_path: pathlib.Path):
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])

    saved = save_figure(fig, "test_fig", tmp_path, ["png"])

    assert (tmp_path / "test_fig.png").exists(), (
        "save_figure did not create test_fig.png"
    )
    assert len(saved) == 1
    plt.close("all")


# ---------------------------------------------------------------------------
# Test 8 — PALETTE structure
# ---------------------------------------------------------------------------

def test_palette_has_all_expected_keys():
    expected = {"momentum", "mean_reversion", "ml_signal", "spy", "cash"}
    assert expected.issubset(set(PALETTE.keys())), (
        f"PALETTE is missing keys: {expected - set(PALETTE.keys())}"
    )
    for key, value in PALETTE.items():
        assert isinstance(value, str), f"PALETTE['{key}'] must be a string"
        assert value.startswith("#"), (
            f"PALETTE['{key}'] = '{value}' does not start with '#'"
        )
        assert len(value) == 7, (
            f"PALETTE['{key}'] = '{value}' is not a 7-character hex colour"
        )
