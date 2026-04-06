"""
Performance metrics module for the algorithmic trading backtester.

This module is a pure-computation layer: it accepts a BacktestResult and
an optional benchmark BacktestResult and returns a flat dict of floats,
ints, and None values.  It has no knowledge of the engine, strategies,
or data-loading logic.

All values are cast to Python built-in types (float / int / None) before
being returned — never np.float64, np.int64, or pd.Series.

Ratio metrics (Sharpe, Sortino, Calmar, beta, IR, correlation) are rounded
to 4 decimal places using Python's built-in round(), not np.round.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from backtester.engine import BacktestResult


# ---------------------------------------------------------------------------
# Module-level constant
# ---------------------------------------------------------------------------

TRADING_DAYS_PER_YEAR: int = 252


# ---------------------------------------------------------------------------
# Private helper (also exported at module level as drawdown_series)
# ---------------------------------------------------------------------------

def _drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """
    Compute the full drawdown series as a percentage.

    At each date the drawdown is ``(equity - running_peak) / running_peak * 100``.
    The result is always <= 0 (zero means the portfolio is at a new high).

    Parameters
    ----------
    equity_curve : pd.Series
        Daily portfolio value indexed by date.

    Returns
    -------
    pd.Series
        Drawdown in percentage terms; same index as ``equity_curve``.
    """
    running_max = equity_curve.cummax()
    return (equity_curve - running_max) / running_max * 100


# Export _drawdown_series at module level under the public name
drawdown_series = _drawdown_series


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_metrics(
    result: BacktestResult,
    benchmark: BacktestResult | None = None,
    risk_free_rate: float = 0.0,
) -> dict:
    """
    Compute a comprehensive suite of performance metrics from a BacktestResult.

    Parameters
    ----------
    result : BacktestResult
        Completed backtest output produced by ``Backtester.run()``.
    benchmark : BacktestResult | None
        Optional buy-and-hold benchmark result.  When provided, benchmark-
        relative metrics (alpha, beta, IR, correlation) are included.
        When None, those keys are absent from the returned dict.
    risk_free_rate : float
        Annualised risk-free rate as a decimal (e.g. 0.04 for 4%).
        Defaults to 0.0.

    Returns
    -------
    dict
        Flat mapping of metric name → value.  Values are Python float,
        int, or None — never NumPy or Pandas scalar types.

    Notes
    -----
    win_rate_pct is a proxy: it counts the fraction of trades whose
    direction is "BUY" rather than computing realised P&L per round-trip.
    A true win-rate calculation would require matching every BUY with its
    subsequent SELL and comparing entry and exit prices, which is
    intentionally deferred to Phase 4 analytics.
    """
    equity: pd.Series = result.equity_curve
    returns: pd.Series = result.returns
    trades: pd.DataFrame = result.trades
    positions: pd.DataFrame = result.positions

    n_days: int = len(returns)
    daily_rf: float = risk_free_rate / TRADING_DAYS_PER_YEAR

    # ── Returns ───────────────────────────────────────────────────────────────
    total_return_ratio: float = float(equity.iloc[-1] / equity.iloc[0])
    total_return_pct: float = (total_return_ratio - 1.0) * 100.0

    # Annualised return: compound the total return to a 252-day year
    if total_return_ratio <= 0.0:
        ann_return_pct: float = -100.0
    else:
        ann_return_pct = float(
            (total_return_ratio ** (TRADING_DAYS_PER_YEAR / n_days) - 1.0) * 100.0
        )

    ann_vol_pct: float = float(returns.std() * math.sqrt(TRADING_DAYS_PER_YEAR) * 100.0)
    best_day_pct: float = float(returns.max() * 100.0)
    worst_day_pct: float = float(returns.min() * 100.0)
    positive_days_pct: float = float((returns > 0).sum() / n_days * 100.0)

    # ── Risk-adjusted ────────────────────────────────────────────────────────
    excess_returns: pd.Series = returns - daily_rf
    excess_std: float = float(excess_returns.std())

    if excess_std == 0.0 or math.isnan(excess_std):
        sharpe: float = 0.0
    else:
        sharpe = round(
            float(excess_returns.mean() / excess_std * math.sqrt(TRADING_DAYS_PER_YEAR)),
            4,
        )

    # Sortino: denominator uses downside deviation only
    downside_returns: pd.Series = returns[returns < daily_rf]
    if len(downside_returns) == 0:
        downside_std_ann: float = 0.0
    else:
        raw_std: float = float(downside_returns.std())
        downside_std_ann = 0.0 if math.isnan(raw_std) else raw_std * math.sqrt(TRADING_DAYS_PER_YEAR)

    if downside_std_ann == 0.0:
        sortino: float = 0.0
    else:
        sortino = round(
            float((ann_return_pct / 100.0 - risk_free_rate) / downside_std_ann),
            4,
        )

    # ── Drawdown ──────────────────────────────────────────────────────────────
    dd_series: pd.Series = _drawdown_series(equity)
    max_dd_pct: float = float(dd_series.min())

    assert max_dd_pct <= 0.0, (
        f"max_drawdown_pct must be <= 0 but got {max_dd_pct:.6f}.  "
        "This indicates a bug in _drawdown_series."
    )

    if max_dd_pct == 0.0:
        # No drawdown — at or above initial equity at all times
        max_dd_duration: int = 0
        recovery_days: int | None = None
    else:
        trough_date: pd.Timestamp = dd_series.idxmin()
        peak_date: pd.Timestamp = equity.loc[:trough_date].idxmax()
        max_dd_duration = int((trough_date - peak_date).days)

        peak_value: float = float(equity.loc[peak_date])
        after_trough: pd.Series = equity.loc[trough_date:]
        recoveries: pd.Series = after_trough[after_trough > peak_value]
        if len(recoveries) == 0:
            recovery_days = None
        else:
            recovery_date: pd.Timestamp = recoveries.index[0]
            recovery_days = int((recovery_date - trough_date).days)

    # Calmar: annualised return divided by absolute max drawdown
    if max_dd_pct == 0.0:
        calmar: float = 0.0
    else:
        calmar = round(float(ann_return_pct / abs(max_dd_pct)), 4)

    # ── Trading activity ─────────────────────────────────────────────────────
    n_trades: int = int(len(trades))
    initial_cap: float = float(equity.iloc[0])

    if n_trades == 0:
        win_rate_pct: float = 0.0
        avg_trade_duration: float | None = None
        comm_pct: float = 0.0
        slip_pct: float = 0.0
    else:
        # Proxy win-rate: fraction of trades that are BUYs (see docstring)
        win_rate_pct = float((trades["direction"] == "BUY").sum() / n_trades * 100.0)

        # Avg trade duration: pair BUY → next SELL per ticker
        durations: list[int] = []
        for ticker in trades["ticker"].unique():
            t = trades[trades["ticker"] == ticker].sort_values("date")
            buys = t[t["direction"] == "BUY"]
            sells = t[t["direction"] == "SELL"]
            for _, buy_row in buys.iterrows():
                after = sells[sells["date"] > buy_row["date"]]
                if len(after) > 0:
                    durations.append(int((after.iloc[0]["date"] - buy_row["date"]).days))

        avg_trade_duration = (
            float(sum(durations) / len(durations)) if durations else None
        )

        comm_pct = float(trades["commission"].sum() / initial_cap * 100.0)
        slip_pct = float(trades["slippage"].sum() / initial_cap * 100.0)

    cost_pct: float = comm_pct + slip_pct

    # Turnover: average daily absolute weight change × 252 × 100
    if positions is None or positions.empty:
        turnover_ann: float = 0.0
    else:
        daily_turnover: float = float(positions.diff().abs().sum(axis=1).mean())
        turnover_ann = daily_turnover * TRADING_DAYS_PER_YEAR * 100.0

    # ── Assemble base metrics dict ────────────────────────────────────────────
    metrics: dict = {
        # Returns
        "total_return_pct":           total_return_pct,
        "annualized_return_pct":      ann_return_pct,
        "annualized_volatility_pct":  ann_vol_pct,
        "best_day_pct":               best_day_pct,
        "worst_day_pct":              worst_day_pct,
        "positive_days_pct":          positive_days_pct,
        # Risk-adjusted
        "sharpe_ratio":               sharpe,
        "sortino_ratio":              sortino,
        "calmar_ratio":               calmar,
        # Drawdown
        "max_drawdown_pct":           max_dd_pct,
        "max_drawdown_duration_days": max_dd_duration,
        "recovery_days":              recovery_days,
        # Trading activity
        "n_trades":                   n_trades,
        "win_rate_pct":               win_rate_pct,
        "avg_trade_duration_days":    avg_trade_duration,
        "total_commission_pct":       comm_pct,
        "total_slippage_pct":         slip_pct,
        "total_cost_pct":             cost_pct,
        "turnover_annual_pct":        turnover_ann,
    }

    # ── Benchmark-relative (only when benchmark is provided) ─────────────────
    if benchmark is not None:
        bench_equity: pd.Series = benchmark.equity_curve
        bench_returns: pd.Series = benchmark.returns

        bench_n_days: int = len(bench_returns)
        bench_total_ratio: float = float(bench_equity.iloc[-1] / bench_equity.iloc[0])
        bench_total_ret: float = (bench_total_ratio - 1.0) * 100.0

        if bench_total_ratio <= 0.0:
            bench_ann_ret: float = -100.0
        else:
            bench_ann_ret = float(
                (bench_total_ratio ** (TRADING_DAYS_PER_YEAR / bench_n_days) - 1.0) * 100.0
            )

        alpha_pct: float = float(ann_return_pct - bench_ann_ret)

        # Align returns by date index for covariance-based metrics
        aligned: pd.DataFrame = pd.concat(
            [returns.rename("strat"), bench_returns.rename("bench")],
            axis=1,
            join="inner",
        ).dropna()

        if len(aligned) < 2:
            beta: float = 0.0
            corr: float = 0.0
            ir: float = 0.0
        else:
            strat_al: pd.Series = aligned["strat"]
            bench_al: pd.Series = aligned["bench"]

            bench_var: float = float(bench_al.var())
            if bench_var == 0.0 or math.isnan(bench_var):
                beta = 0.0
            else:
                beta = round(float(strat_al.cov(bench_al) / bench_var), 4)

            import warnings as _warnings
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore", RuntimeWarning)
                corr_raw: float = float(strat_al.corr(bench_al))
            corr = round(0.0 if math.isnan(corr_raw) else corr_raw, 4)

            active: pd.Series = strat_al - bench_al
            active_std: float = float(active.std())
            if active_std == 0.0 or math.isnan(active_std):
                ir = 0.0
            else:
                ir = round(
                    float(active.mean() / active_std * math.sqrt(TRADING_DAYS_PER_YEAR)),
                    4,
                )

        # Benchmark drawdown
        bench_dd: pd.Series = _drawdown_series(bench_equity)
        bench_max_dd: float = float(bench_dd.min())

        # Benchmark Sharpe
        bench_excess: pd.Series = bench_returns - daily_rf
        bench_exc_std: float = float(bench_excess.std())
        if bench_exc_std == 0.0 or math.isnan(bench_exc_std):
            bench_sharpe: float = 0.0
        else:
            bench_sharpe = round(
                float(bench_excess.mean() / bench_exc_std * math.sqrt(TRADING_DAYS_PER_YEAR)),
                4,
            )

        metrics.update({
            "alpha_pct":                      alpha_pct,
            "beta":                            beta,
            "correlation_with_benchmark":      corr,
            "information_ratio":               ir,
            "benchmark_total_return_pct":      bench_total_ret,
            "benchmark_annualized_return_pct": bench_ann_ret,
            "benchmark_max_drawdown_pct":      bench_max_dd,
            "benchmark_sharpe_ratio":          bench_sharpe,
        })

    return metrics
