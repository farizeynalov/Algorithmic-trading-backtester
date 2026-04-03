"""
Backtesting engine.

Implementation plan (Phase 3.1)
---------------------------------
Event-driven loop that:
1. Iterates over each trading day in chronological order.
2. Feeds OHLCV data up to (but not including) the current bar to the strategy
   so that no look-ahead bias is introduced.
3. Computes target position weights from the strategy signal.
4. Calculates trade sizes accounting for current holdings, capital, and the
   cost model from `costs.py`.
5. Records daily portfolio value, positions, and cash for downstream analysis.

Key design considerations:
- Supports both single-asset and multi-asset (portfolio) strategies.
- Position sizing defaults to equal-weight but is overridable.
- Handles dividends and stock splits via adjusted close prices.
"""

from __future__ import annotations

import pandas as pd

from backtester.costs import CostModel
from backtester.metrics import compute_metrics
from config import INITIAL_CAPITAL
from strategies.base import BaseStrategy


class BacktestResult:
    """
    Container for the outputs of a completed backtest run.

    Attributes
    ----------
    equity_curve : pd.Series
        Daily portfolio value indexed by date.
    positions : pd.DataFrame
        Daily position weights per ticker.
    trades : pd.DataFrame
        Log of every executed trade with date, ticker, direction, size, cost.
    metrics : dict
        Computed performance metrics (Sharpe, CAGR, max drawdown, …).
    """

    def __init__(
        self,
        equity_curve: pd.Series,
        positions: pd.DataFrame,
        trades: pd.DataFrame,
        metrics: dict,
    ) -> None:
        self.equity_curve = equity_curve
        self.positions = positions
        self.trades = trades
        self.metrics = metrics

    def __repr__(self) -> str:
        sharpe = self.metrics.get("sharpe_ratio", float("nan"))
        cagr = self.metrics.get("cagr", float("nan"))
        return f"BacktestResult(sharpe={sharpe:.3f}, cagr={cagr:.2%})"


class BacktestEngine:
    """
    Orchestrates a strategy backtest over a historical price dataset.

    Parameters
    ----------
    strategy : BaseStrategy
        Concrete strategy instance to be tested.
    cost_model : CostModel | None
        Transaction cost model.  If None, uses the default from config.
    initial_capital : float
        Starting portfolio value in USD.
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        cost_model: CostModel | None = None,
        initial_capital: float = INITIAL_CAPITAL,
    ) -> None:
        self.strategy = strategy
        self.cost_model = cost_model or CostModel()
        self.initial_capital = initial_capital

    def run(self, prices: pd.DataFrame) -> BacktestResult:
        """
        Execute the backtest over `prices` and return a BacktestResult.

        Parameters
        ----------
        prices : pd.DataFrame
            Adjusted closing prices with DatetimeIndex and tickers as columns.

        Returns
        -------
        BacktestResult
            Equity curve, trade log, positions, and performance metrics.
        """
        raise NotImplementedError(
            "BacktestEngine.run is not yet implemented. "
            "Scheduled for Phase 3.1."
        )
