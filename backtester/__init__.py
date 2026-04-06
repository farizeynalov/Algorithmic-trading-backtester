"""
backtester package

Core backtesting engine, performance metrics, and transaction cost modelling.
"""

from backtester.data_loader import DataLoader
from backtester.engine import Backtester, BacktestResult
from backtester.costs import (
    CostModel,
    FlatBpsCostModel,
    TieredCommissionModel,
    SpreadSlippageModel,
    make_cost_model,
)
from backtester.metrics import compute_metrics, drawdown_series

__all__ = [
    "DataLoader",
    "Backtester",
    "BacktestResult",
    "CostModel",
    "FlatBpsCostModel",
    "TieredCommissionModel",
    "SpreadSlippageModel",
    "make_cost_model",
    "compute_metrics",
    "drawdown_series",
]
