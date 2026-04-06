"""
backtester package

Core backtesting engine, performance metrics, and transaction cost modelling.
"""

from backtester.data_loader import DataLoader
from backtester.engine import Backtester, BacktestResult

__all__ = ["DataLoader", "Backtester", "BacktestResult"]
