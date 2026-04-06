"""
Abstract base class for all trading strategies.

Moved here from strategies/base.py so that backtester/ can import it without
creating a circular dependency (strategies/ imports from backtester/, so
backtester/ must not import from strategies/).

Signal convention
-----------------
Every strategy must return a pd.DataFrame of floats in the closed interval
[-1, 1], with the same DatetimeIndex and columns as the price data passed in:

    +1.0  => full long  (allocate 100% to this instrument)
     0.0  => flat       (no position)
    -1.0  => full short (short 100% in this instrument)

Fractional values express partial positioning.  The backtesting engine
interprets the signal as a *target weight* and handles all rebalancing.

All concrete strategy classes must inherit from BaseStrategy and implement
both abstract methods.
"""

from abc import ABC, abstractmethod

import pandas as pd


class BaseStrategy(ABC):
    """
    Abstract base class that every trading strategy must subclass.

    Subclasses are responsible for:
    - Implementing `generate_signals` to produce position targets from price data.
    - Implementing `get_name` to return a human-readable identifier used in
      reports and plot legends.
    """

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute position signals from market data.

        Parameters
        ----------
        data : pd.DataFrame
            Wide-format adjusted close prices with a DatetimeIndex.
            Columns are ticker symbols.

        Returns
        -------
        pd.DataFrame
            Float DataFrame aligned to `data.index` and `data.columns`
            (or a subset of columns) with values in [-1, 1].
            Index must be a subset of `data.index` so the engine can join
            signals with prices without introducing look-ahead bias.
        """
        raise NotImplementedError

    @abstractmethod
    def get_name(self) -> str:
        """
        Return a short, unique, human-readable name for this strategy.

        Used in report headings, plot legends, and log messages.

        Returns
        -------
        str
            Example: "MomentumStrategy(lookback=20)"
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.get_name()
