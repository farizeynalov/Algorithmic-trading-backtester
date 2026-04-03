"""
Abstract base class for all trading strategies.

Signal convention
-----------------
Every strategy must return a pd.Series of floats in the closed interval [-1, 1]:

    +1.0  => full long  (allocate 100% of capital to this instrument)
     0.0  => flat       (no position)
    -1.0  => full short (short 100% of capital in this instrument)

Fractional values are allowed to express partial positioning, e.g. 0.5 for a
half-sized long.  The backtesting engine interprets the signal as a *target
weight* and handles the rebalancing logic.

All concrete strategy classes must inherit from BaseStrategy and implement
both abstract methods.
"""

from abc import ABC, abstractmethod

import pandas as pd


class BaseStrategy(ABC):
    """
    Abstract base class that every trading strategy must subclass.

    Subclasses are responsible for:
    - Implementing `generate_signals` to produce position targets from OHLCV data.
    - Implementing `get_name` to return a human-readable identifier used in
      reports and plot legends.
    """

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Compute position signals from market data.

        Parameters
        ----------
        data : pd.DataFrame
            OHLCV DataFrame with a DatetimeIndex.  Expected columns:
            ['Open', 'High', 'Low', 'Close', 'Volume'].  Additional
            pre-computed feature columns are allowed and may be used.

        Returns
        -------
        pd.Series
            Float series aligned to `data.index` with values in [-1, 1].
            Index must match `data.index` exactly so the engine can join
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
