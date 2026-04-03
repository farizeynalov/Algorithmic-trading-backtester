"""
Mean-reversion strategy.

Implementation plan (Phase 2.2)
---------------------------------
Exploits short-term price over-extension using z-score of price relative to a
rolling moving average.  When price is statistically stretched above its mean
the strategy goes short, and vice versa.

Key design choices:
- Z-score computed over a configurable rolling window.
- Entry threshold (e.g., |z| > 2) and exit threshold (e.g., |z| < 0.5).
- Optional Bollinger Band overlay for visual confirmation.
"""

import pandas as pd

from strategies.base import BaseStrategy


class MeanReversionStrategy(BaseStrategy):
    """
    Z-score mean-reversion strategy.

    Computes a rolling z-score of closing price over `window` days and
    generates a fade signal: go long when z < -`entry_z`, short when
    z > +`entry_z`, and exit when |z| < `exit_z`.

    Parameters
    ----------
    window : int
        Rolling window (in trading days) for mean and std computation.
        Default is 20.
    entry_z : float
        Absolute z-score threshold to enter a position.  Default is 2.0.
    exit_z : float
        Absolute z-score threshold to exit an existing position.  Default is 0.5.
    """

    def __init__(
        self,
        window: int = 20,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
    ) -> None:
        self.window = window
        self.entry_z = entry_z
        self.exit_z = exit_z

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Compute mean-reversion signals from closing prices.

        Parameters
        ----------
        data : pd.DataFrame
            OHLCV data with DatetimeIndex.

        Returns
        -------
        pd.Series
            Signal series in [-1, 1].
        """
        raise NotImplementedError(
            "MeanReversionStrategy.generate_signals is not yet implemented. "
            "Scheduled for Phase 2.2."
        )

    def get_name(self) -> str:
        return (
            f"MeanReversionStrategy(window={self.window}, "
            f"entry_z={self.entry_z}, exit_z={self.exit_z})"
        )
