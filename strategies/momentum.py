"""
Momentum strategy.

Implementation plan (Phase 2.1)
--------------------------------
Cross-sectional and time-series momentum signals based on:
- Lookback returns (e.g., 12-month return, skipping the most recent month to
  avoid short-term reversal as per Jegadeesh & Titman 1993).
- Optional volatility scaling: divide raw signal by rolling realised volatility
  to equalise risk contribution across instruments.

Signal output: +1 for top momentum decile, -1 for bottom decile, 0 otherwise.
"""

import pandas as pd

from strategies.base import BaseStrategy


class MomentumStrategy(BaseStrategy):
    """
    Time-series momentum strategy.

    Ranks instruments by their trailing return over `lookback` trading days
    (skipping the most recent `skip` days) and goes long/short accordingly.

    Parameters
    ----------
    lookback : int
        Number of trading days used to compute the trailing return signal.
        Default is 252 (approx. 1 year).
    skip : int
        Most-recent days to exclude from the return window to avoid
        short-term reversal contamination.  Default is 21 (approx. 1 month).
    """

    def __init__(self, lookback: int = 252, skip: int = 21) -> None:
        self.lookback = lookback
        self.skip = skip

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Compute momentum signals from closing prices.

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
            "MomentumStrategy.generate_signals is not yet implemented. "
            "Scheduled for Phase 2.1."
        )

    def get_name(self) -> str:
        return f"MomentumStrategy(lookback={self.lookback}, skip={self.skip})"
