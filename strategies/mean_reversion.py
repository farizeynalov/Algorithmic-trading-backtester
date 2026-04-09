"""
Mean-reversion strategy — Bollinger Band + RSI confirmation.

Prices that deviate significantly from their recent moving average tend to
revert.  Bollinger Bands define "significant deviation" as price crossing
beyond N standard deviations of a rolling window.  An RSI confirmation
filter avoids entering into genuine trending moves disguised as mean-reversion
opportunities.

Signal construction is fully vectorized using a forward-fill pattern as a
stateless substitute for a loop-based position state machine.  This keeps the
implementation testable and free from lookahead bias.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtester.base import BaseStrategy


class MeanReversionStrategy(BaseStrategy):
    """
    Bollinger Band mean-reversion strategy with RSI confirmation.

    Parameters
    ----------
    bb_window : int
        Rolling window for Bollinger Band MA and std.  Default 20.
    bb_std : float
        Number of standard deviations (sigma) for band width.  Default 2.0.
    rsi_window : int
        RSI lookback period.  Default 14.
    rsi_oversold : float
        RSI threshold below which long entry is confirmed.  Default 35.
        Entry requires BOTH price < lower band AND RSI < rsi_oversold.
    rsi_overbought : float
        RSI threshold above which short entry is confirmed.  Default 65.
    exit_at_mean : bool
        If True, exit long when price crosses back above the middle band (MA).
        If False, exit long when price crosses above the upper band.
        Default True (faster exit, more round-trips).
    signal_type : str
        ``"binary"``  — signal is exactly 1.0, -1.0, or 0.0.
        ``"scaled"``  — signal intensity proportional to distance from band.
    """

    def __init__(
        self,
        bb_window: int = 20,
        bb_std: float = 2.0,
        rsi_window: int = 14,
        rsi_oversold: float = 35.0,
        rsi_overbought: float = 65.0,
        exit_at_mean: bool = True,
        signal_type: str = "binary",
    ) -> None:
        assert bb_window >= 5, (
            f"bb_window must be >= 5, got {bb_window}"
        )
        assert bb_std > 0, (
            f"bb_std must be > 0, got {bb_std}"
        )
        assert rsi_window >= 2, (
            f"rsi_window must be >= 2, got {rsi_window}"
        )
        assert 0 < rsi_oversold < 50, (
            f"rsi_oversold must be in (0, 50), got {rsi_oversold}"
        )
        assert 50 < rsi_overbought < 100, (
            f"rsi_overbought must be in (50, 100), got {rsi_overbought}"
        )
        assert signal_type in ("binary", "scaled"), (
            f"signal_type must be 'binary' or 'scaled', got '{signal_type}'"
        )

        self.bb_window = bb_window
        self.bb_std = bb_std
        self.rsi_window = rsi_window
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.exit_at_mean = exit_at_mean
        self.signal_type = signal_type

    def get_name(self) -> str:
        return (
            f"MeanReversion(bb={self.bb_window}/{self.bb_std}std, "
            f"rsi={self.rsi_window}/{self.rsi_oversold}/{self.rsi_overbought}, "
            f"exit={'mean' if self.exit_at_mean else 'band'}, "
            f"type={self.signal_type})"
        )

    def _compute_rsi(self, prices: pd.Series) -> pd.Series:
        """
        Compute Wilder RSI for a single price series using EWM smoothing.

        Wilder smoothing uses span = 2 * window - 1, which is equivalent to
        an alpha of 1/window.  This differs materially from a simple rolling
        mean of gains/losses.

        Parameters
        ----------
        prices : pd.Series
            Single-ticker close price series.

        Returns
        -------
        pd.Series
            RSI values in [0, 100] with the same index as ``prices``.
            NaN positions (during warmup) are filled with 50 (neutral).
        """
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)

        # Wilder smoothing: span = 2 * window - 1
        span = 2 * self.rsi_window - 1
        avg_gain = gain.ewm(span=span, min_periods=self.rsi_window,
                            adjust=False).mean()
        avg_loss = loss.ewm(span=span, min_periods=self.rsi_window,
                            adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(50)  # neutral RSI where avg_loss == 0

        assert rsi.index.equals(prices.index), (
            "RSI index must match input prices index"
        )
        assert float(rsi.min()) >= 0.0 - 1e-9, (
            f"RSI value below 0: min = {float(rsi.min()):.6f}"
        )
        assert float(rsi.max()) <= 100.0 + 1e-9, (
            f"RSI value above 100: max = {float(rsi.max()):.6f}"
        )

        return rsi

    def _compute_bands(
        self, prices: pd.Series
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Compute Bollinger Bands for a single price series.

        Parameters
        ----------
        prices : pd.Series
            Single-ticker close price series.

        Returns
        -------
        tuple[pd.Series, pd.Series, pd.Series]
            (upper_band, middle_band, lower_band) — all with same index.
        """
        middle = prices.rolling(self.bb_window, min_periods=self.bb_window).mean()
        std = prices.rolling(self.bb_window, min_periods=self.bb_window).std()
        upper = middle + self.bb_std * std
        lower = middle - self.bb_std * std

        assert upper.index.equals(prices.index), (
            "Upper band index must match input prices index"
        )
        assert middle.index.equals(prices.index), (
            "Middle band index must match input prices index"
        )
        assert lower.index.equals(prices.index), (
            "Lower band index must match input prices index"
        )

        # Validate ordering where values are not NaN
        valid = ~(upper.isna() | middle.isna() | lower.isna())
        assert (upper[valid] >= middle[valid]).all(), (
            "Upper band must be >= middle band at all non-NaN positions"
        )
        assert (middle[valid] >= lower[valid]).all(), (
            "Middle band must be >= lower band at all non-NaN positions"
        )

        return upper, middle, lower

    def _signals_for_ticker(self, prices: pd.Series) -> pd.Series:
        """
        Compute the mean-reversion signal for a single ticker.

        Fully vectorized: no Python loops over dates or rows.
        The forward-fill pattern acts as a stateless state machine — entry
        conditions set the signal, exit conditions reset it to 0, and
        ffill() propagates the active position between events.

        Parameters
        ----------
        prices : pd.Series
            Single-ticker close price series.

        Returns
        -------
        pd.Series
            Signal in [-1, 1] with the same index as ``prices``.
        """
        # ── Step 1: compute indicators ──────────────────────────────────────
        upper, middle, lower = self._compute_bands(prices)
        rsi = self._compute_rsi(prices)

        # ── Step 2: entry and exit conditions ───────────────────────────────
        long_entry = (prices < lower) & (rsi < self.rsi_oversold)
        short_entry = (prices > upper) & (rsi > self.rsi_overbought)

        if self.exit_at_mean:
            long_exit = prices >= middle
            short_exit = prices <= middle
        else:
            long_exit = prices >= upper
            short_exit = prices <= lower

        # ── Step 3: forward-fill state machine ──────────────────────────────
        # Initialize to NaN so ffill only propagates explicit signals
        raw = pd.Series(np.nan, index=prices.index)

        # Set entry signals
        raw[long_entry] = 1.0
        raw[short_entry] = -1.0

        # Set exit signals — only where not simultaneously entering
        raw[long_exit & ~long_entry] = 0.0
        raw[short_exit & ~short_entry] = 0.0

        # Forward-fill: hold position between entry and exit events
        signal = raw.ffill().fillna(0.0)

        # When entry and exit fire on the same bar, exit takes priority
        signal[long_entry & long_exit] = 0.0
        signal[short_entry & short_exit] = 0.0

        # ── Step 4: apply scaled signal intensity ───────────────────────────
        if self.signal_type == "scaled":
            rolling_std = prices.rolling(
                self.bb_window, min_periods=self.bb_window
            ).std()
            long_intensity = (
                (lower - prices) / (self.bb_std * rolling_std + 1e-9)
            ).clip(0, 1)
            short_intensity = (
                (prices - upper) / (self.bb_std * rolling_std + 1e-9)
            ).clip(0, 1)

            signal = signal.copy()
            signal[signal > 0] = long_intensity[signal > 0]
            signal[signal < -1] = -short_intensity[signal < -1]
            signal = signal.clip(-1, 1)

        # ── Step 5: zero out the warmup period ──────────────────────────────
        warmup = self.bb_window + self.rsi_window
        signal.iloc[:warmup] = 0.0

        assert signal.index.equals(prices.index), (
            "Signal index must match input prices index"
        )
        assert float(signal.min()) >= -1.0 - 1e-9, (
            f"Signal below -1.0: min = {float(signal.min()):.6f}"
        )
        assert float(signal.max()) <= 1.0 + 1e-9, (
            f"Signal above +1.0: max = {float(signal.max()):.6f}"
        )

        return signal

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate mean-reversion signals for all tickers independently.

        Each ticker's signal is computed in isolation from every other ticker
        (no cross-sectional ranking).  The engine's ``_resolve_signals()``
        applies a shift(1) on top to prevent same-day execution.

        Do NOT apply shift() here.

        Parameters
        ----------
        data : pd.DataFrame
            Wide-format close prices with DatetimeIndex.

        Returns
        -------
        pd.DataFrame
            Signals in [-1, 1] with same shape, index, and columns as ``data``.
        """
        signals = data.apply(self._signals_for_ticker, axis=0)

        assert signals.shape == data.shape, (
            f"Signal shape {signals.shape} must match data shape {data.shape}"
        )
        assert signals.columns.tolist() == data.columns.tolist(), (
            "Signal columns must match data columns exactly"
        )

        return signals

    def __repr__(self) -> str:
        return self.get_name()
