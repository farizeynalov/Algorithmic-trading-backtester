"""
Momentum strategy — Jegadeesh-Titman (1993) cross-sectional momentum.

Stocks that have outperformed over the past 12 months (excluding the most
recent month to avoid short-term reversal) tend to continue outperforming
over the next month.  This module implements both a binary ranked variant
and a continuous rank-normalized variant.

Reference: Jegadeesh, N. & Titman, S. (1993). Returns to Buying Winners
and Selling Losers: Implications for Stock Market Efficiency. Journal of
Finance, 48(1), 65–91.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtester.base import BaseStrategy


class MomentumStrategy(BaseStrategy):
    """Cross-sectional momentum based on Jegadeesh and Titman (1993).

    At the end of each month, tickers are ranked by their cumulative
    return over the lookback window excluding the skip window. The top
    ``n_long`` tickers receive a signal of +1.0 (full long); the bottom
    ``n_short`` tickers receive -1.0 (full short, requires
    ``allow_short=True`` in the engine); all others receive 0.0 (flat).

    Signal convention:
        +1.0 — full long (maximum long allocation by the engine)
         0.0 — flat (no position in this ticker)
        -1.0 — full short (maximum short allocation by the engine)

    Note:
        The monthly ``shift(skip_months)`` in ``_compute_momentum_return``
        implements the skip-month exclusion (signal definition).
        The engine's ``shift(1)`` in ``_resolve_signals`` implements
        execution timing (no same-day execution). These are orthogonal:
        removing either shift would introduce a different form of bias.

    Args:
        lookback_months: Months over which to measure past return.
            Default 12 (Jegadeesh-Titman specification).
        skip_months: Most recent months excluded from the lookback to
            avoid short-term reversal contamination. Default 1 (JT
            spec). Effective window: [T - lookback - skip, T - skip].
        n_long: Top-momentum tickers to go long. Default 5.
        n_short: Bottom-momentum tickers to short. Default 0
            (long-only strategy).
        signal_type: ``"ranked"`` — binary ±1 / 0; ``"continuous"``
            — every ticker receives a signal proportional to its
            cross-sectional rank, normalised to roughly (-0.5, +0.5).
    """

    def __init__(
        self,
        lookback_months: int = 12,
        skip_months: int = 1,
        n_long: int = 5,
        n_short: int = 0,
        signal_type: str = "ranked",
    ) -> None:
        assert lookback_months >= 2, (
            f"lookback_months must be >= 2, got {lookback_months}"
        )
        assert skip_months >= 0, (
            f"skip_months must be >= 0, got {skip_months}"
        )
        assert n_long >= 1, (
            f"n_long must be >= 1, got {n_long}"
        )
        assert n_short >= 0, (
            f"n_short must be >= 0, got {n_short}"
        )
        assert signal_type in ("ranked", "continuous"), (
            f"signal_type must be 'ranked' or 'continuous', got '{signal_type}'"
        )

        self.lookback_months = lookback_months
        self.skip_months = skip_months
        self.n_long = n_long
        self.n_short = n_short
        self.signal_type = signal_type

    def get_name(self) -> str:
        return (
            f"Momentum(lookback={self.lookback_months}m, "
            f"skip={self.skip_months}m, "
            f"long={self.n_long}, short={self.n_short}, "
            f"type={self.signal_type})"
        )

    def _compute_momentum_return(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Compute raw cross-sectional momentum returns at monthly frequency,
        then forward-fill to a daily index.

        Two operations on the monthly series serve two distinct purposes —
        it is important not to confuse them with the engine's execution shift:

        (1) ``pct_change(lookback_months)``
            Defines the LENGTH of the lookback window.  At monthly date T,
            this gives the cumulative return from T-lookback_months to T.
            This is part of the SIGNAL DEFINITION, not execution timing.

        (2) ``shift(skip_months)``
            Enforces the SKIP-MONTH EXCLUSION (Jegadeesh-Titman's 12-1 rule).
            Shifting forward by skip_months months means that at monthly date T,
            we see the momentum value from T-skip_months, which measured the
            return from T-lookback-skip to T-skip.  This removes the most
            recent skip_months of returns from the signal, avoiding the
            well-documented short-term reversal in the most recent month.
            This is also part of the SIGNAL DEFINITION, not execution timing.

        Neither of these is the execution shift.  The engine's
        ``_resolve_signals()`` applies a separate shift(1) on the DAILY signals
        to enforce that a signal observed on day T can only be acted upon on
        day T+1 (no same-day execution).

        Parameters
        ----------
        prices : pd.DataFrame
            Wide-format daily close prices with DatetimeIndex.

        Returns
        -------
        pd.DataFrame
            Daily momentum returns (NaN during the warmup period).
        """
        # Resample to month-end prices to avoid day-of-month noise
        monthly: pd.DataFrame = prices.resample("ME").last()

        # (1) Lookback return:  (price_T / price_{T-lookback}) - 1
        # (2) Skip-month shift: move signal forward so we exclude [T-skip, T]
        momentum_monthly: pd.DataFrame = (
            monthly.pct_change(self.lookback_months).shift(self.skip_months)
        )

        # Forward-fill monthly momentum values to every business day.
        # During a given month, all days use the most recent month-end signal.
        # NaN values (warmup period) remain NaN until propagated by ffill.
        momentum_daily: pd.DataFrame = momentum_monthly.reindex(
            prices.index, method="ffill"
        )

        return momentum_daily

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate cross-sectional momentum signals for all tickers.

        During the warmup period (first ``lookback_months + skip_months``
        month-end dates), all signals are 0.0 because there is no
        complete lookback window of return data yet.

        Warning:
            Do not apply additional ``shift()`` inside this method —
            the engine's ``_resolve_signals()`` handles the
            execution-lag shift (T → T+1) structurally.

        Args:
            data: Wide-format close prices with DatetimeIndex — the
                same object passed to ``Backtester(data)``.

        Returns:
            Signals in [-1, 1] with the same index and columns as
            ``data``. Warmup-period rows are 0.0 (flat).
        """
        momentum_daily: pd.DataFrame = self._compute_momentum_return(data)

        # Initialise all signals to 0.0 — covers both warmup and flat rows
        signals: pd.DataFrame = pd.DataFrame(
            0.0, index=data.index, columns=data.columns
        )

        # Only process dates where at least one ticker has a valid momentum value
        valid_mask: pd.Series = ~momentum_daily.isna().all(axis=1)
        valid_mom: pd.DataFrame = momentum_daily.loc[valid_mask]

        if valid_mask.sum() == 0:
            return signals

        if self.signal_type == "ranked":
            # ── Ranked: binary long/short signals ─────────────────────────
            # Rank descending: rank 1 = highest momentum ticker
            ranks_desc: pd.DataFrame = valid_mom.rank(
                axis=1, ascending=False, method="first", na_option="keep"
            )
            # Top n_long get +1
            long_sig: pd.DataFrame = (
                (ranks_desc <= self.n_long).astype(float).fillna(0.0)
            )
            result: pd.DataFrame = long_sig

            if self.n_short > 0:
                # Rank ascending: rank 1 = lowest momentum (worst) ticker
                ranks_asc: pd.DataFrame = valid_mom.rank(
                    axis=1, ascending=True, method="first", na_option="keep"
                )
                # Bottom n_short get -1
                short_sig: pd.DataFrame = (
                    -(ranks_asc <= self.n_short).astype(float).fillna(0.0)
                )
                result = long_sig + short_sig

            signals.loc[valid_mask] = result.values

        elif self.signal_type == "continuous":
            # ── Continuous: rank-normalized signals for all tickers ────────
            # Rank ascending: rank 1 = lowest momentum, rank N = highest
            ranks_asc = valid_mom.rank(
                axis=1, ascending=True, method="first", na_option="keep"
            )
            # Centre by row mean and scale by row range → roughly (-0.5, 0.5)
            row_means: pd.Series = ranks_asc.mean(axis=1, skipna=True)
            row_ranges: pd.Series = (
                ranks_asc.max(axis=1, skipna=True)
                - ranks_asc.min(axis=1, skipna=True)
            )
            # Add a tiny positive offset (1e-10) so that the median-ranked
            # ticker never gets exactly 0.0 (spec requirement for continuous mode)
            normalized: pd.DataFrame = (
                ranks_asc.sub(row_means, axis=0)
                .div(row_ranges + 1e-9, axis=0)
                + 1e-10
            )
            # Tickers with NaN momentum (missing data) fall back to 0
            normalized = normalized.fillna(0.0)
            signals.loc[valid_mask] = normalized.values

        assert signals.columns.tolist() == data.columns.tolist(), (
            "Signal columns must match data columns exactly"
        )
        assert float(signals.min().min()) >= -1.0 - 1e-9, (
            f"Signal below -1.0: min = {signals.min().min():.6f}"
        )
        assert float(signals.max().max()) <= 1.0 + 1e-9, (
            f"Signal above +1.0: max = {signals.max().max():.6f}"
        )

        return signals

    def __repr__(self) -> str:
        return self.get_name()
