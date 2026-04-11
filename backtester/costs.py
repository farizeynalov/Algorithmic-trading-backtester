"""
Transaction cost and slippage models for the algorithmic trading backtester.

Three concrete models are provided, each implementing the CostModel interface:

FlatBpsCostModel      — constant basis-point fee on every trade notional;
                        matches the inline arithmetic used before Phase 2.2
                        and is the default when no model is specified.

TieredCommissionModel — per-dollar commission rate decreases as trade size
                        grows, modelling typical broker fee schedules (e.g.
                        Interactive Brokers tiered pricing).

SpreadSlippageModel   — slippage widens with realized volatility, penalizing
                        rebalances during high-volatility drawdown periods;
                        commission remains flat bps.

All models share the same interface: they receive only (notional, ticker, date)
and return a non-negative float in USD — they have zero knowledge of engine
internals.

Rationale for SpreadSlippageModel
----------------------------------
During the 2020 COVID crash and the 2022 rate-hike period, realized volatility
spiked 3–5× above its median level.  A flat slippage model treats a $10,000
rebalance the same whether markets are calm or in free-fall.  SpreadSlippageModel
scales the effective spread by the ratio of current vol to median vol, so trades
executed during the worst drawdown days are automatically penalized more heavily
— precisely the behaviour a realistic backtest requires.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class CostModel(ABC):
    """Abstract interface for all transaction cost models.

    Subclasses implement ``commission()`` and ``slippage()`` separately
    so callers can inspect each cost component. ``total()`` is provided
    here and is the single enforcement point for the non-negativity
    guarantee — subclasses must not duplicate the check.

    The ``notional`` parameter passed to every method is always the
    absolute USD value of the trade (``abs(dollar_trade)``), so it is
    always >= 0. Callers must never pass a signed dollar amount.
    """

    @abstractmethod
    def commission(self, notional: float, ticker: str, date: pd.Timestamp) -> float:
        """Return the commission charge in USD for a trade of ``notional``."""
        ...

    @abstractmethod
    def slippage(self, notional: float, ticker: str, date: pd.Timestamp) -> float:
        """Return the slippage charge in USD for a trade of ``notional``."""
        ...

    def total(self, notional: float, ticker: str, date: pd.Timestamp) -> float:
        """
        Return total friction cost (commission + slippage) in USD.

        This is the single assertion point for non-negativity: no subclass
        should duplicate the check.
        """
        c = self.commission(notional, ticker, date)
        s = self.slippage(notional, ticker, date)
        total_cost = c + s
        assert total_cost >= 0, (
            f"{type(self).__name__}.total() returned a negative cost "
            f"({total_cost:.6f}) for notional={notional:.2f}, "
            f"ticker={ticker}, date={date}.  "
            "Commission and slippage must both be >= 0."
        )
        return total_cost

    @abstractmethod
    def describe(self) -> dict:
        """
        Return a dict summarising the model's parameters.

        Snapshotted into BacktestResult.config so every result is
        self-documenting about how costs were computed.
        """
        ...


# ---------------------------------------------------------------------------
# FlatBpsCostModel
# ---------------------------------------------------------------------------

class FlatBpsCostModel(CostModel):
    """
    Constant basis-point fee applied uniformly to every trade notional.

    This is the simplest possible cost model and matches the inline
    arithmetic used in engine.py before Phase 2.2.  It is the default
    model when no ``cost_model`` key is provided in config.

    Parameters
    ----------
    commission_bps : float
        One-way commission charge in basis points (1 bp = 0.01%).
    slippage_bps : float
        One-way slippage charge in basis points.
    """

    def __init__(
        self,
        commission_bps: float = 5,
        slippage_bps: float = 2,
    ) -> None:
        assert commission_bps >= 0, (
            f"commission_bps must be >= 0, got {commission_bps}"
        )
        assert slippage_bps >= 0, (
            f"slippage_bps must be >= 0, got {slippage_bps}"
        )
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps

    def commission(self, notional: float, ticker: str, date: pd.Timestamp) -> float:
        return notional * self.commission_bps / 10_000

    def slippage(self, notional: float, ticker: str, date: pd.Timestamp) -> float:
        return notional * self.slippage_bps / 10_000

    def describe(self) -> dict:
        return {
            "model":          "FlatBps",
            "commission_bps": self.commission_bps,
            "slippage_bps":   self.slippage_bps,
        }

    def __repr__(self) -> str:
        return (
            f"FlatBpsCostModel("
            f"commission_bps={self.commission_bps}, "
            f"slippage_bps={self.slippage_bps})"
        )


# ---------------------------------------------------------------------------
# TieredCommissionModel
# ---------------------------------------------------------------------------

class TieredCommissionModel(CostModel):
    """
    Commission schedule whose per-dollar rate decreases with trade size.

    Models typical broker tiered pricing (e.g. IBKR): small orders pay
    a higher bps rate, large orders qualify for a lower rate.

    Parameters
    ----------
    tiers : list[tuple[float, float]]
        List of ``(threshold, bps)`` pairs sorted ascending by threshold.
        The effective bps for a trade is the bps of the highest threshold
        that the notional meets or exceeds.

        Example::

            [(0, 10), (10_000, 7), (50_000, 5), (100_000, 3)]

        A $500 trade pays 10 bps; a $15,000 trade pays 7 bps.

    slippage_bps : float
        Flat slippage charge in basis points (independent of tier).
    """

    def __init__(
        self,
        tiers: list[tuple[float, float]],
        slippage_bps: float = 2,
    ) -> None:
        assert len(tiers) > 0, (
            "tiers must be non-empty"
        )
        for i in range(1, len(tiers)):
            assert tiers[i][0] > tiers[i - 1][0], (
                f"tiers must be sorted ascending by threshold.  "
                f"Found tiers[{i-1}]={tiers[i-1]} then tiers[{i}]={tiers[i]}."
            )
        for threshold, bps in tiers:
            assert bps > 0, (
                f"All tier bps values must be > 0; "
                f"got bps={bps} for threshold={threshold}"
            )
        assert slippage_bps >= 0, (
            f"slippage_bps must be >= 0, got {slippage_bps}"
        )
        self.tiers = tiers
        self.slippage_bps = slippage_bps

    def commission(self, notional: float, ticker: str, date: pd.Timestamp) -> float:
        # Walk tiers from highest threshold to lowest; use first match
        for threshold, bps in reversed(self.tiers):
            if notional >= threshold:
                return notional * bps / 10_000
        # Fallback: use lowest tier (should never reach here since tiers[0][0] is
        # typically 0, meaning every notional qualifies)
        return notional * self.tiers[0][1] / 10_000

    def slippage(self, notional: float, ticker: str, date: pd.Timestamp) -> float:
        return notional * self.slippage_bps / 10_000

    def describe(self) -> dict:
        return {
            "model":        "TieredCommission",
            "tiers":        self.tiers,
            "slippage_bps": self.slippage_bps,
        }

    def __repr__(self) -> str:
        return (
            f"TieredCommissionModel("
            f"tiers={self.tiers}, "
            f"slippage_bps={self.slippage_bps})"
        )


# ---------------------------------------------------------------------------
# SpreadSlippageModel
# ---------------------------------------------------------------------------

class SpreadSlippageModel(CostModel):
    """
    Slippage model that scales with realized volatility.

    During high-volatility periods (market crashes, rate-shock episodes),
    bid-ask spreads widen significantly.  This model adjusts the effective
    slippage spread in proportion to how much today's realized vol exceeds
    the historical median, ensuring that rebalances during drawdowns pay
    a realistically higher friction cost.

    If ``realized_vol`` is None or the trade date is not in the vol index,
    the model degrades gracefully to flat-bps slippage — identical to
    ``FlatBpsCostModel``.

    Parameters
    ----------
    base_spread_bps : float
        Effective spread in a normal volatility environment (bps).
    vol_scalar : float
        Sensitivity of spread to excess volatility.  A value of 0.5 means
        that a vol 2× the median doubles the extra spread by 50%.
    realized_vol : pd.Series | None
        Series indexed by date; values are annualized realized volatility
        (e.g. the 60-day rolling vol computed in the EDA notebook).
    commission_bps : float
        Flat commission charge in basis points (independent of vol).

    Rationale
    ---------
    During the 2020 COVID crash and the 2022 rate-hike period visible in
    the EDA notebook, realized volatility spiked 3–5× above its median
    level.  A flat slippage model underestimates costs precisely when they
    matter most — at the peak of a drawdown when the strategy most wants
    to rebalance.  SpreadSlippageModel penalizes those periods automatically
    by scaling the spread with the vol ratio.
    """

    def __init__(
        self,
        base_spread_bps: float = 2,
        vol_scalar: float = 0.5,
        realized_vol: pd.Series | None = None,
        commission_bps: float = 5,
    ) -> None:
        assert base_spread_bps >= 0, (
            f"base_spread_bps must be >= 0, got {base_spread_bps}"
        )
        assert vol_scalar >= 0, (
            f"vol_scalar must be >= 0, got {vol_scalar}"
        )
        assert commission_bps >= 0, (
            f"commission_bps must be >= 0, got {commission_bps}"
        )
        self.base_spread_bps = base_spread_bps
        self.vol_scalar = vol_scalar
        self.realized_vol = realized_vol
        self.commission_bps = commission_bps

        # Pre-compute median once at construction time (avoids recomputing per trade)
        self._vol_median: float | None = (
            float(realized_vol.median()) if realized_vol is not None else None
        )

    def slippage(self, notional: float, ticker: str, date: pd.Timestamp) -> float:
        # Degrade gracefully when vol data is unavailable
        if (
            self.realized_vol is None
            or self._vol_median is None
            or self._vol_median == 0.0
            or date not in self.realized_vol.index
        ):
            return notional * self.base_spread_bps / 10_000

        daily_vol: float = float(self.realized_vol.loc[date])
        vol_ratio: float = daily_vol / self._vol_median

        # Spread widens linearly with excess vol; floor at base_spread_bps
        effective_spread_bps: float = self.base_spread_bps * (
            1.0 + self.vol_scalar * (vol_ratio - 1.0)
        )
        effective_spread_bps = max(effective_spread_bps, self.base_spread_bps)

        return notional * effective_spread_bps / 10_000

    def commission(self, notional: float, ticker: str, date: pd.Timestamp) -> float:
        return notional * self.commission_bps / 10_000

    def describe(self) -> dict:
        return {
            "model":                  "SpreadSlippage",
            "base_spread_bps":        self.base_spread_bps,
            "vol_scalar":             self.vol_scalar,
            "commission_bps":         self.commission_bps,
            "realized_vol_available": self.realized_vol is not None,
        }

    def __repr__(self) -> str:
        return (
            f"SpreadSlippageModel("
            f"base_spread_bps={self.base_spread_bps}, "
            f"vol_scalar={self.vol_scalar}, "
            f"commission_bps={self.commission_bps}, "
            f"realized_vol={'provided' if self.realized_vol is not None else 'None'})"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_cost_model(config: dict) -> CostModel:
    """
    Instantiate the appropriate CostModel from a config dict.

    Reads ``config["cost_model"]`` to select the model class, then
    pulls remaining parameters from the same dict so callers only need
    one config object for the entire engine.

    Accepted values for ``config["cost_model"]``
    ---------------------------------------------
    ``"flat"`` (default)
        FlatBpsCostModel using ``commission_bps`` and ``slippage_bps``.
    ``"tiered"``
        TieredCommissionModel using ``tiers`` and ``slippage_bps``.
    ``"spread"``
        SpreadSlippageModel using ``base_spread_bps``, ``vol_scalar``,
        ``realized_vol``, and ``commission_bps``.

    Parameters
    ----------
    config : dict
        Engine configuration dict (merged with defaults in Backtester.__init__).

    Returns
    -------
    CostModel
        Concrete cost model instance ready for use.

    Raises
    ------
    ValueError
        If ``config["cost_model"]`` is not one of the accepted values.
    """
    model_name: str = config.get("cost_model", "flat")

    if model_name == "flat":
        return FlatBpsCostModel(
            commission_bps=config.get("commission_bps", 5),
            slippage_bps=config.get("slippage_bps", 2),
        )

    if model_name == "tiered":
        return TieredCommissionModel(
            tiers=config.get(
                "tiers",
                [(0, 10), (10_000, 7), (50_000, 5), (100_000, 3)],
            ),
            slippage_bps=config.get("slippage_bps", 2),
        )

    if model_name == "spread":
        return SpreadSlippageModel(
            base_spread_bps=config.get("base_spread_bps", 2),
            vol_scalar=config.get("vol_scalar", 0.5),
            realized_vol=config.get("realized_vol", None),
            commission_bps=config.get("commission_bps", 5),
        )

    raise ValueError(
        f"Unknown cost_model '{model_name}'.  "
        "Accepted values: 'flat', 'tiered', 'spread'."
    )
