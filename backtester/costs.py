"""
Transaction cost and slippage model.

Implementation plan (Phase 3.1)
---------------------------------
Models two sources of friction that erode live trading returns:

1. Commission / transaction cost
   - Flat basis-point fee applied to the notional value of each trade.
   - Configurable via TRANSACTION_COST_BPS in config.py.

2. Slippage
   - Assumed market-impact cost modelled as a fixed BPS spread around mid.
   - More sophisticated models (square-root impact) can be swapped in later.

The CostModel.apply() method is called by BacktestEngine on each rebalance
event and reduces the portfolio value accordingly before recording daily PnL.
"""

from __future__ import annotations

from config import SLIPPAGE_BPS, TRANSACTION_COST_BPS


class CostModel:
    """
    Simple linear transaction cost and slippage model.

    Parameters
    ----------
    transaction_cost_bps : int
        One-way commission in basis points (1 bp = 0.01%).
    slippage_bps : int
        One-way slippage in basis points.
    """

    def __init__(
        self,
        transaction_cost_bps: int = TRANSACTION_COST_BPS,
        slippage_bps: int = SLIPPAGE_BPS,
    ) -> None:
        self.transaction_cost_bps = transaction_cost_bps
        self.slippage_bps = slippage_bps
        self._total_bps: float = (transaction_cost_bps + slippage_bps) / 10_000

    def compute_cost(self, notional: float) -> float:
        """
        Compute the total friction cost for a trade of a given notional value.

        Parameters
        ----------
        notional : float
            Absolute USD value of the trade (always positive).

        Returns
        -------
        float
            Cost in USD to be subtracted from the portfolio.
        """
        raise NotImplementedError(
            "CostModel.compute_cost is not yet implemented. "
            "Scheduled for Phase 3.1."
        )

    def __repr__(self) -> str:
        return (
            f"CostModel(transaction_cost_bps={self.transaction_cost_bps}, "
            f"slippage_bps={self.slippage_bps})"
        )
