"""
Core backtesting engine for the algorithmic trading backtester.

Two non-negotiable invariants are enforced throughout:

INVARIANT 1 — NO LOOKAHEAD BIAS
    A signal generated using data up to and including date T can only be
    executed at the open of date T+1 at the earliest.  This is enforced
    *structurally* by the .shift(1) call in _resolve_signals — it is
    architecturally impossible for any signal to consume data it could not
    have known at execution time.

INVARIANT 2 — NO SILENT FAILURES
    Every assumption the engine makes (data alignment, signal shape, date
    coverage) is explicitly asserted with a descriptive error message.
    Wrong inputs raise errors; they never produce silently wrong results.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backtester.base import BaseStrategy
from backtester.costs import CostModel, make_cost_model


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG: dict = {
    "initial_capital":   100_000,
    "commission_bps":    5,           # basis points per trade (one-way)
    "slippage_bps":      2,           # basis points per trade (one-way)
    "position_sizing":   "equal_weight",  # "equal_weight" | "signal_weighted"
    "allow_short":       False,
    "max_position_size": 0.25,        # max fraction of portfolio per ticker
    "rebalance_buffer":  0.01,        # min abs weight delta to trigger a trade
    "benchmark":         "SPY",
    "cost_model":        "flat",      # "flat" | "tiered" | "spread"
    "risk_free_rate":    0.0,         # annualised, decimal (e.g. 0.04 = 4%)
}

# Canonical column order for the trades DataFrame (preserved even when empty)
_TRADE_COLUMNS: list[str] = [
    "date", "ticker", "direction", "quantity",
    "price", "commission", "slippage", "net_cost",
]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """Immutable container for the outputs of a completed backtest run."""

    strategy_name: str
    equity_curve: pd.Series    # index=date, values=portfolio value in USD
    returns: pd.Series         # index=date, daily portfolio simple returns
    positions: pd.DataFrame    # index=date, cols=tickers, values=weights
    trades: pd.DataFrame       # one row per executed trade
    metrics: dict              # populated by Phase 2.3 — empty dict for now
    config: dict               # frozen snapshot of all engine params


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------

class Backtester:
    """Event-driven, single-pass backtesting engine.

    Two non-negotiable invariants are enforced at all times:

    **Invariant 1 — No Lookahead Bias**: enforced *structurally* by
    shifting all signals forward one bar in ``_resolve_signals()``.
    It is architecturally impossible for a signal to consume data from
    dates after its computation date.

    **Invariant 2 — No Silent Failures**: every assumption about data
    alignment, signal shape, and date coverage is explicitly asserted
    with a descriptive error message. Wrong inputs raise errors; they
    never produce silently wrong results.

    Args:
        data: Wide-format adjusted close prices (``df_wide`` from
            DataLoader). Index must be a DatetimeIndex; columns are
            ticker symbols.
        config: Optional overrides for any engine parameter. Keys not
            present fall back to ``_DEFAULT_CONFIG`` defaults:

            - ``initial_capital`` (float): Starting portfolio value.
              Default 100_000.
            - ``commission_bps`` (int): One-way commission in bps.
              Default 5.
            - ``slippage_bps`` (int): One-way slippage in bps.
              Default 2.
            - ``position_sizing`` (str): ``"equal_weight"`` or
              ``"signal_weighted"``. Default ``"equal_weight"``.
            - ``allow_short`` (bool): Allow short positions. Default
              False.
            - ``max_position_size`` (float): Max per-ticker weight.
              Default 0.25.
            - ``rebalance_buffer`` (float): Min absolute weight delta
              before a trade is executed. Default 0.01.
            - ``benchmark`` (str): Benchmark ticker column. Default
              ``"SPY"``.
            - ``cost_model`` (str): ``"flat"``, ``"tiered"``, or
              ``"spread"``. Default ``"flat"``.
            - ``risk_free_rate`` (float): Annualised decimal.
              Default 0.0.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        config: dict | None = None,
    ) -> None:
        assert isinstance(data, pd.DataFrame), (
            f"data must be a pd.DataFrame, got {type(data).__name__}"
        )
        assert isinstance(data.index, pd.DatetimeIndex), (
            "data.index must be a pd.DatetimeIndex.  "
            "Pass df_wide from DataLoader (wide-format close prices)."
        )

        merged: dict = {**_DEFAULT_CONFIG, **(config or {})}

        assert merged["benchmark"] in data.columns, (
            f"Benchmark ticker '{merged['benchmark']}' is not a column in data.  "
            f"Available tickers: {sorted(data.columns.tolist())}"
        )
        assert merged["position_sizing"] in ("equal_weight", "signal_weighted"), (
            f"position_sizing must be 'equal_weight' or 'signal_weighted', "
            f"got '{merged['position_sizing']}'"
        )
        assert 0 < merged["max_position_size"] <= 1.0, (
            f"max_position_size must be in (0, 1], got {merged['max_position_size']}"
        )

        self.data: pd.DataFrame = data
        self.config: dict = merged

        # Instantiate cost model from config and snapshot its parameters
        self.cost_model: CostModel = make_cost_model(self.config)
        self.config["cost_model_params"] = self.cost_model.describe()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_signals(self, raw_signals: pd.DataFrame) -> pd.DataFrame:
        """Enforce Invariant 1 by shifting all signals forward by one bar.

        A strategy's signal on date T is derived from data through date T.
        Applying ``shift(1)`` ensures that the signal only affects
        positions starting on date T+1. This is the structural guarantee
        of Invariant 1: it is architecturally impossible for any signal
        value computed on date T to influence the portfolio on date T.

        Row 0 of the shifted DataFrame is always 0.0 (the NaN produced
        by shift(1) is filled with 0, meaning the portfolio is flat on
        the first date — no position before the first signal executes).

        Args:
            raw_signals: Strategy output in [-1, 1] with
                ``index ⊆ self.data.index`` and
                ``columns ⊆ self.data.columns``.

        Returns:
            Shifted signals with the same shape; row 0 is always 0.0.

        Raises:
            AssertionError: If ``raw_signals`` index or columns are
                not subsets of ``self.data``.
        """
        assert isinstance(raw_signals, pd.DataFrame), (
            f"raw_signals must be a pd.DataFrame, got {type(raw_signals).__name__}"
        )
        assert set(raw_signals.index).issubset(set(self.data.index)), (
            "raw_signals.index must be a subset of self.data.index.  "
            f"Dates in signals but not in data: "
            f"{sorted(set(raw_signals.index) - set(self.data.index))[:5]}"
        )
        assert set(raw_signals.columns).issubset(set(self.data.columns)), (
            "raw_signals.columns must be a subset of self.data.columns.  "
            f"Unknown tickers: "
            f"{sorted(set(raw_signals.columns) - set(self.data.columns))}"
        )

        # shift(1): signal on day T executes on day T+1
        shifted = raw_signals.shift(1)

        # Row 0 is NaN after the shift — fill with 0 (flat / no position)
        shifted = shifted.fillna(0.0)

        return shifted

    def _compute_weights(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Convert shifted signals in [-1, 1] into target portfolio weights.

        Two modes are controlled by ``self.config["position_sizing"]``:

        ``"equal_weight"``: Every non-zero signal receives an equal share
        of the portfolio. Longs get +1/n_active; shorts get -1/n_active
        when ``allow_short=True``. Rows where all signals are zero stay
        at 0.0 (fully cash).

        ``"signal_weighted"``: Long weights are normalised so they sum to
        1; short weights are normalised so their absolute values sum to 1.
        Zero signals are excluded from both pools.

        In both modes individual weights are capped at ``±max_position_size``
        and the total absolute exposure is asserted to be ≤ 100%
        (Invariant 2 enforcement).

        Args:
            signals: Shifted signals from ``_resolve_signals()``.

        Returns:
            Portfolio weights with the same shape as ``signals``.
        """
        allow_short: bool = self.config["allow_short"]
        max_pos: float = self.config["max_position_size"]
        mode: str = self.config["position_sizing"]

        if mode == "equal_weight":
            # Build a direction matrix: +1 for longs, -1 for shorts, 0 for flat
            direction = pd.DataFrame(
                0.0, index=signals.index, columns=signals.columns
            )
            direction[signals > 0] = 1.0
            if allow_short:
                direction[signals < 0] = -1.0

            # Count active (non-zero) positions per row
            n_active = direction.abs().sum(axis=1)

            # Equal weight = 1/n_active; rows with n_active==0 stay at 0
            weights = direction.div(
                n_active.replace(0, np.nan), axis=0
            ).fillna(0.0)

        elif mode == "signal_weighted":
            # Normalise positive signals so they sum to 1 per row
            long_signals = signals.clip(lower=0.0)
            long_sums = long_signals.sum(axis=1).replace(0, np.nan)
            long_weights = long_signals.div(long_sums, axis=0).fillna(0.0)

            if allow_short:
                # Normalise negative signals so their abs values sum to 1
                short_signals = signals.clip(upper=0.0)
                short_abs_sums = short_signals.abs().sum(axis=1).replace(0, np.nan)
                # Dividing a negative series by a positive scalar keeps sign
                short_weights = short_signals.div(
                    short_abs_sums, axis=0
                ).fillna(0.0)
                weights = long_weights + short_weights
            else:
                weights = long_weights

        else:
            raise ValueError(
                f"Unknown position_sizing mode: '{mode}'.  "
                "Must be 'equal_weight' or 'signal_weighted'."
            )

        # Apply per-position size cap
        if allow_short:
            weights = weights.clip(lower=-max_pos, upper=max_pos)
        else:
            weights = weights.clip(lower=0.0, upper=max_pos)

        # INVARIANT 2: gross exposure must not exceed 100%
        max_abs_exposure = float(weights.abs().sum(axis=1).max())
        assert max_abs_exposure <= 1.0 + 1e-9, (
            f"Portfolio gross exposure exceeds 100%: "
            f"max abs-weight row sum = {max_abs_exposure:.6f}.  "
            "This indicates a bug in the weight-computation logic."
        )

        return weights

    def _execute_trades(
        self,
        weights_today: pd.Series,
        weights_prev: pd.Series,
        prices_today: pd.Series,
        portfolio_value: float,
        date: pd.Timestamp,
    ) -> tuple[pd.Series, list[dict], float]:
        """Compute position changes, apply rebalance buffer, record costs.

        The rebalance buffer (``config["rebalance_buffer"]``) suppresses
        trades where the absolute weight delta is smaller than the
        threshold. This avoids costly micro-rebalances that would not
        meaningfully change portfolio exposure.

        Args:
            weights_today: Target weights after rebalancing, aligned to
                ``self.data.columns``.
            weights_prev: Current holdings as weights, aligned to
                ``self.data.columns``.
            prices_today: Closing prices for all tickers on this date.
            portfolio_value: Current portfolio value in USD after today's
                P&L and before costs are subtracted.
            date: Trade execution date — used to tag each trade record
                and passed to the cost model for vol-scaling.

        Returns:
            A three-tuple of:
            - ``weights_today`` (pd.Series): target weights that become
              the new current weights after execution.
            - ``trades`` (list[dict]): one dict per executed trade leg
              with keys matching ``_TRADE_COLUMNS``.
            - ``total_cost`` (float): total commission + slippage in USD.
        """
        allow_short: bool = self.config["allow_short"]
        buffer: float = self.config["rebalance_buffer"]

        weight_delta: pd.Series = weights_today - weights_prev

        # Apply rebalance buffer: suppress micro-rebalances
        weight_delta[weight_delta.abs() < buffer] = 0.0

        trades: list[dict] = []
        total_cost: float = 0.0

        for ticker in weight_delta.index:
            delta: float = float(weight_delta[ticker])
            if delta == 0.0:
                continue

            price: float = float(prices_today[ticker])
            if pd.isna(price) or price <= 0.0:
                # Skip untradeable tickers rather than crashing
                continue

            dollar_trade: float = delta * portfolio_value
            shares: float = dollar_trade / price
            notional: float = abs(dollar_trade)
            commission: float = self.cost_model.commission(notional, ticker, date)
            slippage: float = self.cost_model.slippage(notional, ticker, date)
            # net_cost: positive = cash outflow (buy); negative = cash inflow (sell)
            net_cost: float = dollar_trade + commission + slippage

            prev_w: float = float(weights_prev[ticker])
            target_w: float = float(weights_today[ticker])

            if delta > 0.0:
                direction = "COVER" if prev_w < 0.0 else "BUY"
            else:
                direction = "SHORT" if (allow_short and target_w < 0.0) else "SELL"

            trades.append({
                "date":       date,
                "ticker":     ticker,
                "direction":  direction,
                "quantity":   shares,
                "price":      price,
                "commission": commission,
                "slippage":   slippage,
                "net_cost":   net_cost,
            })
            total_cost += commission + slippage

        return weights_today, trades, total_cost

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        strategy: BaseStrategy,
        _skip_metrics: bool = False,
    ) -> BacktestResult:
        """Execute the strategy backtest over ``self.data``.

        ``strategy.generate_signals()`` is called exactly once before the
        loop begins. Invariant 1 is enforced via ``_resolve_signals()``
        before weights are computed.

        Args:
            strategy: Concrete strategy instance implementing
                ``generate_signals()`` and ``get_name()``.
            _skip_metrics: Internal use only. When True, skips the
                ``compute_metrics()`` call and returns an empty metrics
                dict. Do not set this parameter externally — it exists
                only to prevent infinite recursion when
                ``run_benchmark()`` calls ``run()`` internally.

        Returns:
            BacktestResult containing equity curve, daily returns,
            position history, trade log, metrics dict, and config
            snapshot.

        Raises:
            AssertionError: If signal shape, columns, or date coverage
                fail validation (Invariant 2).
        """
        # ── Pre-run: generate and validate signals (called exactly once) ──────
        raw_signals: pd.DataFrame = strategy.generate_signals(self.data)

        assert isinstance(raw_signals, pd.DataFrame), (
            f"strategy.generate_signals() must return a pd.DataFrame, "
            f"got {type(raw_signals).__name__}.  Strategy: {strategy.get_name()}"
        )
        assert set(raw_signals.columns).issubset(set(self.data.columns)), (
            f"Signal columns {sorted(raw_signals.columns.tolist())} are not a "
            f"subset of data columns {sorted(self.data.columns.tolist())}.  "
            f"Strategy: {strategy.get_name()}"
        )

        # Enforce INVARIANT 1 structurally via shift(1)
        shifted_signals: pd.DataFrame = self._resolve_signals(raw_signals)

        # Convert shifted signals to target weights
        target_weights: pd.DataFrame = self._compute_weights(shifted_signals)

        # Pad any missing tickers with 0 so the weights cover all data columns
        target_weights = target_weights.reindex(
            columns=self.data.columns, fill_value=0.0
        )

        # Align date indices (inner join; strategy may cover a subset of dates)
        aligned_index: pd.DatetimeIndex = target_weights.index.intersection(
            self.data.index
        )
        target_weights = target_weights.loc[aligned_index]

        assert len(aligned_index) >= 60, (
            f"Aligned date index has only {len(aligned_index)} rows.  "
            "At least 60 trading days are required for a meaningful backtest.  "
            f"Strategy: {strategy.get_name()}"
        )

        # ── Initialise state ──────────────────────────────────────────────────
        initial_capital: float = float(self.config["initial_capital"])
        portfolio_value: float = initial_capital
        current_weights: pd.Series = pd.Series(0.0, index=self.data.columns)

        equity_curve: dict[pd.Timestamp, float] = {}
        daily_returns: dict[pd.Timestamp, float] = {}
        positions_history: dict[pd.Timestamp, pd.Series] = {}
        all_trades: list[dict] = []

        # ── Main loop ─────────────────────────────────────────────────────────
        prev_date: pd.Timestamp | None = None

        for date in aligned_index:
            prices: pd.Series = self.data.loc[date]

            assert not prices.isna().any(), (
                f"NaN prices on {date.date()} for tickers: "
                f"{prices[prices.isna()].index.tolist()}.  "
                "Forward-fill your price data before running the backtest."
            )

            target: pd.Series = target_weights.loc[date]

            # Step 1: Apply overnight price return to current holdings
            pnl: float = 0.0
            if prev_date is not None:
                prev_prices: pd.Series = self.data.loc[prev_date]
                safe_prev = prev_prices.replace(0.0, np.nan)
                price_return: pd.Series = (prices / safe_prev - 1.0).fillna(0.0)
                pnl = float((current_weights * price_return).sum() * portfolio_value)
                portfolio_value += pnl

            # Step 2: Execute rebalancing trades and compute friction costs
            new_weights, trades, total_cost = self._execute_trades(
                target, current_weights, prices, portfolio_value, date
            )

            # Step 3: Subtract transaction costs from portfolio value
            portfolio_value -= total_cost

            # Step 4: Record state for this date
            equity_curve[date] = portfolio_value
            denom: float = max(portfolio_value + total_cost - pnl, 1.0)
            daily_returns[date] = pnl / denom
            positions_history[date] = new_weights.copy()
            all_trades.extend(trades)

            current_weights = new_weights
            prev_date = date

        # ── Post-loop assembly ────────────────────────────────────────────────
        equity_series = pd.Series(equity_curve)
        returns_series = pd.Series(daily_returns)

        positions_df = pd.DataFrame(positions_history).T
        positions_df.index.name = "date"

        if all_trades:
            trades_df = pd.DataFrame(all_trades)[_TRADE_COLUMNS]
            trades_df = trades_df.reset_index(drop=True)
        else:
            trades_df = pd.DataFrame(columns=_TRADE_COLUMNS)

        final_value: float = float(equity_series.iloc[-1])
        total_return_pct: float = (final_value / initial_capital - 1.0) * 100.0

        print(
            f"[Backtest] {strategy.get_name()} | "
            f"{aligned_index[0].date()} → {aligned_index[-1].date()} | "
            f"Trades: {len(all_trades):,} | "
            f"Final: ${final_value:,.2f} | "
            f"Return: {total_return_pct:+.2f}%"
        )

        result = BacktestResult(
            strategy_name=strategy.get_name(),
            equity_curve=equity_series,
            returns=returns_series,
            positions=positions_df,
            trades=trades_df,
            metrics={},
            config=self.config.copy(),
        )

        if not _skip_metrics:
            # Deferred import breaks the engine ↔ metrics circular dependency
            from backtester.metrics import compute_metrics

            benchmark_result: BacktestResult | None = None
            if self.config["benchmark"] in self.data.columns:
                try:
                    benchmark_result = self.run_benchmark()
                except Exception:
                    benchmark_result = None

            result.metrics = compute_metrics(
                result,
                benchmark=benchmark_result,
                risk_free_rate=self.config.get("risk_free_rate", 0.0),
            )

        return result

    def run_benchmark(self) -> BacktestResult:
        """Run a buy-and-hold strategy on ``config["benchmark"]``.

        The benchmark is routed through the identical engine loop as any
        strategy — it pays the same commission, slippage, and rebalance
        buffer as the strategy runs — ensuring an apples-to-apples
        performance comparison. Costs are applied identically so that
        any cost advantage or disadvantage is attributable to strategy
        behaviour, not to inconsistent friction assumptions.

        Returns:
            BacktestResult containing the benchmark equity curve and
            trade log. The metrics dict is empty (``_skip_metrics=True``
            is set internally to avoid infinite recursion).
        """
        benchmark_ticker: str = self.config["benchmark"]

        class _BenchmarkHold(BaseStrategy):
            """Trivial buy-and-hold for a single benchmark ticker."""

            def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
                signals = pd.DataFrame(0.0, index=data.index, columns=data.columns)
                if benchmark_ticker in signals.columns:
                    signals[benchmark_ticker] = 1.0
                return signals

            def get_name(self) -> str:
                return f"BuyAndHold({benchmark_ticker})"

        return self.run(_BenchmarkHold(), _skip_metrics=True)

    def compare(
        self,
        results: list[BacktestResult],
        benchmark: BacktestResult | None = None,
    ) -> pd.DataFrame:
        """Build a summary comparison table for multiple backtest results.

        The ``cost_model`` column (added in Phase 2.2) records the cost
        model name from each result's config so the comparison table is
        self-documenting about how costs were computed for each row.

        Args:
            results: Strategy results to compare.
            benchmark: Optional benchmark result; appended as the last
                row if provided.

        Returns:
            DataFrame indexed by strategy name with columns:
            ``final_value``, ``total_return_pct``, ``n_trades``,
            ``total_costs``, ``cost_model``.
        """
        all_results: list[BacktestResult] = list(results) + (
            [benchmark] if benchmark is not None else []
        )

        rows: list[dict] = []
        for r in all_results:
            initial: float = float(r.config.get("initial_capital", 100_000))
            final_value: float = float(r.equity_curve.iloc[-1])
            total_return_pct: float = (final_value / initial - 1.0) * 100.0
            n_trades: int = len(r.trades)

            if n_trades > 0:
                total_costs: float = (
                    float(r.trades["commission"].sum())
                    + float(r.trades["slippage"].sum())
                )
            else:
                total_costs = 0.0

            rows.append({
                "strategy":         r.strategy_name,
                "final_value":      round(final_value, 2),
                "total_return_pct": round(total_return_pct, 4),
                "n_trades":         n_trades,
                "total_costs":      round(total_costs, 2),
                "cost_model":       r.config.get("cost_model", "unknown"),
            })

        summary = pd.DataFrame(rows).set_index("strategy")

        print("\n" + "=" * 60)
        print("STRATEGY COMPARISON")
        print("=" * 60)
        print(summary.to_string())
        print("=" * 60 + "\n")

        return summary
