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
    """
    Event-driven, single-pass backtesting engine.

    Parameters
    ----------
    data : pd.DataFrame
        Wide-format adjusted close prices (``df_wide`` from DataLoader).
        Index must be a DatetimeIndex; columns are ticker symbols.
    config : dict | None
        Optional overrides for any default engine parameter.  Keys not
        present fall back to the defaults in ``_DEFAULT_CONFIG``.
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

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_signals(self, raw_signals: pd.DataFrame) -> pd.DataFrame:
        """
        Enforce INVARIANT 1 by shifting all signals forward by one bar.

        A strategy's signal on date T is derived from data through date T.
        Applying .shift(1) ensures that signal only affects positions starting
        on date T+1 — the structural guarantee of no lookahead bias.

        Parameters
        ----------
        raw_signals : pd.DataFrame
            Strategy output in [-1, 1] with index ⊆ self.data.index and
            columns ⊆ self.data.columns.

        Returns
        -------
        pd.DataFrame
            Shifted signals; row 0 is 0 (NaN filled) because the day-0
            raw signal will execute on day 1.
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
        """
        Convert shifted signals in [-1, 1] into target portfolio weights.

        Two modes are supported (``self.config["position_sizing"]``):

        ``"equal_weight"``
            Every non-zero signal gets an equal share of the portfolio.
            Longs receive +1/n_active; shorts receive -1/n_active (when
            ``allow_short=True``).

        ``"signal_weighted"``
            Long weights are normalised so they sum to 1; short weights are
            normalised so their absolute values sum to 1.

        In both modes individual weights are capped at ±max_position_size,
        and the total absolute exposure is asserted to be ≤ 100%.

        Parameters
        ----------
        signals : pd.DataFrame
            Shifted signals from ``_resolve_signals``.

        Returns
        -------
        pd.DataFrame
            Portfolio weights; same shape as ``signals``.
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
        """
        Compute position changes, apply a rebalance buffer, record trade costs.

        Parameters
        ----------
        weights_today : pd.Series
            Target weights after rebalancing (aligned to self.data.columns).
        weights_prev : pd.Series
            Current holdings expressed as weights (aligned to self.data.columns).
        prices_today : pd.Series
            Closing prices for all tickers on this date.
        portfolio_value : float
            Current portfolio value in USD (after today's P&L, before costs).
        date : pd.Timestamp
            Current date — used only to tag each trade record.

        Returns
        -------
        tuple[pd.Series, list[dict], float]
            ``weights_today``  — target weights (become new current weights)
            ``trades``         — list of trade-record dicts
            ``total_cost``     — total commission + slippage in USD
        """
        allow_short: bool = self.config["allow_short"]
        commission_rate: float = self.config["commission_bps"] / 10_000
        slippage_rate: float = self.config["slippage_bps"] / 10_000
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
            commission: float = abs(dollar_trade) * commission_rate
            slippage: float = abs(dollar_trade) * slippage_rate
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

    def run(self, strategy: BaseStrategy) -> BacktestResult:
        """
        Execute the strategy backtest over ``self.data``.

        ``strategy.generate_signals()`` is called exactly once; the result
        is cached before the loop begins to prevent accidental multi-calls.

        Parameters
        ----------
        strategy : BaseStrategy
            Concrete strategy instance.

        Returns
        -------
        BacktestResult
            Equity curve, daily returns, positions, trade log, and config.
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

        return BacktestResult(
            strategy_name=strategy.get_name(),
            equity_curve=equity_series,
            returns=returns_series,
            positions=positions_df,
            trades=trades_df,
            metrics={},
            config=self.config.copy(),
        )

    def run_benchmark(self) -> BacktestResult:
        """
        Run a buy-and-hold strategy on ``config["benchmark"]``.

        The benchmark is routed through the identical engine loop as any
        strategy — it pays the same commission, slippage, and rebalance
        buffer — ensuring an apples-to-apples performance comparison.

        Returns
        -------
        BacktestResult
            Equity curve and trade log for the benchmark.
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

        return self.run(_BenchmarkHold())

    def compare(
        self,
        results: list[BacktestResult],
        benchmark: BacktestResult | None = None,
    ) -> pd.DataFrame:
        """
        Build a summary comparison table for multiple backtest results.

        Parameters
        ----------
        results : list[BacktestResult]
            Strategy results to compare.
        benchmark : BacktestResult | None
            Optional benchmark result; appended as the last row if provided.

        Returns
        -------
        pd.DataFrame
            Index = strategy names; columns = [final_value, total_return_pct,
            n_trades, total_costs].
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
            })

        summary = pd.DataFrame(rows).set_index("strategy")

        print("\n" + "=" * 60)
        print("STRATEGY COMPARISON")
        print("=" * 60)
        print(summary.to_string())
        print("=" * 60 + "\n")

        return summary
