"""
Streamlit interactive dashboard for the algorithmic trading backtester.

Launch from the project root with:
    streamlit run app/streamlit_app.py

Design principles:
- FAST: @st.cache_data / @st.cache_resource prevent re-runs on widget changes.
- CLEAR: every section opens with a plain-English explanation.
- HONEST: drawdowns and costs are prominent, not buried.
- SELF-CONTAINED: ROOT path detection works regardless of launch directory.
"""

from __future__ import annotations

import math
import pathlib
import sys

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# ─── Path setup — must run before any project imports ────────────────────────
ROOT = pathlib.Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtester import DataLoader, Backtester, BacktestResult, drawdown_series
from strategies import MomentumStrategy, MeanReversionStrategy, MLSignalStrategy
from analysis.visualizations import (
    plot_equity_curves,
    plot_drawdowns,
    plot_metrics_comparison,
    plot_rolling_metrics,
    plot_correlation_matrix,
    plot_monthly_returns_heatmap,
    plot_trade_analysis,
    plot_position_concentration,
    save_figure,
    PALETTE,
)
from config import (
    DEFAULT_TICKERS,
    BENCHMARK_TICKER,
    START_DATE,
    END_DATE,
    INITIAL_CAPITAL,
)


# ─── Cached data loader ───────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading market data...")
def load_data(tickers: tuple, start: str, end: str) -> pd.DataFrame:
    """Download and cache wide-format adjusted close prices.

    Parameters
    ----------
    tickers : tuple
        Tuple (not list) of ticker symbols — required for cache hashing.
    start : str
        Start date in "YYYY-MM-DD" format.
    end : str
        End date in "YYYY-MM-DD" format.
    """
    loader = DataLoader(list(tickers), start, end)
    return loader.load_wide()


# ─── Cached strategy runners ──────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def run_strategy(
    strategy_name: str,
    strategy_params: tuple,
    data_key: str,
) -> BacktestResult:
    """Run a single strategy backtest and cache the result globally.

    Parameters
    ----------
    strategy_name : str
        One of "momentum", "mean_reversion", "ml_signal".
    strategy_params : tuple
        Tuple of (key, value) pairs — hashable, used as part of the cache key.
    data_key : str
        String encoding tickers + date range; ensures cache invalidation when
        the data configuration changes.

    Notes
    -----
    Prices are read from st.session_state["prices"], which must be set by the
    caller before this function is invoked (guaranteed by the page render order).
    On a cache hit the function body never re-executes, so session_state access
    is only triggered on a genuine cache miss.
    """
    prices = st.session_state["prices"]
    bt = Backtester(prices, config={
        "allow_short": False,
        "initial_capital": INITIAL_CAPITAL,
    })
    params = dict(strategy_params)

    if strategy_name == "momentum":
        strategy = MomentumStrategy(**params)
    elif strategy_name == "mean_reversion":
        strategy = MeanReversionStrategy(**params)
    elif strategy_name == "ml_signal":
        strategy = MLSignalStrategy(**params)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name!r}")

    return bt.run(strategy)


@st.cache_resource(show_spinner=False)
def run_benchmark(data_key: str) -> BacktestResult:
    """Run SPY buy-and-hold benchmark and cache the result globally.

    Parameters
    ----------
    data_key : str
        Cache discriminator — same format as in run_strategy().
    """
    prices = st.session_state["prices"]
    bt = Backtester(prices, config={
        "allow_short": False,
        "initial_capital": INITIAL_CAPITAL,
    })
    return bt.run_benchmark()


# ─── Helper: collect results for selected strategies ──────────────────────────

def _collect_results(strategy_choice: str, data_key: str) -> tuple:
    """Run the selected strategies and return (results, benchmark).

    Reads all parameter values from st.session_state (populated by sidebar
    widgets).  Falls back to sensible defaults if a widget was not rendered
    (e.g. viewing "Momentum" mode hides mean-reversion sliders).

    Parameters
    ----------
    strategy_choice : str
        The value selected from the strategy selectbox.
    data_key : str
        Forwarded to run_strategy() and run_benchmark() as the cache key.

    Returns
    -------
    tuple[list[BacktestResult], BacktestResult]
        (results_list, benchmark_result)
    """
    results: list[BacktestResult] = []

    # ── Momentum ─────────────────────────────────────────────────────────────
    if strategy_choice in ("Momentum", "All Three (comparison)"):
        mom_params = tuple({
            "lookback_months": st.session_state.get("mom_lookback", 12),
            "skip_months":     st.session_state.get("mom_skip", 1),
            "n_long":          st.session_state.get("mom_n_long", 5),
        }.items())
        with st.spinner("Running Momentum strategy..."):
            try:
                results.append(run_strategy("momentum", mom_params, data_key))
            except Exception as exc:
                st.error(f"Momentum strategy failed: {exc}")

    # ── Mean Reversion ────────────────────────────────────────────────────────
    if strategy_choice in ("Mean Reversion", "All Three (comparison)"):
        mr_params = tuple({
            "bb_window":    st.session_state.get("mr_bb_window", 20),
            "bb_std":       st.session_state.get("mr_bb_std", 2.0),
            "rsi_oversold": st.session_state.get("mr_rsi_oversold", 35),
        }.items())
        with st.spinner("Running Mean Reversion strategy..."):
            try:
                results.append(run_strategy("mean_reversion", mr_params, data_key))
            except Exception as exc:
                st.error(f"Mean Reversion strategy failed: {exc}")

    # ── ML Signal ─────────────────────────────────────────────────────────────
    if strategy_choice in ("ML Signal", "All Three (comparison)"):
        ml_params = tuple({
            "model_type":          st.session_state.get("ml_model_type", "xgboost"),
            "n_long":              st.session_state.get("ml_n_long", 5),
            "min_train_years":     3,
            "retrain_freq_months": st.session_state.get("ml_retrain", 3),
        }.items())

        if "ml_result" not in st.session_state:
            st.session_state["ml_result"] = None

        if st.session_state["ml_result"] is None:
            if st.button("▶ Run ML Signal strategy (30-60 seconds)"):
                with st.spinner("Running ML Signal strategy..."):
                    st.session_state["ml_result"] = run_strategy(
                        "ml_signal", ml_params, data_key)
            else:
                st.info(
                    "Click the button above to run the ML Signal "
                    "strategy. Momentum and Mean Reversion results "
                    "are shown below."
                )
        ml_result = st.session_state["ml_result"]
        if ml_result is not None:
            results.append(ml_result)

    benchmark = run_benchmark(data_key)
    return results, benchmark


# ─── Main app ─────────────────────────────────────────────────────────────────

def main() -> None:
    """Render the full Streamlit dashboard.

    This function is only called when the script is run as __main__ — i.e.
    by Streamlit.  It is never called on import, which keeps the module safe
    to use in unit tests without a running Streamlit server.
    """

    # ── Page config — must be the very first st call ──────────────────────────
    st.set_page_config(
        page_title="Algo Trading Backtester",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # SIDEBAR
    # ═══════════════════════════════════════════════════════════════════════════
    with st.sidebar:
        st.title("⚙️ Configuration")
        st.markdown("---")

        # ── Data settings ──────────────────────────────────────────────────────
        st.subheader("Data")
        start_date = st.date_input(
            "Start date",
            value=pd.Timestamp(START_DATE),
            min_value=pd.Timestamp("2010-01-01"),
            max_value=pd.Timestamp("2023-01-01"),
        )
        end_date = st.date_input(
            "End date",
            value=pd.Timestamp(END_DATE),
            min_value=pd.Timestamp("2011-01-01"),
            max_value=pd.Timestamp("2024-12-31"),
        )
        st.markdown("---")

        # ── Strategy selector ──────────────────────────────────────────────────
        st.subheader("Strategy")
        strategy_choice = st.selectbox(
            "Select strategy",
            options=["Momentum", "Mean Reversion", "ML Signal", "All Three (comparison)"],
            index=3,
        )
        st.markdown("---")

        # ── Strategy-specific parameters ───────────────────────────────────────
        if strategy_choice in ("Momentum", "All Three (comparison)"):
            with st.expander(
                "Momentum parameters",
                expanded=(strategy_choice == "Momentum"),
            ):
                st.slider("Lookback months", 3, 24, 12, 1, key="mom_lookback")
                st.slider("Skip months", 0, 3, 1, 1, key="mom_skip")
                st.slider("N long positions", 1, 10, 5, 1, key="mom_n_long")

        if strategy_choice in ("Mean Reversion", "All Three (comparison)"):
            with st.expander(
                "Mean reversion parameters",
                expanded=(strategy_choice == "Mean Reversion"),
            ):
                st.slider("BB window", 10, 50, 20, 5, key="mr_bb_window")
                st.slider("BB std", 1.0, 3.0, 2.0, 0.25, key="mr_bb_std")
                st.slider("RSI oversold", 20, 45, 35, 5, key="mr_rsi_oversold")

        if strategy_choice in ("ML Signal", "All Three (comparison)"):
            with st.expander(
                "ML signal parameters",
                expanded=(strategy_choice == "ML Signal"),
            ):
                st.selectbox(
                    "Model type",
                    ["xgboost", "random_forest", "logistic"],
                    key="ml_model_type",
                )
                st.slider("N long positions", 1, 10, 5, 1, key="ml_n_long")
                st.slider("Retrain every N months", 1, 12, 3, 1, key="ml_retrain")
                st.caption(
                    "⚠️ ML strategy takes 30–60s on first run. "
                    "Results are cached — parameter changes only "
                    "rerun if you modify the settings above."
                )

        st.markdown("---")
        st.caption("Built with Python · Streamlit · XGBoost")
        st.caption("Data: Yahoo Finance via yfinance")

    # ═══════════════════════════════════════════════════════════════════════════
    # MAIN PANEL
    # ═══════════════════════════════════════════════════════════════════════════

    st.title("📈 Algorithmic Trading Backtester")
    st.markdown(
        "Interactive backtesting of three quantitative strategies on "
        f"S&P 500 stocks ({START_DATE} → {END_DATE}). "
        "Adjust parameters in the sidebar and results update automatically."
    )

    # ── Data loading ──────────────────────────────────────────────────────────
    tickers_tuple = tuple(DEFAULT_TICKERS + [BENCHMARK_TICKER])
    data_key = f"{tickers_tuple}_{start_date}_{end_date}"

    with st.spinner("Loading market data..."):
        prices = load_data(tickers_tuple, str(start_date), str(end_date))
        st.session_state["prices"] = prices

    # ── Validation warnings ────────────────────────────────────────────────────
    n_trading_days = len(prices)
    if n_trading_days < 252:
        st.warning(
            f"Only {n_trading_days} trading days available for the selected date range. "
            "At least 252 trading days (≈1 year) are recommended for a meaningful backtest."
        )

    date_range_years = (
        pd.Timestamp(str(end_date)) - pd.Timestamp(str(start_date))
    ).days / 365.25
    if strategy_choice in ("ML Signal", "All Three (comparison)") and date_range_years < 4:
        st.warning(
            "ML strategy requires at least 4 years of data for "
            "meaningful walk-forward validation."
        )

    # ── Strategy execution ─────────────────────────────────────────────────────
    results, benchmark = _collect_results(strategy_choice, data_key)

    if not results:
        st.error(
            "No strategies produced results. "
            "Try widening the date range or checking the configuration."
        )
        return

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB LAYOUT
    # ═══════════════════════════════════════════════════════════════════════════
    tabs = st.tabs([
        "📊 Overview",
        "📈 Equity Curves",
        "📉 Drawdowns",
        "🎯 Metrics",
        "🔄 Rolling Performance",
        "🔗 Correlation",
        "📅 Monthly Returns",
        "🔬 Trade Analysis",
        "⚖️ Positions",
    ])

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 0 — OVERVIEW
    # ═══════════════════════════════════════════════════════════════════════════
    with tabs[0]:
        st.markdown("### Performance at a glance")
        st.markdown(
            "Key performance indicators for each strategy versus the SPY benchmark. "
            "The delta shows the Sharpe ratio — a value above 1 is generally considered good."
        )

        # KPI card row (up to 4 columns: strategies + SPY)
        kpi_items = results[:3] + [benchmark]
        kpi_cols = st.columns(len(kpi_items))

        for col, r in zip(kpi_cols, kpi_items):
            m = r.metrics
            if m:
                ann_ret = float(m.get("annualized_return_pct") or 0)
                sharpe  = float(m.get("sharpe_ratio") or 0)
                alpha   = float(m.get("alpha_pct") or 0)
            else:
                # Benchmark — compute inline since _skip_metrics=True
                init_v  = float(r.equity_curve.iloc[0])
                final_v = float(r.equity_curve.iloc[-1])
                n_days  = len(r.equity_curve)
                ratio   = final_v / init_v if init_v > 0 else 1.0
                ann_ret = ((ratio ** (252 / n_days)) - 1) * 100 if ratio > 0 else -100.0
                std_r   = float(r.returns.std())
                sharpe  = float(r.returns.mean() / std_r * math.sqrt(252)) if std_r > 0 else 0.0
                alpha   = 0.0

            delta_color = "normal" if alpha >= 0 else "inverse"
            col.metric(
                label=r.strategy_name[:22],
                value=f"{ann_ret:.1f}%",
                delta=f"Sharpe {sharpe:.2f}",
                delta_color=delta_color,
            )

        st.markdown("---")

        # ── Summary table ──────────────────────────────────────────────────────
        st.markdown("### Strategy comparison summary")
        summary_rows = []
        for r in results:
            m = r.metrics
            summary_rows.append({
                "Ann. Return (%)": round(float(m.get("annualized_return_pct") or 0), 2),
                "Sharpe":          round(float(m.get("sharpe_ratio") or 0), 2),
                "Max DD (%)":      round(float(m.get("max_drawdown_pct") or 0), 2),
                "Alpha (%)":       round(float(m.get("alpha_pct") or 0), 2),
                "Costs (%)":       round(float(m.get("total_cost_pct") or 0), 4),
                "Trades":          int(m.get("n_trades") or 0),
            })
        summary_df = pd.DataFrame(
            summary_rows,
            index=[r.strategy_name for r in results],
        )

        def _hl_max(s: pd.Series) -> list[str]:
            return ["background-color: #d4edda" if v == s.max() else "" for v in s]

        def _hl_min(s: pd.Series) -> list[str]:
            return ["background-color: #d4edda" if v == s.min() else "" for v in s]

        styled = (
            summary_df.style
            .apply(_hl_max, subset=["Sharpe"])
            # Max DD is negative; max() = closest to 0 = shallowest = best
            .apply(_hl_max, subset=["Max DD (%)"])
            .apply(_hl_min, subset=["Costs (%)"])
            .format({
                "Ann. Return (%)": "{:.2f}",
                "Sharpe":          "{:.2f}",
                "Max DD (%)":      "{:.2f}",
                "Alpha (%)":       "{:.2f}",
                "Costs (%)":       "{:.4f}",
                "Trades":          "{:d}",
            })
        )
        st.dataframe(styled, use_container_width=True)

        # ── Plain-English interpretation ───────────────────────────────────────
        st.markdown("---")
        if len(results) > 1:
            best_sharpe = max(results, key=lambda r: float(r.metrics.get("sharpe_ratio") or 0))
            best_dd     = max(results, key=lambda r: float(r.metrics.get("max_drawdown_pct") or -999))
            best_alpha  = max(results, key=lambda r: float(r.metrics.get("alpha_pct") or 0))
            st.markdown("**What does this mean?**")
            st.markdown(
                f"The **{best_sharpe.strategy_name}** strategy achieved the highest "
                f"risk-adjusted return (Sharpe: {float(best_sharpe.metrics.get('sharpe_ratio') or 0):.2f}). "
                f"The **{best_dd.strategy_name}** strategy had the shallowest maximum drawdown "
                f"({float(best_dd.metrics.get('max_drawdown_pct') or 0):.1f}%). "
                f"The **{best_alpha.strategy_name}** strategy generated the highest alpha "
                f"({float(best_alpha.metrics.get('alpha_pct') or 0):.2f}%) over the SPY benchmark. "
                "Past performance does not guarantee future results — evaluate strategies "
                "across multiple market regimes before drawing conclusions."
            )
        else:
            r = results[0]
            m = r.metrics
            st.markdown("**What does this mean?**")
            st.markdown(
                f"**{r.strategy_name}** achieved a Sharpe ratio of "
                f"{float(m.get('sharpe_ratio') or 0):.2f} "
                f"with a maximum drawdown of {float(m.get('max_drawdown_pct') or 0):.1f}% "
                f"and alpha of {float(m.get('alpha_pct') or 0):.2f}% vs. the SPY benchmark."
            )

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 1 — EQUITY CURVES
    # ═══════════════════════════════════════════════════════════════════════════
    with tabs[1]:
        st.markdown("### Normalized equity curves (base = 100)")
        st.markdown(
            "All strategies rebased to 100 at the common start of the backtest. "
            "Green shading = outperforming SPY benchmark. "
            "Red shading = underperforming."
        )
        fig, _ = plot_equity_curves(results, benchmark)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        with st.expander("Show underlying data"):
            equity_df = pd.DataFrame(
                {r.strategy_name: r.equity_curve for r in results}
            )
            st.dataframe(equity_df.tail(20), use_container_width=True)
            st.download_button(
                "Download equity curves CSV",
                equity_df.to_csv(),
                "equity_curves.csv",
                "text/csv",
            )

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 2 — DRAWDOWNS
    # ═══════════════════════════════════════════════════════════════════════════
    with tabs[2]:
        st.markdown("### Drawdown comparison")
        st.markdown(
            "Drawdown = percentage decline from the most recent portfolio peak. "
            "Lower (more negative) = larger loss from the high-water mark. "
            "Shallower drawdowns and faster recoveries are signs of resilience."
        )
        fig, _ = plot_drawdowns(results, benchmark)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Drawdown statistics table
        st.markdown("**Drawdown statistics**")
        dd_rows = []
        for r in results:
            m = r.metrics
            rec = m.get("recovery_days")
            dd_rows.append({
                "Strategy":        r.strategy_name,
                "Max DD (%)":      round(float(m.get("max_drawdown_pct") or 0), 2),
                "Duration (days)": int(m.get("max_drawdown_duration_days") or 0),
                "Recovery (days)": "Not yet recovered" if rec is None else int(rec),
            })

        # Compute benchmark drawdown stats manually (_skip_metrics=True)
        bench_dd_s   = drawdown_series(benchmark.equity_curve)
        bench_max_dd = float(bench_dd_s.min())
        bench_trough = bench_dd_s.idxmin()
        bench_peak   = benchmark.equity_curve.loc[:bench_trough].idxmax()
        bench_dur    = int((bench_trough - bench_peak).days)
        bench_pv     = float(benchmark.equity_curve.loc[bench_peak])
        recoveries   = benchmark.equity_curve.loc[bench_trough:]
        recoveries   = recoveries[recoveries > bench_pv]
        bench_rec    = (
            "Not yet recovered"
            if recoveries.empty
            else int((recoveries.index[0] - bench_trough).days)
        )
        dd_rows.append({
            "Strategy":        benchmark.strategy_name,
            "Max DD (%)":      round(bench_max_dd, 2),
            "Duration (days)": bench_dur,
            "Recovery (days)": bench_rec,
        })

        st.dataframe(
            pd.DataFrame(dd_rows).set_index("Strategy"),
            use_container_width=True,
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 3 — METRICS
    # ═══════════════════════════════════════════════════════════════════════════
    with tabs[3]:
        st.markdown("### Performance metrics comparison")
        st.markdown(
            "Eight key metrics displayed as horizontal bar charts. "
            "Longer bars are better for all metrics except Max Drawdown "
            "(where a bar closer to zero is better)."
        )
        fig, _ = plot_metrics_comparison(results, benchmark)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Full 27-metric table
        st.markdown("**Complete metrics table (all 27 indicators)**")
        full_rows = {r.strategy_name: r.metrics for r in results}
        metrics_df = pd.DataFrame(full_rows).T
        float_cols = metrics_df.select_dtypes(include="float").columns
        metrics_df[float_cols] = metrics_df[float_cols].round(4)
        st.dataframe(metrics_df, use_container_width=True)

        with st.expander("📖 Metric glossary"):
            st.markdown("""
**Return metrics**
- **total_return_pct** — Total return over the full backtest period (%)
- **annualized_return_pct** — Compound annual growth rate, CAGR (%)
- **annualized_volatility_pct** — Annualised standard deviation of daily returns (%)
- **best_day_pct** — Single best daily return (%)
- **worst_day_pct** — Single worst daily return (%)
- **positive_days_pct** — Fraction of trading days with positive return (%)

**Risk-adjusted metrics**
- **sharpe_ratio** — Excess return per unit of total risk (annualised); above 1 is generally good
- **sortino_ratio** — Like Sharpe but penalises only downside volatility
- **calmar_ratio** — Annualised return divided by absolute max drawdown; above 0.5 is solid

**Drawdown metrics**
- **max_drawdown_pct** — Peak-to-trough decline (%; always ≤ 0)
- **max_drawdown_duration_days** — Calendar days from equity peak to trough
- **recovery_days** — Calendar days from trough back to prior peak (None if not yet recovered)

**Trading activity**
- **n_trades** — Total number of executed trades
- **win_rate_pct** — Fraction of BUY-direction trades (proxy for win rate, %)
- **avg_trade_duration_days** — Mean calendar days between BUY and next SELL per ticker
- **total_commission_pct** — Total commissions as % of initial capital
- **total_slippage_pct** — Total slippage as % of initial capital
- **total_cost_pct** — All-in transaction costs as % of initial capital
- **turnover_annual_pct** — Annualised portfolio turnover (%)

**Benchmark-relative metrics**
- **alpha_pct** — Annualised return minus benchmark annualised return (%)
- **beta** — Sensitivity of strategy returns to benchmark returns (1 = market-like)
- **correlation_with_benchmark** — Pearson correlation of daily returns with SPY
- **information_ratio** — Active return per unit of tracking error (annualised)
- **benchmark_total_return_pct** — Benchmark total return over the period (%)
- **benchmark_annualized_return_pct** — Benchmark CAGR (%)
- **benchmark_max_drawdown_pct** — Benchmark max drawdown (%)
- **benchmark_sharpe_ratio** — Benchmark Sharpe ratio
""")

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 4 — ROLLING PERFORMANCE
    # ═══════════════════════════════════════════════════════════════════════════
    with tabs[4]:
        st.markdown("### Rolling performance metrics")
        st.markdown(
            "Performance statistics computed over a sliding window. "
            "Shaded regions mark significant market regime changes "
            "(COVID-19 in 2020, rate-hike cycle in 2022)."
        )
        window_input = st.select_slider(
            "Rolling window",
            options=[21, 42, 63, 126, 252],
            value=63,
            format_func=lambda x: f"{x} days ({x // 21} months)",
        )
        fig, _ = plot_rolling_metrics(results, benchmark, window=window_input)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 5 — CORRELATION
    # ═══════════════════════════════════════════════════════════════════════════
    with tabs[5]:
        st.markdown("### Return correlation matrix")
        st.markdown(
            "Pearson correlation of daily returns across strategies and the benchmark. "
            "Values close to 0 indicate low co-movement — a signal that combining "
            "strategies into a portfolio can reduce overall portfolio risk."
        )
        fig, _ = plot_correlation_matrix(results, benchmark)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        if strategy_choice == "All Three (comparison)" and len(results) > 1:
            series_dict = {r.strategy_name[:20]: r.returns for r in results + [benchmark]}
            corr = pd.concat(series_dict, axis=1).dropna().corr()
            st.dataframe(corr.round(4), use_container_width=True)

            n = len(corr)
            off_vals = [
                corr.iloc[i, j]
                for i in range(n)
                for j in range(n)
                if i != j
            ]
            min_corr = min(off_vals) if off_vals else 1.0
            if min_corr < 0.3:
                st.success(
                    f"Low correlation detected (min off-diagonal: {min_corr:.3f}) — "
                    "combining these strategies may provide meaningful diversification."
                )
            elif min_corr < 0.6:
                st.info(
                    f"Moderate correlation (min off-diagonal: {min_corr:.3f}) — "
                    "some diversification benefit from combining strategies."
                )
            else:
                st.warning(
                    f"High correlation (min off-diagonal: {min_corr:.3f}) — "
                    "limited diversification benefit from combining these strategies."
                )

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 6 — MONTHLY RETURNS
    # ═══════════════════════════════════════════════════════════════════════════
    with tabs[6]:
        st.markdown("### Monthly return calendars")
        st.markdown(
            "Each cell shows the total return for that calendar month. "
            "Green = positive month, red = negative. "
            "The 'Full Year' column is the compounded annual return."
        )
        strategy_selector = st.selectbox(
            "Select strategy to view",
            options=[r.strategy_name for r in results],
            key="monthly_selector",
        )
        selected = next(r for r in results if r.strategy_name == strategy_selector)

        fig, _ = plot_monthly_returns_heatmap(selected)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        monthly = (
            selected.returns
            .resample("ME")
            .apply(lambda x: (1 + x).prod() - 1) * 100
        )
        col1, col2 = st.columns(2)
        col1.metric(
            "Best month",
            f"{monthly.max():.1f}%",
            monthly.idxmax().strftime("%b %Y"),
        )
        col2.metric(
            "Worst month",
            f"{monthly.min():.1f}%",
            monthly.idxmin().strftime("%b %Y"),
            delta_color="inverse",
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 7 — TRADE ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════
    with tabs[7]:
        st.markdown("### Trade analysis")
        st.markdown(
            "Breakdown of executed trades: direction mix, monthly transaction costs, "
            "trade size distribution, and cumulative cost drag on capital."
        )
        strategy_selector_2 = st.selectbox(
            "Select strategy to view",
            options=[r.strategy_name for r in results],
            key="trade_selector",
        )
        selected_2 = next(r for r in results if r.strategy_name == strategy_selector_2)

        fig, _ = plot_trade_analysis(selected_2)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        if not selected_2.trades.empty:
            st.dataframe(
                selected_2.trades.sort_values("date", ascending=False),
                use_container_width=True,
                height=300,
            )
            st.download_button(
                "Download trade log CSV",
                selected_2.trades.to_csv(index=False),
                f"trades_{selected_2.strategy_name}.csv",
                "text/csv",
            )
        else:
            st.info("No trades were executed for this strategy.")

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 8 — POSITIONS
    # ═══════════════════════════════════════════════════════════════════════════
    with tabs[8]:
        st.markdown("### Portfolio position weights")
        st.markdown(
            "Left heatmap: weight allocated to each stock over time (weekly sample). "
            "Right bar chart: time-averaged weight per ticker. "
            "Together they reveal concentration risk and holding patterns."
        )
        strategy_selector_3 = st.selectbox(
            "Select strategy to view",
            options=[r.strategy_name for r in results],
            key="position_selector",
        )
        selected_3 = next(r for r in results if r.strategy_name == strategy_selector_3)

        fig, _ = plot_position_concentration(selected_3)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        if not selected_3.positions.empty:
            latest = selected_3.positions.iloc[-1]
            latest_nonzero = latest[latest.abs() > 0.01].sort_values(ascending=False)
            if not latest_nonzero.empty:
                st.markdown("**Most recent portfolio weights:**")
                st.dataframe(
                    latest_nonzero.rename("Weight").to_frame(),
                    use_container_width=True,
                )
            else:
                st.info("No active positions at the end of the backtest period.")
        else:
            st.info("No position data available for this strategy.")

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "**Algo Trading Backtester** · "
        "Built as part of NYU CDS MS Data Science portfolio · "
        "Data via Yahoo Finance · "
        "Not financial advice."
    )


# ─── Entry point ──────────────────────────────────────────────────────────────
# Streamlit runs this script as __main__, so main() is called when launched
# with `streamlit run app/streamlit_app.py`.  When imported in pytest (where
# __name__ != "__main__"), main() is NOT called — keeping imports safe.

if __name__ == "__main__":
    main()
