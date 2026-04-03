"""
Streamlit dashboard for the algorithmic trading backtester.

Implementation plan (Phase 5)
-------------------------------
Interactive web UI that lets a user:
1. Select tickers from DEFAULT_TICKERS (or enter custom symbols).
2. Choose a strategy (Momentum / Mean Reversion / ML Signal) and tune its
   parameters via sidebar widgets.
3. Click "Run Backtest" to execute the full pipeline and display:
   - Equity curve chart (vs SPY benchmark)
   - Drawdown chart
   - Monthly returns heatmap
   - Summary metrics table (Sharpe, CAGR, Max Drawdown, Win Rate)
4. Export results as CSV or PDF report.

Run locally with:
    streamlit run app/streamlit_app.py
"""

import streamlit as st

from config import BENCHMARK_TICKER, DEFAULT_TICKERS, END_DATE, START_DATE


def main() -> None:
    """Entry point for the Streamlit dashboard."""
    st.set_page_config(
        page_title="Algo Trading Backtester",
        page_icon=":chart_with_upwards_trend:",
        layout="wide",
    )

    st.title("Algorithmic Trading Backtester")
    st.markdown(
        "Select a strategy and parameters in the sidebar, then click **Run Backtest**."
    )

    # --- Sidebar controls (to be wired up in Phase 5) ---
    with st.sidebar:
        st.header("Configuration")

        tickers = st.multiselect(
            label="Tickers",
            options=DEFAULT_TICKERS,
            default=["AAPL", "MSFT", "NVDA"],
        )

        strategy_name = st.selectbox(
            label="Strategy",
            options=["Momentum", "Mean Reversion", "ML Signal"],
        )

        st.subheader("Date Range")
        start_date = st.text_input("Start date", value=START_DATE)
        end_date = st.text_input("End date", value=END_DATE)

        run_button = st.button("Run Backtest", type="primary")

    # --- Main panel placeholder ---
    if not run_button:
        st.info("Configure your backtest in the sidebar and click **Run Backtest**.")
        return

    st.warning(
        "Backtesting engine not yet implemented — coming in Phase 3. "
        "Stay tuned!"
    )


if __name__ == "__main__":
    main()
