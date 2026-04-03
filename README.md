![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

# Algorithmic Trading Backtester

A professional-grade backtesting framework for systematic equity strategies built in Python.  The project implements three distinct alpha strategies — momentum, mean reversion, and a machine-learning signal — and evaluates them over a decade of S&P 500 data using a realistic transaction cost model.  Results are surfaced through a full performance analytics suite and an interactive Streamlit dashboard.

## Project Structure

```
algo-trading-backtester/
├── data/
│   ├── raw/                  # cached parquet files from yfinance
│   └── processed/            # cleaned, feature-engineered data
├── strategies/
│   ├── __init__.py
│   ├── base.py               # abstract base class for all strategies
│   ├── momentum.py           # 12-month cross-sectional momentum
│   ├── mean_reversion.py     # z-score mean reversion
│   └── ml_signal.py          # XGBoost-based predictive signal
├── backtester/
│   ├── __init__.py
│   ├── engine.py             # core event-driven backtesting loop
│   ├── metrics.py            # Sharpe, CAGR, drawdown, Sortino, …
│   └── costs.py              # transaction cost + slippage model
├── analysis/
│   ├── __init__.py
│   └── visualizations.py     # equity curves, drawdown, heatmaps
├── app/
│   └── streamlit_app.py      # interactive Streamlit dashboard
├── notebooks/
│   └── 01_eda.ipynb          # exploratory data analysis
├── tests/
│   ├── __init__.py
│   └── test_metrics.py       # pytest unit tests
├── .gitignore
├── requirements.txt
├── config.py                 # global configuration
└── README.md
```

## Strategies Implemented

| Strategy | Description |
|---|---|
| **Momentum** | Ranks S&P 500 stocks by trailing 12-month return (skipping 1 month) and goes long winners / short losers. |
| **Mean Reversion** | Fades short-term price over-extension using a rolling z-score with configurable entry and exit thresholds. |
| **ML Signal** | Gradient-boosted XGBoost model trained on technical indicators and multi-horizon return features to predict next-day direction. |

## Performance Results

> Results will be populated after Phase 3 backtests are complete.

| Strategy | Sharpe Ratio | CAGR | Max Drawdown | Win Rate |
|---|---|---|---|---|
| Momentum | TBD | TBD | TBD | TBD |
| Mean Reversion | TBD | TBD | TBD | TBD |
| ML Signal | TBD | TBD | TBD | TBD |
| SPY (benchmark) | TBD | TBD | TBD | TBD |

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-username/algo-trading-backtester.git
cd algo-trading-backtester

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the Streamlit dashboard
streamlit run app/streamlit_app.py
```

## Usage

```python
import pandas as pd
from strategies import MomentumStrategy
from backtester.engine import BacktestEngine

# Load price data (adjusted close)
prices = pd.read_parquet("data/raw/prices.parquet")

# Instantiate a strategy and run a backtest
strategy = MomentumStrategy(lookback=252, skip=21)
engine = BacktestEngine(strategy=strategy)
result = engine.run(prices)

# Inspect results
print(result.metrics)
result.equity_curve.plot(title=str(strategy))
```
