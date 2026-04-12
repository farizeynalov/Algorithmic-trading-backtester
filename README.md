# 📈 Algorithmic Trading Backtester

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![Tests: 75 passing](https://img.shields.io/badge/tests-75%20passing-brightgreen)
![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit&logoColor=white)](https://algorithmic-trading-backtester-farizeynalov.streamlit.app)
![Code style: PEP8](https://img.shields.io/badge/code%20style-PEP8-black)

A production-quality backtesting framework implementing three quantitative trading
strategies on S&P 500 equities, with a full performance analytics suite and
interactive Streamlit dashboard.

| [Live App](https://algorithmic-trading-backtester-farizeynalov.streamlit.app) | [Strategies](#strategies) | [Results](#results) | [Architecture](#architecture) | [Setup](#setup) |

---

## Strategies

### 1. Momentum (Jegadeesh-Titman)

The momentum strategy implements the cross-sectional momentum factor documented by
Jegadeesh and Titman (1993): stocks that have outperformed their peers over the
trailing 12 months tend to continue outperforming over the next month. At each
month-end, tickers are ranked by their 12-month cumulative return (excluding the
most recent month — the "skip month"). Excluding the skip month is critical: the
most recent month exhibits short-term reversal rather than continuation, so
including it would contaminate the signal. The top `n_long` tickers receive a
signal of +1; the bottom `n_short` receive -1. The continuous variant replaces the
binary ranking with a cross-sectional rank normalized to (-0.5, +0.5) so all
tickers carry a non-zero signal proportional to their rank.

**Reference:** Jegadeesh, N. & Titman, S. (1993). Returns to Buying Winners and
Selling Losers: Implications for Stock Market Efficiency. *Journal of Finance*,
48(1), 65–91.

| Parameter | Default | Description |
|---|---|---|
| `lookback_months` | 12 | Return measurement window (months) |
| `skip_months` | 1 | Recent months excluded (reversal avoidance) |
| `n_long` | 5 | Stocks held long each period |
| `n_short` | 0 | Stocks shorted each period (0 = long-only) |
| `signal_type` | `ranked` | `ranked` = binary ±1; `continuous` = cross-sectional rank |

---

### 2. Mean Reversion (Bollinger Band + RSI)

The mean-reversion strategy fades short-term price dislocations using Bollinger
Bands as the entry trigger and a Wilder RSI as a confirmation filter. A long
position is entered when price falls below the lower band **and** RSI is below
`rsi_oversold`, avoiding entries into genuine downtrends. Similarly, a short
position (when `allow_short=True` in the engine) is entered when price exceeds the
upper band **and** RSI is above `rsi_overbought`. Signal construction is fully
vectorized using a forward-fill pattern as a stateless substitute for a loop-based
state machine: entry conditions write ±1 into a sparse Series, exit conditions
write 0, and `ffill()` propagates the active position between events. This keeps
the implementation auditable and avoids any date-ordered loop that could
accidentally introduce lookahead bias. Signals are computed per-ticker
independently — there is no cross-sectional ranking.

| Parameter | Default | Description |
|---|---|---|
| `bb_window` | 20 | Bollinger Band rolling window (days) |
| `bb_std` | 2.0 | Band width in standard deviations |
| `rsi_window` | 14 | RSI lookback period (Wilder EWM) |
| `rsi_oversold` | 35 | RSI threshold confirming long entry |
| `rsi_overbought` | 65 | RSI threshold confirming short entry |
| `exit_at_mean` | `True` | Exit at middle band (True) or opposite band (False) |
| `signal_type` | `binary` | `binary` = ±1; `scaled` = intensity ∝ band distance |

---

### 3. ML Signal (Walk-Forward XGBoost)

The ML signal strategy uses a supervised classifier to predict whether each ticker
will outperform the cross-sectional median return over the next `forward_days`
trading days. The critical implementation constraint is **walk-forward validation**:
the model is trained on an expanding window of past data and generates predictions
only for a strictly out-of-sample future period. Training on the full sample before
predicting — a common mistake — would constitute lookahead bias, because the model
would have seen distributional information from dates after the prediction date. An
equally important constraint is the StandardScaler: it is `fit_transform`-ed
exclusively on the training set and then `transform`-ed (never re-fit) on the
prediction set. Fitting the scaler on prediction data is a subtle but consequential
form of distributional lookahead bias.

| Parameter | Default | Description |
|---|---|---|
| `model_type` | `xgboost` | `xgboost`, `random_forest`, or `logistic` |
| `forward_days` | 5 | Prediction horizon (trading days) |
| `n_long` | 5 | Top-ranked tickers to hold long each period |
| `n_short` | 0 | Bottom-ranked tickers to short (0 = long-only) |
| `min_train_years` | 3 | Minimum history before first training |
| `retrain_freq_months` | 3 | Months between model retrains |
| `scale_features` | `True` | Apply StandardScaler before fitting |

#### Features

| Feature | Description |
|---|---|
| `ret_1d` | 1-day price return |
| `ret_5d` | 5-day price return |
| `ret_21d` | 21-day price return |
| `ret_63d` | 63-day price return |
| `vol_21d` | 21-day annualised realised volatility |
| `vol_63d` | 63-day annualised realised volatility |
| `mom_12_1` | 12-month minus 1-month Jegadeesh-Titman momentum |
| `rsi_14` | 14-period Wilder RSI |
| `bb_pct` | Bollinger Band %B (position within the band) |
| `price_to_52w_high` | Price ÷ 52-week rolling high |
| `price_to_52w_low` | Price ÷ 52-week rolling low |
| `vol_ratio_21d` | Volume ÷ 21-day mean volume (0 if volumes unavailable) |
| `vol_trend` | 21-day volume percentage change (0 if volumes unavailable) |

---

## Architecture

The framework separates concerns into four layers. The `backtester/` core has zero
knowledge of which strategy is running — it only receives a signal DataFrame and
enforces two non-negotiable invariants. The `strategies/` layer depends on
`backtester/` but the reverse dependency is architecturally prohibited. The
`analysis/` and `app/` layers are consumers of everything below them.

```
config.py
    │
    ├── backtester/
    │   ├── base.py          ← BaseStrategy (abstract)
    │   ├── data_loader.py   ← DataLoader (yfinance + parquet cache)
    │   ├── costs.py         ← CostModel hierarchy
    │   ├── engine.py        ← Backtester, BacktestResult
    │   └── metrics.py       ← compute_metrics, drawdown_series
    │
    ├── strategies/          ← depend on backtester/, never reverse
    │   ├── momentum.py
    │   ├── mean_reversion.py
    │   └── ml_signal.py
    │
    ├── analysis/
    │   └── visualizations.py ← 8 reusable plot functions
    │
    ├── notebooks/           ← consume all of the above
    └── app/
        └── streamlit_app.py ← consume all of the above
```

### Key Design Decisions

#### 1. No-Lookahead Guarantee

The engine structurally prevents lookahead bias. Inside `_resolve_signals()`, every
signal DataFrame is shifted forward by exactly one bar with `raw_signals.shift(1)`
before the portfolio loop begins. This means a signal computed using data through
date T can only influence positions starting on date T+1 — the earliest possible
execution date. The guarantee is structural, not documentary: there is no comment
saying "remember not to use future data." Instead, it is architecturally impossible
for any strategy implementation to consume data it could not have known at
execution time. The test suite verifies this directly:
`assert positions.iloc[0].abs().sum() == 0` confirms that the first row of
positions is always flat.

#### 2. CostModel Abstraction

Transaction costs are extracted from the engine into a separate class hierarchy so
that cost assumptions can be changed without touching engine logic. `CostModel` is
an abstract interface with three concrete implementations: `FlatBpsCostModel`
applies a constant basis-point fee uniformly, `TieredCommissionModel` replicates
broker tiered pricing schedules, and `SpreadSlippageModel` scales slippage with
realized volatility. The last model is the most realistic: during the 2020 COVID
crash and 2022 rate-hike period, realized volatility spiked 3–5× above its median
level. A flat-cost model would treat a rebalance during peak drawdown the same as
a rebalance in calm markets. `SpreadSlippageModel` automatically penalizes those
periods more heavily — exactly when flat-cost models are most optimistic.

#### 3. Walk-Forward ML Integrity

The ML strategy uses an expanding training window that grows with each retrain
cycle. The first training date is after `min_train_years` of data; subsequent
retrains occur every `retrain_freq_months` months. Two constraints enforce
integrity. First, the `StandardScaler` is `fit_transform`-ed on training data and
then only `transform`-ed (never re-fit) on prediction data — fitting on prediction
data would leak distributional statistics from future dates into the feature
scaling. Second, `compute_target()` uses `shift(-forward_days)` which looks
forward in time and must only be called on the training slice; the walk-forward
loop drops NaN targets (produced by the forward shift at the tail of each window)
before fitting. The test suite includes a `MockModel` test
(`test_ml_signal.py::test_walk_forward_no_future_data`) that programmatically
verifies no future observations enter training: it records which dates were seen
during each training window and asserts that none exceed the declared training
cutoff.

---

## Results

> Results below are from a live backtest run on Yahoo Finance
> data. To reproduce, run `notebooks/05_comparison_dashboard.ipynb`.

| Strategy | Ann. Return | Sharpe | Sortino | Max DD | Alpha | Costs |
|---|---|---|---|---|---|---|
| Momentum | 33.64% | 1.30 | 1.62 | -36.39% | 30.17% | 12.04% |
| Mean Reversion | 15.22% | 0.98 | 0.95 | -28.83% | 11.75% | 65.11% |
| ML Signal (XGB) | 18.40% | 1.12 | 0.86 | -36.73% | 14.93% | 169.36% |
| Combined (EW) | 27.20% | 1.27 | — | -33.40% | 23.70% | — | 
| SPY (benchmark) | 3.47% | 0.80 | 0.95 | -9.26% | — | — |

> **Notes:**
> - Backtest period: 2015-01-01 → 2024-12-31
> - Universe: 20 S&P 500 stocks + SPY benchmark
> - Initial capital: $100,000
> - Transaction costs: 5 bps commission + 2 bps slippage per trade
> - Mean Reversion run with bb_std=1.5, rsi_oversold=45.
>   Exit condition uses price crossover (not level) to avoid
>   signal cancellation on entry bars.
> - ML Signal high costs (169%) reflect excessive turnover
>   (5,067 trades). A turnover constraint would reduce this
>   significantly in a production implementation.

### Figures

**Figure 1** (`01_equity_curves`): Normalized equity curves (base = 100) for all
three strategies vs SPY benchmark, with fill shading green/red where above/below.

**Figure 2** (`02_drawdowns`): Drawdown depth comparison for all strategies and
SPY, with the peak-to-trough depth annotated at each strategy's worst trough.

**Figure 3** (`03_metrics_comparison`): 2×4 grid of horizontal bar charts
comparing 8 key performance metrics across strategies and benchmark.

**Figure 4** (`04_rolling_metrics`): Three stacked subplots showing 63-day rolling
annualised return, rolling Sharpe ratio, and rolling maximum drawdown, with COVID
and rate-hike regime shading.

**Figure 5** (`05_correlation_matrix`): Pearson return-correlation heatmap across
all strategies and SPY, with a diversification interpretation note.

**Figure 6** (`06_monthly_returns_*`): Monthly return calendar heatmap (rows =
years, cols = Jan–Dec + Full Year) for each individual strategy.

**Figure 7** (`07_trade_analysis_*`): 2×2 panel showing trade direction breakdown,
monthly transaction costs, trade size histogram, and cumulative cost drag for each
strategy.

**Figure 8** (`08_positions_*`): Portfolio weight heatmap over time (weekly sample)
and mean weight per ticker for each strategy.

**Figure 9** (`09_combined_portfolio`): All three strategies plus an equal-weight
combined portfolio vs SPY, showing diversification benefit.

**Figure 10** (`10_combined_monthly_returns`): Monthly return calendar heatmap for
the equal-weight combined portfolio.

---

## Metrics Reference

All metrics are returned by `backtester.compute_metrics()`. Metrics marked with
an asterisk (*) are present only when a benchmark `BacktestResult` is provided.

| Metric | Description |
|---|---|
| `total_return_pct` | Cumulative return over full backtest (%) |
| `annualized_return_pct` | CAGR scaled to 252 trading days (%) |
| `annualized_volatility_pct` | Annualised std dev of daily log returns (%) |
| `best_day_pct` | Single best daily return (%) |
| `worst_day_pct` | Single worst daily return (%) |
| `positive_days_pct` | Fraction of days with positive return (%) |
| `sharpe_ratio` | Annualised excess return / return std dev |
| `sortino_ratio` | Ann. excess return / downside deviation |
| `calmar_ratio` | Annualised return / \|max drawdown\| |
| `max_drawdown_pct` | Peak-to-trough decline; always ≤ 0 (%) |
| `max_drawdown_duration_days` | Calendar days from drawdown peak to trough |
| `recovery_days` | Days from trough to new equity high; None if unrecovered |
| `n_trades` | Total number of executed trade legs |
| `win_rate_pct` | % of trades with direction=BUY (proxy) |
| `avg_trade_duration_days` | Mean calendar days between BUY→SELL pairs |
| `total_commission_pct` | Total commission paid as % of starting capital |
| `total_slippage_pct` | Total slippage paid as % of starting capital |
| `total_cost_pct` | `total_commission_pct` + `total_slippage_pct` |
| `turnover_annual_pct` | Avg annual weight change across all tickers (%) |
| `alpha_pct` * | Annualised return minus benchmark return (%) |
| `beta` * | Cov(strategy, benchmark) / Var(benchmark) |
| `correlation_with_benchmark` * | Pearson correlation with benchmark returns |
| `information_ratio` * | Active return mean / active return std, ann. |
| `benchmark_total_return_pct` * | Benchmark cumulative return (%) |
| `benchmark_annualized_return_pct` * | Benchmark CAGR (%) |
| `benchmark_max_drawdown_pct` * | Benchmark peak-to-trough decline (%) |
| `benchmark_sharpe_ratio` * | Benchmark annualised Sharpe ratio |

\* Present only when `benchmark=BacktestResult` is provided to `compute_metrics()`.

---

## Setup

Assumes Python 3.11 is installed. All other dependencies are installed via pip.

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/algo-trading-backtester.git
cd algo-trading-backtester

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify the test suite
pytest tests/ -v

# 5. Run the EDA notebook
jupyter notebook notebooks/01_eda.ipynb

# 6. Run the full comparison dashboard
jupyter notebook notebooks/05_comparison_dashboard.ipynb
# Note: the ML strategy takes 30-60 seconds on first run.
# Results are cached to data/raw/ after the first execution.
```

---

## Running the Streamlit App

```bash
streamlit run app/streamlit_app.py
```

The app opens at http://localhost:8501. Nine tabs cover: strategy configuration,
equity curves, drawdowns, metrics comparison, rolling metrics, correlation matrix,
monthly returns, trade analysis, and position concentration.

**Caching note:** ML strategy results are cached after the first run. Subsequent
parameter changes for momentum and mean reversion update instantly. Changing ML
parameters (model type, retrain frequency, n positions) triggers a fresh run only
when those specific settings change.

### Deployment to Streamlit Community Cloud

1. Fork the repository on GitHub.
2. Connect your fork to [share.streamlit.io](https://share.streamlit.io).
3. Set the main file path: `app/streamlit_app.py`
4. Streamlit Cloud will pick up `requirements.txt` from the repository root automatically.

---

## Testing

```bash
# Run full suite
pytest tests/ -v

# Run specific module
pytest tests/test_engine.py -v
pytest tests/test_ml_signal.py -v  # slower — model fitting

# Run with coverage
pytest tests/ --cov=backtester --cov=strategies --cov=analysis
```

| File | Tests | What it covers |
|---|---|---|
| `test_engine.py` | 9 | Backtester loop, lookahead guard, position limits |
| `test_metrics.py` | 8 | All 27 metrics, drawdown series, benchmark-relative |
| `test_momentum.py` | 8 | Signal shape, warmup, n_long, lookahead via engine |
| `test_mean_reversion.py` | 10 | BB bands, RSI, ffill signals, exit rule comparison |
| `test_ml_signal.py` | 10 | Walk-forward integrity, scaler leakage, feature shape |
| `test_visualizations.py` | 8 | Figure objects, file export, PALETTE consistency |
| `test_streamlit_app.py` | 22 | Cache key hashing, path resolution, color config, data loading |
| **Total** | **75** | |

---

## Contributing

Pull requests are welcome. Please:
- Run `pytest tests/ -v` and confirm all 75 tests pass before opening a PR.
- Follow PEP 8 style throughout.
- Add tests for any new public function or class.

## License

MIT License. See [LICENSE](LICENSE) for details.
