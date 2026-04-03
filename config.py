"""
Global configuration for the algorithmic trading backtester.

Centralises all tuneable parameters — tickers, date ranges, directory paths,
and cost model assumptions — so every module imports from a single source of
truth rather than hard-coding values.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Universe & benchmark
# ---------------------------------------------------------------------------

DEFAULT_TICKERS: list[str] = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "META", "BRK-B", "LLY", "AVGO", "JPM",
    "UNH", "XOM", "TSLA", "V", "MA",
    "PG", "JNJ", "COST", "HD", "MRK",
]

BENCHMARK_TICKER: str = "SPY"

# ---------------------------------------------------------------------------
# Backtest date range
# ---------------------------------------------------------------------------

START_DATE: str = "2015-01-01"
END_DATE: str = "2024-12-31"

# ---------------------------------------------------------------------------
# Directory layout  (all paths relative to this file's location)
# ---------------------------------------------------------------------------

_PROJECT_ROOT: Path = Path(__file__).parent

DATA_DIR: Path = _PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"

# Ensure directories exist at import time so downstream code can write freely
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Transaction cost model
# ---------------------------------------------------------------------------

TRANSACTION_COST_BPS: int = 5   # basis points charged per trade (one-way)
SLIPPAGE_BPS: int = 2           # additional slippage per trade (one-way)

# ---------------------------------------------------------------------------
# Capital
# ---------------------------------------------------------------------------

INITIAL_CAPITAL: float = 100_000.0  # USD
