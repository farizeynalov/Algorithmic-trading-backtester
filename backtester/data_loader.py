"""
Data ingestion module for the algorithmic trading backtester.

Responsible for fetching, caching, validating, and serving OHLCV market data.
All data is sourced via yfinance and cached as per-ticker parquet files under
RAW_DIR to avoid redundant network calls on subsequent runs.
"""

from __future__ import annotations

import sys
import pathlib
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Config import — config.py lives at project root, one level above this file
# ---------------------------------------------------------------------------
_PROJECT_ROOT = pathlib.Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import DEFAULT_TICKERS, START_DATE, END_DATE, RAW_DIR  # noqa: E402


class DataLoader:
    """Fetches, caches, validates, and serves OHLCV market data.

    Data is retrieved via yfinance one ticker at a time and stored as
    parquet files in ``RAW_DIR``. On subsequent calls the cache is read
    instead of re-downloading, making iterative development fast.

    The recommended instantiation path is ``DataLoader.from_config()``,
    which reads all parameters from ``config.py`` so callers never need
    to hard-code tickers or dates.

    Attributes:
        tickers: Ticker symbols that will be loaded.
        start_date: Start of the date range in "YYYY-MM-DD" format.
        end_date: End of the date range in "YYYY-MM-DD" format.
        cache: When True, read from and write to parquet files in
            RAW_DIR. When False, always re-download from yfinance.

    Example:
        >>> loader = DataLoader.from_config()
        >>> df_long = loader.load()
        >>> report = loader.validate(df_long)
        >>> returns = loader.get_returns()
    """

    def __init__(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
        cache: bool = True,
    ) -> None:
        """Initialise the DataLoader.

        Args:
            tickers: List of ticker symbols to load.
            start_date: Start of the date range in "YYYY-MM-DD" format.
            end_date: End of the date range in "YYYY-MM-DD" format.
            cache: When True, read from and write to parquet cache files in
                RAW_DIR.  Set to False to always re-download from yfinance.
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.cache = cache
        RAW_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _cache_path(self, ticker: str) -> pathlib.Path:
        """Return the parquet cache path for a single ticker.

        Args:
            ticker: Ticker symbol.

        Returns:
            A ``pathlib.Path`` of the form
            ``RAW_DIR/{ticker}_{start_date}_{end_date}.parquet``.
        """
        return RAW_DIR / f"{ticker}_{self.start_date}_{self.end_date}.parquet"

    def _fetch_single(self, ticker: str) -> pd.DataFrame:
        """Download OHLCV data for one ticker from yfinance.

        Args:
            ticker: Ticker symbol.

        Returns:
            A cleaned DataFrame with a DatetimeIndex named ``date`` and
            columns ``open``, ``high``, ``low``, ``close``, ``volume``,
            ``ticker``.

        Raises:
            ValueError: If yfinance returns an empty DataFrame.
        """
        raw = yf.download(
            ticker,
            start=self.start_date,
            end=self.end_date,
            auto_adjust=True,
            progress=False,
        )

        if raw.empty:
            raise ValueError(f"No data returned for {ticker}")

        # Flatten MultiIndex columns that yfinance sometimes produces
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        # Standardise to lowercase column names
        raw.columns = [c.lower() for c in raw.columns]

        # Keep only the five OHLCV columns (drop any extras like 'dividends')
        ohlcv_cols = [c for c in ("open", "high", "low", "close", "volume") if c in raw.columns]
        raw = raw[ohlcv_cols].copy()

        raw["ticker"] = ticker
        raw.index.name = "date"
        raw.index = pd.to_datetime(raw.index)

        return raw

    def _load_single(self, ticker: str) -> pd.DataFrame:
        """Load one ticker from cache or fetch from yfinance.

        Args:
            ticker: Ticker symbol.

        Returns:
            Cleaned OHLCV DataFrame for the ticker.
        """
        cache_file = self._cache_path(ticker)

        if self.cache and cache_file.exists():
            print(f"Loading {ticker} from cache...")
            return pd.read_parquet(cache_file)

        print(f"Fetching {ticker} from yfinance...")
        df = self._fetch_single(ticker)

        if self.cache:
            df.to_parquet(cache_file)

        return df

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, tickers: Optional[list[str]] = None) -> pd.DataFrame:
        """Load OHLCV data for multiple tickers in long format.

        Iterates over ``tickers`` (or ``self.tickers`` when not supplied),
        calling ``_load_single`` for each.  Failed tickers are skipped with a
        warning rather than aborting the whole load.

        Args:
            tickers: Optional override list of ticker symbols.  When None,
                ``self.tickers`` is used.

        Returns:
            A long-format DataFrame indexed by ``date`` with columns
            ``ticker``, ``open``, ``high``, ``low``, ``close``, ``volume``,
            sorted by (date, ticker).

        Raises:
            RuntimeError: If fewer than 2 tickers loaded successfully.
        """
        symbols = tickers if tickers is not None else self.tickers
        frames: list[pd.DataFrame] = []

        for ticker in symbols:
            try:
                frames.append(self._load_single(ticker))
            except Exception as exc:  # noqa: BLE001
                print(f"WARNING: skipping {ticker} — {exc}", file=sys.stderr)

        if len(frames) < 2:
            raise RuntimeError(
                "Insufficient data: fewer than 2 tickers loaded successfully."
            )

        combined = pd.concat(frames).sort_index()
        combined = combined.sort_values(["date", "ticker"])
        return combined

    def load_wide(self, tickers: Optional[list[str]] = None) -> pd.DataFrame:
        """Load wide-format close prices (one column per ticker).

        Args:
            tickers: Optional override list of ticker symbols. When
                None, ``self.tickers`` is used.

        Returns:
            Wide-format adjusted close prices indexed by ``date``,
            one column per ticker symbol.
        """
        df_long = self.load(tickers)
        wide = df_long.pivot_table(index="date", columns="ticker", values="close")
        wide.columns.name = None
        return wide

    def get_returns(self, tickers: Optional[list[str]] = None) -> pd.DataFrame:
        """Compute daily log returns for all tickers.

        Args:
            tickers: Optional override list of ticker symbols. When
                None, ``self.tickers`` is used.

        Returns:
            Wide-format log returns with the same columns as
            ``load_wide()``; the first row is always dropped because
            ``log(price / shift(1))`` is NaN on the first date.
        """
        prices = self.load_wide(tickers)
        log_returns = np.log(prices / prices.shift(1))
        return log_returns.iloc[1:]

    def validate(self, df: pd.DataFrame) -> dict:
        """Run data-quality checks on a long-format OHLCV DataFrame.

        Args:
            df: Long-format DataFrame as returned by ``load()``.

        Returns:
            A dict with the following keys:

            - ``tickers_found`` (list[str]): Tickers present in *df*.
            - ``tickers_missing`` (list[str]): Tickers requested but absent.
            - ``date_range`` (tuple): ``(min_date, max_date)`` of the index.
            - ``total_rows`` (int): Number of rows in *df*.
            - ``missing_values`` (dict): Column → NaN count for columns that
              have at least one NaN.
            - ``gaps`` (dict): Ticker → list of (gap_start, gap_end) tuples
              where the business-day gap exceeds 5 days.
            - ``stale_tickers`` (list[str]): Tickers whose last date is more
              than 5 calendar days before the overall max date.
        """
        tickers_found: list[str] = sorted(df["ticker"].unique().tolist())
        tickers_missing: list[str] = sorted(
            set(self.tickers) - set(tickers_found)
        )
        date_range = (df.index.min(), df.index.max())
        total_rows = len(df)

        # Missing values (exclude 'ticker' column from check)
        numeric_cols = [c for c in df.columns if c != "ticker"]
        nan_counts = df[numeric_cols].isna().sum()
        missing_values: dict[str, int] = {
            col: int(count) for col, count in nan_counts.items() if count > 0
        }

        # Business-day gaps per ticker (> 5 bdays between consecutive dates)
        gaps: dict[str, list[tuple]] = {}
        max_date = df.index.max()

        for ticker in tickers_found:
            ticker_dates = df[df["ticker"] == ticker].index.sort_values()
            ticker_gaps: list[tuple] = []
            for i in range(1, len(ticker_dates)):
                bdays = len(
                    pd.bdate_range(ticker_dates[i - 1], ticker_dates[i])
                ) - 1  # -1 because bdate_range is inclusive on both ends
                if bdays > 5:
                    ticker_gaps.append((ticker_dates[i - 1], ticker_dates[i]))
            if ticker_gaps:
                gaps[ticker] = ticker_gaps

        # Stale tickers: last date more than 5 days before overall max
        stale_tickers: list[str] = [
            ticker
            for ticker in tickers_found
            if (max_date - df[df["ticker"] == ticker].index.max()).days > 5
        ]

        report = {
            "tickers_found": tickers_found,
            "tickers_missing": tickers_missing,
            "date_range": date_range,
            "total_rows": total_rows,
            "missing_values": missing_values,
            "gaps": gaps,
            "stale_tickers": stale_tickers,
        }

        # Human-readable summary
        print("=" * 60)
        print("DATA VALIDATION REPORT")
        print("=" * 60)
        print(f"  Tickers found   : {len(tickers_found)}")
        print(f"  Tickers missing : {tickers_missing if tickers_missing else 'none'}")
        print(f"  Date range      : {date_range[0].date()} → {date_range[1].date()}")
        print(f"  Total rows      : {total_rows:,}")
        if missing_values:
            print(f"  Missing values  : {missing_values}")
        else:
            print("  Missing values  : none")
        if gaps:
            print(f"  Tickers w/ gaps : {list(gaps.keys())}")
        else:
            print("  Date gaps       : none")
        if stale_tickers:
            print(f"  Stale tickers   : {stale_tickers}")
        else:
            print("  Stale tickers   : none")
        print("=" * 60)

        return report

    # ------------------------------------------------------------------
    # Class-level convenience constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls) -> "DataLoader":
        """Instantiate DataLoader using project-wide defaults from config.py.

        This is the recommended instantiation path for all notebooks and
        production code. Direct construction is reserved for tests that
        need non-default tickers or date ranges.

        Returns:
            A ``DataLoader`` configured with ``DEFAULT_TICKERS``,
            ``START_DATE``, and ``END_DATE`` from ``config.py``, with
            caching enabled.

        Example:
            >>> loader = DataLoader.from_config()
            >>> df_long = loader.load()
        """
        return cls(
            tickers=DEFAULT_TICKERS,
            start_date=START_DATE,
            end_date=END_DATE,
            cache=True,
        )
