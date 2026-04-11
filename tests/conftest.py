"""
Shared test fixtures for the algorithmic trading backtester test suite.

This module provides session-scoped fixtures that are available to all
test files without explicit import. File-local fixtures defined in
individual test modules are intentionally preserved — this conftest only
adds a session-scoped synthetic price fixture for future tests.
"""

import numpy as np
import pandas as pd
import pytest

import sys
import pathlib

_PROJECT_ROOT = pathlib.Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import DEFAULT_TICKERS, BENCHMARK_TICKER


@pytest.fixture(scope="session")
def synthetic_prices() -> pd.DataFrame:
    """Session-scoped wide-format price DataFrame for integration tests.

    Generates a reproducible 10-year GBM price series covering all
    DEFAULT_TICKERS plus the BENCHMARK_TICKER. The random seed is fixed
    at 42 for deterministic behaviour across test runs.

    Returns:
        Wide-format DataFrame of shape (n_days, n_tickers) indexed by
        business dates from 2015-01-01 to 2024-12-31.
    """
    dates = pd.bdate_range("2015-01-01", "2024-12-31")
    tickers = DEFAULT_TICKERS + [BENCHMARK_TICKER]
    np.random.seed(42)
    prices = pd.DataFrame(
        100 * np.exp(
            np.random.randn(len(dates), len(tickers)).cumsum(axis=0) * 0.008
        ),
        index=dates,
        columns=tickers,
    )
    return prices
