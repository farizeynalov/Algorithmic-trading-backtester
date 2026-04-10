"""
tests/test_streamlit_app.py

Unit tests for the non-Streamlit logic in app/streamlit_app.py.

Streamlit apps cannot be driven end-to-end inside pytest without the
streamlit.testing library.  These tests instead validate:

  1. load_data()         — returns a real DataFrame for valid inputs
  2. Strategy params     — tuples are hashable (required for cache keys)
  3. PALETTE consistency — theme primaryColor matches PALETTE["momentum"]
  4. ROOT resolution     — path points to project root with expected layout
  5. save_figure()       — creates output directories and files correctly

To isolate the tests from Streamlit's runtime, a minimal mock is injected
into sys.modules["streamlit"] before the app module is imported.  The mock
turns @st.cache_data / @st.cache_resource into no-op passthrough decorators,
and stubs out all other st.* calls.  Because app/streamlit_app.py guards
its rendering code with `if __name__ == "__main__": main()`, importing the
module does NOT execute any UI logic.
"""

from __future__ import annotations

import pathlib
import sys
import tomllib
from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import pandas as pd
import pytest

# ─── Project root on path ─────────────────────────────────────────────────────
_HERE = pathlib.Path(__file__).parent
ROOT = _HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ─── Inject streamlit mock BEFORE importing the app module ───────────────────
# cache_data / cache_resource must be passthrough decorators so that
# load_data, run_strategy, run_benchmark remain plain callable functions.

def _passthrough_cache(**kwargs):
    """Return a no-op decorator — makes @st.cache_data a transparent wrapper."""
    def decorator(func):
        return func
    return decorator


_mock_st = MagicMock()
_mock_st.cache_data    = _passthrough_cache
_mock_st.cache_resource = _passthrough_cache
_mock_st.session_state = {}

sys.modules["streamlit"] = _mock_st

# ─── Import app module (safe — main() is guarded by __name__ == "__main__") ──
# Add app/ to sys.path so we can do a plain `import streamlit_app`.
_APP_DIR = ROOT / "app"
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

import streamlit_app  # noqa: E402  (must come after mock injection)


# ═════════════════════════════════════════════════════════════════════════════
# Test 1 — load_data() returns a well-formed DataFrame
# ═════════════════════════════════════════════════════════════════════════════

class TestLoadData:
    """Verify that load_data() fetches real price data from Yahoo Finance."""

    def test_returns_dataframe(self):
        """load_data with a small universe and short window returns a DataFrame."""
        df = streamlit_app.load_data(
            tickers=("AAPL", "SPY"),
            start="2023-01-01",
            end="2023-06-30",
        )
        assert isinstance(df, pd.DataFrame), (
            f"Expected pd.DataFrame, got {type(df).__name__}"
        )

    def test_has_expected_columns(self):
        """Returned DataFrame contains the requested tickers as columns."""
        df = streamlit_app.load_data(
            tickers=("AAPL", "SPY"),
            start="2023-01-01",
            end="2023-06-30",
        )
        assert "AAPL" in df.columns, f"AAPL not in columns: {df.columns.tolist()}"
        assert "SPY"  in df.columns, f"SPY not in columns: {df.columns.tolist()}"

    def test_has_datetime_index(self):
        """Index must be a DatetimeIndex (required by the engine)."""
        df = streamlit_app.load_data(
            tickers=("AAPL", "SPY"),
            start="2023-01-01",
            end="2023-06-30",
        )
        assert isinstance(df.index, pd.DatetimeIndex), (
            f"Expected DatetimeIndex, got {type(df.index).__name__}"
        )

    def test_not_empty(self):
        """DataFrame must contain at least one row of price data."""
        df = streamlit_app.load_data(
            tickers=("AAPL", "SPY"),
            start="2023-01-01",
            end="2023-06-30",
        )
        assert not df.empty, "DataFrame should not be empty for a valid date range"


# ═════════════════════════════════════════════════════════════════════════════
# Test 2 — Strategy parameter tuples are hashable
# ═════════════════════════════════════════════════════════════════════════════

class TestStrategyParamsTuples:
    """
    run_strategy() uses strategy_params as part of its @st.cache_resource key.
    The cache requires the argument to be hashable — tuples of (key, value)
    pairs satisfy this; plain dicts do not.
    """

    def test_momentum_params_hashable(self):
        mom_params = tuple({
            "lookback_months": 12,
            "skip_months":     1,
            "n_long":          5,
        }.items())
        assert hash(mom_params) is not None

    def test_momentum_params_round_trips(self):
        original = {"lookback_months": 12, "skip_months": 1, "n_long": 5}
        as_tuple = tuple(original.items())
        assert dict(as_tuple) == original

    def test_mean_reversion_params_hashable(self):
        mr_params = tuple({
            "bb_window":    20,
            "bb_std":       2.0,
            "rsi_oversold": 35,
        }.items())
        assert hash(mr_params) is not None

    def test_mean_reversion_params_round_trips(self):
        original = {"bb_window": 20, "bb_std": 2.0, "rsi_oversold": 35}
        as_tuple = tuple(original.items())
        assert dict(as_tuple) == original

    def test_ml_params_hashable(self):
        ml_params = tuple({
            "model_type":          "xgboost",
            "n_long":              5,
            "min_train_years":     3,
            "retrain_freq_months": 3,
        }.items())
        assert hash(ml_params) is not None

    def test_ml_params_round_trips(self):
        original = {
            "model_type":          "xgboost",
            "n_long":              5,
            "min_train_years":     3,
            "retrain_freq_months": 3,
        }
        as_tuple = tuple(original.items())
        assert dict(as_tuple) == original

    def test_different_params_produce_different_hashes(self):
        mom_params = tuple({"lookback_months": 12, "skip_months": 1, "n_long": 5}.items())
        mr_params  = tuple({"bb_window": 20, "bb_std": 2.0, "rsi_oversold": 35}.items())
        # Tuples with different content must not be equal
        assert mom_params != mr_params


# ═════════════════════════════════════════════════════════════════════════════
# Test 3 — PALETTE colours match the Streamlit theme config
# ═════════════════════════════════════════════════════════════════════════════

class TestPaletteColors:
    """
    The app uses PALETTE["momentum"] (#6C63FF) as the visual accent for the
    momentum strategy.  The Streamlit theme's primaryColor must match so that
    interactive widgets share the same visual language.
    """

    def test_palette_has_all_five_keys(self):
        from analysis.visualizations import PALETTE
        required = {"momentum", "mean_reversion", "ml_signal", "spy", "cash"}
        assert required == set(PALETTE.keys()), (
            f"PALETTE is missing keys: {required - set(PALETTE.keys())}"
        )

    def test_momentum_color_is_correct_hex(self):
        from analysis.visualizations import PALETTE
        assert PALETTE["momentum"] == "#6C63FF"

    def test_momentum_palette_matches_theme_primary_color(self):
        from analysis.visualizations import PALETTE
        config_path = ROOT / "app" / ".streamlit" / "config.toml"
        assert config_path.exists(), f"config.toml not found at {config_path}"
        with open(config_path, "rb") as fh:
            config = tomllib.load(fh)
        primary = config.get("theme", {}).get("primaryColor", "")
        assert primary.upper() == PALETTE["momentum"].upper(), (
            f"Theme primaryColor {primary!r} does not match "
            f"PALETTE['momentum'] {PALETTE['momentum']!r}"
        )


# ═════════════════════════════════════════════════════════════════════════════
# Test 4 — ROOT path resolution
# ═════════════════════════════════════════════════════════════════════════════

class TestRootPathResolution:
    """
    ROOT = pathlib.Path(__file__).parent.parent from inside app/streamlit_app.py
    must resolve to the project root regardless of where streamlit is launched.
    """

    def test_root_exists(self):
        assert streamlit_app.ROOT.exists(), (
            f"ROOT does not exist: {streamlit_app.ROOT}"
        )

    def test_root_is_directory(self):
        assert streamlit_app.ROOT.is_dir(), (
            f"ROOT is not a directory: {streamlit_app.ROOT}"
        )

    def test_config_py_exists_in_root(self):
        assert (streamlit_app.ROOT / "config.py").exists(), (
            f"config.py not found under ROOT ({streamlit_app.ROOT})"
        )

    def test_backtester_package_exists(self):
        assert (streamlit_app.ROOT / "backtester").is_dir(), (
            f"backtester/ not found under ROOT ({streamlit_app.ROOT})"
        )

    def test_strategies_package_exists(self):
        assert (streamlit_app.ROOT / "strategies").is_dir(), (
            f"strategies/ not found under ROOT ({streamlit_app.ROOT})"
        )


# ═════════════════════════════════════════════════════════════════════════════
# Test 5 — save_figure creates directories and writes files
# ═════════════════════════════════════════════════════════════════════════════

class TestFiguresDirCreation:
    """
    save_figure() must create the output directory if it does not exist and
    write at least one file per requested format.
    """

    def test_creates_output_directory(self, tmp_path):
        from analysis.visualizations import save_figure

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])

        output_dir = tmp_path / "nested" / "output"
        assert not output_dir.exists(), "Pre-condition: directory should not exist"

        save_figure(fig, "test_chart", output_dir, formats=["png"])

        assert output_dir.exists(), "save_figure should create the output directory"
        plt.close(fig)

    def test_file_is_written(self, tmp_path):
        from analysis.visualizations import save_figure

        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])

        output_dir = tmp_path / "figs"
        saved = save_figure(fig, "equity_curve", output_dir, formats=["png"])

        assert len(saved) == 1
        assert saved[0].exists(), f"File was not written: {saved[0]}"
        assert saved[0].suffix == ".png"
        assert saved[0].stat().st_size > 0, "Written file should not be empty"
        plt.close(fig)

    def test_multiple_formats(self, tmp_path):
        from analysis.visualizations import save_figure

        fig, ax = plt.subplots()
        ax.scatter([1, 2], [1, 2])

        output_dir = tmp_path / "multi"
        saved = save_figure(fig, "drawdown", output_dir, formats=["png", "pdf"])

        assert len(saved) == 2
        suffixes = {p.suffix for p in saved}
        assert ".png" in suffixes
        assert ".pdf" in suffixes
        for p in saved:
            assert p.exists(), f"Expected file not written: {p}"
        plt.close(fig)
