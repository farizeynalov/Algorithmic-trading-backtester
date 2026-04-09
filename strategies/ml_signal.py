"""
ML signal strategy — walk-forward supervised classification.

A machine learning model predicts whether each stock will outperform the
cross-sectional median return over the next N days.  The model is trained on
an expanding window of past data and generates predictions only for the
out-of-sample future — the only valid way to use supervised ML in a backtest.

Walk-forward integrity guarantees
----------------------------------
1. Training data is always strictly before the training cutoff date.
2. The StandardScaler is fit ONLY on training data; prediction data is
   transformed with the already-fitted scaler (no fit_transform on test set).
3. compute_target() uses shift(-forward_days) and must never be called on a
   slice that includes prediction-period prices.  The current design calls it
   once on the full dataset, which is safe because _train_predict_window drops
   NaN targets and uses only rows where date <= train_cutoff.
4. Features are computed once on the full price history — slicing happens on
   the pre-computed feature DataFrame, not on raw prices inside each window.

Lazy imports
------------
sklearn and xgboost are imported inside _make_model() and
_train_predict_window() so this module remains importable even if those
packages are not installed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from backtester.base import BaseStrategy


# ---------------------------------------------------------------------------
# Expected feature columns (used for validation)
# ---------------------------------------------------------------------------

_PRICE_FEATURES = [
    "ret_1d", "ret_5d", "ret_21d", "ret_63d",
    "vol_21d", "vol_63d",
    "mom_12_1",
    "rsi_14",
    "bb_pct",
    "price_to_52w_high",
    "price_to_52w_low",
]

_VOLUME_FEATURES = ["vol_ratio_21d", "vol_trend"]

ALL_FEATURES = _PRICE_FEATURES + _VOLUME_FEATURES


# ---------------------------------------------------------------------------
# Module-level feature engineering
# ---------------------------------------------------------------------------

def compute_features(
    prices: pd.DataFrame,
    volumes: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compute technical features for all tickers.

    All features are computed in wide format (same shape as ``prices``) and
    then stacked to long format.  NaN rows (warmup period) are dropped after
    forward-filling each feature independently.

    Parameters
    ----------
    prices : pd.DataFrame
        Wide-format adjusted close prices (DatetimeIndex × tickers).
    volumes : pd.DataFrame | None
        Wide-format daily volumes (same shape as prices), or None.
        If None, volume-based features are filled with 0.0.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with MultiIndex (date, ticker).
        Columns: all feature names in ALL_FEATURES.
    """
    feat: dict[str, pd.DataFrame] = {}

    # ── Price-based returns ──────────────────────────────────────────────────
    ret_1d = prices.pct_change(1)
    feat["ret_1d"]  = ret_1d
    feat["ret_5d"]  = prices.pct_change(5)
    feat["ret_21d"] = prices.pct_change(21)
    feat["ret_63d"] = prices.pct_change(63)

    # ── Rolling volatility (annualised) ──────────────────────────────────────
    feat["vol_21d"] = ret_1d.rolling(21, min_periods=21).std() * np.sqrt(252)
    feat["vol_63d"] = ret_1d.rolling(63, min_periods=63).std() * np.sqrt(252)

    # ── 12-1 momentum (Jegadeesh-Titman pattern) ─────────────────────────────
    monthly = prices.resample("ME").last()
    mom_monthly = monthly.pct_change(12).shift(1)
    feat["mom_12_1"] = mom_monthly.reindex(prices.index, method="ffill")

    # ── RSI-14 (Wilder EWM smoothing, same formula as MeanReversionStrategy) ─
    span = 2 * 14 - 1  # Wilder span for rsi_window=14
    gain = ret_1d.clip(lower=0)
    loss = (-ret_1d).clip(lower=0)
    avg_gain = gain.ewm(span=span, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(span=span, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = (100 - (100 / (1 + rs))).fillna(50)
    feat["rsi_14"] = rsi

    # ── Bollinger Band %B (bb_window=20, bb_std=2.0) ─────────────────────────
    bb_middle = prices.rolling(20, min_periods=20).mean()
    bb_std_   = prices.rolling(20, min_periods=20).std()
    bb_upper  = bb_middle + 2.0 * bb_std_
    bb_lower  = bb_middle - 2.0 * bb_std_
    feat["bb_pct"] = (prices - bb_lower) / (bb_upper - bb_lower + 1e-9)

    # ── Proximity to 52-week high / low ──────────────────────────────────────
    feat["price_to_52w_high"] = prices / prices.rolling(252, min_periods=252).max()
    feat["price_to_52w_low"]  = prices / prices.rolling(252, min_periods=252).min()

    # ── Volume-based features ─────────────────────────────────────────────────
    if volumes is not None:
        feat["vol_ratio_21d"] = volumes / volumes.rolling(21, min_periods=21).mean()
        feat["vol_trend"]     = volumes.pct_change(21)
    else:
        feat["vol_ratio_21d"] = pd.DataFrame(
            0.0, index=prices.index, columns=prices.columns
        )
        feat["vol_trend"] = pd.DataFrame(
            0.0, index=prices.index, columns=prices.columns
        )

    # ── Stack to long format ─────────────────────────────────────────────────
    stacked: dict[str, pd.Series] = {}
    for name, df in feat.items():
        s = df.stack(future_stack=True)
        s.index.names = ["date", "ticker"]
        stacked[name] = s

    result = pd.DataFrame(stacked)

    # Forward-fill each feature independently, then drop any remaining NaNs
    result = result.groupby(level="ticker").ffill()
    result = result.dropna()

    assert result.index.nlevels == 2, (
        f"compute_features must return a DataFrame with 2-level MultiIndex, "
        f"got {result.index.nlevels} levels"
    )
    for col in ALL_FEATURES:
        assert col in result.columns, (
            f"Expected feature column '{col}' not found in compute_features output. "
            f"Present columns: {result.columns.tolist()}"
        )

    inf_mask = ~np.isfinite(result.values)
    assert not inf_mask.any(), (
        f"compute_features produced {inf_mask.sum()} infinite values. "
        "Check pct_change or division operations."
    )

    return result


# ---------------------------------------------------------------------------
# Module-level target engineering
# ---------------------------------------------------------------------------

def compute_target(
    prices: pd.DataFrame,
    forward_days: int = 5,
) -> pd.Series:
    """
    Compute the binary cross-sectional outperformance target.

    For each date and ticker: 1 if the ticker's forward return exceeds the
    cross-sectional median on that date, 0 otherwise.

    Parameters
    ----------
    prices : pd.DataFrame
        Wide-format close prices.
    forward_days : int
        Prediction horizon in trading days.

    Returns
    -------
    pd.Series
        Binary target with MultiIndex (date, ticker).

    .. warning::
        This function uses ``shift(-forward_days)``, which looks forward in
        time.  It must NEVER be called on data that extends into the prediction
        period — only on the training slice.  The walk-forward design in
        ``_train_predict_window`` enforces this by dropping tail NaN targets
        before fitting.
    """
    # WARNING: uses shift(-forward_days) — only call on training data.
    # Calling on the full price history is lookahead bias if not sliced.
    fwd_return = prices.shift(-forward_days) / prices - 1

    # Stack to long format
    fwd_long = fwd_return.stack(future_stack=True)
    fwd_long.index.names = ["date", "ticker"]

    # Cross-sectional binary target: 1 if above median on that date
    target = fwd_long.groupby(level="date").transform(
        lambda x: (x > x.median()).astype(int)
    )
    return target


# ---------------------------------------------------------------------------
# Strategy class
# ---------------------------------------------------------------------------

class MLSignalStrategy(BaseStrategy):
    """
    Walk-forward ML classification strategy.

    A classifier is trained on an expanding window of technical features to
    predict whether each ticker will outperform the cross-sectional median
    return over the next ``forward_days`` trading days.  Tickers are then
    ranked by predicted probability and the top ``n_long`` are held long.

    Parameters
    ----------
    model_type : str
        ``"xgboost"`` | ``"random_forest"`` | ``"logistic"``
    forward_days : int
        Prediction horizon.  Default 5 (one trading week).
    n_long : int
        Number of top-ranked tickers to hold long each period.
    n_short : int
        Number of bottom-ranked tickers to short.  Default 0 (long-only).
    min_train_years : int
        Minimum years of data before first model training.
    retrain_freq_months : int
        How often (in months) to retrain the model on expanded data.
    feature_importance_threshold : float
        Drop features with importance below this fraction of max importance.
        Only applies to xgboost/random_forest.  Default 0.0 (keep all).
    scale_features : bool
        Apply StandardScaler before fitting.  Mandatory for logistic.
    """

    def __init__(
        self,
        model_type: str = "xgboost",
        forward_days: int = 5,
        n_long: int = 5,
        n_short: int = 0,
        min_train_years: int = 3,
        retrain_freq_months: int = 3,
        feature_importance_threshold: float = 0.0,
        scale_features: bool = True,
    ) -> None:
        assert model_type in ("xgboost", "random_forest", "logistic"), (
            f"model_type must be 'xgboost', 'random_forest', or 'logistic', "
            f"got '{model_type}'"
        )
        assert forward_days >= 1, (
            f"forward_days must be >= 1, got {forward_days}"
        )
        assert n_long >= 1, (
            f"n_long must be >= 1, got {n_long}"
        )
        assert n_short >= 0, (
            f"n_short must be >= 0, got {n_short}"
        )
        assert min_train_years >= 1, (
            f"min_train_years must be >= 1, got {min_train_years}"
        )
        assert retrain_freq_months >= 1, (
            f"retrain_freq_months must be >= 1, got {retrain_freq_months}"
        )
        assert 0.0 <= feature_importance_threshold < 1.0, (
            f"feature_importance_threshold must be in [0, 1), "
            f"got {feature_importance_threshold}"
        )

        self.model_type = model_type
        self.forward_days = forward_days
        self.n_long = n_long
        self.n_short = n_short
        self.min_train_years = min_train_years
        self.retrain_freq_months = retrain_freq_months
        self.feature_importance_threshold = feature_importance_threshold
        self.scale_features = scale_features

        # Populated during generate_signals()
        self.models_: dict[pd.Timestamp, Any] = {}
        self.scalers_: dict[pd.Timestamp, Any] = {}
        self.feature_names_: list[str] | None = None
        self.training_dates_: list[pd.Timestamp] = []
        self.feature_importances_: pd.DataFrame | None = None
        self.proba_wide_: pd.DataFrame | None = None

    def get_name(self) -> str:
        return (
            f"MLSignal({self.model_type}, fwd={self.forward_days}d, "
            f"long={self.n_long}, retrain={self.retrain_freq_months}m)"
        )

    def _make_model(self) -> Any:
        """Instantiate an unfitted classifier based on self.model_type."""
        if self.model_type == "xgboost":
            from xgboost import XGBClassifier
            return XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            )
        elif self.model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(
                n_estimators=200,
                max_depth=6,
                min_samples_leaf=20,
                random_state=42,
                n_jobs=-1,
            )
        elif self.model_type == "logistic":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(
                C=0.1,
                max_iter=1000,
                random_state=42,
                n_jobs=-1,
            )

    def _get_retraining_dates(
        self,
        all_dates: pd.DatetimeIndex,
    ) -> list[pd.Timestamp]:
        """
        Compute the expanding-window retraining schedule.

        The first training date is after ``min_train_years`` of data.
        Subsequent dates are every ``retrain_freq_months`` calendar months.

        Parameters
        ----------
        all_dates : pd.DatetimeIndex
            Full DatetimeIndex of the price data.

        Returns
        -------
        list[pd.Timestamp]
            Ordered list of dates on which the model is retrained.
        """
        first_train_idx = int(self.min_train_years * 252)
        if first_train_idx >= len(all_dates):
            raise ValueError(
                f"Insufficient data: need {self.min_train_years} years "
                f"({first_train_idx} days), got {len(all_dates)} days."
            )
        first_date = all_dates[first_train_idx]

        retraining_dates = [first_date]
        current = first_date
        while True:
            next_date = current + pd.DateOffset(months=self.retrain_freq_months)
            future = all_dates[all_dates >= next_date]
            if len(future) == 0:
                break
            retraining_dates.append(future[0])
            current = future[0]

        return retraining_dates

    def _train_predict_window(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        train_cutoff: pd.Timestamp,
        predict_start: pd.Timestamp,
        predict_end: pd.Timestamp,
    ) -> pd.Series:
        """
        Train on data up to ``train_cutoff``; predict ``predict_start``→``predict_end``.

        WALK-FORWARD INTEGRITY:
        - Training set: features with date <= train_cutoff.
        - NaN targets (last forward_days rows of training window) are dropped.
          These NaNs come from compute_target's shift(-forward_days) — they
          have no label yet and must not be used for training.
          This is NOT lookahead — it is the correct labelling constraint.
        - Scaler is fit on training data ONLY, then applied via transform()
          (never fit_transform()) to prediction data.
        - feature_cols determined from training set, applied consistently
          to prediction set.

        Parameters
        ----------
        features : pd.DataFrame
            Pre-computed long-format features (date, ticker MultiIndex).
        target : pd.Series
            Binary target with same MultiIndex as features.
        train_cutoff : pd.Timestamp
            Last date allowed in the training set.
        predict_start, predict_end : pd.Timestamp
            Inclusive prediction window.

        Returns
        -------
        pd.Series
            Predicted P(outperform) with same MultiIndex as the prediction slice.
        """
        # ── Training set: date <= train_cutoff ────────────────────────────────
        train_mask = features.index.get_level_values("date") <= train_cutoff
        X_train = features[train_mask]
        y_train = target[train_mask]

        # Drop NaN targets — these are the tail rows where shift(-forward_days)
        # produced NaN because there is not yet a realized future return.
        # Safe to drop: they are strictly inside the training window, not peeked.
        valid = y_train.notna()
        X_train = X_train[valid]
        y_train = y_train[valid]

        pred_mask = (
            (features.index.get_level_values("date") >= predict_start)
            & (features.index.get_level_values("date") <= predict_end)
        )

        if len(X_train) < 100:
            # Not enough training data — return neutral 0.5 probability
            return pd.Series(0.5, index=features[pred_mask].index)

        # ── Feature selection (optional) ──────────────────────────────────────
        if (
            self.feature_importance_threshold > 0
            and self.model_type in ("xgboost", "random_forest")
        ):
            from sklearn.ensemble import RandomForestClassifier as RFC
            selector = RFC(n_estimators=50, random_state=42, n_jobs=-1)
            selector.fit(X_train.values, y_train.values)
            importance = selector.feature_importances_
            keep = importance >= (importance.max() * self.feature_importance_threshold)
            X_train = X_train.iloc[:, keep]
            feature_cols = X_train.columns.tolist()
        else:
            feature_cols = X_train.columns.tolist()

        # ── Scaling — fit on training data ONLY ───────────────────────────────
        scaler = None
        if self.scale_features:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train_arr = scaler.fit_transform(X_train[feature_cols].values)
        else:
            X_train_arr = X_train[feature_cols].values

        # ── Fit model ─────────────────────────────────────────────────────────
        model = self._make_model()
        model.fit(X_train_arr, y_train.values.astype(int))

        # Store model and scaler keyed by train_cutoff for inspection
        self.models_[train_cutoff] = model
        self.scalers_[train_cutoff] = scaler

        # Accumulate feature importances if the model exposes them
        if hasattr(model, "feature_importances_"):
            imp = pd.Series(
                model.feature_importances_,
                index=feature_cols,
                name=train_cutoff,
            )
            if self.feature_importances_ is None:
                self.feature_importances_ = imp.to_frame().T
            else:
                self.feature_importances_ = pd.concat(
                    [self.feature_importances_, imp.to_frame().T]
                )

        # ── Predict on out-of-sample window ───────────────────────────────────
        X_pred = features[pred_mask][feature_cols]

        if len(X_pred) == 0:
            return pd.Series(dtype=float)

        # transform() only — scaler was fit on training data above
        if scaler is not None:
            X_pred_arr = scaler.transform(X_pred.values)
        else:
            X_pred_arr = X_pred.values

        # predict_proba: column 1 = P(class=1) = P(outperform)
        proba = model.predict_proba(X_pred_arr)[:, 1]
        return pd.Series(proba, index=X_pred.index)

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Orchestrate walk-forward training and generate cross-sectional signals.

        Steps
        -----
        1. Compute features on the full price history (once).
        2. Compute targets on the full price history (safe — see note below).
        3. Determine retraining schedule.
        4. Walk-forward loop: train → predict for each window.
        5. Convert predicted probabilities to ranked long/short signals.
        6. Reindex to the full data.index (warmup period → 0.0).

        Note on target safety
        ~~~~~~~~~~~~~~~~~~~~~
        compute_target() is called once on the full dataset.  This is safe
        because _train_predict_window slices training rows with
        ``date <= train_cutoff`` and then drops NaN targets.  The tail NaNs
        produced by shift(-forward_days) fall in the prediction window where
        we call predict_proba() — they never contaminate the training set.

        Parameters
        ----------
        data : pd.DataFrame
            Wide-format close prices (DatetimeIndex × tickers).

        Returns
        -------
        pd.DataFrame
            Binary signals {-1, 0, 1} with same shape as ``data``.
        """
        # Reset accumulated state so multiple calls don't cross-contaminate
        self.models_ = {}
        self.scalers_ = {}
        self.feature_importances_ = None
        self.proba_wide_ = None

        # Step 1 — features (computed once on full history)
        features = compute_features(data, volumes=None)
        self.feature_names_ = features.columns.tolist()

        # Step 2 — targets (computed once on full history — safe, see docstring)
        target = compute_target(data, self.forward_days)

        # Align target to feature index (features drop warmup NaN rows)
        target = target.reindex(features.index)

        # Step 3 — retraining schedule
        retraining_dates = self._get_retraining_dates(data.index)
        self.training_dates_ = retraining_dates
        print(
            f"Walk-forward: {len(retraining_dates)} training windows, "
            f"first={retraining_dates[0].date()}, "
            f"last={retraining_dates[-1].date()}"
        )

        # Step 4 — walk-forward loop
        all_probas: list[pd.Series] = []
        for i, train_cutoff in enumerate(retraining_dates):
            predict_start = train_cutoff + pd.Timedelta(days=1)
            predict_end = (
                retraining_dates[i + 1] - pd.Timedelta(days=1)
                if i + 1 < len(retraining_dates)
                else data.index[-1]
            )
            probas = self._train_predict_window(
                features, target, train_cutoff, predict_start, predict_end
            )
            all_probas.append(probas)
            print(
                f"  [{i + 1}/{len(retraining_dates)}] "
                f"train->{train_cutoff.date()} "
                f"predict {predict_start.date()}->{predict_end.date()} "
                f"({len(probas)} obs)"
            )

        # Step 5 — combine and unstack to wide format
        proba_long = pd.concat(all_probas)
        proba_long.index.names = ["date", "ticker"]
        proba_wide = proba_long.unstack(level="ticker")
        self.proba_wide_ = proba_wide

        # Step 6 — rank probabilities cross-sectionally → binary signals
        signals = pd.DataFrame(
            0.0, index=proba_wide.index, columns=proba_wide.columns
        )
        for date in proba_wide.index:
            row = proba_wide.loc[date].dropna()
            if len(row) < self.n_long:
                continue
            top = row.nlargest(self.n_long).index
            signals.loc[date, top] = 1.0
            if self.n_short > 0 and len(row) >= self.n_long + self.n_short:
                bottom = row.nsmallest(self.n_short).index
                signals.loc[date, bottom] = -1.0

        # Step 7 — reindex to full data.index; warmup period fills with 0.0
        signals = signals.reindex(data.index, fill_value=0.0)

        # Step 8 — validate
        assert signals.shape == data.shape, (
            f"Signal shape {signals.shape} != data shape {data.shape}"
        )
        assert float(signals.values.min()) >= -1.0 - 1e-9, (
            f"Signal below -1.0: min = {float(signals.values.min()):.6f}"
        )
        assert float(signals.values.max()) <= 1.0 + 1e-9, (
            f"Signal above +1.0: max = {float(signals.values.max()):.6f}"
        )

        return signals

    def __repr__(self) -> str:
        return self.get_name()
