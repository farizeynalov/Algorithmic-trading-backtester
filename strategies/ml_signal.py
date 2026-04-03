"""
ML-based signal strategy.

Implementation plan (Phase 2.3)
---------------------------------
Uses a gradient-boosted tree (XGBoost) trained on a rich feature set to predict
the sign and magnitude of next-day returns.  The model output (a probability or
regression score) is mapped to a position signal in [-1, 1].

Feature set candidates:
- Technical indicators from the `ta` library (RSI, MACD, ATR, Bollinger %B, …)
- Rolling return features at multiple horizons (1d, 5d, 21d, 63d)
- Volume z-scores and price-volume trend
- Macro proxies (VIX level, yield curve slope) if available

Training protocol:
- Walk-forward expanding window to avoid look-ahead bias.
- Hyperparameter tuning via TimeSeriesSplit cross-validation.
- SHAP values for post-hoc interpretability.
"""

import pandas as pd
from sklearn.base import BaseEstimator

from strategies.base import BaseStrategy


class MLSignalStrategy(BaseStrategy):
    """
    Gradient-boosted ML signal strategy.

    Wraps a scikit-learn compatible estimator that is trained offline and
    serialised to disk.  At signal-generation time the model is loaded and
    used to score new observations.

    Parameters
    ----------
    model : BaseEstimator | None
        A fitted scikit-learn compatible model.  Pass None to use the
        default XGBoost configuration (must be trained before first use).
    feature_window : int
        Number of days of history required to compute features for a single
        prediction.  Default is 63.
    """

    def __init__(
        self,
        model: BaseEstimator | None = None,
        feature_window: int = 63,
    ) -> None:
        self.model = model
        self.feature_window = feature_window

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Score observations with the fitted model and return position signals.

        Parameters
        ----------
        data : pd.DataFrame
            OHLCV data with DatetimeIndex.

        Returns
        -------
        pd.Series
            Signal series in [-1, 1].
        """
        raise NotImplementedError(
            "MLSignalStrategy.generate_signals is not yet implemented. "
            "Scheduled for Phase 2.3."
        )

    def get_name(self) -> str:
        model_name = type(self.model).__name__ if self.model is not None else "untrained"
        return f"MLSignalStrategy(model={model_name}, feature_window={self.feature_window})"
