"""
strategies package

Exposes all concrete strategy classes through a single import point so callers
can do:

    from strategies import MomentumStrategy, MeanReversionStrategy, MLSignalStrategy
"""

from strategies.base import BaseStrategy
from strategies.momentum import MomentumStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.ml_signal import MLSignalStrategy

__all__ = [
    "BaseStrategy",
    "MomentumStrategy",
    "MeanReversionStrategy",
    "MLSignalStrategy",
]
