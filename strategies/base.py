"""
Backward-compatibility shim — BaseStrategy now lives in backtester/base.py.

Existing imports of the form::

    from strategies.base import BaseStrategy

continue to work via this re-export.
"""

from backtester.base import BaseStrategy  # noqa: F401 (re-export)

__all__ = ["BaseStrategy"]
