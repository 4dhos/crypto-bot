"""
utils.py
────────
Shared utilities: structured logging, retry decorator, timestamp helpers,
and miscellaneous helpers used across the entire system.
"""

import logging
import time
import functools
from logging.handlers import RotatingFileHandler
from typing import Callable, Any

import config


# ── LOGGING ───────────────────────────────────────────────────────────────────

def setup_logging() -> logging.Logger:
    """Configure and return the root logger with rotating file + stdout handlers."""
    logger = logging.getLogger("v10k")
    if logger.handlers:
        return logger  # already configured (idempotent)

    logger.setLevel(getattr(logging, config.LOG_LEVEL.upper(), logging.INFO))

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Rotating file handler
    fh = RotatingFileHandler(
        config.LOG_FILE,
        maxBytes=config.LOG_MAX_BYTES,
        backupCount=config.LOG_BACKUP_COUNT,
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


log = setup_logging()


# ── RETRY DECORATOR ───────────────────────────────────────────────────────────

def retry(
    max_attempts: int = 3,
    delay: float = 2.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
    label: str = "",
) -> Callable:
    """
    Exponential-backoff retry decorator.

    Parameters
    ----------
    max_attempts : int   – total attempts before raising
    delay        : float – initial wait in seconds
    backoff      : float – multiplier applied to delay on each failure
    exceptions   : tuple – exception types to catch and retry
    label        : str   – human-readable label for log messages
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            _delay = delay
            _label = label or func.__name__
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    if attempt == max_attempts:
                        log.error(
                            "[%s] All %d attempts failed. Last error: %s",
                            _label, max_attempts, exc,
                        )
                        raise
                    log.warning(
                        "[%s] Attempt %d/%d failed (%s). Retrying in %.1fs …",
                        _label, attempt, max_attempts, exc, _delay,
                    )
                    time.sleep(_delay)
                    _delay *= backoff
        return wrapper
    return decorator


# ── TIMESTAMP HELPERS ─────────────────────────────────────────────────────────

def now_ms() -> int:
    """Current UTC time in milliseconds (used for ccxt API calls)."""
    return int(time.time() * 1000)


def ms_to_iso(ms: int) -> str:
    """Convert millisecond timestamp to a human-readable ISO string."""
    import datetime
    return datetime.datetime.utcfromtimestamp(ms / 1000).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )


# ── SYMBOL HELPERS ────────────────────────────────────────────────────────────

def base_currency(symbol: str) -> str:
    """Extract base currency from 'BTC/USDT' → 'BTC'."""
    return symbol.split("/")[0]


def is_stable(symbol: str) -> bool:
    """Return True if the base currency is a known stablecoin."""
    return base_currency(symbol) in config.STABLECOINS


def fmt_float(value: float, decimals: int = 4) -> str:
    """Format a float for clean log / Telegram output."""
    return f"{value:.{decimals}f}"


# ── SAFE DIVISION ─────────────────────────────────────────────────────────────

def safe_div(numerator: float, denominator: float, fallback: float = 0.0) -> float:
    """Divide without ZeroDivisionError; return fallback when denom is zero."""
    if denominator == 0:
        return fallback
    return numerator / denominator
