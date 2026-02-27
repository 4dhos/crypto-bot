"""
persistence.py
──────────────
SQLite-backed persistence layer.  Stores open positions and their current
state so the bot survives server reboots, crashes, and redeployments.

Schema
──────
positions
  symbol          TEXT  PRIMARY KEY   e.g. "BTC/USDT"
  entry_price     REAL                average filled entry price
  quantity        REAL                total position size in base currency
  stop_loss       REAL                current active stop-loss price
  leverage        INTEGER             leverage set on this position
  atr_at_entry    REAL                ATR value at the time of entry
  pyramid_filled  INTEGER (0/1)       whether the pyramid add was executed
  entry_time_ms   INTEGER             entry timestamp in milliseconds
  candles_elapsed INTEGER             15m candles counted since entry
  side            TEXT                "long" (only longs in V3.0)
"""

import sqlite3
import json
from typing import Optional

import config
from utils import log


# ── SCHEMA ────────────────────────────────────────────────────────────────────

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS positions (
    symbol          TEXT    PRIMARY KEY,
    entry_price     REAL    NOT NULL,
    quantity        REAL    NOT NULL,
    stop_loss       REAL    NOT NULL,
    leverage        INTEGER NOT NULL,
    atr_at_entry    REAL    NOT NULL,
    pyramid_filled  INTEGER NOT NULL DEFAULT 0,
    entry_time_ms   INTEGER NOT NULL,
    candles_elapsed INTEGER NOT NULL DEFAULT 0,
    side            TEXT    NOT NULL DEFAULT 'long',
    extra_json      TEXT    NOT NULL DEFAULT '{}'
);
"""


# ── CONNECTION FACTORY ────────────────────────────────────────────────────────

def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(config.DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_db() -> None:
    """Create tables if they don't exist.  Safe to call on every startup."""
    with _get_conn() as conn:
        conn.execute(_CREATE_TABLE)
        conn.commit()
    log.info("[persistence] Database ready at '%s'", config.DB_PATH)


# ── WRITE OPERATIONS ──────────────────────────────────────────────────────────

def upsert_position(
    symbol: str,
    entry_price: float,
    quantity: float,
    stop_loss: float,
    leverage: int,
    atr_at_entry: float,
    entry_time_ms: int,
    pyramid_filled: bool = False,
    candles_elapsed: int = 0,
    side: str = "long",
    extra: Optional[dict] = None,
) -> None:
    """Insert or replace a position record (full overwrite on conflict)."""
    extra_json = json.dumps(extra or {})
    sql = """
    INSERT INTO positions
        (symbol, entry_price, quantity, stop_loss, leverage, atr_at_entry,
         pyramid_filled, entry_time_ms, candles_elapsed, side, extra_json)
    VALUES
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(symbol) DO UPDATE SET
        entry_price     = excluded.entry_price,
        quantity        = excluded.quantity,
        stop_loss       = excluded.stop_loss,
        leverage        = excluded.leverage,
        atr_at_entry    = excluded.atr_at_entry,
        pyramid_filled  = excluded.pyramid_filled,
        entry_time_ms   = excluded.entry_time_ms,
        candles_elapsed = excluded.candles_elapsed,
        side            = excluded.side,
        extra_json      = excluded.extra_json;
    """
    with _get_conn() as conn:
        conn.execute(sql, (
            symbol, entry_price, quantity, stop_loss, leverage, atr_at_entry,
            int(pyramid_filled), entry_time_ms, candles_elapsed, side, extra_json,
        ))
        conn.commit()
    log.debug("[persistence] Upserted position: %s", symbol)


def update_stop_loss(symbol: str, new_stop: float) -> None:
    """Update only the stop-loss for an existing position."""
    with _get_conn() as conn:
        conn.execute(
            "UPDATE positions SET stop_loss = ? WHERE symbol = ?",
            (new_stop, symbol),
        )
        conn.commit()
    log.debug("[persistence] Updated SL for %s → %.6f", symbol, new_stop)


def mark_pyramid_filled(symbol: str) -> None:
    """Flag that the pyramid add has been executed for this position."""
    with _get_conn() as conn:
        conn.execute(
            "UPDATE positions SET pyramid_filled = 1 WHERE symbol = ?",
            (symbol,),
        )
        conn.commit()
    log.debug("[persistence] Pyramid marked filled for %s", symbol)


def increment_candle_count(symbol: str) -> int:
    """Increment the candle counter and return the new count."""
    with _get_conn() as conn:
        conn.execute(
            "UPDATE positions SET candles_elapsed = candles_elapsed + 1 WHERE symbol = ?",
            (symbol,),
        )
        conn.commit()
        row = conn.execute(
            "SELECT candles_elapsed FROM positions WHERE symbol = ?", (symbol,)
        ).fetchone()
    count = row["candles_elapsed"] if row else 0
    log.debug("[persistence] Candle count for %s: %d", symbol, count)
    return count


def close_position(symbol: str) -> None:
    """Remove a position from the database (trade is closed)."""
    with _get_conn() as conn:
        conn.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))
        conn.commit()
    log.info("[persistence] Closed/removed position: %s", symbol)


# ── READ OPERATIONS ───────────────────────────────────────────────────────────

def get_open_positions() -> list[dict]:
    """Return all open positions as a list of plain dicts."""
    with _get_conn() as conn:
        rows = conn.execute("SELECT * FROM positions").fetchall()
    positions = []
    for row in rows:
        d = dict(row)
        d["pyramid_filled"] = bool(d["pyramid_filled"])
        d["extra"] = json.loads(d.pop("extra_json", "{}"))
        positions.append(d)
    return positions


def get_position(symbol: str) -> Optional[dict]:
    """Return a single position dict, or None if not found."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM positions WHERE symbol = ?", (symbol,)
        ).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["pyramid_filled"] = bool(d["pyramid_filled"])
    d["extra"] = json.loads(d.pop("extra_json", "{}"))
    return d


def count_open_positions() -> int:
    """Return the number of currently tracked open positions."""
    with _get_conn() as conn:
        row = conn.execute("SELECT COUNT(*) AS n FROM positions").fetchone()
    return row["n"] if row else 0
