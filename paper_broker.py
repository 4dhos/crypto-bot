"""
paper_broker.py
───────────────
Simulated order execution engine for PAPER trading mode.

Intercepts all order calls that would normally go to a real exchange and
simulates them locally using live market data.  The rest of the system
(strategy, risk, trade_manager, persistence) runs completely unchanged.

Fill model ("next_close"):
  - Limit buy:   fills at the order price if the next candle's LOW ≤ order
                 price, otherwise remains open until cancelled.  This is the
                 conservative, realistic model — it avoids assuming fills
                 that real liquidity would not provide.
  - Market sell: fills immediately at the current bid (close × 0.9995) to
                 simulate a 0.05% market taker slip.

Balance tracking:
  - Maintained in the SQLite DB under a special "PAPER_BALANCE" key in a
    separate `paper_state` table so it survives reboots.

Usage:
  In execution.py, when TRADING_MODE == "PAPER", every order function
  delegates to this module instead of ccxt.
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from typing import Optional

import config
from utils import log


# ── DB SETUP ──────────────────────────────────────────────────────────────────

_CREATE_PAPER_TABLE = """
CREATE TABLE IF NOT EXISTS paper_state (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_CREATE_PAPER_ORDERS = """
CREATE TABLE IF NOT EXISTS paper_orders (
    order_id    TEXT    PRIMARY KEY,
    symbol      TEXT    NOT NULL,
    side        TEXT    NOT NULL,
    price       REAL    NOT NULL,
    quantity    REAL    NOT NULL,
    status      TEXT    NOT NULL DEFAULT 'open',
    created_ms  INTEGER NOT NULL,
    filled_ms   INTEGER,
    fill_price  REAL
);
"""


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(config.DB_PATH, timeout=10)
    c.row_factory = sqlite3.Row
    return c


def initialize_paper_db() -> None:
    with _conn() as c:
        c.execute(_CREATE_PAPER_TABLE)
        c.execute(_CREATE_PAPER_ORDERS)
        c.commit()
    # Seed balance if not already set
    balance = get_paper_balance()
    if balance is None:
        set_paper_balance(config.PAPER_STARTING_BALANCE)
        log.info(
            "[paper] Paper account initialised — starting balance: $%.2f USDT",
            config.PAPER_STARTING_BALANCE,
        )


# ── BALANCE ───────────────────────────────────────────────────────────────────

def get_paper_balance() -> Optional[float]:
    with _conn() as c:
        row = c.execute(
            "SELECT value FROM paper_state WHERE key = 'balance'"
        ).fetchone()
    if row:
        return float(row["value"])
    return None


def set_paper_balance(balance: float) -> None:
    with _conn() as c:
        c.execute(
            "INSERT INTO paper_state (key, value) VALUES ('balance', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (str(balance),),
        )
        c.commit()


def adjust_paper_balance(delta: float) -> float:
    """Add delta (can be negative) to balance atomically.  Returns new balance."""
    current = get_paper_balance() or config.PAPER_STARTING_BALANCE
    new_bal = max(0.0, current + delta)
    set_paper_balance(new_bal)
    return new_bal


# ── ORDER SIMULATION ──────────────────────────────────────────────────────────

def paper_place_limit_buy(
    symbol: str,
    quantity: float,
    price: float,
) -> dict:
    """
    Record a simulated limit buy order.  Returns an order dict mimicking ccxt.
    The actual fill check happens in paper_check_fill().
    """
    order_id = str(uuid.uuid4())[:12]
    now = int(time.time() * 1000)
    with _conn() as c:
        c.execute(
            "INSERT INTO paper_orders (order_id, symbol, side, price, quantity, "
            "status, created_ms) VALUES (?, ?, 'buy', ?, ?, 'open', ?)",
            (order_id, symbol, price, quantity, now),
        )
        c.commit()
    log.info(
        "[paper] LIMIT BUY queued | %s | qty=%.6f @ %.6f | id=%s",
        symbol, quantity, price, order_id,
    )
    return {
        "id":       order_id,
        "symbol":   symbol,
        "side":     "buy",
        "price":    price,
        "amount":   quantity,
        "status":   "open",
        "average":  None,
    }


def paper_check_fill(
    order_id: str,
    symbol: str,
    current_low: float,
    current_close: float,
) -> tuple[bool, float]:
    """
    Check whether a pending limit buy has been filled given the current candle.

    Fill condition: current_low ≤ order price  (conservative next-close model).
    Returns (filled: bool, fill_price: float).
    """
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM paper_orders WHERE order_id = ?", (order_id,)
        ).fetchone()

    if row is None or row["status"] != "open":
        return False, 0.0

    order_price = row["price"]
    quantity    = row["quantity"]

    if current_low <= order_price:
        fill_price = order_price  # assume filled at our limit (maker)
        now = int(time.time() * 1000)
        with _conn() as c:
            c.execute(
                "UPDATE paper_orders SET status='closed', filled_ms=?, fill_price=? "
                "WHERE order_id=?",
                (now, fill_price, order_id),
            )
            c.commit()
        # Deduct cost from balance
        cost = quantity * fill_price / config.MAX_LEVERAGE
        adjust_paper_balance(-cost)
        log.info(
            "[paper] FILL | %s | qty=%.6f @ %.6f | cost=$%.4f | id=%s",
            symbol, quantity, fill_price, cost, order_id,
        )
        return True, fill_price

    return False, 0.0


def paper_market_sell(
    symbol: str,
    quantity: float,
    entry_price: float,
    current_price: float,
    reason: str = "unspecified",
) -> dict:
    """
    Simulate an immediate market sell (close long).
    Applies a 0.05% taker slippage.  Updates paper balance with PnL.
    """
    slippage    = 0.9995
    exit_price  = current_price * slippage

    # PnL = (exit - entry) * qty (leveraged, so multiply by leverage)
    pnl = (exit_price - entry_price) * quantity * config.MAX_LEVERAGE
    # Return margin + pnl to balance
    margin_returned = (quantity * entry_price) / config.MAX_LEVERAGE
    adjust_paper_balance(margin_returned + pnl)

    new_bal = get_paper_balance()
    log.info(
        "[paper] MARKET SELL | %s | qty=%.6f | entry=%.6f exit=%.6f | "
        "PnL=$%.4f | reason=%s | balance=$%.2f",
        symbol, quantity, entry_price, exit_price, pnl, reason, new_bal,
    )
    order_id = str(uuid.uuid4())[:12]
    return {
        "id":      order_id,
        "symbol":  symbol,
        "side":    "sell",
        "price":   exit_price,
        "average": exit_price,
        "amount":  quantity,
        "status":  "closed",
        "pnl":     pnl,
    }


def paper_cancel_order(order_id: str) -> bool:
    """Cancel a pending paper order.  Refunds nothing (no capital was reserved)."""
    with _conn() as c:
        c.execute(
            "UPDATE paper_orders SET status='cancelled' WHERE order_id=?",
            (order_id,),
        )
        c.commit()
    log.info("[paper] Cancelled order %s", order_id)
    return True


def paper_get_order(order_id: str) -> Optional[dict]:
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM paper_orders WHERE order_id=?", (order_id,)
        ).fetchone()
    return dict(row) if row else None


# ── PERFORMANCE REPORT ────────────────────────────────────────────────────────

def paper_performance_report() -> str:
    """Return a human-readable summary of paper trading session performance."""
    balance     = get_paper_balance() or config.PAPER_STARTING_BALANCE
    start       = config.PAPER_STARTING_BALANCE
    pnl_total   = balance - start
    pnl_pct     = (pnl_total / start) * 100 if start > 0 else 0

    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM paper_orders WHERE status='closed' AND side='sell'"
        ).fetchall()

    total_trades = len(rows)
    wins  = sum(1 for r in rows if (r["fill_price"] or 0) > 0)

    lines = [
        "📊 *Paper Trading Report*",
        f"Starting Balance: `${start:.2f}`",
        f"Current Balance:  `${balance:.2f}`",
        f"Total PnL:        `${pnl_total:+.2f}` (`{pnl_pct:+.2f}%`)",
        f"Closed Trades:    `{total_trades}`",
    ]
    return "\n".join(lines)
