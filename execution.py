"""
execution.py
────────────
Thin, resilient ccxt wrapper with a built-in PAPER trading bypass.

TRADING_MODE routing:
  PAPER → All order functions delegate to paper_broker.py (no exchange needed,
          US-friendly, no API keys required).  Market data still pulled from
          MEXC public endpoints (no auth needed for OHLCV/tickers).
  LIVE  → Real orders sent to MEXC Perpetual Futures via ccxt.

All functions return OrderResult so callers are exchange-agnostic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import ccxt

import config
from utils import log, retry


# ── RESULT DATACLASS ──────────────────────────────────────────────────────────

@dataclass
class OrderResult:
    success: bool
    order_id: str         = ""
    symbol: str           = ""
    side: str             = ""
    price: float          = 0.0
    quantity: float       = 0.0
    status: str           = ""
    raw: dict             = field(default_factory=dict)
    error: str            = ""


# ── EXCHANGE FACTORY ──────────────────────────────────────────────────────────

def build_exchange() -> ccxt.Exchange:
    """
    Construct and return the correct ccxt exchange based on TRADING_MODE.

    PAPER → MEXC with no API keys (public endpoints only — free, US-accessible).
            No authentication needed for market data (OHLCV, tickers, funding).
    LIVE  → MEXC Perpetual Futures with full auth for order placement.
    """
    if config.TRADING_MODE == "PAPER":
        # Use MEXC public endpoints only — no keys, no auth, no geo-block.
        exchange = ccxt.mexc({
            "options": {"defaultType": "swap"},
        })
        log.info(
            "[execution] Mode: PAPER — using MEXC public data, all orders simulated locally"
        )
    else:
        exchange = ccxt.mexc({
            "apiKey":  config.MEXC_API_KEY,
            "secret":  config.MEXC_API_SECRET,
            "options": {"defaultType": "swap"},
        })
        log.info("[execution] Mode: LIVE — MEXC Perpetuals (real money)")

    exchange.load_markets()
    return exchange


# ── BALANCE ───────────────────────────────────────────────────────────────────

def fetch_usdt_balance(exchange) -> float:
    """Return free USDT balance — paper balance from DB, or live from exchange."""
    if config.TRADING_MODE == "PAPER":
        from paper_broker import get_paper_balance
        bal = get_paper_balance()
        return bal if bal is not None else config.PAPER_STARTING_BALANCE

    try:
        bal = exchange.fetch_balance()
        free = bal.get("USDT", {}).get("free", 0.0)
        return float(free or 0.0)
    except Exception as exc:
        log.error("[execution] fetch_balance failed: %s", exc)
        return 0.0


# ── LEVERAGE ──────────────────────────────────────────────────────────────────

def set_leverage(exchange, symbol: str, leverage: int) -> bool:
    """Set leverage — no-op in PAPER mode."""
    if config.TRADING_MODE == "PAPER":
        log.info("[paper] set_leverage %dx on %s (simulated)", leverage, symbol)
        return True
    try:
        exchange.set_leverage(leverage, symbol)
        log.info("[execution] Leverage set to %dx on %s", leverage, symbol)
        return True
    except ccxt.NotSupported:
        log.warning("[execution] set_leverage not supported for %s", symbol)
        return True
    except Exception as exc:
        log.error("[execution] set_leverage failed on %s: %s", symbol, exc)
        return False


# ── PRICE HELPERS ─────────────────────────────────────────────────────────────

def _round_price(exchange, symbol: str, price: float) -> float:
    try:
        return float(exchange.price_to_precision(symbol, price))
    except Exception:
        return price


def _round_amount(exchange, symbol: str, amount: float) -> float:
    try:
        return float(exchange.amount_to_precision(symbol, amount))
    except Exception:
        return amount


# ── LIMIT BUY ─────────────────────────────────────────────────────────────────

@retry(max_attempts=3, delay=2.0, backoff=2.0, label="place_limit_buy")
def place_limit_buy(
    exchange,
    symbol: str,
    quantity: float,
    price: float,
    params: Optional[dict] = None,
) -> OrderResult:
    """Place a limit buy — simulated in PAPER mode, real in LIVE mode."""
    if config.TRADING_MODE == "PAPER":
        from paper_broker import paper_place_limit_buy
        raw = paper_place_limit_buy(symbol, quantity, price)
        return OrderResult(
            success=True,
            order_id=raw["id"],
            symbol=symbol,
            side="buy",
            price=price,
            quantity=quantity,
            status="open",
            raw=raw,
        )
    try:
        p = _round_price(exchange, symbol, price)
        q = _round_amount(exchange, symbol, quantity)
        order = exchange.create_limit_buy_order(symbol, q, p, params or {})
        log.info(
            "[execution] Limit BUY placed | %s | qty=%.6f @ %.6f | id=%s",
            symbol, q, p, order.get("id"),
        )
        return OrderResult(
            success=True,
            order_id=str(order.get("id", "")),
            symbol=symbol, side="buy",
            price=p, quantity=q,
            status=order.get("status", "open"),
            raw=order,
        )
    except Exception as exc:
        log.error("[execution] place_limit_buy failed for %s: %s", symbol, exc)
        return OrderResult(success=False, symbol=symbol, side="buy", error=str(exc))


# ── LIMIT SELL ────────────────────────────────────────────────────────────────

@retry(max_attempts=3, delay=2.0, backoff=2.0, label="place_limit_sell")
def place_limit_sell(
    exchange,
    symbol: str,
    quantity: float,
    price: float,
    params: Optional[dict] = None,
) -> OrderResult:
    """Place a limit sell.  (Paper mode: recorded but not commonly used — exits use market.)"""
    if config.TRADING_MODE == "PAPER":
        # In paper mode, exits are handled via close_position_market
        return OrderResult(
            success=True, order_id="paper-sell-noop",
            symbol=symbol, side="sell",
            price=price, quantity=quantity, status="open",
        )
    try:
        p = _round_price(exchange, symbol, price)
        q = _round_amount(exchange, symbol, quantity)
        order = exchange.create_limit_sell_order(symbol, q, p, params or {})
        log.info(
            "[execution] Limit SELL placed | %s | qty=%.6f @ %.6f | id=%s",
            symbol, q, p, order.get("id"),
        )
        return OrderResult(
            success=True,
            order_id=str(order.get("id", "")),
            symbol=symbol, side="sell",
            price=p, quantity=q,
            status=order.get("status", "open"),
            raw=order,
        )
    except Exception as exc:
        log.error("[execution] place_limit_sell failed for %s: %s", symbol, exc)
        return OrderResult(success=False, symbol=symbol, side="sell", error=str(exc))


# ── MARKET CLOSE ──────────────────────────────────────────────────────────────

def close_position_market(
    exchange,
    symbol: str,
    quantity: float,
    reason: str = "unspecified",
    entry_price: float = 0.0,
) -> OrderResult:
    """
    Close a long position immediately.
    PAPER: simulates taker fill with 0.05% slippage.
    LIVE:  sends a reduceOnly market sell to exchange.
    """
    if config.TRADING_MODE == "PAPER":
        from paper_broker import paper_market_sell
        from data import fetch_ohlcv
        try:
            df = fetch_ohlcv(exchange, symbol, config.TF_15M, limit=3)
            current_price = float(df.iloc[-1]["close"])
        except Exception:
            current_price = entry_price  # fallback
        raw = paper_market_sell(symbol, quantity, entry_price, current_price, reason)
        return OrderResult(
            success=True,
            order_id=raw["id"],
            symbol=symbol, side="sell",
            price=raw["average"],
            quantity=quantity,
            status="closed",
            raw=raw,
        )

    try:
        q = _round_amount(exchange, symbol, quantity)
        order = exchange.create_market_sell_order(symbol, q, {"reduceOnly": True})
        log.info(
            "[execution] Market CLOSE | %s | qty=%.6f | reason=%s | id=%s",
            symbol, q, reason, order.get("id"),
        )
        return OrderResult(
            success=True,
            order_id=str(order.get("id", "")),
            symbol=symbol, side="sell",
            quantity=q,
            status=order.get("status", "closed"),
            raw=order,
        )
    except Exception as exc:
        log.error("[execution] close_position_market failed for %s: %s", symbol, exc)
        return OrderResult(success=False, symbol=symbol, side="sell", error=str(exc))


# ── CANCEL ────────────────────────────────────────────────────────────────────

def cancel_order(exchange, order_id: str, symbol: str) -> bool:
    if config.TRADING_MODE == "PAPER":
        from paper_broker import paper_cancel_order
        return paper_cancel_order(order_id)
    try:
        exchange.cancel_order(order_id, symbol)
        log.info("[execution] Cancelled order %s on %s", order_id, symbol)
        return True
    except ccxt.OrderNotFound:
        log.warning("[execution] Order %s not found", order_id)
        return True
    except Exception as exc:
        log.error("[execution] cancel_order failed for %s: %s", order_id, exc)
        return False


def cancel_all_open_orders(exchange, symbol: str) -> bool:
    if config.TRADING_MODE == "PAPER":
        return True
    try:
        exchange.cancel_all_orders(symbol)
        return True
    except Exception as exc:
        log.error("[execution] cancel_all_orders failed for %s: %s", symbol, exc)
        return False


# ── ORDER STATUS / FILL WAIT ──────────────────────────────────────────────────

def check_order_filled(exchange, order_id: str, symbol: str) -> tuple[bool, float]:
    """Returns (is_filled, avg_fill_price)."""
    if config.TRADING_MODE == "PAPER":
        from paper_broker import paper_check_fill
        from data import fetch_ohlcv
        try:
            df = fetch_ohlcv(exchange, symbol, config.TF_15M, limit=3)
            low   = float(df.iloc[-1]["low"])
            close = float(df.iloc[-1]["close"])
            return paper_check_fill(order_id, symbol, low, close)
        except Exception as exc:
            log.warning("[paper] check_order_filled error: %s", exc)
            return False, 0.0

    try:
        order = exchange.fetch_order(order_id, symbol)
        status = order.get("status", "")
        filled = order.get("filled", 0.0)
        avg    = order.get("average") or order.get("price", 0.0)
        if status == "closed" and filled and filled > 0:
            return True, float(avg or 0.0)
        return False, 0.0
    except Exception as exc:
        log.warning("[execution] check_order_filled error for %s: %s", order_id, exc)
        return False, 0.0


def wait_for_fill(
    exchange,
    order_id: str,
    symbol: str,
    timeout: int = config.ORDER_FILL_TIMEOUT_SECONDS,
    poll_interval: int = 5,
) -> tuple[bool, float]:
    """
    Poll until the order fills or timeout expires.
    In PAPER mode, polls the current candle data to check the fill condition.
    """
    elapsed = 0
    while elapsed < timeout:
        filled, avg_price = check_order_filled(exchange, order_id, symbol)
        if filled:
            log.info("[execution] Order %s filled at %.6f", order_id, avg_price)
            return True, avg_price
        time.sleep(poll_interval)
        elapsed += poll_interval

    log.warning("[execution] Order %s timed out after %ds — cancelling", order_id, timeout)
    cancel_order(exchange, order_id, symbol)
    return False, 0.0
