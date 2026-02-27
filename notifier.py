"""
notifier.py
───────────
Telegram push-notification dispatcher.

All sends are handled in a background daemon thread via a queue so that a
Telegram outage or slow network never blocks the trading loop.

Usage
─────
    from notifier import notify
    notify("ENTRY", symbol="ETH/USDT", price=2500.0, size=0.04, sl=2420.0)
"""

import queue
import threading
import time
from typing import Any

import requests

import config
from utils import log, fmt_float


# ── INTERNAL QUEUE ────────────────────────────────────────────────────────────

_queue: queue.Queue = queue.Queue()
_STOP_SENTINEL = object()


# ── MESSAGE FORMATTERS ────────────────────────────────────────────────────────

def _fmt_entry(symbol: str, price: float, size: float, sl: float, leverage: int) -> str:
    return (
        "🟢 *ENTRY FIRED*\n"
        f"Symbol:    `{symbol}`\n"
        f"Entry:     `{fmt_float(price, 6)}`\n"
        f"Size:      `{fmt_float(size, 6)}`\n"
        f"Stop Loss: `{fmt_float(sl, 6)}`\n"
        f"Leverage:  `{leverage}x`"
    )


def _fmt_stop_update(symbol: str, old_sl: float, new_sl: float, reason: str) -> str:
    return (
        "🔁 *STOP UPDATED*\n"
        f"Symbol:  `{symbol}`\n"
        f"Old SL:  `{fmt_float(old_sl, 6)}`\n"
        f"New SL:  `{fmt_float(new_sl, 6)}`\n"
        f"Reason:  {reason}"
    )


def _fmt_pyramid(symbol: str, price: float, add_size: float, new_sl: float) -> str:
    return (
        "📐 *PYRAMID ADD*\n"
        f"Symbol:  `{symbol}`\n"
        f"Add at:  `{fmt_float(price, 6)}`\n"
        f"Add Qty: `{fmt_float(add_size, 6)}`\n"
        f"New SL:  `{fmt_float(new_sl, 6)}`"
    )


def _fmt_exit(symbol: str, entry: float, exit_price: float, pnl_pct: float, reason: str) -> str:
    emoji = "✅" if pnl_pct >= 0 else "🔴"
    return (
        f"{emoji} *POSITION CLOSED*\n"
        f"Symbol: `{symbol}`\n"
        f"Entry:  `{fmt_float(entry, 6)}`\n"
        f"Exit:   `{fmt_float(exit_price, 6)}`\n"
        f"PnL:    `{fmt_float(pnl_pct * 100, 2)}%`\n"
        f"Reason: {reason}"
    )


def _fmt_error(message: str) -> str:
    return (
        "🚨 *CRITICAL ERROR*\n"
        f"```\n{message}\n```"
    )


def _fmt_filter_halt(reason: str) -> str:
    return (
        "⏸️ *MACRO FILTER HALT*\n"
        f"Reason: {reason}"
    )


_FORMATTERS = {
    "ENTRY":        _fmt_entry,
    "STOP_UPDATE":  _fmt_stop_update,
    "PYRAMID":      _fmt_pyramid,
    "EXIT":         _fmt_exit,
    "ERROR":        _fmt_error,
    "FILTER_HALT":  _fmt_filter_halt,
}


# ── SEND FUNCTION ─────────────────────────────────────────────────────────────

def _send(text: str) -> None:
    """Synchronous Telegram send. Called from background thread only."""
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        log.debug("[notifier] Telegram not configured, skipping send.")
        return
    url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": config.TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }
    for attempt in range(3):
        try:
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code == 200:
                return
            log.warning("[notifier] Telegram HTTP %d: %s", resp.status_code, resp.text)
        except Exception as exc:
            log.warning("[notifier] Telegram send attempt %d failed: %s", attempt + 1, exc)
        time.sleep(2 ** attempt)
    log.error("[notifier] All Telegram send attempts exhausted.")


# ── BACKGROUND WORKER ─────────────────────────────────────────────────────────

def _worker() -> None:
    while True:
        item = _queue.get()
        if item is _STOP_SENTINEL:
            break
        try:
            _send(item)
        except Exception as exc:
            log.error("[notifier] Worker error: %s", exc)
        finally:
            _queue.task_done()


_thread = threading.Thread(target=_worker, daemon=True, name="TelegramWorker")
_thread.start()


# ── PUBLIC API ────────────────────────────────────────────────────────────────

def notify(event: str, **kwargs: Any) -> None:
    """
    Queue a Telegram notification.  Non-blocking — returns immediately.

    Parameters
    ----------
    event : str
        One of: "ENTRY", "STOP_UPDATE", "PYRAMID", "EXIT", "ERROR", "FILTER_HALT"
    **kwargs :
        Arguments forwarded to the matching formatter function.
    """
    formatter = _FORMATTERS.get(event.upper())
    if formatter is None:
        log.warning("[notifier] Unknown event type: %s", event)
        return
    try:
        text = formatter(**kwargs)
    except TypeError as exc:
        log.error("[notifier] Formatter error for event '%s': %s", event, exc)
        return
    _queue.put(text)
    log.debug("[notifier] Queued '%s' notification.", event)


def shutdown() -> None:
    """Gracefully drain the queue and stop the worker thread."""
    _queue.put(_STOP_SENTINEL)
    _thread.join(timeout=15)
