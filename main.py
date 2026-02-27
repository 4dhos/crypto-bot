"""
main.py
───────
Entry point for the Velocity-10k Protocol V3.0.

Main loop sequence every MAIN_LOOP_SLEEP_SECONDS:
  1. Manage open trades (stop-loss, time-stop, ratchet, pyramid).
  2. Run macro filters → determines active direction (long/short/both).
  3. Check position capacity.
  4. Scan for candidates.
  5. Evaluate entry trigger for each candidate in the active direction.
  6. Size, place order, persist, alert.
"""

from __future__ import annotations

import signal
import sys
import time

import config
import notifier
import persistence
from execution import build_exchange, fetch_usdt_balance, wait_for_fill, set_leverage
from execution import place_limit_buy, place_limit_sell
from filters import all_filters_pass
from notifier import notify
from persistence import count_open_positions, get_position, upsert_position
from risk import calculate_position_size, validate_min_notional
from scanner import scan_candidates
from strategy import evaluate_entry
from trade_manager import run_trade_management_tick
from utils import log, now_ms

# ── GRACEFUL SHUTDOWN ─────────────────────────────────────────────────────────

_running = True

def _handle_signal(signum, frame):
    global _running
    log.info("Shutdown signal received. Stopping …")
    _running = False

signal.signal(signal.SIGINT,  _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ── STARTUP ───────────────────────────────────────────────────────────────────

def startup() -> None:
    log.info("=" * 60)
    log.info("  Velocity-10k Protocol V3.0 — starting up")
    log.info("  Mode: %s  |  Direction: %s", config.TRADING_MODE, config.TRADE_DIRECTION)
    log.info("=" * 60)
    persistence.initialize_db()
    if config.TRADING_MODE == "PAPER":
        from paper_broker import initialize_paper_db, get_paper_balance
        initialize_paper_db()
        bal = get_paper_balance()
        log.info("[main] Paper balance: $%.2f USDT", bal)
    notify("ERROR", message=(
        f"🚀 V3.0 started\n"
        f"Mode: *{config.TRADING_MODE}*\n"
        f"Direction: *{config.TRADE_DIRECTION}*"
    ))


# ── ENTRY EXECUTION ───────────────────────────────────────────────────────────

def attempt_entry(exchange, symbol: str, balance: float, direction: str) -> None:
    """
    Full entry execution for one candidate symbol in the given direction.
    direction: "long" or "short"
    """
    if get_position(symbol) is not None:
        log.info("[main] Already holding %s — skipping", symbol)
        return

    sig = evaluate_entry(exchange, symbol, direction=direction)
    if not sig.fired:
        log.info("[main] No %s signal on %s: %s", direction, symbol, sig.fail_reason)
        return

    sizing = calculate_position_size(
        balance=balance,
        entry_price=sig.entry_price,
        stop_loss=sig.initial_stop,
        atr=sig.atr,
    )
    if not sizing.valid:
        log.warning("[main] Sizing failed for %s: %s", symbol, sizing.fail_reason)
        return

    if not validate_min_notional(sizing.quantity, sig.entry_price):
        return

    if not set_leverage(exchange, symbol, sizing.leverage):
        log.error("[main] Could not set leverage for %s — aborting", symbol)
        return

    # Place entry order in correct direction
    if direction == "long":
        order_res = place_limit_buy(exchange, symbol, sizing.quantity, sig.entry_price)
    else:
        order_res = place_limit_sell(exchange, symbol, sizing.quantity, sig.entry_price)

    if not order_res.success:
        log.error("[main] Entry order failed for %s: %s", symbol, order_res.error)
        notify("ERROR", message=f"Entry order failed for {symbol}: {order_res.error}")
        return

    filled, avg_fill = wait_for_fill(
        exchange, order_res.order_id, symbol,
        timeout=config.ORDER_FILL_TIMEOUT_SECONDS,
    )
    if not filled:
        log.warning("[main] Entry timed out for %s", symbol)
        return

    actual_entry = avg_fill or sig.entry_price
    if direction == "long":
        actual_stop = actual_entry - (config.ATR_STOP_MULTIPLIER * sig.atr)
    else:
        actual_stop = actual_entry + (config.ATR_STOP_MULTIPLIER * sig.atr)

    upsert_position(
        symbol=symbol,
        entry_price=actual_entry,
        quantity=sizing.quantity,
        stop_loss=actual_stop,
        leverage=sizing.leverage,
        atr_at_entry=sig.atr,
        entry_time_ms=now_ms(),
        pyramid_filled=False,
        candles_elapsed=0,
        side=direction,
    )

    notify("ENTRY", symbol=symbol, price=actual_entry, size=sizing.quantity,
           sl=actual_stop, leverage=sizing.leverage)

    log.info(
        "[main] ✅ ENTERED %s %s | qty=%.6f entry=%.6f sl=%.6f lev=%dx",
        direction.upper(), symbol, sizing.quantity,
        actual_entry, actual_stop, sizing.leverage,
    )


# ── MAIN LOOP ─────────────────────────────────────────────────────────────────

def main() -> None:
    startup()

    try:
        exchange = build_exchange()
    except Exception as exc:
        log.critical("[main] Exchange connection failed: %s", exc)
        notify("ERROR", message=f"Exchange connection failed: {exc}")
        sys.exit(1)

    while _running:
        tick_start = time.time()
        try:
            # Step 1: Manage open positions
            run_trade_management_tick(exchange)

            # Step 2: Capacity check
            if count_open_positions() >= config.MAX_CONCURRENT_POSITIONS:
                log.info("[main] At capacity — skipping scan")
                _sleep(config.MAIN_LOOP_SLEEP_SECONDS, tick_start)
                continue

            # Step 3: Macro filters → get active direction
            passed, reason, active_direction = all_filters_pass(exchange)
            if not passed:
                log.info("[main] Filters FAILED — %s", reason)
                notify("FILTER_HALT", reason=reason)
                _sleep(config.FILTER_FAIL_SLEEP_SECONDS, tick_start)
                continue

            log.info("[main] Filters passed — active direction: %s", active_direction)

            # Step 4: Scan
            candidates = scan_candidates(exchange)
            if not candidates:
                log.info("[main] No candidates this cycle")
                _sleep(config.MAIN_LOOP_SLEEP_SECONDS, tick_start)
                continue

            # Step 5: Balance
            balance = fetch_usdt_balance(exchange)
            if balance <= 0:
                log.error("[main] Zero balance — skipping entries")
                _sleep(config.MAIN_LOOP_SLEEP_SECONDS, tick_start)
                continue

            log.info("[main] Balance: $%.2f USDT", balance)

            # Step 6: Entry attempts
            # Determine which directions to try this cycle
            directions = []
            if active_direction in ("long", "both"):
                directions.append("long")
            if active_direction in ("short", "both"):
                directions.append("short")

            for sym in candidates:
                if count_open_positions() >= config.MAX_CONCURRENT_POSITIONS:
                    break
                for direction in directions:
                    try:
                        attempt_entry(exchange, sym, balance, direction)
                    except Exception as exc:
                        log.error("[main] Entry crash %s %s: %s", direction, sym, exc)
                        notify("ERROR", message=f"Entry crash {direction} {sym}: {exc}")

        except KeyboardInterrupt:
            break
        except Exception as exc:
            log.error("[main] Main loop error: %s", exc, exc_info=True)
            notify("ERROR", message=f"Main loop crash: {exc}")

        _sleep(config.MAIN_LOOP_SLEEP_SECONDS, tick_start)

    log.info("[main] Shutting down …")
    notifier.shutdown()
    log.info("[main] Goodbye.")


def _sleep(target: float, tick_start: float) -> None:
    remaining = max(0.0, target - (time.time() - tick_start))
    if remaining > 0:
        time.sleep(remaining)


if __name__ == "__main__":
    main()