"""
trade_manager.py
────────────────
Manages every open trade (long or short) on every tick.

For each open position the manager checks, in order:
  1. Stop-Loss Hit        — price hits SL → market close
  2. Time Stop            — 12 candles elapsed, profit < 0.5 ATR → close
  3. Breakeven Ratchet    — profit ≥ 1.5 ATR → move SL to breakeven ± buffer
  4. Pyramid Trigger      — profit ≥ 1.0 ATR, pyramid not filled → add 50%

All math is direction-aware:
  Long:  profit = current_price - entry_price  (stop below entry)
  Short: profit = entry_price - current_price  (stop above entry)
"""

from __future__ import annotations

import config
from data import fetch_ohlcv
from execution import close_position_market, place_limit_buy, place_limit_sell, wait_for_fill
from notifier import notify
from persistence import (
    close_position, get_open_positions, increment_candle_count,
    mark_pyramid_filled, update_stop_loss,
)
from risk import calculate_pyramid_size, validate_min_notional
from utils import log


def _get_current_price(exchange, symbol: str) -> float:
    try:
        df = fetch_ohlcv(exchange, symbol, config.TF_15M, limit=3)
        return float(df.iloc[-1]["close"])
    except Exception as exc:
        log.error("[tm] Price fetch failed for %s: %s", symbol, exc)
        return 0.0


def _handle_position(exchange, pos: dict) -> None:
    symbol         = pos["symbol"]
    entry_price    = pos["entry_price"]
    quantity       = pos["quantity"]
    stop_loss      = pos["stop_loss"]
    leverage       = pos["leverage"]
    atr            = pos["atr_at_entry"]
    pyramid_filled = pos["pyramid_filled"]
    side           = pos.get("side", "long")   # "long" or "short"

    current_price = _get_current_price(exchange, symbol)
    if current_price <= 0:
        return

    # Direction-aware profit calculation
    if side == "long":
        profit_in_price = current_price - entry_price
        sl_hit          = current_price <= stop_loss
    else:  # short
        profit_in_price = entry_price - current_price
        sl_hit          = current_price >= stop_loss

    profit_in_atr = profit_in_price / atr if atr > 0 else 0

    log.info(
        "[tm] %s %s | price=%.6f entry=%.6f sl=%.6f profit=%.4f ATR",
        side.upper(), symbol, current_price, entry_price, stop_loss, profit_in_atr,
    )

    # ── 1. STOP-LOSS ──────────────────────────────────────────────────────────
    if sl_hit:
        log.info("[tm] STOP-LOSS hit on %s %s at %.6f", side, symbol, current_price)
        res = close_position_market(
            exchange, symbol, quantity, reason="stop_loss", entry_price=entry_price
        )
        if res.success:
            pnl_pct = profit_in_price / entry_price
            notify("EXIT", symbol=symbol, entry=entry_price,
                   exit_price=current_price, pnl_pct=pnl_pct, reason="Stop-Loss")
            close_position(symbol)
        else:
            notify("ERROR", message=f"Stop-loss close FAILED for {symbol}: {res.error}")
        return

    # ── 2. TIME STOP ──────────────────────────────────────────────────────────
    candle_count = increment_candle_count(symbol)
    if candle_count >= config.TIME_STOP_CANDLES:
        if profit_in_atr < config.ATR_TIME_STOP_THRESHOLD:
            log.info("[tm] TIME STOP %s after %d candles (profit %.4f ATR)",
                     symbol, candle_count, profit_in_atr)
            res = close_position_market(
                exchange, symbol, quantity, reason="time_stop", entry_price=entry_price
            )
            if res.success:
                pnl_pct = profit_in_price / entry_price
                notify("EXIT", symbol=symbol, entry=entry_price,
                       exit_price=current_price, pnl_pct=pnl_pct,
                       reason=f"Time Stop ({candle_count} candles)")
                close_position(symbol)
            else:
                notify("ERROR", message=f"Time-stop close FAILED for {symbol}: {res.error}")
            return

    # ── 3. BREAKEVEN RATCHET ──────────────────────────────────────────────────
    if profit_in_atr >= config.ATR_BREAKEVEN_TRIGGER:
        if side == "long":
            new_sl = entry_price * (1 + config.BREAKEVEN_FEE_BUFFER_PCT)
            better = new_sl > stop_loss
        else:
            new_sl = entry_price * (1 - config.BREAKEVEN_FEE_BUFFER_PCT)
            better = new_sl < stop_loss

        if better:
            update_stop_loss(symbol, new_sl)
            notify("STOP_UPDATE", symbol=symbol, old_sl=stop_loss,
                   new_sl=new_sl, reason=f"Breakeven Ratchet (+{config.ATR_BREAKEVEN_TRIGGER} ATR)")
            log.info("[tm] Breakeven ratchet %s | %.6f → %.6f", symbol, stop_loss, new_sl)
            stop_loss = new_sl

    # ── 4. PYRAMID ────────────────────────────────────────────────────────────
    if profit_in_atr >= config.ATR_PYRAMID_TRIGGER and not pyramid_filled:
        log.info("[tm] PYRAMID trigger %s %s at %.6f", side, symbol, current_price)

        # Move SL to breakeven first
        if side == "long":
            new_sl_be = entry_price * (1 + config.BREAKEVEN_FEE_BUFFER_PCT)
            better_be = new_sl_be > stop_loss
        else:
            new_sl_be = entry_price * (1 - config.BREAKEVEN_FEE_BUFFER_PCT)
            better_be = new_sl_be < stop_loss

        if better_be:
            update_stop_loss(symbol, new_sl_be)
            notify("STOP_UPDATE", symbol=symbol, old_sl=stop_loss,
                   new_sl=new_sl_be, reason="Breakeven before Pyramid")

        add_qty = calculate_pyramid_size(quantity)
        if not validate_min_notional(add_qty, current_price):
            mark_pyramid_filled(symbol)
            return

        # Place add order in the right direction
        if side == "long":
            add_price = current_price * (1 - config.MAKER_OFFSET_PCT)
            pyr_res   = place_limit_buy(exchange, symbol, add_qty, add_price)
        else:
            add_price = current_price * (1 + config.MAKER_OFFSET_PCT)
            pyr_res   = place_limit_sell(exchange, symbol, add_qty, add_price)

        if not pyr_res.success:
            notify("ERROR", message=f"Pyramid order failed for {symbol}: {pyr_res.error}")
            return

        filled, avg_fill = wait_for_fill(
            exchange, pyr_res.order_id, symbol, timeout=config.ORDER_FILL_TIMEOUT_SECONDS
        )
        if filled:
            mark_pyramid_filled(symbol)
            notify("PYRAMID", symbol=symbol, price=avg_fill or add_price,
                   add_size=add_qty, new_sl=new_sl_be)
        else:
            log.warning("[tm] Pyramid timed out for %s", symbol)
            mark_pyramid_filled(symbol)


def run_trade_management_tick(exchange) -> None:
    """Called once per main loop tick. Manages all open positions."""
    positions = get_open_positions()
    if not positions:
        log.debug("[tm] No open positions.")
        return
    log.info("[tm] Managing %d position(s) …", len(positions))
    for pos in positions:
        try:
            _handle_position(exchange, pos)
        except Exception as exc:
            sym = pos.get("symbol", "UNKNOWN")
            log.error("[tm] Error managing %s: %s", sym, exc)
            notify("ERROR", message=f"Trade manager crash on {sym}: {exc}")