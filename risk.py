"""
risk.py
───────
Position sizing and risk management calculations.

Sizing formula:
    raw_qty = (balance * BASE_RISK) / (ATR_STOP_MULTIPLIER * ATR)

Half-Kelly cap:
    Max equity at risk on a single trade = MAX_EQUITY_RISK_PCT (20%).
    dollar_risk = qty * stop_distance
    If dollar_risk > max_allowed → scale qty down.

Leverage:
    Capped at MAX_LEVERAGE (5x).  Effective leverage = (qty * price) / balance.
    If effective leverage would exceed the cap, qty is reduced further.
"""

from __future__ import annotations

from dataclasses import dataclass

import config
from utils import log, safe_div


@dataclass
class SizingResult:
    valid: bool
    quantity: float          = 0.0   # base currency units to buy
    leverage: int            = 1
    dollar_risk: float       = 0.0   # USD at risk (stop distance × qty)
    equity_risk_pct: float   = 0.0   # fraction of balance at risk
    notional: float          = 0.0   # total position value in USD
    fail_reason: str         = ""


def calculate_position_size(
    balance: float,
    entry_price: float,
    stop_loss: float,
    atr: float,
    leverage: int | None = None,
) -> SizingResult:
    """
    Calculate a safe position size.

    Parameters
    ----------
    balance     : current total account equity (USDT)
    entry_price : planned limit entry price
    stop_loss   : initial stop-loss price
    atr         : current ATR_14 on the 15m chart
    leverage    : override leverage; defaults to MAX_LEVERAGE

    Returns
    -------
    SizingResult with valid=True and a safe quantity, or valid=False + reason.
    """
    result = SizingResult(valid=False)

    if balance <= 0:
        result.fail_reason = "balance is zero or negative"
        return result

    if entry_price <= 0 or stop_loss <= 0:
        result.fail_reason = "invalid price or stop"
        return result

    if atr <= 0:
        result.fail_reason = "ATR is zero or negative"
        return result

    stop_distance = abs(entry_price - stop_loss)
    if stop_distance == 0:
        result.fail_reason = "stop distance is zero"
        return result

    used_leverage = min(leverage or config.MAX_LEVERAGE, config.MAX_LEVERAGE)

    # ── Raw volatility-based sizing ────────────────────────────────────────────
    # Size so that hitting the stop costs BASE_RISK × balance
    # stop_distance ≈ ATR_STOP_MULTIPLIER × ATR, so:
    #   qty = (balance × BASE_RISK) / stop_distance
    dollar_risk_target = balance * config.BASE_RISK_PER_TRADE
    raw_qty = safe_div(dollar_risk_target, stop_distance, fallback=0.0)

    if raw_qty <= 0:
        result.fail_reason = "computed quantity is zero"
        return result

    # ── Kelly cap ─────────────────────────────────────────────────────────────
    # Maximum dollar risk allowed = balance × MAX_EQUITY_RISK_PCT
    max_dollar_risk = balance * config.MAX_EQUITY_RISK_PCT
    if stop_distance * raw_qty > max_dollar_risk:
        capped_qty = safe_div(max_dollar_risk, stop_distance)
        log.info(
            "[risk] Kelly cap applied: %.6f → %.6f (risk $%.2f > $%.2f)",
            raw_qty, capped_qty, stop_distance * raw_qty, max_dollar_risk,
        )
        raw_qty = capped_qty

    # ── Leverage cap ──────────────────────────────────────────────────────────
    # Notional = qty × entry_price.  Max notional = balance × leverage.
    max_notional = balance * used_leverage
    notional = raw_qty * entry_price

    if notional > max_notional:
        capped_qty = safe_div(max_notional, entry_price)
        log.info(
            "[risk] Leverage cap applied: notional $%.2f > max $%.2f, qty %.6f → %.6f",
            notional, max_notional, raw_qty, capped_qty,
        )
        raw_qty = capped_qty
        notional = raw_qty * entry_price

    if raw_qty <= 0:
        result.fail_reason = "quantity reduced to zero by caps"
        return result

    # ── Warn if below minimum risk threshold ───────────────────────────────────
    actual_risk_pct = safe_div(stop_distance * raw_qty, balance)
    if actual_risk_pct < config.MIN_EQUITY_RISK_PCT:
        log.warning(
            "[risk] Equity risk %.4f%% is below MIN threshold %.4f%% — "
            "trade may be too small to be meaningful",
            actual_risk_pct * 100,
            config.MIN_EQUITY_RISK_PCT * 100,
        )

    result.valid           = True
    result.quantity        = raw_qty
    result.leverage        = used_leverage
    result.dollar_risk     = stop_distance * raw_qty
    result.equity_risk_pct = actual_risk_pct
    result.notional        = notional

    log.info(
        "[risk] Size for entry %.6f | qty=%.6f | notional=$%.2f | "
        "risk=$%.2f (%.2f%%) | lev=%dx",
        entry_price,
        raw_qty,
        notional,
        result.dollar_risk,
        result.equity_risk_pct * 100,
        used_leverage,
    )

    return result


def calculate_pyramid_size(original_quantity: float) -> float:
    """
    Return the secondary order quantity for a pyramid add.
    = PYRAMID_SIZE_RATIO (50%) of the original position size.
    """
    return original_quantity * config.PYRAMID_SIZE_RATIO


def validate_min_notional(quantity: float, price: float, min_usd: float = 5.0) -> bool:
    """
    Quick guard: ensure the order is above the exchange's typical minimum
    notional (default $5).  Returns False if too small.
    """
    notional = quantity * price
    if notional < min_usd:
        log.warning(
            "[risk] Order notional $%.4f is below minimum $%.2f — skipping",
            notional, min_usd,
        )
        return False
    return True
