"""
risk.py
───────
Dynamic position sizing based on account equity and current market performance.
"""
from __future__ import annotations
from dataclasses import dataclass
import config
from utils import log, safe_div
from learning_engine import get_risk_multiplier  # NEW ML MODULE

@dataclass
class SizingResult:
    valid: bool
    quantity: float          = 0.0
    leverage: int            = 1
    dollar_risk: float       = 0.0
    equity_risk_pct: float   = 0.0
    notional: float          = 0.0
    fail_reason: str         = ""

def calculate_position_size(
    balance: float, entry_price: float, stop_loss: float, atr: float, leverage: int | None = None
) -> SizingResult:
    result = SizingResult(valid=False)
    if balance <= 0 or entry_price <= 0 or atr <= 0:
        result.fail_reason = "Invalid metrics"
        return result

    stop_distance = abs(entry_price - stop_loss)
    if stop_distance == 0:
        return result

    used_leverage = min(leverage or config.MAX_LEVERAGE, config.MAX_LEVERAGE)

    # ── DYNAMIC ASYMMETRIC RISK ────────────────────────────────────────────────
    # If account < $250, risk 8% to grow fast. If > $2000, risk 2% to preserve wealth.
    if balance < 250:
        base_risk = 0.08
    elif balance < 1000:
        base_risk = 0.05
    else:
        base_risk = 0.02
        
    # Apply self-learning multiplier (reduces risk during losing streaks)
    market_health_multiplier = get_risk_multiplier()
    adjusted_risk = base_risk * market_health_multiplier

    dollar_risk_target = balance * adjusted_risk
    raw_qty = safe_div(dollar_risk_target, stop_distance, fallback=0.0)

    # Leverage Cap
    max_notional = balance * used_leverage
    notional = raw_qty * entry_price

    if notional > max_notional:
        raw_qty = safe_div(max_notional, entry_price)

    result.valid = True
    result.quantity = raw_qty
    result.leverage = used_leverage
    result.dollar_risk = stop_distance * raw_qty
    result.equity_risk_pct = adjusted_risk
    
    log.info(f"[risk] Sized {raw_qty:.4f} at {adjusted_risk*100:.1f}% risk (ML Multiplier: {market_health_multiplier}x)")
    return result

def calculate_pyramid_size(original_quantity: float) -> float:
    return original_quantity * config.PYRAMID_SIZE_RATIO

def validate_min_notional(quantity: float, price: float, min_usd: float = 5.0) -> bool:
    return (quantity * price) >= min_usd