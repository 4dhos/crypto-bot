"""
strategy.py
───────────
Entry trigger evaluation for both LONG and SHORT signals on the 15m chart.

LONG  conditions (all 5 must pass):
  1. Close > 20-period Donchian UPPER band  (upside breakout)
  2. BBW expanding from compression          (squeeze release)
  3. Price within 3% of EMA50               (not overextended up)
  4. Volume > 1.5× 20-period Vol MA         (confirmed breakout)
  5. Price > Anchored VWAP                  (buying pressure)

SHORT conditions (all 5 must pass — mirror image):
  1. Close < 20-period Donchian LOWER band  (downside breakout)
  2. BBW expanding from compression          (same squeeze logic)
  3. Price within 3% of EMA50               (not overextended down)
  4. Volume > 1.5× 20-period Vol MA         (confirmed breakdown)
  5. Price < Anchored VWAP                  (selling pressure)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import config
from data import fetch_ohlcv, build_15m_indicators
from utils import log


# ── RESULT DATACLASS ──────────────────────────────────────────────────────────

@dataclass
class SignalResult:
    fired: bool               = False
    side: str                 = ""        # "long" or "short"
    symbol: str               = ""
    entry_price: float        = 0.0
    atr: float                = 0.0
    initial_stop: float       = 0.0
    ema50: float              = 0.0
    avwap: float              = 0.0
    dc_upper: float           = 0.0
    dc_lower: float           = 0.0
    close: float              = 0.0
    conditions: dict          = field(default_factory=dict)
    fail_reason: str          = ""


# ── SHARED CONDITION CHECKS ───────────────────────────────────────────────────

def _check_bbw_expansion(last, prev) -> tuple[bool, str]:
    for col in ("bbw", "bbw_sma"):
        if col not in last.index:
            return False, f"column '{col}' missing"
    cur_bbw  = last["bbw"];   cur_sma  = last["bbw_sma"]
    prev_bbw = prev.get("bbw", float("nan"))
    prev_sma = prev.get("bbw_sma", float("nan"))
    if any(v != v for v in (cur_bbw, cur_sma, prev_bbw, prev_sma)):
        return False, "BBW or BBW_SMA is NaN"
    if cur_bbw > cur_sma and prev_bbw <= prev_sma:
        return True, f"BBW {cur_bbw:.4f} crossing above SMA {cur_sma:.4f}"
    return False, f"BBW not expanding (cur={cur_bbw:.4f} sma={cur_sma:.4f})"


def _check_gravity(last) -> tuple[bool, str]:
    ema_key = f"ema_{config.EMA_PERIOD}"
    if ema_key not in last.index:
        return False, "ema_50 missing"
    ema = last[ema_key]
    close = last["close"]
    if ema != ema:
        return False, "ema_50 is NaN"
    dist = abs(close - ema) / ema
    if dist <= config.GRAVITY_PCT:
        return True, f"distance {dist:.4%} ≤ {config.GRAVITY_PCT:.4%}"
    return False, f"overextended {dist:.4%} > {config.GRAVITY_PCT:.4%}"


def _check_volume_spike(last) -> tuple[bool, str]:
    vol_key = f"vol_ma_{config.VOLUME_MA_PERIOD}"
    if vol_key not in last.index:
        return False, "vol_ma missing"
    vol = last["volume"]; vol_ma = last[vol_key]
    if vol_ma != vol_ma or vol_ma == 0:
        return False, "vol_ma NaN or zero"
    ratio = vol / vol_ma
    if ratio >= config.VOLUME_SPIKE_MULT:
        return True, f"vol {ratio:.2f}x ≥ {config.VOLUME_SPIKE_MULT}x"
    return False, f"vol {ratio:.2f}x < {config.VOLUME_SPIKE_MULT}x"


# ── LONG CONDITIONS ───────────────────────────────────────────────────────────

def _check_donchian_long(last, prev) -> tuple[bool, str]:
    dc_key = f"dc_upper_{config.DONCHIAN_PERIOD}"
    if dc_key not in last.index:
        return False, "dc_upper missing"
    dc_upper = prev.get(dc_key, float("nan"))
    if dc_upper != dc_upper:
        return False, "dc_upper NaN"
    if last["close"] > dc_upper:
        return True, f"close {last['close']:.4f} > DC_upper {dc_upper:.4f}"
    return False, f"close {last['close']:.4f} ≤ DC_upper {dc_upper:.4f}"


def _check_avwap_long(last) -> tuple[bool, str]:
    if "avwap" not in last.index or last["avwap"] != last["avwap"]:
        return False, "avwap missing or NaN"
    if last["close"] > last["avwap"]:
        return True, f"close {last['close']:.4f} > AVWAP {last['avwap']:.4f}"
    return False, f"close {last['close']:.4f} ≤ AVWAP {last['avwap']:.4f}"


# ── SHORT CONDITIONS ──────────────────────────────────────────────────────────

def _check_donchian_short(last, prev) -> tuple[bool, str]:
    dc_key = f"dc_lower_{config.DONCHIAN_PERIOD}"
    if dc_key not in last.index:
        return False, "dc_lower missing"
    dc_lower = prev.get(dc_key, float("nan"))
    if dc_lower != dc_lower:
        return False, "dc_lower NaN"
    if last["close"] < dc_lower:
        return True, f"close {last['close']:.4f} < DC_lower {dc_lower:.4f}"
    return False, f"close {last['close']:.4f} ≥ DC_lower {dc_lower:.4f}"


def _check_avwap_short(last) -> tuple[bool, str]:
    if "avwap" not in last.index or last["avwap"] != last["avwap"]:
        return False, "avwap missing or NaN"
    if last["close"] < last["avwap"]:
        return True, f"close {last['close']:.4f} < AVWAP {last['avwap']:.4f}"
    return False, f"close {last['close']:.4f} ≥ AVWAP {last['avwap']:.4f}"


# ── MAIN EVALUATORS ───────────────────────────────────────────────────────────

def evaluate_entry(exchange, symbol: str, direction: str = "long") -> SignalResult:
    """
    Fetch 15m data and evaluate entry conditions for the given direction.

    Parameters
    ----------
    exchange  : ccxt exchange instance
    symbol    : e.g. "ETH/USDT"
    direction : "long" or "short"
    """
    result = SignalResult(symbol=symbol, side=direction)
    try:
        df = fetch_ohlcv(exchange, symbol, config.TF_15M, config.OHLCV_LIMIT_15M)
        df = build_15m_indicators(df)
        if len(df) < 3:
            result.fail_reason = "insufficient candle data"
            return result

        last = df.iloc[-1]
        prev = df.iloc[-2]

        atr_key = f"atr_{config.ATR_PERIOD}"
        atr = last.get(atr_key, float("nan"))
        if atr != atr or atr <= 0:
            result.fail_reason = "ATR NaN or zero"
            return result

        # Shared conditions
        c_bbw,  r_bbw  = _check_bbw_expansion(last, prev)
        c_grv,  r_grv  = _check_gravity(last)
        c_vol,  r_vol  = _check_volume_spike(last)

        # Direction-specific conditions
        if direction == "long":
            c_dc,   r_dc   = _check_donchian_long(last, prev)
            c_vwap, r_vwap = _check_avwap_long(last)
            entry_price    = last["close"] * (1 - config.MAKER_OFFSET_PCT)
            initial_stop   = entry_price - (config.ATR_STOP_MULTIPLIER * atr)
        else:  # short
            c_dc,   r_dc   = _check_donchian_short(last, prev)
            c_vwap, r_vwap = _check_avwap_short(last)
            entry_price    = last["close"] * (1 + config.MAKER_OFFSET_PCT)
            initial_stop   = entry_price + (config.ATR_STOP_MULTIPLIER * atr)

        conditions = {
            "donchian": (c_dc,   r_dc),
            "bbw":      (c_bbw,  r_bbw),
            "gravity":  (c_grv,  r_grv),
            "volume":   (c_vol,  r_vol),
            "avwap":    (c_vwap, r_vwap),
        }
        result.conditions = conditions

        all_pass = all(v for v, _ in conditions.values())
        icon = "FIRE" if all_pass else "skip"
        log.info(
            "[strategy] %s %s | DC:%s BBW:%s GRV:%s VOL:%s AVWAP:%s → %s",
            symbol, direction.upper(),
            "✓" if c_dc else "✗", "✓" if c_bbw else "✗",
            "✓" if c_grv else "✗", "✓" if c_vol else "✗",
            "✓" if c_vwap else "✗", icon,
        )

        if not all_pass:
            failed = [k for k, (v, _) in conditions.items() if not v]
            result.fail_reason = f"failed: {failed}"
            return result

        result.fired        = True
        result.entry_price  = entry_price
        result.atr          = float(atr)
        result.initial_stop = initial_stop
        result.ema50        = float(last.get(f"ema_{config.EMA_PERIOD}", 0.0))
        result.avwap        = float(last.get("avwap", 0.0))
        result.dc_upper     = float(last.get(f"dc_upper_{config.DONCHIAN_PERIOD}", 0.0))
        result.dc_lower     = float(last.get(f"dc_lower_{config.DONCHIAN_PERIOD}", 0.0))
        result.close        = float(last["close"])
        return result

    except Exception as exc:
        log.error("[strategy] Error evaluating %s %s: %s", symbol, direction, exc)
        result.fail_reason = str(exc)
        return result