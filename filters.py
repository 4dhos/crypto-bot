"""
filters.py
──────────
Macro gate filters for both LONG and SHORT directions.

LONG  gates: BTC bullish regime + broad market strength + sentiment floor
SHORT gates: BTC bearish regime + broad market weakness + sentiment cap

all_filters_pass() returns (passed, reason, direction) where direction
tells the caller whether the current regime favours "long", "short", or
"both" — so the scanner and strategy know which signals to look for.
"""

from __future__ import annotations

import config
from data import (
    fetch_ohlcv,
    build_4h_indicators,
    fetch_fear_greed_index,
    fetch_tickers,
    get_top_symbols_by_volume,
)
from utils import log


# ── SENTIMENT ─────────────────────────────────────────────────────────────────

def check_fear_greed() -> tuple[bool, str, int]:
    """
    Returns (passed, reason, value).
    Halts ALL trading if value < FEAR_GREED_FLOOR (extreme panic).
    """
    value = fetch_fear_greed_index()
    if value == -1:
        return False, "Fear & Greed Index unavailable — halting as precaution", -1
    if value < config.FEAR_GREED_FLOOR:
        return False, f"Extreme Fear — F&G {value} < floor {config.FEAR_GREED_FLOOR}", value
    log.info("[filters] Fear & Greed: %d", value)
    return True, "OK", value


# ── BTC REGIME ────────────────────────────────────────────────────────────────

def check_btc_regime(exchange) -> tuple[str, str]:
    """
    Returns (regime, reason) where regime is "bullish", "bearish", or "neutral".
    Bullish:  price > EMA50 AND ADX > threshold
    Bearish:  price < EMA50 AND ADX > threshold
    Neutral:  ADX too weak (choppy — avoid both directions)
    """
    try:
        df = fetch_ohlcv(exchange, "BTC/USDT", config.TF_4H, config.OHLCV_LIMIT_4H)
        df = build_4h_indicators(df)
        last  = df.iloc[-1]
        close = last["close"]
        ema   = last[f"ema_{config.EMA_PERIOD}"]
        adx   = last[f"adx_{config.ADX_PERIOD}"]

        if adx < config.ADX_THRESHOLD:
            return "neutral", f"ADX {adx:.2f} < {config.ADX_THRESHOLD} — market too choppy"

        if close > ema:
            log.info("[filters] BTC BULLISH (close=%.2f EMA=%.2f ADX=%.2f)", close, ema, adx)
            return "bullish", "OK"
        else:
            log.info("[filters] BTC BEARISH (close=%.2f EMA=%.2f ADX=%.2f)", close, ema, adx)
            return "bearish", "OK"

    except Exception as exc:
        log.error("[filters] BTC regime error: %s", exc)
        return "neutral", f"BTC regime check error: {exc}"


# ── MARKET BREADTH ────────────────────────────────────────────────────────────

def check_market_breadth(exchange, tickers: dict | None = None) -> tuple[str, str]:
    """
    Returns ("strong", reason), ("weak", reason), or ("neutral", reason).
    Strong: ≥ BREADTH_MIN_ABOVE (6) of top 10 above EMA50  → supports longs
    Weak:   ≤ BREADTH_MAX_ABOVE (4) of top 10 above EMA50  → supports shorts
    Neutral: 5/10 — no clear directional edge
    """
    try:
        if tickers is None:
            tickers = fetch_tickers(exchange)

        top_symbols = get_top_symbols_by_volume(tickers, n=config.BREADTH_TOP_N)
        if not top_symbols:
            return "neutral", "Could not fetch top symbols"

        above = 0
        checked = 0
        for sym in top_symbols[:config.BREADTH_TOP_N]:
            try:
                df = fetch_ohlcv(exchange, sym, config.TF_4H, config.OHLCV_LIMIT_4H)
                df = build_4h_indicators(df)
                last = df.iloc[-1]
                if last["close"] > last[f"ema_{config.EMA_PERIOD}"]:
                    above += 1
                checked += 1
            except Exception:
                pass

        if checked == 0:
            return "neutral", "No symbols could be checked"

        log.info("[filters] Breadth: %d/%d above EMA%d", above, checked, config.EMA_PERIOD)

        if above >= config.BREADTH_MIN_ABOVE:
            return "strong", f"{above}/{checked} coins above EMA50"
        elif above <= config.BREADTH_MAX_ABOVE:
            return "weak", f"Only {above}/{checked} coins above EMA50"
        else:
            return "neutral", f"{above}/{checked} coins above EMA50 — no clear edge"

    except Exception as exc:
        log.error("[filters] Breadth error: %s", exc)
        return "neutral", str(exc)


# ── COMBINED GATE ─────────────────────────────────────────────────────────────

def all_filters_pass(
    exchange,
    tickers: dict | None = None,
) -> tuple[bool, str, str]:
    """
    Run all macro filters and return (passed, reason, active_direction).

    active_direction: "long", "short", "both", or "" (halt)

    Logic:
      - Extreme fear (<floor)       → halt everything
      - BTC neutral (weak ADX)      → halt everything
      - BTC bullish + breadth strong → allow longs
      - BTC bearish + breadth weak   → allow shorts
      - Config TRADE_DIRECTION acts as an additional override:
          if set to "LONG",  shorts are never returned even if regime is bearish
          if set to "SHORT", longs are never returned even if regime is bullish
          if set to "BOTH",  direction follows the regime
    """
    # ── 1. Sentiment floor — halts everything ─────────────────────────────────
    fg_ok, fg_reason, fg_value = check_fear_greed()
    if not fg_ok:
        return False, f"[Sentiment] {fg_reason}", ""

    # ── 2. BTC Regime ─────────────────────────────────────────────────────────
    regime, regime_reason = check_btc_regime(exchange)
    if regime == "neutral":
        return False, f"[Regime] {regime_reason}", ""

    # ── 3. Market Breadth ─────────────────────────────────────────────────────
    breadth, breadth_reason = check_market_breadth(exchange, tickers=tickers)

    # ── 4. Resolve active direction ───────────────────────────────────────────
    cfg_dir = config.TRADE_DIRECTION  # "LONG" | "SHORT" | "BOTH"

    want_long  = cfg_dir in ("LONG",  "BOTH")
    want_short = cfg_dir in ("SHORT", "BOTH")

    can_long  = regime == "bullish" and breadth == "strong" and want_long
    # For shorts also require F&G ≤ SHORT_CAP (don't short into euphoria)
    can_short = (
        regime == "bearish"
        and breadth == "weak"
        and want_short
        and fg_value <= config.FEAR_GREED_SHORT_CAP
    )

    if can_long and can_short:
        return True, "OK", "both"
    elif can_long:
        return True, "OK", "long"
    elif can_short:
        return True, "OK", "short"
    else:
        reason = (
            f"[Regime={regime} Breadth={breadth} F&G={fg_value}] "
            f"No valid direction for cfg={cfg_dir}"
        )
        return False, reason, ""