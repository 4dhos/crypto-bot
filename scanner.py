"""
scanner.py
──────────
Asset selection pipeline.

Steps:
  1. Fetch top UNIVERSE_SIZE coins by 24h volume (excluding stablecoins).
  2. Reject any with funding rate > FUNDING_RATE_CAP.
  3. Rank remaining coins by RS Slope (coin/BTC on 1H, 24 bars).
  4. Reject any that are too correlated (> 0.85) with existing open positions.
  5. Return up to TOP_CANDIDATES symbols.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from data import (
    fetch_ohlcv,
    fetch_funding_rate,
    fetch_tickers,
    get_top_symbols_by_volume,
    compute_rs_slope,
)
from persistence import get_open_positions
from utils import log


# ── CORRELATION SHIELD ────────────────────────────────────────────────────────

def _open_position_symbols() -> list[str]:
    return [p["symbol"] for p in get_open_positions()]


def _is_too_correlated(
    exchange,
    candidate: str,
    open_symbols: list[str],
) -> bool:
    """
    Fetch 1H closes for `candidate` and each open position.
    If any pairwise Pearson correlation exceeds CORRELATION_THRESHOLD, reject.
    """
    if not open_symbols:
        return False

    try:
        cand_df = fetch_ohlcv(exchange, candidate, config.TF_1H, config.OHLCV_LIMIT_1H)
        cand_close = cand_df["close"].rename(candidate)
    except Exception as exc:
        log.warning("[scanner] Could not fetch 1H data for %s: %s", candidate, exc)
        return False

    for sym in open_symbols:
        if sym == candidate:
            return True  # already holding this, skip
        try:
            pos_df = fetch_ohlcv(exchange, sym, config.TF_1H, config.OHLCV_LIMIT_1H)
            pos_close = pos_df["close"].rename(sym)

            merged = pd.concat([cand_close, pos_close], axis=1).dropna()
            if len(merged) < 10:
                continue

            corr = merged.corr().iloc[0, 1]
            log.debug(
                "[scanner] Correlation %s vs %s = %.4f", candidate, sym, corr
            )
            if corr > config.CORRELATION_THRESHOLD:
                log.info(
                    "[scanner] %s rejected — correlation %.4f with %s > %.2f",
                    candidate, corr, sym, config.CORRELATION_THRESHOLD,
                )
                return True

        except Exception as exc:
            log.warning("[scanner] Correlation check error %s vs %s: %s", candidate, sym, exc)

    return False


# ── RS SLOPE RANKING ──────────────────────────────────────────────────────────

def _rank_by_rs_slope(
    exchange,
    symbols: list[str],
    btc_df: pd.DataFrame,
) -> list[tuple[str, float]]:
    """
    Return [(symbol, slope), ...] sorted by slope descending (strongest first).
    """
    ranked = []
    for sym in symbols:
        slope = compute_rs_slope(exchange, sym, btc_df, lookback=config.RS_SLOPE_LOOKBACK)
        if slope > -900:   # filter out error sentinel
            ranked.append((sym, slope))
            log.debug("[scanner] RS slope %s: %.6f", sym, slope)

    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


# ── MAIN SCANNER ──────────────────────────────────────────────────────────────

def scan_candidates(exchange, tickers: dict | None = None) -> list[str]:
    """
    Run the full asset selection pipeline and return up to TOP_CANDIDATES
    symbols ready for the entry trigger check.

    Parameters
    ----------
    exchange : ccxt exchange instance
    tickers  : pre-fetched ticker dict (avoids duplicate API call)

    Returns
    -------
    list[str]  ordered by RS slope strength, best first
    """
    log.info("[scanner] Starting asset scan …")

    # ── 1. Universe ──────────────────────────────────────────────────────────
    if tickers is None:
        try:
            tickers = fetch_tickers(exchange)
        except Exception as exc:
            log.error("[scanner] Failed to fetch tickers: %s", exc)
            return []

    universe = get_top_symbols_by_volume(tickers, n=config.UNIVERSE_SIZE)
    log.info("[scanner] Universe: %d symbols", len(universe))

    if not universe:
        return []

    # ── 2. Funding Rate Filter ────────────────────────────────────────────────
    eligible: list[str] = []
    for sym in universe:
        rate = fetch_funding_rate(exchange, sym)
        if rate > config.FUNDING_RATE_CAP:
            log.info(
                "[scanner] %s rejected — funding rate %.5f > cap %.5f",
                sym, rate, config.FUNDING_RATE_CAP,
            )
            continue
        eligible.append(sym)

    log.info("[scanner] After funding filter: %d symbols", len(eligible))

    if not eligible:
        return []

    # ── 3. BTC 1H data for RS slope baseline ─────────────────────────────────
    try:
        btc_df = fetch_ohlcv(exchange, "BTC/USDT", config.TF_1H, config.OHLCV_LIMIT_1H)
    except Exception as exc:
        log.error("[scanner] Cannot fetch BTC 1H data for RS baseline: %s", exc)
        return []

    # Remove BTC itself from the candidate list
    eligible = [s for s in eligible if not s.startswith("BTC/")]

    # ── 4. RS Slope Ranking ───────────────────────────────────────────────────
    ranked = _rank_by_rs_slope(exchange, eligible, btc_df)
    log.info("[scanner] Top RS slopes: %s", ranked[:5])

    # ── 5. Correlation Shield ─────────────────────────────────────────────────
    open_syms = _open_position_symbols()
    candidates: list[str] = []

    for sym, slope in ranked:
        if len(candidates) >= config.TOP_CANDIDATES:
            break
        if _is_too_correlated(exchange, sym, open_syms):
            continue
        candidates.append(sym)
        log.info("[scanner] Candidate accepted: %s (RS slope=%.6f)", sym, slope)

    log.info("[scanner] Final candidates: %s", candidates)
    return candidates
