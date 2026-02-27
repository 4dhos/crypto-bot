"""
scanner.py
──────────
Asset selection pipeline adapted for the FVG architecture.
"""
from __future__ import annotations
import config
from data import fetch_tickers, rank_by_volatility
from utils import log

def scan_candidates(exchange, tickers: dict | None = None) -> list[str]:
    """
    Returns the most volatile candidates for FVG hunting.
    """
    log.info("[scanner] Starting asset scan …")

    if tickers is None:
        try:
            tickers = fetch_tickers(exchange)
        except Exception as exc:
            log.error("[scanner] Failed to fetch tickers: %s", exc)
            return []

    # Get the top candidates based strictly on 24h volatility range
    candidates = rank_by_volatility(tickers, n_input=50, n_output=config.TOP_CANDIDATES)
    log.info(f"[scanner] Final candidates: {candidates}")
    
    return candidates
