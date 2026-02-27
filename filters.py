"""
filters.py
──────────
Macro gates using Asyncio for speed, and NLP for sentiment tracking.
Adapted for the FVG / SMC Architecture.
"""
from __future__ import annotations
import asyncio
import ccxt.async_support as ccxt_async
import pandas as pd
import config
from data import fetch_tickers, rank_by_volatility
from sentiment_engine import get_market_sentiment
from utils import log

async def _fetch_async_breadth(exchange_id: str, symbol: str) -> bool:
    try:
        exchange = getattr(ccxt_async, exchange_id)({'enableRateLimit': False})
        ohlcv = await exchange.fetch_ohlcv(symbol, config.TF_4H, limit=config.EMA_PERIOD + 10)
        await exchange.close()
        
        if not ohlcv or len(ohlcv) < config.EMA_PERIOD: return False
        
        closes = pd.Series([c[4] for c in ohlcv])
        ema = closes.ewm(span=config.EMA_PERIOD, adjust=False).mean().iloc[-1]
        return closes.iloc[-1] > ema
    except Exception:
        return False

def check_market_breadth(exchange, tickers: dict | None = None) -> tuple[str, str]:
    if tickers is None: tickers = fetch_tickers(exchange)
    
    # Use the new volatility ranker instead of the old volume function
    top_symbols = rank_by_volatility(tickers, n_input=50, n_output=config.BREADTH_TOP_N)
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tasks = [_fetch_async_breadth(exchange.id, sym) for sym in top_symbols]
    results = loop.run_until_complete(asyncio.gather(*tasks))
    loop.close()

    above = sum(results)
    checked = len(results)
    
    if checked == 0:
        return "neutral", "No symbols checked"
        
    if above >= config.BREADTH_MIN_ABOVE: return "strong", f"{above}/{checked} coins > EMA50"
    elif above <= config.BREADTH_MAX_ABOVE: return "weak", f"Only {above}/{checked} coins > EMA50"
    return "neutral", f"{above}/{checked} coins > EMA50"

def all_filters_pass(exchange, tickers: dict | None = None) -> tuple[bool, str, str]:
    breadth, b_reason = check_market_breadth(exchange, tickers)
    
    # NLP Macro Filter
    sentiment = get_market_sentiment()
    
    want_long  = config.TRADE_DIRECTION in ("LONG",  "BOTH")
    want_short = config.TRADE_DIRECTION in ("SHORT", "BOTH")

    can_long = breadth == "strong" and sentiment['signal'] != "bearish" and want_long
    can_short = breadth == "weak" and sentiment['signal'] != "bullish" and want_short

    if can_long and can_short: return True, "OK", "both"
    elif can_long: return True, "OK", "long"
    elif can_short: return True, "OK", "short"
    return False, f"Breadth:{breadth}, NLP:{sentiment['signal']}", ""