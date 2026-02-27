"""
data.py
────────
Calculates Price Action Microstructure: Sweeps and Fair Value Gaps (FVGs).
Includes automatic pagination to bypass exchange API limits for deep history.
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
import pandas_ta as ta
import config
from utils import log

def fetch_ohlcv(exchange, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    all_candles = []
    
    # Calculate how far back in time to go (in milliseconds)
    tf_ms = exchange.parse_timeframe(timeframe) * 1000
    since = exchange.milliseconds() - (limit * tf_ms)
    
    # Pagination Loop: Fetch in chunks of 1000 to avoid Exchange API bans
    while len(all_candles) < limit:
        try:
            chunk = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1000)
            if not chunk:
                break # No more data available
            
            all_candles.extend(chunk)
            # Move the 'since' timestamp forward to the last candle fetched + 1 millisecond
            since = chunk[-1][0] + 1
            
            # Respect API rate limits so we don't get banned
            time.sleep(exchange.rateLimit / 1000.0 if exchange.rateLimit else 0.1)
            
        except Exception as e:
            print(f"  [!] API limit or error on {symbol}: {e}")
            break

    # Trim to the exact limit requested
    all_candles = all_candles[-limit:]

    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("datetime", inplace=True)
    df = df.astype(float)
    return df

def build_15m_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates Sweeps, FVGs, and Multi-Timeframe EMAs."""
    d = df.copy()
    
    # 1. Macro Trend (MTFA: 800 EMA on 15m chart equals the 50 EMA on 4H chart)
    d["ema_50"] = ta.ema(d["close"], length=50)   # Keep for AI features
    d["ema_800"] = ta.ema(d["close"], length=800) # THE NEW MACRO TREND FILTER
    
    d["atr_14"] = ta.atr(d["high"], d["low"], d["close"], length=14)
    d["vol_ma"] = ta.sma(d["volume"], length=20)

    # 2. Fractal S&R (Swing Highs / Swing Lows)
    d["swing_low"] = (d["low"] < d["low"].shift(1)) & (d["low"] < d["low"].shift(2)) & \
                     (d["low"] < d["low"].shift(-1)) & (d["low"] < d["low"].shift(-2))
    d["swing_high"] = (d["high"] > d["high"].shift(1)) & (d["high"] > d["high"].shift(2)) & \
                      (d["high"] > d["high"].shift(-1)) & (d["high"] > d["high"].shift(-2))

    # 3. Fair Value Gaps (FVGs)
    d["fvg_bull_top"] = d["low"]  
    d["fvg_bull_bot"] = d["high"].shift(2) 
    d["is_bull_fvg"] = d["fvg_bull_top"] > d["fvg_bull_bot"]

    d["fvg_bear_bot"] = d["high"] 
    d["fvg_bear_top"] = d["low"].shift(2) 
    d["is_bear_fvg"] = d["fvg_bear_bot"] < d["fvg_bear_top"]

    return d

def fetch_tickers(exchange): 
    return exchange.fetch_tickers()

def rank_by_volatility(tickers: dict, n_input: int = 30, n_output: int = 15) -> list[str]:
    rows = []
    for sym, t in tickers.items():
        if not sym.endswith("/USDT"): 
            continue
            
        vol = t.get("quoteVolume")
        hi = t.get("high")
        lo = t.get("low")
        close = t.get("last")
        
        # 1. ERROR FIX: Skip any coin that has missing (None) data from the exchange
        if close is None or hi is None or lo is None or vol is None:
            continue
            
        # 2. Convert to floats to ensure math works safely
        close_price = float(close)
        if close_price <= 0: 
            continue
            
        volatility = (float(hi) - float(lo)) / close_price
        rows.append((sym, float(vol), volatility))
    
    # Sort by Volume first (to get liquid coins)
    rows.sort(key=lambda x: x[1], reverse=True)
    liquid = rows[:n_input]
    
    # Then sort by Volatility (to get the biggest movers)
    liquid.sort(key=lambda x: x[2], reverse=True)
    
    return [r[0] for r in liquid[:n_output]]