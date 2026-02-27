"""
strategy.py
───────────
Liquidity Sweep + Fair Value Gap (FVG) Strategy with strict 1:3 R:R.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import pandas as pd
import config
from data import fetch_ohlcv, build_15m_indicators
from utils import log

@dataclass
class SignalResult:
    fired: bool = False
    side: str = ""
    symbol: str = ""
    entry_price: float = 0.0
    initial_stop: float = 0.0
    take_profit: float = 0.0
    fail_reason: str = ""

def evaluate_entry(exchange, symbol: str, direction: str = "long") -> SignalResult:
    result = SignalResult(symbol=symbol, side=direction)
    try:
        df = fetch_ohlcv(exchange, symbol, config.TF_15M, 100)
        df = build_15m_indicators(df)
        
        # Look at the last 10 closed candles for a setup
        window = df.iloc[-12:-2] # Avoid the currently forming unclosed candle
        
        if direction == "long":
            # 1. Did we sweep a recent swing low?
            recent_lows = window["low"].min()
            
            # 2. Did a bullish FVG form AFTER the lowest point?
            bull_fvgs = window[window["is_bull_fvg"] == True]
            
            if bull_fvgs.empty:
                result.fail_reason = "No Bullish FVG found"
                return result
                
            last_fvg = bull_fvgs.iloc[-1]
            fvg_top = last_fvg["fvg_bull_top"]
            
            # 3. Entry is exactly at the top of the FVG
            entry_price = fvg_top
            
            # 4. Stop Loss is structurally below the absolute sweep low
            stop_loss = recent_lows * 0.999 # Add a tiny 0.1% buffer below the wick
            
            risk = entry_price - stop_loss
            if risk <= 0: 
                result.fail_reason = "Invalid Risk structure"
                return result
                
            # 5. Fixed 1:3 R:R Take Profit
            take_profit = entry_price + (risk * 3.0)
            
            result.fired = True
            result.entry_price = entry_price
            result.initial_stop = stop_loss
            result.take_profit = take_profit
            return result
            
        else:
            # Bearish FVG Logic
            recent_highs = window["high"].max()
            bear_fvgs = window[window["is_bear_fvg"] == True]
            
            if bear_fvgs.empty:
                result.fail_reason = "No Bearish FVG found"
                return result
                
            last_fvg = bear_fvgs.iloc[-1]
            fvg_bot = last_fvg["fvg_bear_bot"]
            
            entry_price = fvg_bot
            stop_loss = recent_highs * 1.001
            risk = stop_loss - entry_price
            take_profit = entry_price - (risk * 3.0)
            
            result.fired = True
            result.entry_price = entry_price
            result.initial_stop = stop_loss
            result.take_profit = take_profit
            return result

    except Exception as exc:
        result.fail_reason = f"Error: {exc}"
        return result