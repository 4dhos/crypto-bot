"""
strategy.py
───────────
Entry trigger evaluation for 15m momentum breakouts, upgraded with ML filtering.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import pandas as pd
import ccxt
import config
from data import fetch_ohlcv, build_15m_indicators
from utils import log
from ml_predictor import predict_trade_success

@dataclass
class SignalResult:
    fired: bool               = False
    side: str                 = ""
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

def evaluate_entry(exchange, symbol: str, direction: str = "long") -> SignalResult:
    result = SignalResult(symbol=symbol, side=direction)
    try:
        df = fetch_ohlcv(exchange, symbol, config.TF_15M, config.OHLCV_LIMIT_15M)
        df = build_15m_indicators(df)
        if len(df) < 3:
            result.fail_reason = "insufficient candle data"
            return result

        last = df.iloc[-1]
        prev = df.iloc[-2]

        atr = last.get(f"atr_{config.ATR_PERIOD}")
        if pd.isna(atr) or atr <= 0:
            result.fail_reason = "ATR NaN or zero"
            return result

        # Feature Extraction for ML
        bbw = last.get("bbw", 0)
        vol = last["volume"]
        vol_ma = last.get(f"vol_ma_{config.VOLUME_MA_PERIOD}", 1)
        ema = last.get(f"ema_{config.EMA_PERIOD}", last["close"])
        
        c_bbw = (bbw > last.get("bbw_sma", 0)) and (prev.get("bbw", 0) <= prev.get("bbw_sma", 0))
        c_vol = (vol / vol_ma) >= config.VOLUME_SPIKE_MULT
        c_grv = (abs(last["close"] - ema) / ema) <= config.GRAVITY_PCT

        if direction == "long":
            dc_up = prev.get(f"dc_upper_{config.DONCHIAN_PERIOD}")
            c_dc = last["close"] > dc_up if not pd.isna(dc_up) else False
            c_vwap = last["close"] > last.get("avwap", float('inf'))
            entry_price = last["close"] * (1 + config.TAKER_SLIPPAGE_PCT)
            initial_stop = entry_price - (config.ATR_STOP_MULTIPLIER * atr)
        else:
            dc_lo = prev.get(f"dc_lower_{config.DONCHIAN_PERIOD}")
            c_dc = last["close"] < dc_lo if not pd.isna(dc_lo) else False
            c_vwap = last["close"] < last.get("avwap", float('-inf'))
            entry_price = last["close"] * (1 - config.TAKER_SLIPPAGE_PCT)
            initial_stop = entry_price + (config.ATR_STOP_MULTIPLIER * atr)

        if not (c_bbw and c_vol and c_grv and c_dc and c_vwap):
            result.fail_reason = "Technical conditions not met"
            return result

        # ── MACHINE LEARNING FILTER ──
        if config.USE_ML_FILTER:
            features = {
                "bbw": float(bbw),
                "vol_ratio": float(vol / vol_ma),
                "ema_dist": float(abs(last["close"] - ema) / ema),
                "atr_pct": float(atr / last["close"]),
                "hour": int(last.name.hour),
                "day_of_week": int(last.name.dayofweek)
            }
            prob_win = predict_trade_success(features)
            
            log.info(f"[ML] {symbol} {direction.upper()} | Win Probability: {prob_win*100:.1f}%")
            
            if prob_win < config.ML_PROB_THRESHOLD:
                result.fail_reason = f"ML Filter Blocked (Probability {prob_win*100:.1f}% < {config.ML_PROB_THRESHOLD*100}%)"
                return result

        result.fired = True
        result.entry_price = entry_price
        result.atr = float(atr)
        result.initial_stop = initial_stop
        return result

    except Exception as exc:
        result.fail_reason = f"Error: {exc}"
        return result