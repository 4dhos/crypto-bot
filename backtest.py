"""
backtest.py
───────────
High-speed NumPy backtester with ML Data Collection & Filtering.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pandas_ta as ta
import csv
import os
import config
from data import fetch_ohlcv, fetch_tickers, rank_by_volatility
from execution import build_exchange
from utils import setup_logging
from ml_predictor import predict_trade_success

def _precompute_indicators(df: pd.DataFrame) -> dict:
    d = df.copy()
    d["ema"] = ta.ema(d["close"], length=config.EMA_PERIOD)
    d["atr"] = ta.atr(d["high"], d["low"], d["close"], length=config.ATR_PERIOD)
    d["vol_ma"] = ta.sma(d["volume"], length=config.VOLUME_MA_PERIOD)
    
    dc = ta.donchian(d["high"], d["low"], lower_length=config.DONCHIAN_PERIOD, upper_length=config.DONCHIAN_PERIOD)
    if dc is not None and not dc.empty:
        d["dc_up"] = dc.iloc[:, 2] 
        d["dc_lo"] = dc.iloc[:, 0] 
    else:
        d["dc_up"], d["dc_lo"] = np.nan, np.nan

    bb = ta.bbands(d["close"], length=config.BB_PERIOD, std=config.BB_STD)
    if bb is not None and not bb.empty:
        d["bbw"] = (bb.iloc[:, 0] - bb.iloc[:, 2]) / bb.iloc[:, 1] * 100
        d["bbw_sma"] = ta.sma(d["bbw"], length=config.BBW_SMA_PERIOD)
    else:
        d["bbw"], d["bbw_sma"] = np.nan, np.nan

    d["avwap"] = ta.vwap(d["high"], d["low"], d["close"], d["volume"])
    d.ffill(inplace=True)

    # Add Time features for the AI
    d["hour"] = d.index.hour.values
    d["day_of_week"] = d.index.dayofweek.values

    return {col: d[col].values for col in d.columns}

def run_backtest(n_coins=15, limit=8000, balance=100.0):
    exchange = build_exchange()
    try:
        tickers = fetch_tickers(exchange)
        symbols = rank_by_volatility(tickers, n_input=30, n_output=n_coins)
    except Exception as e:
        print(f"Failed to fetch market data: {e}")
        return

    data_arrays = {}
    print(f"Downloading data for {n_coins} coins...")
    for sym in symbols:
        try:
            df = fetch_ohlcv(exchange, sym, "15m", limit)
            if len(df) > 100:
                data_arrays[sym] = _precompute_indicators(df)
        except Exception:
            continue

    if not data_arrays: return

    equity_curve = [balance]
    open_trades = {}
    trade_history = []
    ml_training_data = []
    
    taker_fee = config.BACKTEST_TAKER_FEE
    slippage = config.BACKTEST_SLIPPAGE

    n_bars = min(len(arr["close"]) for arr in data_arrays.values())
    mode_str = "🧠 ML FILTER ACTIVE" if config.USE_ML_FILTER else "📡 DATA COLLECTION MODE"
    print(f"Running simulation ({mode_str}) for {n_bars} bars (~{n_bars * 15 / 60 / 24:.1f} days)...")

    for bar in range(2, n_bars):
        if balance <= 5: break

        # 1. Manage Open Trades
        for sym in list(open_trades.keys()):
            trade = open_trades[sym]
            arr = data_arrays[sym]
            hi, lo, close = arr["high"][bar], arr["low"][bar], arr["close"][bar]
            
            sl_hit = False
            exit_price = 0.0

            if trade["side"] == "long" and lo <= trade["stop"]:
                sl_hit, exit_price = True, trade["stop"] * (1 - slippage)
            elif trade["side"] == "short" and hi >= trade["stop"]:
                sl_hit, exit_price = True, trade["stop"] * (1 + slippage)

            candles_held = bar - trade["entry_bar"]
            if not sl_hit and candles_held >= config.TIME_STOP_CANDLES:
                sl_hit, exit_price = True, close * (1 - slippage) if trade["side"] == "long" else close * (1 + slippage)

            if sl_hit:
                if trade["side"] == "long":
                    pnl = (exit_price - trade["entry"]) * trade["qty"]
                else:
                    pnl = (trade["entry"] - exit_price) * trade["qty"]
                
                net_pnl = pnl - (exit_price * trade["qty"] * taker_fee)
                balance += net_pnl
                
                # Record the outcome for the AI Data
                is_win = 1 if net_pnl > 0 else 0
                trade_record = trade["features"]
                trade_record["is_win"] = is_win
                ml_training_data.append(trade_record)

                trade_history.append({"sym": sym, "pnl": net_pnl})
                del open_trades[sym]

        # 2. Check Signals
        for sym, arr in data_arrays.items():
            if sym in open_trades or len(open_trades) >= config.MAX_CONCURRENT_POSITIONS: continue

            close, vol = arr["close"][bar], arr["volume"][bar]
            atr, ema, vol_ma = arr["atr"][bar-1], arr["ema"][bar-1], arr["vol_ma"][bar-1]
            bbw, bbw_sma = arr["bbw"][bar], arr["bbw_sma"][bar]
            
            if np.isnan(atr) or atr == 0 or np.isnan(vol_ma) or vol_ma == 0: continue
            
            c_bbw = (bbw > bbw_sma) and (arr["bbw"][bar-1] <= arr["bbw_sma"][bar-1])
            c_vol = (vol / vol_ma) >= config.VOLUME_SPIKE_MULT
            c_grv = (abs(close - ema) / ema) <= config.GRAVITY_PCT

            if not (c_bbw and c_vol and c_grv): continue

            is_long = close > arr["dc_up"][bar-1] and close > arr["avwap"][bar-1]
            is_short = close < arr["dc_lo"][bar-1] and close < arr["avwap"][bar-1]

            if is_long or is_short:
                side = "long" if is_long else "short"
                entry_price = close * (1 + slippage) if side == "long" else close * (1 - slippage)
                stop_loss = entry_price - (1.5 * atr) if side == "long" else entry_price + (1.5 * atr)
                
                features = {
                    "bbw": bbw, "vol_ratio": vol / vol_ma, "ema_dist": abs(close - ema) / ema,
                    "atr_pct": atr / close, "hour": arr["hour"][bar], "day_of_week": arr["day_of_week"][bar]
                }

                # AI FILTER INTERVENTION
                if config.USE_ML_FILTER:
                    prob = predict_trade_success(features)
                    if prob < config.ML_PROB_THRESHOLD:
                        continue # AI rejected the trade

                qty = (balance * 0.08) / abs(entry_price - stop_loss)
                if (qty * entry_price) <= (balance * config.MAX_LEVERAGE):
                    balance -= (entry_price * qty * taker_fee)
                    open_trades[sym] = {"side": side, "entry": entry_price, "stop": stop_loss, "qty": qty, "entry_bar": bar, "features": features}
        
        equity_curve.append(balance)

    print("\n" + "="*50)
    print(f"FINAL BALANCE : ${balance:.2f} ({((balance-100)/100)*100:.2f}%)")
    
    if trade_history:
        wins = [t for t in trade_history if t['pnl'] > 0]
        print(f"TOTAL TRADES  : {len(trade_history)} (Win Rate: {len(wins)/len(trade_history)*100:.1f}%)")
    print("="*50)

    # Save Data for AI
    if not config.USE_ML_FILTER and ml_training_data:
        df = pd.DataFrame(ml_training_data)
        file_exists = os.path.exists("trade_data.csv")
        df.to_csv("trade_data.csv", mode='a', header=not file_exists, index=False)
        print(f"💾 Saved {len(ml_training_data)} trades to trade_data.csv for AI Training.")

if __name__ == "__main__":
    setup_logging()
    # Test for 3 months
    run_backtest(n_coins=15, limit=8000, balance=100.0)