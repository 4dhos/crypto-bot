"""
backtest.py
───────────
High-speed backtester with Disk Caching and Precision Timers.
"""
from __future__ import annotations
import os
import time
import pickle
import numpy as np
import pandas as pd
import ccxt
from data import fetch_ohlcv, fetch_tickers, rank_by_volatility, build_15m_indicators

def _precompute_fvg_arrays(df: pd.DataFrame) -> dict:
    d = build_15m_indicators(df)
    d["hour"] = d.index.hour.values
    d["day_of_week"] = d.index.dayofweek.values
    d.fillna(0, inplace=True)
    return {col: d[col].values for col in d.columns}

def download_and_prep_data(n_coins=30, limit=35000, verbose=True):
    # THE UPGRADE: The Local Cache File
    cache_file = f"market_data_cache_{n_coins}c_{limit}L.pkl"
    
    if os.path.exists(cache_file):
        if verbose: print(f"⚡ FAST BOOT: Loading data from local cache ({cache_file})... Instant load!")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    if verbose: print("🌐 Connecting to Binance.US to harvest historical data...")
    exchange = ccxt.binanceus({'enableRateLimit': True})
    
    try:
        tickers = fetch_tickers(exchange)
        symbols = rank_by_volatility(tickers, n_input=150, n_output=80)
    except Exception as e:
        if verbose: print(f"❌ Failed to fetch market data: {e}")
        return {}

    raw_dfs = {}
    if verbose: print(f"📥 Searching for top {n_coins} volatile coins...")
    
    for sym in symbols:
        if len(raw_dfs) >= n_coins: break
        try:
            df = fetch_ohlcv(exchange, sym, "15m", limit)
            if len(df) > 1000: 
                raw_dfs[sym] = df
                if verbose: print(f"  ✅ {sym} | Harvested {len(df)} candles")
        except Exception:
            continue

    if not raw_dfs: return {}

    max_len = max(len(df) for df in raw_dfs.values())
    data_arrays = {}
    
    for sym, df in raw_dfs.items():
        arrs = _precompute_fvg_arrays(df)
        pad_size = max_len - len(df)
        
        padded_arrs = {}
        for col, val_array in arrs.items():
            if pad_size > 0:
                padded_arrs[col] = np.pad(val_array, (pad_size, 0), 'constant', constant_values=0)
            else:
                padded_arrs[col] = val_array
        data_arrays[sym] = padded_arrs

    # THE UPGRADE: Save the massive dataset to the hard drive for next time
    if verbose: print(f"💾 Saving arrays to {cache_file} for instant loading next time...")
    with open(cache_file, "wb") as f:
        pickle.dump(data_arrays, f)

    return data_arrays

# ... [run_simulation stays exactly the same] ...

    # THE PAD FIX: Aligning the arrays
    max_len = max(len(df) for df in raw_dfs.values())
    data_arrays = {}
    
    for sym, df in raw_dfs.items():
        arrs = _precompute_fvg_arrays(df)
        pad_size = max_len - len(df)
        padded_arrs = {}
        for col, val_array in arrs.items():
            if pad_size > 0:
                padded_arrs[col] = np.pad(val_array, (pad_size, 0), 'constant', constant_values=0)
            else:
                padded_arrs[col] = val_array
        data_arrays[sym] = padded_arrs

    if verbose: print(f"⏱️ Total API Download Time: {time.time() - dl_start:.1f} seconds.")

    # Save to disk so we never have to wait 20 minutes again
    with open(cache_file, 'wb') as f:
        pickle.dump(data_arrays, f)
    if verbose: print(f"💾 Saved {len(data_arrays)} coins to {cache_file} for instant loading next time.")

    return data_arrays

def run_simulation(data_arrays, balance=100.0, rr_ratio=3.0, risk_pct=0.02, 
                   sweep_lookback=10, fvg_depth=0.0, sl_buffer=0.001, be_trigger=1.5,
                   data_range=(0.0, 1.0), verbose=True, save_csv=False):
    
    if not data_arrays: return balance, 0.0, 0
    
    open_trades = {}
    trade_history = []
    ml_training_data = [] 
    
    n_bars = len(list(data_arrays.values())[0]["close"])
    start_idx = int(n_bars * data_range[0])
    end_idx = int(n_bars * data_range[1])
    start_idx = max(start_idx, sweep_lookback + 850) 
    
    for bar in range(start_idx, end_idx):
        if balance <= 5: break 

        for sym in list(open_trades.keys()):
            trade = open_trades[sym]
            hi, lo = data_arrays[sym]["high"][bar], data_arrays[sym]["low"][bar]
            exit_price, is_closed = 0.0, False

            if trade["side"] == "long":
                if lo <= trade["stop"]: exit_price, is_closed = trade["stop"], True
                elif hi >= trade["tp"]: exit_price, is_closed = trade["tp"], True
                
                if not is_closed and hi >= trade["entry"] + ((trade["entry"] - trade["initial_stop"]) * be_trigger):
                    if trade["stop"] < trade["entry"]: trade["stop"] = trade["entry"] * 1.002 

            else: 
                if hi >= trade["stop"]: exit_price, is_closed = trade["stop"], True
                elif lo <= trade["tp"]: exit_price, is_closed = trade["tp"], True
                
                if not is_closed and lo <= trade["entry"] - ((trade["initial_stop"] - trade["entry"]) * be_trigger):
                    if trade["stop"] > trade["entry"]: trade["stop"] = trade["entry"] * 0.998

            if is_closed:
                pnl = (exit_price - trade["entry"]) * trade["qty"] if trade["side"] == "long" else (trade["entry"] - exit_price) * trade["qty"]
                net_pnl = pnl - (exit_price * trade["qty"] * 0.0006)
                balance += net_pnl
                
                if save_csv: ml_training_data.append({**trade["features"], "is_win": 1 if net_pnl > 0 else 0})
                trade_history.append({"sym": sym, "pnl": net_pnl})
                del open_trades[sym]

        for sym, arr in data_arrays.items():
            if sym in open_trades or len(open_trades) >= 5: continue
            if arr["close"][bar] == 0: continue

            ema_macro = arr["ema_800"][bar-1] + 1e-9 
            close_val = arr["close"][bar-1]
            
            is_killzone = int(arr["hour"][bar]) in [7, 8, 9, 10, 13, 14, 15, 16] 
            vol_spike = arr["volume"][bar-1] > (arr["vol_ma"][bar-1] * 1.5)      
            
            if not (is_killzone and vol_spike): continue
            
            if close_val > ema_macro and arr["is_bull_fvg"][bar-1]:
                recent_low = np.min(arr["low"][bar-sweep_lookback:bar])
                fvg_top, fvg_bot = arr["fvg_bull_top"][bar-1], arr["fvg_bull_bot"][bar-1]
                entry_price = fvg_top - ((fvg_top - fvg_bot) * fvg_depth)
                
                if arr["low"][bar] <= entry_price:
                    stop_loss = recent_low * (1.0 - sl_buffer)
                    risk = entry_price - stop_loss
                    
                    if risk > (entry_price * 0.005):
                        tp = entry_price + (risk * rr_ratio)
                        qty = (balance * risk_pct) / risk
                        
                        features = {
                            "vol_ratio": float(arr["volume"][bar-1] / (arr["vol_ma"][bar-1] + 1e-9)),
                            "ema_dist_15m": float(abs(entry_price - (arr["ema_50"][bar-1] + 1e-9)) / (arr["ema_50"][bar-1] + 1e-9)),
                            "atr_pct": float(arr["atr_14"][bar-1] / entry_price),
                            "hour": int(arr["hour"][bar])
                        }
                        
                        if (qty * entry_price) <= (balance * 10): 
                            balance -= (entry_price * qty * 0.0006)
                            open_trades[sym] = {"side": "long", "entry": entry_price, "initial_stop": stop_loss, "stop": stop_loss, "tp": tp, "qty": qty, "features": features}
            
            elif close_val < ema_macro and arr["is_bear_fvg"][bar-1]:
                recent_high = np.max(arr["high"][bar-sweep_lookback:bar])
                fvg_bot, fvg_top = arr["fvg_bear_bot"][bar-1], arr["fvg_bear_top"][bar-1]
                entry_price = fvg_bot + ((fvg_top - fvg_bot) * fvg_depth)
                
                if arr["high"][bar] >= entry_price:
                    stop_loss = recent_high * (1.0 + sl_buffer)
                    risk = stop_loss - entry_price
                    
                    if risk > (entry_price * 0.005):
                        tp = entry_price - (risk * rr_ratio)
                        qty = (balance * risk_pct) / risk
                        
                        features = {
                            "vol_ratio": float(arr["volume"][bar-1] / (arr["vol_ma"][bar-1] + 1e-9)),
                            "ema_dist_15m": float(abs(entry_price - (arr["ema_50"][bar-1] + 1e-9)) / (arr["ema_50"][bar-1] + 1e-9)),
                            "atr_pct": float(arr["atr_14"][bar-1] / entry_price),
                            "hour": int(arr["hour"][bar])
                        }
                        
                        if (qty * entry_price) <= (balance * 10): 
                            balance -= (entry_price * qty * 0.0006)
                            open_trades[sym] = {"side": "short", "entry": entry_price, "initial_stop": stop_loss, "stop": stop_loss, "tp": tp, "qty": qty, "features": features}

    win_rate = len([t for t in trade_history if t['pnl'] > 0]) / len(trade_history) if trade_history else 0.0
        
    if save_csv and ml_training_data:
        df = pd.DataFrame(ml_training_data)
        df.to_csv("trade_data.csv", mode='a', header=not os.path.exists("trade_data.csv"), index=False)

    return balance, win_rate, len(trade_history)