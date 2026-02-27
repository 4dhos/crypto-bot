"""
backtest.py
───────────
Multi-asset walk-forward backtester for Velocity-10k V3.0.

Key improvements over v1:
  1. VOLATILITY FILTERING — coins are ranked by 24h price range (high-low)/close,
     not just volume. We only trade the coins that are actually moving.
  2. MINIMUM HISTORY GUARD — any coin with fewer than MIN_CANDLE_COVERAGE (90%)
     of the requested bars is dropped before simulation. One new/thin coin can
     no longer collapse the entire backtest window.
  3. PAGINATION — fetches beyond the 1000-bar API cap automatically. Default
     is now 17,500 bars (~6 months of 15m data).
  4. SHARED TIMELINE — all coins are trimmed to the same number of most-recent
     bars after the min-coverage filter is applied.

Usage:
    python backtest.py                       # 50 volatile coins, ~6 months
    python backtest.py --limit 35000         # ~1 year
    python backtest.py --direction LONG      # longs only
    python backtest.py --coins 30            # narrower universe
    python backtest.py --balance 500         # different starting equity
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import pandas_ta as ta

import config
from data import fetch_ohlcv, fetch_tickers, rank_by_volatility
from execution import build_exchange
from utils import log, setup_logging


# ── CONSTANTS ─────────────────────────────────────────────────────────────────

# A coin must have at least this fraction of requested bars or it is dropped.
MIN_CANDLE_COVERAGE = 0.90

# Fetch this many coins by volume before volatility-ranking to n_coins.
# e.g. if n_coins=50 we pull top-150 by volume then keep the 50 most volatile.
VOLUME_POOL_MULTIPLIER = 3


# ── TRADE RECORD ──────────────────────────────────────────────────────────────

@dataclass
class BacktestTrade:
    symbol: str
    side: str
    entry_bar: int
    entry_price: float
    exit_bar: int         = 0
    exit_price: float     = 0.0
    quantity: float       = 0.0
    stop_loss: float      = 0.0
    atr: float            = 0.0
    rs_slope: float       = 0.0
    volatility_rank: float = 0.0   # 24h range % at time of selection
    pnl_pct: float        = 0.0
    pnl_usd: float        = 0.0
    exit_reason: str      = ""
    pyramid_added: bool   = False
    candles_held: int     = 0
    entry_balance: float  = 0.0
    exit_balance: float   = 0.0


# ── DATA LOADING ──────────────────────────────────────────────────────────────

def _load_universe(
    exchange,
    n_coins: int,
    limit: int,
) -> dict[str, pd.DataFrame]:
    """
    1. Fetch tickers and rank top (n_coins × VOLUME_POOL_MULTIPLIER) by volume.
    2. Re-rank by 24h volatility, keep top n_coins.
    3. Download `limit` × 15m candles for each selected coin (paginated).
    4. Drop any coin with fewer than MIN_CANDLE_COVERAGE × limit bars.
    5. Trim all survivors to the same length (most recent bars).
    """
    pool_size = n_coins * VOLUME_POOL_MULTIPLIER

    print(f"\n[backtest] Fetching tickers to rank universe …")
    try:
        tickers = fetch_tickers(exchange)
    except Exception as exc:
        print(f"[backtest] ERROR: {exc}")
        return {}

    # Volatility-first ranking
    symbols = rank_by_volatility(tickers, n_input=pool_size, n_output=n_coins)
    print(f"[backtest] Selected {len(symbols)} most volatile coins from "
          f"top-{pool_size} by volume.")
    print(f"[backtest] Top coins: {[s.split('/')[0] for s in symbols[:10]]} …")
    print(f"\n[backtest] Downloading {limit:,} × 15m candles per coin "
          f"(~{limit*15/60/24:.0f} days) …\n")

    min_bars = int(limit * MIN_CANDLE_COVERAGE)
    data: dict[str, pd.DataFrame] = {}

    for i, sym in enumerate(symbols):
        try:
            df = fetch_ohlcv(exchange, sym, "15m", limit)
            df = df.reset_index(drop=False)
            if "datetime" in df.columns:
                df = df.rename(columns={"datetime": "ts"})

            n = len(df)
            if n < min_bars:
                print(f"  [{i+1:2d}/{len(symbols)}] ✗ {sym:25s} "
                      f"only {n} bars (need {min_bars}) — SKIPPED")
                continue

            data[sym] = df
            print(f"  [{i+1:2d}/{len(symbols)}] ✓ {sym:25s} {n:,} bars")

        except Exception as exc:
            print(f"  [{i+1:2d}/{len(symbols)}] ✗ {sym:25s} {exc}")
        time.sleep(0.15)

    if not data:
        print("[backtest] No coins passed the minimum history filter.")
        return {}

    # Trim all coins to the same length (shortest survivor)
    min_len = min(len(df) for df in data.values())
    for sym in data:
        data[sym] = data[sym].iloc[-min_len:].reset_index(drop=True)

    print(f"\n[backtest] {len(data)} coins loaded, each trimmed to "
          f"{min_len:,} bars (~{min_len*15/60/24:.0f} days).\n")
    return data


# ── INDICATOR PRECOMPUTATION ──────────────────────────────────────────────────

def _precompute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorised indicator computation on the full DataFrame.
    Signal logic uses bar i-1 for Donchian to prevent lookahead bias.
    """
    d = df.copy()

    d[f"ema_{config.EMA_PERIOD}"]          = ta.ema(d["close"], length=config.EMA_PERIOD)
    d[f"atr_{config.ATR_PERIOD}"]          = ta.atr(d["high"], d["low"], d["close"],
                                                      length=config.ATR_PERIOD)
    d[f"vol_ma_{config.VOLUME_MA_PERIOD}"] = ta.sma(d["volume"],
                                                      length=config.VOLUME_MA_PERIOD)

    # Donchian channels
    dc = ta.donchian(d["high"], d["low"],
                     lower_length=config.DONCHIAN_PERIOD,
                     upper_length=config.DONCHIAN_PERIOD)
    if dc is not None and not dc.empty:
        uc = [c for c in dc.columns if "DCU" in c or "UPPER" in c.upper()]
        lc = [c for c in dc.columns if "DCL" in c or "LOWER" in c.upper()]
        if uc: d[f"dc_upper_{config.DONCHIAN_PERIOD}"] = dc[uc[0]]
        if lc: d[f"dc_lower_{config.DONCHIAN_PERIOD}"] = dc[lc[0]]

    # Bollinger Band width
    bb = ta.bbands(d["close"], length=config.BB_PERIOD, std=config.BB_STD)
    if bb is not None and not bb.empty:
        bbu = [c for c in bb.columns if c.startswith("BBU")]
        bbl = [c for c in bb.columns if c.startswith("BBL")]
        bbm = [c for c in bb.columns if c.startswith("BBM")]
        bbw = [c for c in bb.columns if c.startswith("BBB")]
        if bbu and bbl and bbm:
            d["bbw"] = bb[bbw[0]] if bbw else \
                       (bb[bbu[0]] - bb[bbl[0]]) / bb[bbm[0]] * 100
    if "bbw" in d.columns:
        d["bbw_sma"] = ta.sma(d["bbw"], length=config.BBW_SMA_PERIOD)

    # Rolling Anchored VWAP (reanchor every 96 bars = 24 hours)
    avwap = [np.nan] * len(d)
    anchor = 0
    for i in range(len(d)):
        if i - anchor >= 96 or i == 0:
            anchor = i
        seg     = d.iloc[anchor: i + 1]
        tp      = (seg["high"] + seg["low"] + seg["close"]) / 3
        cum_vol = seg["volume"].sum()
        avwap[i] = (tp * seg["volume"]).sum() / cum_vol if cum_vol > 0 else np.nan
    d["avwap"] = avwap

    # Momentum slope (normalised close over RS_SLOPE_LOOKBACK bars)
    window = config.RS_SLOPE_LOOKBACK
    slopes = [0.0] * len(d)
    closes = d["close"].values
    for i in range(window, len(d)):
        seg = closes[i - window: i]
        if seg[0] > 0:
            x = np.arange(len(seg))
            slopes[i] = np.polyfit(x, seg / seg[0], 1)[0]
    d["rs_slope"] = slopes

    return d


# ── SIGNAL EVALUATION ─────────────────────────────────────────────────────────

def _check_signal(
    d: pd.DataFrame, i: int, direction: str
) -> tuple[bool, float, float, float]:
    """
    Returns (fired, entry_price, atr, momentum_slope).
    Uses bar i-1 Donchian to avoid look-ahead.
    """
    if i < 2:
        return False, 0.0, 0.0, 0.0

    def g(col):
        if col not in d.columns: return None
        v = d[col].iat[i]
        return float(v) if pd.notna(v) else None

    def gp(col):
        if col not in d.columns: return None
        v = d[col].iat[i - 1]
        return float(v) if pd.notna(v) else None

    close  = g("close");    volume = g("volume")
    ema    = g(f"ema_{config.EMA_PERIOD}")
    atr    = g(f"atr_{config.ATR_PERIOD}")
    vol_ma = g(f"vol_ma_{config.VOLUME_MA_PERIOD}")
    bbw    = g("bbw");      bbw_sma = g("bbw_sma")
    p_bbw  = gp("bbw");     p_bsma  = gp("bbw_sma")
    avwap  = g("avwap");    slope   = g("rs_slope") or 0.0
    dc_up  = gp(f"dc_upper_{config.DONCHIAN_PERIOD}")
    dc_lo  = gp(f"dc_lower_{config.DONCHIAN_PERIOD}")

    if any(v is None for v in [close, volume, ema, atr, vol_ma,
                                bbw, bbw_sma, p_bbw, p_bsma, avwap]):
        return False, 0.0, 0.0, 0.0
    if atr <= 0 or vol_ma <= 0:
        return False, 0.0, 0.0, 0.0

    # Shared conditions
    if not (bbw > bbw_sma and p_bbw <= p_bsma):   return False, 0.0, 0.0, 0.0
    if abs(close - ema) / ema > config.GRAVITY_PCT: return False, 0.0, 0.0, 0.0
    if (volume / vol_ma) < config.VOLUME_SPIKE_MULT: return False, 0.0, 0.0, 0.0

    if direction == "long":
        if dc_up is None or not (close > dc_up and close > avwap):
            return False, 0.0, 0.0, 0.0
        entry = close * (1 - config.MAKER_OFFSET_PCT)
        return True, entry, atr, slope

    else:  # short
        if dc_lo is None or not (close < dc_lo and close < avwap):
            return False, 0.0, 0.0, 0.0
        entry = close * (1 + config.MAKER_OFFSET_PCT)
        return True, entry, atr, -slope   # negate: highest = strongest downtrend

    return False, 0.0, 0.0, 0.0


# ── POSITION SIZING ───────────────────────────────────────────────────────────

def _calc_size(balance: float, entry: float, stop: float) -> float:
    stop_dist = abs(entry - stop)
    if stop_dist == 0: return 0.0
    qty = (balance * config.BASE_RISK_PER_TRADE) / stop_dist
    if stop_dist * qty > balance * config.MAX_EQUITY_RISK_PCT:
        qty = (balance * config.MAX_EQUITY_RISK_PCT) / stop_dist
    if qty * entry > balance * config.MAX_LEVERAGE:
        qty = (balance * config.MAX_LEVERAGE) / entry
    return max(qty, 0.0)


# ── METRICS ───────────────────────────────────────────────────────────────────

def _compute_metrics(trades: list[BacktestTrade], equity: list[float]) -> dict:
    if not trades:
        return {}

    pnls   = [t.pnl_usd for t in trades]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gp     = sum(wins)
    gl     = abs(sum(losses))

    peak = equity[0]; max_dd = 0.0
    for eq in equity:
        if eq > peak: peak = eq
        dd = (peak - eq) / peak if peak > 0 else 0
        if dd > max_dd: max_dd = dd

    rets = [(equity[i] - equity[i-1]) / equity[i-1]
            for i in range(1, len(equity)) if equity[i-1] > 0]
    sharpe = 0.0
    if len(rets) > 1 and np.std(rets) > 0:
        sharpe = (np.mean(rets) / np.std(rets)) * math.sqrt(35040)

    by_sym = {}
    for t in trades:
        by_sym.setdefault(t.symbol, []).append(t.pnl_usd)

    top5 = sorted(by_sym.items(), key=lambda x: sum(x[1]), reverse=True)[:5]
    bot5 = sorted(by_sym.items(), key=lambda x: sum(x[1]))[:5]

    return {
        "total_trades":     len(trades),
        "long_trades":      sum(1 for t in trades if t.side == "long"),
        "short_trades":     sum(1 for t in trades if t.side == "short"),
        "win_rate":         len(wins) / len(pnls),
        "profit_factor":    gp / gl if gl > 0 else float("inf"),
        "total_return":     (equity[-1] - equity[0]) / equity[0],
        "start_balance":    equity[0],
        "end_balance":      equity[-1],
        "gross_profit":     gp,
        "gross_loss":       gl,
        "avg_win":          sum(wins)   / len(wins)   if wins   else 0,
        "avg_loss":         sum(losses) / len(losses) if losses else 0,
        "max_drawdown":     max_dd,
        "sharpe":           sharpe,
        "avg_hold_candles": sum(t.candles_held for t in trades) / len(trades),
        "unique_symbols":   len(by_sym),
        "top5_symbols":     top5,
        "bot5_symbols":     bot5,
        "exit_reasons":     {r: sum(1 for t in trades if t.exit_reason == r)
                             for r in set(t.exit_reason for t in trades)},
    }


# ── REPORT ────────────────────────────────────────────────────────────────────

def _print_report(m: dict, n_coins: int, direction: str, n_bars: int) -> None:
    if not m:
        print("\n  No trades generated.\n")
        return
    days = n_bars * 15 / 60 / 24
    print()
    print("=" * 64)
    print("  VELOCITY-10K V3.0  —  MULTI-ASSET BACKTEST")
    print(f"  Universe : {n_coins} most volatile coins")
    print(f"  Direction: {direction}")
    print(f"  Period   : {n_bars:,} × 15m bars  (~{days:.0f} days / {days/30:.1f} months)")
    print(f"  Max slots: {config.MAX_CONCURRENT_POSITIONS}  |  "
          f"Risk/trade: {config.BASE_RISK_PER_TRADE*100:.0f}%")
    print("=" * 64)
    print(f"  Starting Balance  : ${m['start_balance']:.2f}")
    print(f"  Ending Balance    : ${m['end_balance']:.2f}")
    print(f"  Total Return      : {m['total_return']*100:+.2f}%")
    print(f"  Max Drawdown      : {m['max_drawdown']*100:.2f}%")
    print(f"  Sharpe Ratio      : {m['sharpe']:.3f}")
    print("-" * 64)
    print(f"  Total Trades      : {m['total_trades']}  across {m['unique_symbols']} coins")
    print(f"    Long            : {m['long_trades']}")
    print(f"    Short           : {m['short_trades']}")
    print(f"  Win Rate          : {m['win_rate']*100:.1f}%")
    print(f"  Profit Factor     : {m['profit_factor']:.2f}")
    print(f"  Avg Win           : ${m['avg_win']:.4f}")
    print(f"  Avg Loss          : ${m['avg_loss']:.4f}")
    print(f"  Avg Hold          : {m['avg_hold_candles']:.1f} bars  "
          f"({m['avg_hold_candles']*15/60:.1f}h)")
    print("-" * 64)
    print("  Exit Breakdown:")
    for reason, count in sorted(m["exit_reasons"].items(), key=lambda x: -x[1]):
        print(f"    {reason:22s}: {count}")
    print("-" * 64)
    print("  Top 5 Coins by PnL:")
    for sym, pnls in m["top5_symbols"]:
        base = sym.split("/")[0]
        print(f"    {base:12s}  ${sum(pnls):+8.2f}  ({len(pnls)} trades)")
    print("  Bottom 5 Coins by PnL:")
    for sym, pnls in m["bot5_symbols"]:
        base = sym.split("/")[0]
        print(f"    {base:12s}  ${sum(pnls):+8.2f}  ({len(pnls)} trades)")
    print("=" * 64)
    print()


# ── MAIN ENGINE ───────────────────────────────────────────────────────────────

def run_backtest(
    n_coins:   int   = None,
    direction: str   = None,
    limit:     int   = None,
    balance:   float = None,
) -> dict:
    n_coins   = n_coins   or config.BACKTEST_UNIVERSE_SIZE
    direction = (direction or config.BACKTEST_DIRECTION).upper()
    limit     = limit     or config.BACKTEST_LIMIT
    balance   = balance   or config.BACKTEST_STARTING_BALANCE
    max_pos   = config.MAX_CONCURRENT_POSITIONS

    exchange = build_exchange()

    # ── Load & filter data ────────────────────────────────────────────────────
    data = _load_universe(exchange, n_coins, limit)
    if not data:
        return {}

    # ── Precompute indicators ─────────────────────────────────────────────────
    print("[backtest] Computing indicators …")
    ind: dict[str, pd.DataFrame] = {}
    for sym, df in data.items():
        try:
            ind[sym] = _precompute_indicators(df)
        except Exception as exc:
            print(f"  ✗ {sym}: {exc}")
    print(f"  Done — {len(ind)} coins ready.\n")

    n_bars  = min(len(df) for df in ind.values())
    symbols = list(ind.keys())
    print(f"[backtest] Running {n_bars:,} bars × {len(symbols)} coins …\n")

    # ── Simulation state ──────────────────────────────────────────────────────
    trades: list[BacktestTrade]        = []
    equity_curve: list[float]          = [balance]
    open_positions: dict[str, BacktestTrade] = {}
    pending_orders: dict[str, dict]    = {}
    fill_timeout_bars = max(1, config.ORDER_FILL_TIMEOUT_SECONDS // 900)

    directions = []
    if direction in ("LONG",  "BOTH"): directions.append("long")
    if direction in ("SHORT", "BOTH"): directions.append("short")

    def efee(p, q): return p * q * config.BACKTEST_MAKER_FEE
    def xfee(p, q): return p * q * config.BACKTEST_TAKER_FEE

    # ── Bar loop ──────────────────────────────────────────────────────────────
    for bar in range(n_bars):

        # Print progress every ~5%
        if n_bars > 100 and bar % max(1, n_bars // 20) == 0:
            pct = bar / n_bars * 100
            print(f"  … {pct:4.0f}%  |  open={len(open_positions)}  "
                  f"pending={len(pending_orders)}  trades={len(trades)}  "
                  f"balance=${balance:.2f}")

        # ── Fill pending limit orders ─────────────────────────────────────────
        for sym in list(pending_orders):
            if sym in open_positions:
                del pending_orders[sym]; continue
            order = pending_orders[sym]
            df    = ind[sym]
            hi    = float(df["high"].iat[bar])
            lo    = float(df["low"].iat[bar])
            side  = order["side"]
            ep    = order["entry_price"]

            if bar - order["placed_bar"] > fill_timeout_bars:
                del pending_orders[sym]; continue

            if (side == "long" and lo <= ep) or (side == "short" and hi >= ep):
                t = BacktestTrade(
                    symbol=sym, side=side,
                    entry_bar=bar, entry_price=ep,
                    quantity=order["quantity"],
                    stop_loss=order["stop_loss"],
                    atr=order["atr"],
                    rs_slope=order["rs"],
                    volatility_rank=order.get("vol_rank", 0.0),
                    entry_balance=balance,
                )
                balance -= efee(ep, order["quantity"])
                open_positions[sym] = t
                del pending_orders[sym]

        # ── Manage open positions ─────────────────────────────────────────────
        to_close = []
        for sym, t in open_positions.items():
            df    = ind[sym]
            hi    = float(df["high"].iat[bar])
            lo    = float(df["low"].iat[bar])
            close = float(df["close"].iat[bar])
            atr   = t.atr
            t.candles_held += 1

            sl_hit = (t.side == "long" and lo <= t.stop_loss) or \
                     (t.side == "short" and hi >= t.stop_loss)

            if sl_hit:
                slip = config.BACKTEST_SLIPPAGE
                ep   = t.stop_loss * (1 - slip) if t.side == "long" \
                       else t.stop_loss * (1 + slip)
                pnl  = (ep - t.entry_price) * t.quantity * config.MAX_LEVERAGE \
                       if t.side == "long" \
                       else (t.entry_price - ep) * t.quantity * config.MAX_LEVERAGE
                pnl -= xfee(ep, t.quantity)
                balance += pnl
                t.exit_bar = bar; t.exit_price = ep
                t.pnl_usd = pnl; t.pnl_pct = pnl / t.entry_balance if t.entry_balance else 0
                t.exit_reason = "stop_loss"; t.exit_balance = balance
                to_close.append(sym); continue

            profit_atr = ((close - t.entry_price) / atr if t.side == "long"
                          else (t.entry_price - close) / atr) if atr > 0 else 0

            # Time stop
            if (t.candles_held >= config.TIME_STOP_CANDLES
                    and profit_atr < config.ATR_TIME_STOP_THRESHOLD):
                slip = config.BACKTEST_SLIPPAGE
                ep   = close * (1 - slip) if t.side == "long" else close * (1 + slip)
                pnl  = (ep - t.entry_price) * t.quantity * config.MAX_LEVERAGE \
                       if t.side == "long" \
                       else (t.entry_price - ep) * t.quantity * config.MAX_LEVERAGE
                pnl -= xfee(ep, t.quantity)
                balance += pnl
                t.exit_bar = bar; t.exit_price = ep
                t.pnl_usd = pnl; t.pnl_pct = pnl / t.entry_balance if t.entry_balance else 0
                t.exit_reason = "time_stop"; t.exit_balance = balance
                to_close.append(sym); continue

            # Breakeven ratchet
            if profit_atr >= config.ATR_BREAKEVEN_TRIGGER:
                if t.side == "long":
                    new_sl = t.entry_price * (1 + config.BREAKEVEN_FEE_BUFFER_PCT)
                    if new_sl > t.stop_loss: t.stop_loss = new_sl
                else:
                    new_sl = t.entry_price * (1 - config.BREAKEVEN_FEE_BUFFER_PCT)
                    if new_sl < t.stop_loss: t.stop_loss = new_sl

            # Pyramid
            if profit_atr >= config.ATR_PYRAMID_TRIGGER and not t.pyramid_added:
                add_qty   = t.quantity * config.PYRAMID_SIZE_RATIO
                add_price = close * (1 - config.MAKER_OFFSET_PCT) if t.side == "long" \
                            else close * (1 + config.MAKER_OFFSET_PCT)
                t.quantity += add_qty
                t.pyramid_added = True
                balance -= efee(add_price, add_qty)
                if t.side == "long":
                    t.stop_loss = t.entry_price * (1 + config.BREAKEVEN_FEE_BUFFER_PCT)
                else:
                    t.stop_loss = t.entry_price * (1 - config.BREAKEVEN_FEE_BUFFER_PCT)

        for sym in to_close:
            trades.append(open_positions.pop(sym))

        # ── Scan for new signals ──────────────────────────────────────────────
        slots = max_pos - len(open_positions) - len(pending_orders)
        if slots > 0:
            active = set(open_positions) | set(pending_orders)
            sigs   = []

            for sym in symbols:
                if sym in active: continue
                df = ind[sym]
                for d in directions:
                    fired, entry, atr, rs = _check_signal(df, bar, d)
                    if not fired: continue
                    stop = (entry - config.ATR_STOP_MULTIPLIER * atr if d == "long"
                            else entry + config.ATR_STOP_MULTIPLIER * atr)
                    qty = _calc_size(balance, entry, stop)
                    if qty <= 0 or entry * qty < 1.0: continue
                    sigs.append((rs, sym, d, entry, atr, stop, qty))

            # Best signals first (highest momentum slope)
            sigs.sort(key=lambda x: x[0], reverse=True)

            active = set(open_positions) | set(pending_orders)
            for rs, sym, d, entry, atr, stop, qty in sigs[:slots]:
                if sym in active: continue
                pending_orders[sym] = {
                    "side": d, "entry_price": entry,
                    "stop_loss": stop, "quantity": qty,
                    "atr": atr, "rs": rs,
                    "vol_rank": 0.0,
                    "placed_bar": bar,
                }
                active.add(sym)

        equity_curve.append(balance)

    # ── Close remaining positions at last bar ─────────────────────────────────
    for sym, t in open_positions.items():
        df    = ind[sym]
        close = float(df["close"].iat[n_bars - 1])
        slip  = config.BACKTEST_SLIPPAGE
        ep    = close * (1 - slip) if t.side == "long" else close * (1 + slip)
        pnl   = (ep - t.entry_price) * t.quantity * config.MAX_LEVERAGE \
                if t.side == "long" \
                else (t.entry_price - ep) * t.quantity * config.MAX_LEVERAGE
        pnl  -= xfee(ep, t.quantity)
        balance += pnl
        t.exit_bar = n_bars - 1; t.exit_price = ep
        t.pnl_usd = pnl; t.pnl_pct = pnl / t.entry_balance if t.entry_balance else 0
        t.exit_reason = "end_of_data"; t.exit_balance = balance
        trades.append(t)
    equity_curve.append(balance)

    # ── Output ────────────────────────────────────────────────────────────────
    metrics = _compute_metrics(trades, equity_curve)
    _print_report(metrics, len(symbols), direction, n_bars)

    if trades:
        with open("backtest_trades.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(trades[0].__dict__.keys()))
            w.writeheader()
            for t in trades: w.writerow(t.__dict__)
        print(f"  Trade log    → backtest_trades.csv  ({len(trades)} trades)")

    with open("backtest_equity.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bar", "equity_usd"])
        for i, eq in enumerate(equity_curve):
            w.writerow([i, round(eq, 6)])
    print(f"  Equity curve → backtest_equity.csv ({len(equity_curve)} bars)")
    print()

    return metrics


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Velocity-10k V3.0 — Multi-Asset Backtester"
    )
    parser.add_argument("--coins",     type=int,   default=config.BACKTEST_UNIVERSE_SIZE,
                        help=f"Volatile coins to include (default {config.BACKTEST_UNIVERSE_SIZE})")
    parser.add_argument("--limit",     type=int,   default=config.BACKTEST_LIMIT,
                        help="15m candles per coin (default 17500 ≈ 6 months)")
    parser.add_argument("--direction", default=config.BACKTEST_DIRECTION,
                        choices=["LONG", "SHORT", "BOTH"])
    parser.add_argument("--balance",   type=float, default=config.BACKTEST_STARTING_BALANCE,
                        help=f"Starting USDT balance (default ${config.BACKTEST_STARTING_BALANCE})")
    parser.add_argument("--top",       type=int,   default=config.MAX_CONCURRENT_POSITIONS,
                        help=f"Max concurrent positions (default {config.MAX_CONCURRENT_POSITIONS})")
    args = parser.parse_args()

    config.MAX_CONCURRENT_POSITIONS = args.top

    run_backtest(
        n_coins=args.coins,
        direction=args.direction,
        limit=args.limit,
        balance=args.balance,
    )