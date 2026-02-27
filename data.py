"""
data.py
───────
All market data fetching via ccxt and all indicator computation via pandas_ta.

Every function returns either a DataFrame or a scalar value so callers never
touch raw ccxt structures directly.
"""

from __future__ import annotations

import time as _time

import numpy as np
import pandas as pd
import pandas_ta as ta
import requests

import config
from utils import log, retry

# ── OHLCV FETCHING ────────────────────────────────────────────────────────────

_CANDLES_PER_PAGE = 1000   # MEXC hard cap per single API request

@retry(max_attempts=3, delay=2.0, backoff=2.0, label="fetch_ohlcv")
def fetch_ohlcv(exchange, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    """
    Fetch OHLCV candles, paginating automatically when limit > 1000.

    Columns: open, high, low, close, volume
    Index:   DatetimeIndex (UTC)
    """
    tf_ms = {
        "1m": 60_000, "3m": 180_000, "5m": 300_000,
        "15m": 900_000, "30m": 1_800_000,
        "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000,
    }
    step_ms = tf_ms.get(timeframe, 900_000)

    if limit <= _CANDLES_PER_PAGE:
        raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if not raw:
            raise ValueError(f"Empty OHLCV for {symbol} {timeframe}")
    else:
        # Walk backwards in time, collecting pages until we have `limit` bars
        raw    = []
        # Start from now and step backwards
        # Use `since` parameter: fetch `batch_size` bars starting from `since`
        # Strategy: fetch the latest page first, record earliest timestamp,
        # then fetch the page before that, etc.
        since  = None
        remaining = limit

        while remaining > 0:
            batch_size = min(_CANDLES_PER_PAGE, remaining)

            if since is None:
                # First call — get the most recent candles
                batch = exchange.fetch_ohlcv(
                    symbol, timeframe=timeframe, limit=batch_size
                )
            else:
                # Subsequent calls — fetch candles ending just before `since`
                fetch_since = since - batch_size * step_ms
                batch = exchange.fetch_ohlcv(
                    symbol, timeframe=timeframe,
                    since=fetch_since, limit=batch_size
                )

            if not batch:
                break

            # Prepend this older batch to raw (maintain chronological order)
            raw = batch + raw
            since = batch[0][0]      # earliest timestamp in this batch
            remaining -= len(batch)
            _time.sleep(0.12)        # gentle rate limiting

            if len(batch) < batch_size:
                break   # exchange has no more history available

        if not raw:
            raise ValueError(f"Empty paginated OHLCV for {symbol} {timeframe}")

    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp")
    df.set_index("datetime", inplace=True)
    df = df.astype({
        "open": float, "high": float, "low": float,
        "close": float, "volume": float,
    })
    # Trim to exactly `limit` most recent bars
    if len(df) > limit:
        df = df.iloc[-limit:]
    return df


# ── INDICATOR HELPERS ─────────────────────────────────────────────────────────

def add_ema(df: pd.DataFrame, period: int, col: str = "close") -> pd.DataFrame:
    df[f"ema_{period}"] = ta.ema(df[col], length=period)
    return df


def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    adx = ta.adx(df["high"], df["low"], df["close"], length=period)
    if adx is not None and not adx.empty:
        df[f"adx_{period}"] = adx[f"ADX_{period}"]
    else:
        df[f"adx_{period}"] = np.nan
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df[f"atr_{period}"] = ta.atr(df["high"], df["low"], df["close"], length=period)
    return df


def add_donchian(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    dc = ta.donchian(df["high"], df["low"], lower_length=period, upper_length=period)
    if dc is not None and not dc.empty:
        upper_col = [c for c in dc.columns if "UPPER" in c.upper() or "DCU" in c.upper()]
        lower_col = [c for c in dc.columns if "LOWER" in c.upper() or "DCL" in c.upper()]
        if upper_col:
            df[f"dc_upper_{period}"] = dc[upper_col[0]]
        if lower_col:
            df[f"dc_lower_{period}"] = dc[lower_col[0]]
    return df


def add_bbands(df: pd.DataFrame, period: int = 20, std: float = 2.0) -> pd.DataFrame:
    bb = ta.bbands(df["close"], length=period, std=std)
    if bb is not None and not bb.empty:
        bbu = [c for c in bb.columns if c.startswith("BBU")]
        bbl = [c for c in bb.columns if c.startswith("BBL")]
        bbm = [c for c in bb.columns if c.startswith("BBM")]
        bbw = [c for c in bb.columns if c.startswith("BBB")]
        if bbu: df["bb_upper"] = bb[bbu[0]]
        if bbl: df["bb_lower"] = bb[bbl[0]]
        if bbm: df["bb_mid"]   = bb[bbm[0]]
        if bbw:
            df["bbw"] = bb[bbw[0]]
        else:
            if bbu and bbl and bbm:
                df["bbw"] = (bb[bbu[0]] - bb[bbl[0]]) / bb[bbm[0]] * 100
    return df


def add_volume_ma(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    df[f"vol_ma_{period}"] = ta.sma(df["volume"], length=period)
    return df


# ── ANCHORED VWAP ─────────────────────────────────────────────────────────────

def compute_anchored_vwap(df: pd.DataFrame, lookback_bars: int = 96) -> pd.DataFrame:
    window     = df.tail(lookback_bars).copy()
    anchor_idx = window["volume"].idxmax()
    anchor_pos = df.index.get_loc(anchor_idx)
    anchored   = df.iloc[anchor_pos:].copy()
    anchored["tp"]      = (anchored["high"] + anchored["low"] + anchored["close"]) / 3
    anchored["cum_tpv"] = (anchored["tp"] * anchored["volume"]).cumsum()
    anchored["cum_vol"] = anchored["volume"].cumsum()
    anchored["avwap"]   = anchored["cum_tpv"] / anchored["cum_vol"]
    df["avwap"] = np.nan
    df.loc[anchored.index, "avwap"] = anchored["avwap"]
    return df


# ── RELATIVE STRENGTH SLOPE ───────────────────────────────────────────────────

def compute_rs_slope(
    exchange,
    symbol: str,
    btc_df: pd.DataFrame,
    lookback: int = 24,
) -> float:
    try:
        coin_df = fetch_ohlcv(exchange, symbol, config.TF_1H, config.OHLCV_LIMIT_1H)
        merged  = coin_df[["close"]].join(
            btc_df[["close"]].rename(columns={"close": "btc_close"}),
            how="inner",
        ).tail(lookback)
        if len(merged) < lookback // 2:
            return -999.0
        rs_ratio = merged["close"] / merged["btc_close"]
        x = np.arange(len(rs_ratio))
        slope, _ = np.polyfit(x, rs_ratio.values, 1)
        return float(slope)
    except Exception as exc:
        log.warning("[data] RS slope error for %s: %s", symbol, exc)
        return -999.0


# ── FUNDING RATE ──────────────────────────────────────────────────────────────

@retry(max_attempts=3, delay=2.0, backoff=2.0, label="fetch_funding_rate")
def fetch_funding_rate(exchange, symbol: str) -> float:
    try:
        data = exchange.fetch_funding_rate(symbol)
        rate = data.get("fundingRate", 0.0)
        return float(rate) if rate is not None else 0.0
    except Exception as exc:
        log.warning("[data] Funding rate failed for %s: %s", symbol, exc)
        return 0.0


# ── TICKER / VOLUME ───────────────────────────────────────────────────────────

@retry(max_attempts=3, delay=2.0, backoff=2.0, label="fetch_tickers")
def fetch_tickers(exchange) -> dict:
    return exchange.fetch_tickers()


def get_top_symbols_by_volume(
    tickers: dict,
    n: int = 30,
    exclude_stables: bool = True,
) -> list[str]:
    from utils import is_stable
    rows = []
    for sym, t in tickers.items():
        if not sym.endswith("/USDT") and not sym.endswith("/USDT:USDT"):
            continue
        base = sym.split("/")[0]
        if exclude_stables and is_stable(base):
            continue
        vol = t.get("quoteVolume") or t.get("baseVolume", 0) or 0
        rows.append((sym, float(vol)))
    rows.sort(key=lambda x: x[1], reverse=True)
    return [r[0] for r in rows[:n]]


# ── VOLATILITY RANKING ────────────────────────────────────────────────────────

def rank_by_volatility(
    tickers: dict,
    n_input: int = 100,
    n_output: int = 50,
) -> list[str]:
    """
    From the top n_input symbols by volume, return the n_output most volatile.
    Volatility = (high_24h - low_24h) / close  (normalised daily range).
    This ensures we trade coins that are actually MOVING, not just liquid.
    Falls back to volume ranking if price range data is unavailable.
    """
    from utils import is_stable
    rows = []
    for sym, t in tickers.items():
        if not sym.endswith("/USDT") and not sym.endswith("/USDT:USDT"):
            continue
        base = sym.split("/")[0]
        if is_stable(base):
            continue
        vol    = t.get("quoteVolume") or t.get("baseVolume", 0) or 0
        hi     = t.get("high")  or 0.0
        lo     = t.get("low")   or 0.0
        close  = t.get("last")  or t.get("close") or 1.0
        if close <= 0:
            continue
        volatility = (float(hi) - float(lo)) / float(close) if hi and lo else 0.0
        rows.append((sym, float(vol), volatility))

    # First filter to top n_input by volume (ensures liquidity)
    rows.sort(key=lambda x: x[1], reverse=True)
    liquid = rows[:n_input]

    # Then rank by volatility and take top n_output
    liquid.sort(key=lambda x: x[2], reverse=True)
    result = [r[0] for r in liquid[:n_output]]
    log.info(
        "[data] Volatility ranking: top coin=%s (%.2f%% daily range), "
        "bottom=%s (%.2f%%)",
        liquid[0][0], liquid[0][2] * 100 if liquid else 0,
        liquid[n_output - 1][0] if len(liquid) >= n_output else "N/A",
        liquid[n_output - 1][2] * 100 if len(liquid) >= n_output else 0,
    )
    return result


# ── FEAR & GREED INDEX ────────────────────────────────────────────────────────

def fetch_fear_greed_index() -> int:
    try:
        resp = requests.get(config.FEAR_GREED_URL, timeout=10)
        resp.raise_for_status()
        data  = resp.json()
        value = int(data["data"][0]["value"])
        log.info("[data] Fear & Greed Index: %d", value)
        return value
    except Exception as exc:
        log.error("[data] Fear & Greed fetch failed: %s", exc)
        return -1


# ── FULL INDICATOR PIPELINES ──────────────────────────────────────────────────

def build_4h_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = add_ema(df, config.EMA_PERIOD)
    df = add_adx(df, config.ADX_PERIOD)
    return df


def build_1h_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = add_ema(df, config.EMA_PERIOD)
    return df


def build_15m_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = add_ema(df, config.EMA_PERIOD)
    df = add_atr(df, config.ATR_PERIOD)
    df = add_donchian(df, config.DONCHIAN_PERIOD)
    df = add_bbands(df, config.BB_PERIOD, config.BB_STD)
    df = add_volume_ma(df, config.VOLUME_MA_PERIOD)
    if "bbw" in df.columns:
        df["bbw_sma"] = ta.sma(df["bbw"], length=config.BBW_SMA_PERIOD)
    df = compute_anchored_vwap(df, lookback_bars=96)
    return df