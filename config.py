"""
config.py
─────────
Single source of truth for every tunable parameter in the V3.0 system.
Edit values here; never scatter magic numbers across other modules.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── MODE ──────────────────────────────────────────────────────────────────────
# "PAPER" → Paper trading mode (NO API keys needed, US-friendly).
#           Uses live MEXC public market data. All orders are simulated locally.
# "LIVE"  → MEXC Perpetual Futures (real money, requires MEXC API keys)
TRADING_MODE: str = os.getenv("TRADING_MODE", "PAPER").upper()

# ── TRADE DIRECTION ───────────────────────────────────────────────────────────
# "LONG"  → Only take long (buy) trades — bullish regime required
# "SHORT" → Only take short (sell) trades — bearish regime required
# "BOTH"  → Take longs in bull regime, shorts in bear regime
TRADE_DIRECTION: str = os.getenv("TRADE_DIRECTION", "BOTH").upper()

# ── PAPER TRADING SETTINGS ────────────────────────────────────────────────────
PAPER_STARTING_BALANCE: float = 100.0
PAPER_FILL_MODEL: str         = "next_close"   # "optimistic" | "next_close"

# ── MEXC (LIVE mode) ──────────────────────────────────────────────────────────
MEXC_API_KEY: str    = os.getenv("MEXC_API_KEY", "")
MEXC_API_SECRET: str = os.getenv("MEXC_API_SECRET", "")

# ── TELEGRAM ──────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID: str   = os.getenv("TELEGRAM_CHAT_ID", "")

# ── LOOP TIMING ───────────────────────────────────────────────────────────────
MAIN_LOOP_SLEEP_SECONDS: int      = 60
FILTER_FAIL_SLEEP_SECONDS: int    = 300
ORDER_FILL_TIMEOUT_SECONDS: int   = 90

# ── EXCHANGE PARAMETERS ───────────────────────────────────────────────────────
MAX_LEVERAGE: int        = 5
MAKER_OFFSET_PCT: float  = 0.001

# ── RISK PARAMETERS ───────────────────────────────────────────────────────────
BASE_RISK_PER_TRADE: float = 0.20   # 20% of balance risked per trade
MAX_EQUITY_RISK_PCT: float = 0.20   # Kelly cap — matches base risk
MIN_EQUITY_RISK_PCT: float = 0.15

# ── ATR MULTIPLIERS ───────────────────────────────────────────────────────────
ATR_STOP_MULTIPLIER: float      = 2.0
ATR_BREAKEVEN_TRIGGER: float    = 1.5
ATR_PYRAMID_TRIGGER: float      = 1.0
ATR_TIME_STOP_THRESHOLD: float  = 0.5
BREAKEVEN_FEE_BUFFER_PCT: float = 0.001

# ── PYRAMID ───────────────────────────────────────────────────────────────────
PYRAMID_SIZE_RATIO: float = 0.50
MAX_PYRAMID_LAYERS: int   = 1

# ── TIME STOP ─────────────────────────────────────────────────────────────────
TIME_STOP_CANDLES: int = 12

# ── INDICATOR PERIODS ─────────────────────────────────────────────────────────
EMA_PERIOD: int             = 50
ADX_PERIOD: int             = 14
ATR_PERIOD: int             = 14
DONCHIAN_PERIOD: int        = 20
BB_PERIOD: int              = 20
BB_STD: float               = 2.0
BBW_SMA_PERIOD: int         = 20
VOLUME_MA_PERIOD: int       = 20
VOLUME_SPIKE_MULT: float    = 1.5
GRAVITY_PCT: float          = 0.03

# ── TIMEFRAMES ────────────────────────────────────────────────────────────────
TF_4H: str  = "4h"
TF_1H: str  = "1h"
TF_15M: str = "15m"

OHLCV_LIMIT_4H: int  = 100
OHLCV_LIMIT_1H: int  = 48
OHLCV_LIMIT_15M: int = 100

RS_SLOPE_LOOKBACK: int = 24

# ── ASSET UNIVERSE ────────────────────────────────────────────────────────────
UNIVERSE_SIZE: int            = 50   # top 50 coins by 24h volume
BREADTH_TOP_N: int            = 10
BREADTH_MIN_ABOVE: int        = 6
BREADTH_MAX_ABOVE: int        = 4   # breadth for shorts: ≤4/10 above EMA = broad weakness
MAX_CONCURRENT_POSITIONS: int = 5    # 5 simultaneous positions
TOP_CANDIDATES: int           = 5    # scanner returns top 5 signals

STABLECOINS: set = {
    "USDT", "USDC", "BUSD", "DAI", "TUSD", "USDP", "FDUSD",
    "USDD", "GUSD", "FRAX", "LUSD", "SUSD", "USTC",
}

# ── MACRO FILTERS ─────────────────────────────────────────────────────────────
ADX_THRESHOLD: float            = 20.0
FEAR_GREED_FLOOR: int           = 10    # halt ALL trading below this
FEAR_GREED_SHORT_CAP: int       = 40    # only allow shorts when F&G ≤ this
FUNDING_RATE_CAP: float         = 0.0005
CORRELATION_THRESHOLD: float    = 0.85

# ── BACKTEST SETTINGS ─────────────────────────────────────────────────────────
BACKTEST_STARTING_BALANCE: float = 100.0
BACKTEST_UNIVERSE_SIZE: int      = 50          # coins to include in backtest universe
BACKTEST_TIMEFRAME: str          = "15m"        # must match entry trigger TF
BACKTEST_HIGHER_TF: str          = "4h"         # for regime filter in backtest
BACKTEST_LIMIT: int              = 17500        # 17500×15m ≈ 6 months
BACKTEST_DIRECTION: str          = "BOTH"       # "LONG" | "SHORT" | "BOTH"
BACKTEST_TAKER_FEE: float        = 0.0006       # 0.06% taker fee for exits
BACKTEST_MAKER_FEE: float        = 0.0          # 0% maker fee for entries
BACKTEST_SLIPPAGE: float         = 0.0005       # 0.05% slippage on fills

# ── EXTERNAL APIS ─────────────────────────────────────────────────────────────
FEAR_GREED_URL: str = "https://api.alternative.me/fng/?limit=1&format=json"

# ── PERSISTENCE ───────────────────────────────────────────────────────────────
DB_PATH: str = "v10k_state.db"

# ── LOGGING ───────────────────────────────────────────────────────────────────
LOG_FILE: str         = "velocity10k.log"
LOG_LEVEL: str        = "INFO"
LOG_MAX_BYTES: int    = 5_000_000
LOG_BACKUP_COUNT: int = 3