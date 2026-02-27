"""
config.py
─────────
Single source of truth for tunable parameters.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── MODE & ML SETTINGS ────────────────────────────────────────────────────────
TRADING_MODE: str = os.getenv("TRADING_MODE", "PAPER").upper()
TRADE_DIRECTION: str = os.getenv("TRADE_DIRECTION", "BOTH").upper()
PAPER_STARTING_BALANCE: float = 100.0

USE_ML_FILTER: bool = False       # Set to False to collect training data, True to use the AI
ML_PROB_THRESHOLD: float = 0.50   # Minimum AI confidence required to take a trade (50%)

# ── MEXC (LIVE mode) ──────────────────────────────────────────────────────────
MEXC_API_KEY: str    = os.getenv("MEXC_API_KEY", "")
MEXC_API_SECRET: str = os.getenv("MEXC_API_SECRET", "")
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID: str   = os.getenv("TELEGRAM_CHAT_ID", "")

# ── TIMING & EXECUTION ────────────────────────────────────────────────────────
MAIN_LOOP_SLEEP_SECONDS: int      = 60
FILTER_FAIL_SLEEP_SECONDS: int    = 300
ORDER_FILL_TIMEOUT_SECONDS: int   = 90
MAX_LEVERAGE: int         = 10      
TAKER_SLIPPAGE_PCT: float = 0.001   
MAKER_OFFSET_PCT: float   = 0.000   

# ── RISK PARAMETERS ───────────────────────────────────────────────────────────
BASE_RISK_PER_TRADE: float = 0.08   
MAX_EQUITY_RISK_PCT: float = 0.10   
MIN_EQUITY_RISK_PCT: float = 0.02
ATR_STOP_MULTIPLIER: float      = 1.5   
ATR_BREAKEVEN_TRIGGER: float    = 1.2   
ATR_PYRAMID_TRIGGER: float      = 1.5   
ATR_TIME_STOP_THRESHOLD: float  = 0.5
BREAKEVEN_FEE_BUFFER_PCT: float = 0.0015 
PYRAMID_SIZE_RATIO: float = 0.50
MAX_PYRAMID_LAYERS: int   = 1
TIME_STOP_CANDLES: int = 12

# ── INDICATORS & UNIVERSE ─────────────────────────────────────────────────────
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
TF_4H: str  = "4h"
TF_1H: str  = "1h"
TF_15M: str = "15m"
OHLCV_LIMIT_4H: int  = 100
OHLCV_LIMIT_1H: int  = 48
OHLCV_LIMIT_15M: int = 100
RS_SLOPE_LOOKBACK: int = 24

UNIVERSE_SIZE: int            = 50
BREADTH_TOP_N: int            = 10
BREADTH_MIN_ABOVE: int        = 6
BREADTH_MAX_ABOVE: int        = 4
MAX_CONCURRENT_POSITIONS: int = 5
TOP_CANDIDATES: int           = 5

STABLECOINS: set = {"USDT", "USDC", "BUSD", "DAI", "TUSD", "USDP", "FDUSD", "USDD", "GUSD", "FRAX", "LUSD", "SUSD", "USTC"}

# ── MACRO & BACKTEST ──────────────────────────────────────────────────────────
ADX_THRESHOLD: float            = 20.0
FEAR_GREED_FLOOR: int           = 10
FEAR_GREED_SHORT_CAP: int       = 40
FUNDING_RATE_CAP: float         = 0.0005
CORRELATION_THRESHOLD: float    = 0.85

BACKTEST_STARTING_BALANCE: float = 100.0
BACKTEST_UNIVERSE_SIZE: int      = 50
BACKTEST_TIMEFRAME: str          = "15m"
BACKTEST_LIMIT: int              = 17500
BACKTEST_DIRECTION: str          = "BOTH"
BACKTEST_TAKER_FEE: float        = 0.0006   
BACKTEST_SLIPPAGE: float         = 0.001    

DB_PATH: str = "v10k_state.db"
LOG_FILE: str = "velocity10k.log"
LOG_LEVEL: str = "INFO"
LOG_MAX_BYTES: int = 5_000_000
LOG_BACKUP_COUNT: int = 3