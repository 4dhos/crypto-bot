"""
learning_engine.py
──────────────────
Analyzes rolling performance. Adapts risk dynamically based on market regime.
"""
import sqlite3
import config
from utils import log

def get_risk_multiplier() -> float:
    """
    Looks at the last 15 closed trades. 
    If Win Rate < 35%, reduces risk by 50% to prevent drawdown death spiral.
    If Win Rate > 50%, scales risk up by 1.2x to press advantages.
    """
    try:
        conn = sqlite3.connect(config.DB_PATH)
        conn.row_factory = sqlite3.Row
        # Fetch paper_orders or real historical orders from your DB
        # Assuming you use paper_orders for simulation learning
        rows = conn.execute(
            "SELECT * FROM paper_orders WHERE status='closed' ORDER BY filled_ms DESC LIMIT 15"
        ).fetchall()
        conn.close()

        if len(rows) < 5:
            return 1.0 # Not enough data, use baseline

        wins = 0
        for r in rows:
            # Simple approximation: If sell fill price > buy price, it's a win.
            if r["side"] == "sell" and r.get("pnl", 0) > 0:
                wins += 1

        win_rate = wins / len(rows)

        if win_rate < 0.35:
            log.warning(f"[learning] Win rate dropping ({win_rate*100:.1f}%). Throttling risk to 0.5x.")
            return 0.5
        elif win_rate > 0.50:
            log.info(f"[learning] Hot streak detected ({win_rate*100:.1f}%). Pushing risk to 1.2x.")
            return 1.2
            
        return 1.0

    except Exception as e:
        log.debug(f"[learning] History check skipped: {e}")
        return 1.0