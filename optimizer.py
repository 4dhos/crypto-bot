"""
optimizer.py
────────────
Grid Search AI with Out-of-Sample verification.
UPGRADED: Multiprocessing enabled to utilize 100% of CPU cores.
"""
from backtest import download_and_prep_data, run_simulation
from joblib import Parallel, delayed
import time

def eval_setup(data_arrays, rr, lb, depth, buf, be):
    # This function runs independently on a separate CPU core
    final_bal, win_rate, trades = run_simulation(
        data_arrays, 
        balance=100.0, 
        rr_ratio=rr, risk_pct=0.02, 
        sweep_lookback=lb, fvg_depth=depth, sl_buffer=buf, be_trigger=be,
        data_range=(0.0, 0.7), 
        verbose=False
    )
    return {
        "rr": rr, "lookback": lb, "depth": depth, "buffer": buf, "be": be,
        "balance": final_bal, "win_rate": win_rate, "trades": trades
    }

def optimize_strategy():
    print("🧠 Starting Institutional AI Optimizer...")
    
    # 1. Download data (Will load instantly if the .pkl cache file exists)
    print("📥 Fetching market data...")
    # Change limits here as you wish (e.g., limit=105000 for 3 years)
    data_arrays = download_and_prep_data(n_coins=30, limit=35000, verbose=True)
    
    if not data_arrays: return

    # ── HYPERPARAMETER GRID ──
    rr_ratios = [1.5, 2.0, 2.5]
    lookbacks = [10, 20] 
    fvg_depths = [0.0, 0.5]  
    sl_buffers = [0.001, 0.003] 
    be_triggers = [1.0, 1.5] 
    
    # Build a list of all parameter combinations
    tasks = []
    for rr in rr_ratios:
        for lb in lookbacks:
            for depth in fvg_depths:
                for buf in sl_buffers:
                    for be in be_triggers:
                        tasks.append((rr, lb, depth, buf, be))

    total_tests = len(tasks)
    start_time = time.time()
    
    print(f"\n🔬 PHASE 1: IN-SAMPLE TRAINING (Months 1 to 8)")
    print(f"🚀 Launching {total_tests} simulations across ALL CPU cores. Please wait...")

    # MULTIPROCESSING: Runs all combinations in parallel
    # n_jobs=-1 tells the computer to use 100% of available CPU cores
    outcomes = Parallel(n_jobs=-1)(
        delayed(eval_setup)(data_arrays, *params) for params in tasks
    )

    results = [res for res in outcomes if res["trades"] > 15]

    if not results:
        print("\n❌ No profitable combinations found.")
        return

    best = max(results, key=lambda x: x["balance"])
    
    print("\n\n" + "🏆"*20)
    print(" PHASE 2: OUT-OF-SAMPLE VERIFICATION ".center(40, "="))
    print("🏆"*20)
    
    # Stress test the single best strategy on blind data
    oos_bal, oos_win_rate, oos_trades = run_simulation(
        data_arrays, 
        balance=100.0, 
        rr_ratio=best['rr'], risk_pct=0.02, 
        sweep_lookback=best['lookback'], fvg_depth=best['depth'], sl_buffer=best['buffer'], be_trigger=best['be'],
        data_range=(0.7, 1.0), 
        verbose=False
    )
    
    is_profit = best['balance'] - 100
    oos_profit = oos_bal - 100
    
    print(f"🥇 BEST PARAMETERS (Found in {time.time() - start_time:.1f} seconds!):")
    print(f"   Take Profit (R:R)  : 1:{best['rr']}")
    print(f"   Sweep Lookback     : {best['lookback']} candles")
    print(f"   FVG Entry Depth    : {best['depth']*100}%")
    print(f"   Stop-Loss Buffer   : {best['buffer']*100}%")
    print(f"   BE Trailing Trigger: {best['be']} R\n")
    
    print(f"📊 IN-SAMPLE RESULTS (The Past):")
    print(f"   Profit: ${is_profit:.2f} | Win Rate: {best['win_rate']*100:.1f}% | Trades: {best['trades']}")
    
    print(f"\n🔮 OUT-OF-SAMPLE RESULTS (The Future Test):")
    print(f"   Profit: ${oos_profit:.2f} | Win Rate: {oos_win_rate*100:.1f}% | Trades: {oos_trades}")
    
    if oos_profit > 0:
        print("\n✅ VERDICT: PASS. The edge is verified. You are ready to trade.")
    else:
        print("\n❌ VERDICT: FAIL. The strategy did not survive new data.")

if __name__ == "__main__":
    optimize_strategy()