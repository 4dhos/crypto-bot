"""
ml_predictor.py
───────────────
Random Forest probability filter. Learns from historical trade contexts
to predict the probability of success for future breakouts.
"""
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

MODEL_PATH = "ml_model.joblib"
DATA_PATH = "trade_data.csv"

# The data points the AI looks at to make a decision
FEATURES = ["bbw", "vol_ratio", "ema_dist", "atr_pct", "hour", "day_of_week"]

def train_model():
    """Trains the Random Forest AI on your collected trade history."""
    if not os.path.exists(DATA_PATH):
        print(f"❌ Cannot train: {DATA_PATH} not found. Run backtest with USE_ML_FILTER=False first.")
        return

    df = pd.read_csv(DATA_PATH)
    if len(df) < 50:
        print(f"❌ Not enough trades to train AI. Need at least 50, found {len(df)}.")
        return

    # Drop any corrupt rows
    df = df.dropna(subset=FEATURES + ["is_win"])

    X = df[FEATURES]
    y = df["is_win"]

    # Initialize the AI: 100 decision trees, max depth of 5 to prevent overfitting
    print(f"🧠 Training Random Forest AI on {len(df)} trades...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf.fit(X, y)

    joblib.dump(clf, MODEL_PATH)
    
    # Calculate basic accuracy on training set
    accuracy = clf.score(X, y)
    print(f"✅ Model trained and saved to {MODEL_PATH}")
    print(f"📊 Training Accuracy (In-Sample): {accuracy * 100:.1f}%")

def predict_trade_success(features_dict: dict) -> float:
    """Live inference function: Returns probability of trade success (0.0 to 1.0)."""
    if not os.path.exists(MODEL_PATH):
        return 1.0 # Default to taking the trade if no AI is trained yet
    
    try:
        clf = joblib.load(MODEL_PATH)
        X = pd.DataFrame([features_dict], columns=FEATURES)
        # predict_proba returns [[prob_loss, prob_win]]
        prob_win = clf.predict_proba(X)[0][1]
        return prob_win
    except Exception as e:
        print(f"ML Prediction Error: {e}")
        return 1.0

if __name__ == "__main__":
    # If you run this file directly, it trains the model
    train_model()