"""
sentiment_engine.py
───────────────────
Institutional News & Macro NLP Intelligence Layer.
"""
import feedparser
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging

log = logging.getLogger("v10k")

# FinBERT - Pre-trained on financial context
MODEL_NAME = "ProsusAI/finbert"
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
except Exception as e:
    log.warning(f"Could not load NLP model: {e}")
    tokenizer, model = None, None

FEEDS = [
    "https://cryptopanic.com/news/rss/",
    "https://cointelegraph.com/rss"
]

def get_market_sentiment() -> dict:
    """Scrapes RSS and returns sentiment score based on NLP."""
    if model is None:
        return {"signal": "neutral", "score": 0.0}

    scores = []
    for url in FEEDS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:5]: 
                inputs = tokenizer(entry.title, return_tensors="pt", truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    # FinBERT labels: [positive, negative, neutral]
                    scores.append(probs[0][0].item() - probs[0][1].item())
        except Exception:
            pass

    if not scores: return {"signal": "neutral", "score": 0.0}
    
    avg_score = sum(scores) / len(scores)
    
    if avg_score > 0.15: signal = "bullish"
    elif avg_score < -0.15: signal = "bearish"
    else: signal = "neutral"

    return {"signal": signal, "score": avg_score}