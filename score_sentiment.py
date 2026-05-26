import pandas as pd
from transformers import pipeline

_finbert = None

def get_finbert():
    global _finbert
    if _finbert is None:
        _finbert = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert"
        )
    return _finbert

def score_with_finbert(text: str) -> dict:
    if not text or not isinstance(text, str):
        return {"label": "unknown", "score": None, "compound": None}
    try:
        finbert = get_finbert()
        result = finbert(text[:512])[0]
        label = result["label"].lower()
        confidence = result["score"]
        compound = confidence if label == "positive" else -confidence if label == "negative" else 0.0
        return {"label": label, "score": confidence, "compound": compound}
    except Exception as e:
        print(f"[ERROR] FinBERT failed on: {text[:50]}... → {e}")
        return {"label": "unknown", "score": None, "compound": None}


def score_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add FinBERT sentiment columns to a headlines DataFrame."""
    print("[INFO] Running FinBERT — this takes ~30 seconds on first run...")
    
    scores = df["title"].apply(score_with_finbert)
    df["sentiment_label"] = scores.apply(lambda x: x["label"])
    df["sentiment_score"] = scores.apply(lambda x: x["score"])
    df["compound"] = scores.apply(lambda x: x["compound"])
    return df


def load_headlines(filepath: str = "data/headlines.csv") -> pd.DataFrame:
    """Load headlines CSV produced by fetch_news.py."""
    df = pd.read_csv(filepath)
    print(f"[OK] Loaded {len(df)} headlines from {filepath}")
    return df


def save_scored(df: pd.DataFrame, filepath: str = "data/headlines_scored.csv"):
    """Save scored DataFrame to CSV."""
    df.to_csv(filepath, index=False)
    print(f"[OK] Saved scored headlines to {filepath}")

def print_summary(df: pd.DataFrame):
    """Print most positive and most negative headline per ticker."""
    print("\n--- SENTIMENT SUMMARY ---\n")
    
    for ticker in df["ticker"].unique():
        ticker_df = df[df["ticker"] == ticker].dropna(subset=["compound"])
        
        if ticker_df.empty:
            continue
        
        best = ticker_df.loc[ticker_df["compound"].idxmax()]
        worst = ticker_df.loc[ticker_df["compound"].idxmin()]
        avg = ticker_df["compound"].mean()
        
        print(f"{ticker} | avg sentiment: {avg:+.3f}")
        print(f"  Most positive ({best['compound']:+.2f}): {best['title']}")
        print(f"  Most negative ({worst['compound']:+.2f}): {worst['title']}")
        print()


def main():
    df = load_headlines()
    df = score_dataframe(df)
    save_scored(df)
    print_summary(df)


if __name__ == "__main__":
    main()