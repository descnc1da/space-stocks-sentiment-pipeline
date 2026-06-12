import os
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


def add_article_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add SpaceX mention flag and sector article flag to headlines."""
    df['is_spaceX_mention'] = df['title'].str.contains(
        'SpaceX|SPAX|space IPO', case=False, na=False
    ).astype(int)
    df['is_sector_article'] = (
        df.groupby('url')['ticker'].transform('count') > 1
    ).astype(int)
    return df


def add_sentiment_divergence(df: pd.DataFrame) -> pd.DataFrame:
    """Add sector average and divergence per day, excluding sector-wide articles."""
    df['_date'] = pd.to_datetime(df['published_at']).dt.date
    
    # only use company-specific articles for the sector baseline
    company_only = df[df['is_sector_article'] == 0]
    
    daily_sector = (
        company_only
        .groupby('_date')['compound']
        .mean()
        .rename('sector_sentiment')
    )
    
    df = df.join(daily_sector, on='_date').drop(columns=['_date'])
    df['sentiment_divergence'] = df['compound'] - df['sector_sentiment']
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
    df = add_article_flags(df)

    scored_path = "data/headlines_scored.csv"
    if os.path.exists(scored_path):
        existing_scores = pd.read_csv(scored_path)[['ticker', 'url', 'sentiment_label', 'sentiment_score', 'compound']]
        df = df.merge(existing_scores, on=['ticker', 'url'], how='left')
        unscored = df['sentiment_label'].isna()
        if unscored.any():
            print(f"[INFO] Scoring {unscored.sum()} new headlines, skipping {(~unscored).sum()} already scored")
            new_scores = score_dataframe(df[unscored].copy())
            df.loc[unscored, ['sentiment_label', 'sentiment_score', 'compound']] = new_scores[['sentiment_label', 'sentiment_score', 'compound']].values
        else:
            print("[INFO] All headlines already scored, skipping FinBERT.")
    else:
        df = score_dataframe(df)

    df = add_sentiment_divergence(df)
    save_scored(df)
    print_summary(df)


if __name__ == "__main__":
    main()