import sys
from datetime import datetime

from fetch_news import (
    TICKERS,
    fetch_headlines, parse_articles,
    fetch_prices, parse_prices,
    fetch_intraday, parse_intraday,
    save_to_csv, save_prices_to_csv, save_intraday_to_csv
)
from score_sentiment import (
    load_headlines,
    add_article_flags,
    score_dataframe,
    add_sentiment_divergence,
    save_scored,
    print_summary
)
from database import create_schema, insert_headlines, insert_prices, insert_intraday, insert_sentiment
import pandas as pd
import os


def run_fetch():
    print("\n=== STEP 1: Fetching data ===")
    all_articles = []
    all_prices = []
    all_intraday = []

    for ticker in TICKERS:
        articles = fetch_headlines(ticker)
        all_articles.extend(parse_articles(articles, ticker))

        prices = fetch_prices(ticker)
        all_prices.extend(parse_prices(prices, ticker))

        intraday = fetch_intraday(ticker, interval="5m")
        all_intraday.extend(parse_intraday(intraday, ticker))

    save_to_csv(all_articles)
    save_prices_to_csv(all_prices)
    save_intraday_to_csv(all_intraday)

    print(f"Fetched {len(all_articles)} articles, {len(all_prices)} price records, {len(all_intraday)} intraday records")


def run_scoring():
    print("\n=== STEP 2: Scoring sentiment ===")
    if not os.path.exists("data/headlines.csv"):
        print("[WARN] No headlines.csv found — skipping scoring step.")
        return
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


def run_db_store():
    print("\n=== STEP 3: Storing in database ===")
    create_schema()
    if not os.path.exists("data/headlines_scored.csv"):
        print("[WARN] No headlines_scored.csv found — skipping headline/sentiment insert.")
        insert_prices("data/prices.csv")
        insert_intraday("data/intraday.csv")
        return
    scored_df = pd.read_csv("data/headlines_scored.csv")
    url_to_id = insert_headlines(scored_df)
    insert_sentiment(scored_df, url_to_id)
    insert_prices("data/prices.csv")
    insert_intraday("data/intraday.csv")


if __name__ == "__main__":
    start = datetime.now()
    print(f"Pipeline started at {start.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        run_fetch()
        run_scoring()
        run_db_store()
    except Exception as e:
        print(f"\n[FATAL] Pipeline failed: {e}")
        sys.exit(1)

    elapsed = (datetime.now() - start).seconds
    print(f"\nPipeline finished in {elapsed}s")