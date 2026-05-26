import os
import pandas as pd
from newsapi import NewsApiClient
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Load API key from .env
load_dotenv()
API_KEY = os.getenv("NEWS_API_KEY")

# Initialise client
newsapi = NewsApiClient(api_key=API_KEY)

# Your 5 tickers and company names
TICKERS = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Google",
    "AMZN": "Amazon",
    "TSLA": "Tesla"
}

def fetch_headlines(company_name: str, ticker: str, days_back: int = 7) -> list[dict]:
    """Fetch recent headlines for a company. Returns list of article dicts."""
    from_date = (datetime.today() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    
    try:
        response = newsapi.get_everything(
            q=f"{company_name} stock",
            from_param=from_date,
            domains="reuters.com,bloomberg.com,cnbc.com,ft.com,wsj.com,marketwatch.com,finance.yahoo.com",
            language="en",
            sort_by="relevancy",
            page_size=20
        )
        
        if response["status"] != "ok":
            print(f"[ERROR] Bad response for {ticker}: {response.get('message')}")
            return []
        
        articles = response["articles"]
        print(f"[OK] {ticker}: fetched {len(articles)} articles")
        return articles
    
    except Exception as e:
        print(f"[ERROR] Failed to fetch {ticker}: {e}")
        return []


def parse_articles(articles: list[dict], ticker: str) -> list[dict]:
    """Extract the fields we care about from raw API response."""
    parsed = []
    for article in articles:
        parsed.append({
            "ticker": ticker,
            "published_at": article.get("publishedAt"),
            "source": article.get("source", {}).get("name"),
            "title": article.get("title"),
            "description": article.get("description"),
            "url": article.get("url")
        })
    return parsed


def save_to_csv(data: list[dict], filename: str = "data/headlines.csv"):
    """Save parsed articles to CSV, appending if file exists."""
    df = pd.DataFrame(data)
    
    if df.empty:
        print("[WARN] No data to save.")
        return
    
    # Append if file exists, write fresh if not
    if os.path.exists(filename):
        df.to_csv(filename, mode="a", header=False, index=False)
        print(f"[OK] Appended {len(df)} rows to {filename}")
    else:
        df.to_csv(filename, index=False)
        print(f"[OK] Created {filename} with {len(df)} rows")

    # Remove duplicates after every save
    df_full = pd.read_csv(filename)                           # read entire file
    df_full.drop_duplicates(subset=["ticker", "url"], inplace=True)  # remove dupes
    df_full.to_csv(filename, index=False)                     # overwrite clean
    print(f"[OK] Deduped. File now has {len(df_full)} rows")

def main():
    all_articles = []
    
    for ticker, company in TICKERS.items():
        articles = fetch_headlines(company, ticker)
        parsed = parse_articles(articles, ticker)
        all_articles.extend(parsed)
    
    save_to_csv(all_articles)
    print(f"\nDone. Total articles saved: {len(all_articles)}")


if __name__ == "__main__":
    main()