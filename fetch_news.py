import os
import requests
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timedelta


load_dotenv()
API_KEY = os.getenv("EODHD_API_KEY")

EODHD_NEWS_URL = "https://eodhd.com/api/news"

# 5 tickers and company names
TICKERS = {
    "RKLB.US": "Rocket Lab",
    "ASTS.US": "AST SpaceMobile",
    "RDW.US": "Redwire",
    "LUNR.US": "Intuitive Machines",
    "PL.US": "Planet Labs"
}


def fetch_headlines(ticker: str, days_back: int = 30) -> list[dict]:
    """Fetch recent headlines for a company. Returns list of article dicts."""
    from_date = (datetime.today() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    params = {
        "s": ticker,
        "api_token": API_KEY,
        "from": from_date,
        "limit": 100,
        "fmt": "json"
    }
    
    try:
        response = requests.get(EODHD_NEWS_URL, params=params)
        response.raise_for_status()
        articles = response.json()
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
            "published_at": article.get("date"),
            "source": article.get("source", ""),
            "title": article.get("title"),
            "url": article.get("link")
        })
    return parsed


def fetch_prices(ticker: str, days_back: int = 30) -> list[dict]:
    """Fetch end-of-day price and volume data from EODHD."""
    from_date = (datetime.today() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    
    params = {
        "api_token": API_KEY,
        "from": from_date,
        "fmt": "json"
    }
    
    try:
        response = requests.get(
            f"https://eodhd.com/api/eod/{ticker}",
            params=params
        )
        response.raise_for_status()
        data = response.json()
        print(f"[OK] {ticker}: fetched {len(data)} price records")
        return data
    except Exception as e:
        print(f"[ERROR] Failed to fetch prices for {ticker}: {e}")
        return []


def parse_prices(data: list[dict], ticker: str) -> list[dict]:
    """Extract OHLCV fields from EODHD EOD response."""
    parsed = []
    for row in data:
        parsed.append({
            "ticker": ticker,
            "date": row.get("date"),
            "open": row.get("open"),
            "high": row.get("high"),
            "low": row.get("low"),
            "close": row.get("adjusted_close"),
            "volume": row.get("volume")
        })
    return parsed


def fetch_intraday(ticker: str, interval: str = "5m") -> list[dict]:
    """Fetch intraday OHLCV data from EODHD."""
    params = {
        "api_token": API_KEY,
        "interval": interval,
        "fmt": "json"
    }
    try:
        response = requests.get(
            f"https://eodhd.com/api/intraday/{ticker}",
            params=params
        )
        response.raise_for_status()
        data = response.json()
        print(f"[OK] {ticker}: fetched {len(data)} intraday records")
        return data
    except Exception as e:
        print(f"[ERROR] Failed to fetch intraday for {ticker}: {e}")
        return []


def parse_intraday(data: list[dict], ticker: str) -> list[dict]:
    """Extract OHLCV fields from EODHD intraday response."""
    parsed = []
    for row in data:
        parsed.append({
            "ticker": ticker,
            "datetime": row.get("datetime"),
            "open": row.get("open"),
            "high": row.get("high"),
            "low": row.get("low"),
            "close": row.get("close"),
            "volume": row.get("volume")
        })
    return parsed


def save_to_csv(data: list[dict], filename: str = "data/headlines.csv"):
    """Save parsed articles to CSV, appending if file exists."""
    df = pd.DataFrame(data)
    
    if df.empty:
        print("[WARN] No data to save.")
        return
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    if os.path.exists(filename):
        existing = pd.read_csv(filename)
        existing_keys = set(existing['ticker'] + '|' + existing['url'])
        df_new = df[~(df['ticker'] + '|' + df['url']).isin(existing_keys)]
        if not df_new.empty:
            df_new.to_csv(filename, mode="a", header=False, index=False)
            print(f"[OK] Appended {len(df_new)} new rows to {filename}")
        else:
            print(f"[OK] No new articles to append")
    else:
        df.to_csv(filename, index=False)
        print(f"[OK] Created {filename} with {len(df)} rows")


def save_prices_to_csv(data: list[dict], filename: str = "data/prices.csv"):
    """Save price data to CSV, appending if file exists."""
    df = pd.DataFrame(data)
    
    if df.empty:
        print("[WARN] No price data to save.")
        return
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    if os.path.exists(filename):
        existing = pd.read_csv(filename)
        existing_keys = set(existing['ticker'] + '|' + existing['date'].astype(str))
        df_new = df[~(df['ticker'] + '|' + df['date'].astype(str)).isin(existing_keys)]
        if not df_new.empty:
            df_new.to_csv(filename, mode="a", header=False, index=False)
            print(f"[OK] Appended {len(df_new)} new rows to {filename}")
        else:
            print(f"[OK] No new price records to append")
    else:
        df.to_csv(filename, index=False)
        print(f"[OK] Created {filename} with {len(df)} rows")


def save_intraday_to_csv(data: list[dict], filename: str = "data/intraday.csv"):
    """Save intraday data to CSV, appending if file exists."""
    df = pd.DataFrame(data)
    
    if df.empty:
        print("[WARN] No intraday data to save.")
        return
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    if os.path.exists(filename):
        existing = pd.read_csv(filename)
        existing_keys = set(existing['ticker'] + '|' + existing['datetime'].astype(str))
        df_new = df[~(df['ticker'] + '|' + df['datetime'].astype(str)).isin(existing_keys)]
        if not df_new.empty:
            df_new.to_csv(filename, mode="a", header=False, index=False)
            print(f"[OK] Appended {len(df_new)} new intraday rows")
        else:
            print(f"[OK] No new intraday data to append")
    else:
        df.to_csv(filename, index=False)
        print(f"[OK] Created {filename} with {len(df)} rows")


def main():
    all_articles = []
    all_prices = []
    all_intraday = []
    
    for ticker in TICKERS:
        # fetch news
        articles = fetch_headlines(ticker)
        parsed = parse_articles(articles, ticker)
        all_articles.extend(parsed)
        
        # fetch prices
        prices = fetch_prices(ticker)
        parsed_prices = parse_prices(prices, ticker)
        all_prices.extend(parsed_prices)

        # fetch intraday — 5 minute bars
        intraday = fetch_intraday(ticker, interval="5m")
        parsed_intraday = parse_intraday(intraday, ticker)
        all_intraday.extend(parsed_intraday)
    
    save_to_csv(all_articles)
    save_prices_to_csv(all_prices)
    save_intraday_to_csv(all_intraday)
    print(f"\nDone. Total articles saved: {len(all_articles)}")
    print(f"Done. Total price records saved: {len(all_prices)}")
    print(f"Done. Total intraday records saved: {len(all_intraday)}")


if __name__ == "__main__":
    main()