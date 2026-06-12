import sqlite3
import pandas as pd
from datetime import datetime


DB_PATH = "data/pipeline.db"


def get_connection():
    """Return a database connection with foreign keys enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def create_schema():
    """Create all tables if they don't already exist."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            UNIQUE(ticker, date)
        );

        CREATE TABLE IF NOT EXISTS headlines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            published_at TEXT,
            source TEXT,
            title TEXT,
            url TEXT,
            is_spaceX_mention INTEGER DEFAULT 0,
            is_sector_article INTEGER DEFAULT 0,
            fetched_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(ticker, url)
        );

        CREATE TABLE IF NOT EXISTS sentiment (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            headline_id INTEGER NOT NULL UNIQUE,
            label TEXT,
            score REAL,
            compound REAL,
            sentiment_divergence REAL,
            sector_sentiment REAL,
            model_used TEXT DEFAULT 'finbert',
            scored_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (headline_id) REFERENCES headlines(id)
            
        );

        CREATE TABLE IF NOT EXISTS intraday (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            datetime TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            UNIQUE(ticker, datetime)
        );            

        CREATE TABLE IF NOT EXISTS daily_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            ticker TEXT,
            read TEXT,
            was_right INTEGER,
            notes TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_headlines_ticker ON headlines(ticker);
        CREATE INDEX IF NOT EXISTS idx_headlines_date ON headlines(published_at);
        CREATE INDEX IF NOT EXISTS idx_sentiment_headline ON sentiment(headline_id);
        CREATE INDEX IF NOT EXISTS idx_intraday_ticker ON intraday(ticker);
        CREATE INDEX IF NOT EXISTS idx_intraday_datetime ON intraday(datetime);
    """)

    conn.commit()
    conn.close()
    print("[OK] Schema ready")


def insert_headlines(df: pd.DataFrame) -> dict:
    """
    Insert headlines from DataFrame into DB.
    Returns dict mapping url → headline_id for sentiment linking.
    """
    conn = get_connection()
    cursor = conn.cursor()
    inserted = 0
    skipped = 0
    url_to_id = {}

    for _, row in df.iterrows():
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO headlines 
                (ticker, published_at, source, title, url, is_spaceX_mention, is_sector_article)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                row.get("ticker"),
                row.get("published_at"),
                row.get("source"),
                row.get("title"),
                row.get("url"),
                int(row.get("is_spaceX_mention", 0)),
                int(row.get("is_sector_article", 0))
            ))
            if cursor.rowcount > 0:
                inserted += 1
                url_to_id[(row.get("ticker"), row.get("url"))] = cursor.lastrowid
            else:
                # Already existed — fetch its id
                cursor.execute(
                    "SELECT id FROM headlines WHERE ticker = ? AND url = ?",
                    (row.get("ticker"), row.get("url"))
                )
                result = cursor.fetchone()
                if result:
                    url_to_id[(row.get("ticker"), row.get("url"))] = result[0]
                skipped += 1
        except Exception as e:
            print(f"[ERROR] Insert headline failed: {e}")

    conn.commit()
    conn.close()
    print(f"[OK] Headlines: {inserted} inserted, {skipped} skipped (duplicates)")
    return url_to_id


def insert_prices(filename: str = "data/prices.csv"):
    """Load prices CSV and insert into DB."""
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print("[WARN] No prices.csv found, skipping")
        return
    
    conn = get_connection()
    cursor = conn.cursor()
    inserted = 0

    for _, row in df.iterrows():
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO prices
                (ticker, date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                row.get("ticker"),
                row.get("date"),
                row.get("open"),
                row.get("high"),
                row.get("low"),
                row.get("close"),
                row.get("volume")
            ))
            if cursor.rowcount > 0:
                inserted += 1
        except Exception as e:
            print(f"[ERROR] Insert price failed: {e}")

    conn.commit()
    conn.close()
    print(f"[OK] Prices: {inserted} records inserted")


def insert_intraday(filename: str = "data/intraday.csv"):
    """Load intraday CSV and insert into DB."""
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print("[WARN] No intraday.csv found, skipping")
        return
    
    conn = get_connection()
    cursor = conn.cursor()
    inserted = 0

    for _, row in df.iterrows():
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO intraday
                (ticker, datetime, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                row.get("ticker"),
                row.get("datetime"),
                row.get("open"),
                row.get("high"),
                row.get("low"),
                row.get("close"),
                row.get("volume")
            ))
            if cursor.rowcount > 0:
                inserted += 1
        except Exception as e:
            print(f"[ERROR] Insert intraday failed: {e}")

    conn.commit()
    conn.close()
    print(f"[OK] Intraday: {inserted} records inserted")


def insert_sentiment(scored_df: pd.DataFrame, url_to_id: dict):
    """Insert sentiment scores linked to headline IDs."""
    conn = get_connection()
    cursor = conn.cursor()
    inserted = 0
    skipped = 0

    for _, row in scored_df.iterrows():
        headline_id = url_to_id.get((row.get("ticker"), row.get("url")))
        if not headline_id:
            continue
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO sentiment (headline_id, label, score, compound, sentiment_divergence, sector_sentiment, model_used)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                headline_id,
                row.get("sentiment_label"),
                row.get("sentiment_score"),
                row.get("compound"),
                row.get("sentiment_divergence"),
                row.get("sector_sentiment"),
                "finbert"
            ))
            if cursor.rowcount > 0:
                inserted += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"[ERROR] Insert sentiment failed: {e}")

    conn.commit()
    conn.close()
    print(f"[OK] Sentiment: {inserted} scores inserted, {skipped} skipped (duplicates)")


def query_checklist(date: str = None) -> pd.DataFrame:
    """Daily signal checklist — one row per ticker with all signals."""
    target = date or datetime.now().strftime("%Y-%m-%d")
    conn = get_connection()
    query = """
        WITH sector_avg AS (
            SELECT 
                DATE(h.published_at) as date,
                AVG(s.compound) as sector_sentiment,
                SUM(h.is_spaceX_mention) as spaceX_mentions
            FROM sentiment s
            JOIN headlines h ON s.headline_id = h.id
            WHERE DATE(h.published_at) = DATE(?)
            GROUP BY DATE(h.published_at)
        ),
        ticker_sentiment AS (
            SELECT
                h.ticker,
                AVG(s.compound) as ticker_sentiment,
                COUNT(*) as article_count
            FROM sentiment s
            JOIN headlines h ON s.headline_id = h.id
            WHERE DATE(h.published_at) = DATE(?)
            AND h.is_sector_article = 0
            GROUP BY h.ticker
        ),
        volume_signal AS (
            WITH baseline AS (
                SELECT ticker,
                    AVG(volume) as avg_volume,
                    AVG(volume * volume) - AVG(volume) * AVG(volume) as variance
                FROM prices
                WHERE date >= DATE('now', '-20 days')
                GROUP BY ticker
            )
            SELECT p.ticker,
                ROUND((p.volume - b.avg_volume) / SQRT(b.variance), 2) as volume_zscore
            FROM prices p
            JOIN baseline b ON p.ticker = b.ticker
            WHERE p.date = DATE(?)
        )
        SELECT
            t.ticker,
            ROUND(t.ticker_sentiment, 3) as company_sentiment,
            ROUND(sa.sector_sentiment, 3) as sector_sentiment,
            ROUND(t.ticker_sentiment - sa.sector_sentiment, 3) as divergence,
            sa.spaceX_mentions,
            v.volume_zscore,
            t.article_count,
            CASE 
                WHEN sa.spaceX_mentions > 3 THEN 'SECTOR DAY - STAND ASIDE'
                WHEN t.ticker_sentiment - sa.sector_sentiment > 0.15 
                     AND v.volume_zscore > 2.0 THEN 'STRONG SIGNAL'
                WHEN t.ticker_sentiment - sa.sector_sentiment > 0.15 
                     THEN 'WEAK SIGNAL - WAIT FOR VOLUME'
                WHEN t.ticker_sentiment - sa.sector_sentiment < -0.15 
                     AND v.volume_zscore > 2.0 THEN 'NEGATIVE SIGNAL'
                ELSE 'NO SIGNAL'
            END as action
        FROM ticker_sentiment t
        CROSS JOIN sector_avg sa
        LEFT JOIN volume_signal v ON t.ticker = v.ticker
        ORDER BY divergence DESC
    """
    df = pd.read_sql_query(query, conn, params=(target, target, target))
    conn.close()
    return df


def query_top_movers():
    """Tickers ranked by biggest sentiment shift this week vs last week."""
    conn = get_connection()
    query = """
        SELECT 
            h.ticker,
            ROUND(AVG(CASE WHEN DATE(h.published_at) >= DATE('now', '-7 days') 
                THEN s.compound END), 3) as sentiment_this_week,
            ROUND(AVG(CASE WHEN DATE(h.published_at) < DATE('now', '-7 days') 
                THEN s.compound END), 3) as sentiment_last_week,
            ROUND(
                AVG(CASE WHEN DATE(h.published_at) >= DATE('now', '-7 days') 
                    THEN s.compound END) - 
                AVG(CASE WHEN DATE(h.published_at) < DATE('now', '-7 days') 
                    THEN s.compound END)
            , 3) as sentiment_shift
        FROM headlines h
        JOIN sentiment s ON h.id = s.headline_id
        GROUP BY h.ticker
        ORDER BY ABS(sentiment_shift) DESC;
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def query_top_headlines(date: str = None) -> pd.DataFrame:
    """Return top positive and negative headlines for each ticker on a given date."""
    target = date or datetime.now().strftime("%Y-%m-%d")
    conn = get_connection()
    query = """
        SELECT
            h.ticker,
            h.title,
            h.url,
            h.published_at,
            s.compound,
            s.label
        FROM headlines h
        JOIN sentiment s ON h.id = s.headline_id
        WHERE DATE(h.published_at) = DATE(?)
        ORDER BY h.ticker, ABS(s.compound) DESC
    """
    df = pd.read_sql_query(query, conn, params=(target,))
    conn.close()
    return df


def query_rolling_sentiment():
    """Rolling 7-calendar-day average sentiment per ticker."""
    conn = get_connection()
    query = """
        SELECT
            h.ticker,
            DATE(h.published_at) as date,
            AVG(s.compound) as daily_sentiment
        FROM headlines h
        JOIN sentiment s ON h.id = s.headline_id
        GROUP BY h.ticker, DATE(h.published_at)
        ORDER BY h.ticker, date
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        return pd.DataFrame(columns=['ticker', 'date', 'rolling_avg_sentiment'])

    df['date'] = pd.to_datetime(df['date'])
    parts = []
    for ticker, grp in df.groupby('ticker'):
        grp = grp.set_index('date').sort_index()
        grp['rolling_avg_sentiment'] = grp['daily_sentiment'].rolling('7D').mean().round(3)
        grp = grp.reset_index()
        grp['ticker'] = ticker
        parts.append(grp[['ticker', 'date', 'rolling_avg_sentiment']])

    return pd.concat(parts, ignore_index=True)


def query_support_resistance(ticker: str, window: int = 5) -> dict:
    """Find support and resistance levels from price history."""
    conn = get_connection()
    query = """
        SELECT date, high, low, close
        FROM prices
        WHERE ticker = ?
        ORDER BY date
    """
    df = pd.read_sql_query(query, conn, params=(ticker,))
    conn.close()

    if df.empty or len(df) < window * 2:
        return {"support": [], "resistance": []}

    highs = []
    lows = []

    for i in range(window, len(df) - window):
        if df['high'].iloc[i] == df['high'].iloc[i-window:i+window].max():
            highs.append(round(df['high'].iloc[i], 2))
        if df['low'].iloc[i] == df['low'].iloc[i-window:i+window].min():
            lows.append(round(df['low'].iloc[i], 2))

    return {
        "support": sorted(set(lows)),
        "resistance": sorted(set(highs))
    }


def query_intraday_volume_zscore(ticker: str, date: str = None) -> pd.DataFrame:
    """
    Calculate intraday volume z-score per 5-min slot.
    Compares each slot's volume against the average for that
    same time slot across the past 20 trading days.
    """
    target = date or datetime.now().strftime("%Y-%m-%d")
    conn = get_connection()
    query = """
        WITH slot_baseline AS (
            SELECT
                ticker,
                STRFTIME('%H:%M', datetime) as time_slot,
                AVG(volume) as avg_slot_volume,
                AVG(volume * volume) - AVG(volume) * AVG(volume) as variance
            FROM intraday
            WHERE ticker = ?
            AND DATE(datetime) >= DATE(?, '-20 days')
            AND DATE(datetime) < DATE(?)
            GROUP BY ticker, time_slot
        )
        SELECT
            i.datetime,
            STRFTIME('%H:%M', i.datetime) as time_slot,
            i.volume,
            b.avg_slot_volume,
            CASE 
                WHEN b.variance > 0 
                THEN ROUND((i.volume - b.avg_slot_volume) / SQRT(b.variance), 2)
                ELSE 0
            END as volume_zscore
        FROM intraday i
        LEFT JOIN slot_baseline b 
            ON i.ticker = b.ticker 
            AND STRFTIME('%H:%M', i.datetime) = b.time_slot
        WHERE i.ticker = ?
        AND DATE(i.datetime) = DATE(?)
        ORDER BY i.datetime
    """
    df = pd.read_sql_query(
        query, conn,
        params=(ticker, target, target, ticker, target)
    )
    conn.close()
    return df


def main():
    # 1. Create schema
    create_schema()

    # 2. Load scored headlines CSV
    try:
        scored_df = pd.read_csv("data/headlines_scored.csv")
        print(f"[OK] Loaded {len(scored_df)} scored headlines")
    except FileNotFoundError:
        print("[ERROR] Run fetch_news.py and score_sentiment.py first")
        return

    # 3. Insert headlines and get url→id mapping
    url_to_id = insert_headlines(scored_df)

    # 4. Insert sentiment scores
    insert_sentiment(scored_df, url_to_id)

    # 5. Insert prices
    insert_prices()

    # 6. Insert intraday
    insert_intraday()

    # 7. Run and print queries

    print("\n--- DAILY SIGNAL CHECKLIST ---")
    checklist = query_checklist(date="2026-06-03")
    print(checklist.to_string(index=False))

    print("\n--- TOP MOVERS BY SENTIMENT SHIFT ---")
    movers = query_top_movers()
    print(movers.to_string(index=False))

    print("\n--- ROLLING SENTIMENT (7-day avg) ---")
    rolling = query_rolling_sentiment()
    print(rolling.to_string(index=False))


if __name__ == "__main__":
    main()