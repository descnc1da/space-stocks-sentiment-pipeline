# Financial News Sentiment Pipeline
A Python pipeline that fetches financial news headlines for a watchlist of stocks, 
scores them for sentiment using VADER and FinBERT, and stores results in SQLite 
for analysis.

## Projects in this Series
This is Project 1 of a 4-project portfolio built over 8 weeks.

## What It Does
- Fetches the latest headlines for 5 tickers via NewsAPI
- Scores each headline with VADER (fast) and FinBERT (accurate)
- Stores prices, headlines, and sentiment scores in a local SQLite database
- Surfaces top movers and rolling average sentiment per ticker

## Tech Stack
- Python 3.13.5
- NewsAPI · VADER · HuggingFace FinBERT
- SQLite · pandas
- python-dotenv

## Setup
1. Clone the repo  
2. Create a virtual environment: `python -m venv venv && source venv/bin/activate`  
3. Install dependencies: `pip install -r requirements.txt`  
4. Copy `.env.example` to `.env` and add your NewsAPI key  
5. Run: `python run_pipeline.py`

## Project Structure
fetch_news.py       — pulls headlines from NewsAPI
score_sentiment.py  — VADER + FinBERT scoring
database.py         — SQLite schema and queries
run_pipeline.py     — orchestrates the full pipeline
data/               — CSV and database output

## Architecture
*(diagram to be added)*

## Example Output
*(screenshot to be added after first full run)*

## Status
🟡 Week 2 in progress