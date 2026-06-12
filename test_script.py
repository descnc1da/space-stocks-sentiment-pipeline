from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

headlines = [
    "Apple beats earnings expectations by wide margin",
    "Tesla recalls 200,000 vehicles over safety concerns",
    "Microsoft announces layoffs amid restructuring",
    "Amazon stock surges after strong quarterly results"
]

for headline in headlines:
    score = analyzer.polarity_scores(headline)
    print(f"{score['compound']:+.2f} | {headline}")