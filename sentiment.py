import requests
import pandas as pd
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

api_key = "6ef08eee56814dae9d9dab20cce0cacb"

try:
    model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
except:
    sentiment_pipeline = None

def fetch_news(query, api_key):
    url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&pageSize=10&apiKey={api_key}"
    try:
        res = requests.get(url)
        res.raise_for_status()
        data = res.json()
        return [a['title'] + '. ' + a.get('description','') for a in data.get('articles', []) if a.get('description')]
    except:
        return []

def run_sentiment_analysis(stock_name: str) -> str:
    if not sentiment_pipeline:
        return "Sentiment model not loaded."
    
    news_list = fetch_news(stock_name, api_key)
    if not news_list:
        return f"No news articles found for '{stock_name}'."
    
    results = []
    for article in news_list:
        try:
            sentiment = sentiment_pipeline(article)[0]
            results.append({"Sentiment": sentiment["label"], "Confidence": round(sentiment["score"], 2)})
        except:
            continue
    
    df = pd.DataFrame(results)
    if df.empty:
        return f"Could not calculate sentiment for '{stock_name}'."

    label_to_score = {"Positive": 1, "Negative": -1}
    df["Weighted_Score"] = df.apply(lambda r: label_to_score.get(r["Sentiment"],0)*r["Confidence"], axis=1)
    avg_score = df["Weighted_Score"].mean() if not df.empty else 0
    
    if avg_score >= 0.6: overall = "Strongly Positive ✅"
    elif avg_score > 0.3: overall = "Moderately Positive 👍"
    elif avg_score > 0.1: overall = "Slightly Positive ☀"
    elif avg_score > -0.1: overall = "Mixed or Weak 😐"
    elif avg_score > -0.3: overall = "Slightly Negative 🌧"
    elif avg_score > -0.6: overall = "Moderately Negative ⚠"
    else: overall = "Strongly Negative ❌"
    
    return f"Avg Sentiment Score for '{stock_name.title()}': {avg_score:.2f}\nOverall Market Sentiment: {overall}"
