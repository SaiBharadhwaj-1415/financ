import os
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import requests
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import fitz  # PyMuPDF
from pdf2image import convert_from_bytes
import pytesseract

from app.config import GEMINI_API_KEY, NEWS_API_KEY

# -------------------------------
# Trend Prediction
# -------------------------------
def run_trend_prediction(stock_symbol_or_name: str) -> str:
    # --- same logic as your function ---
    stock_symbol = stock_symbol_or_name.upper().strip() + '.NS'
    start_date_str = '2022-09-01'
    end_date = datetime.now()

    try:
        df = yf.download(stock_symbol, start=start_date_str, end=end_date, progress=False)
        if df.empty:
            return f"No data for '{stock_symbol}'."
    except Exception as e:
        return f"Error downloading data: {e}"

    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    data = df[features].fillna(method='ffill')
    dataset = data.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    time_step = 60
    training_data_len = int(np.ceil(len(scaled_data) * 0.95))
    train_data = scaled_data[0:training_data_len, :]
    test_data = scaled_data[training_data_len - time_step:, :]

    def create_dataset(dataset, time_step):
        x, y = [], []
        for i in range(time_step, len(dataset)):
            x.append(dataset[i - time_step:i, :])
            y.append(dataset[i, 3])
        return np.array(x), np.array(y)

    x_train, y_train = create_dataset(train_data, time_step)
    x_test, y_test = create_dataset(test_data, time_step)

    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], len(features))),
        Dropout(0.2),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(128, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(x_train, y_train, batch_size=50, epochs=200, validation_split=0.2, callbacks=[early_stopping], verbose=0)

    predictions = model.predict(x_test)
    dummy_array = np.zeros((len(predictions), len(features)))
    dummy_array[:, 3] = predictions.flatten()
    predictions = scaler.inverse_transform(dummy_array)[:, 3]

    trend_window = 5
    if len(predictions) >= trend_window:
        start_trend_price = predictions[-trend_window]
        end_trend_price = predictions[-1]
        price_change = end_trend_price - start_trend_price
        if price_change > 0:
            return f"Upward trend 📈 for {stock_symbol_or_name}"
        elif price_change < 0:
            return f"Downward trend 📉 for {stock_symbol_or_name}"
        else:
            return f"Neutral trend ↔ for {stock_symbol_or_name}"
    else:
        return "Not enough data to determine trend."

# -------------------------------
# Sentiment Analysis
# -------------------------------
try:
    model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
except Exception:
    sentiment_pipeline = None

def fetch_news(query):
    url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&pageSize=10&apiKey={NEWS_API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        articles = data.get("articles", [])
        return [article["title"] + ". " + article.get("description", "") for article in articles if article.get("description")]
    except Exception:
        return []

def run_sentiment_analysis(stock_name: str) -> str:
    if not sentiment_pipeline:
        return "Sentiment model not loaded."
    news_list = fetch_news(stock_name)
    if not news_list:
        return f"No news found for '{stock_name}'."
    results = []
    for article in news_list:
        sentiment = sentiment_pipeline(article)[0]
        results.append({"Sentiment": sentiment["label"], "Confidence": round(sentiment["score"], 2)})
    df = pd.DataFrame(results)
    if df.empty:
        return f"Could not calculate sentiment for '{stock_name}'."
    label_to_score = {"Positive": 1, "Negative": -1}
    df["Weighted_Score"] = df.apply(lambda row: label_to_score.get(row["Sentiment"], 0) * row["Confidence"], axis=1)
    avg_score = df["Weighted_Score"].mean() if not df.empty else 0
    return f"Average Sentiment Score: {avg_score:.2f}"

# -------------------------------
# Document Analysis
# -------------------------------
def analyze_document_file(file_bytes, filename) -> str:
    content = ""
    try:
        if filename.endswith(".pdf"):
            pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                content += page.get_text()
            pdf_document.close()
            if not content.strip():
                images = convert_from_bytes(file_bytes)
                for img in images:
                    content += pytesseract.image_to_string(img)
        else:
            try:
                content = file_bytes.decode("utf-8")
            except UnicodeDecodeError:
                content = file_bytes.decode("utf-8", errors="ignore")
    except Exception as e:
        return f"Error reading file: {e}"
    if not content.strip():
        return "File is empty or unreadable."
    return f"Document content length: {len(content)} characters"
