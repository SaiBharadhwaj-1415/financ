from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, EmailStr
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import google.generativeai as genai
import fitz  # PyMuPDF
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel as LangModel, Field as LangField
import requests
import os
from passlib.context import CryptContext
from typing import Dict, Optional, Any
from motor.motor_asyncio import AsyncIOMotorClient

# ==========================================================
# FastAPI Setup
# ==========================================================
app = FastAPI(title="Financial AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================================
# MongoDB Setup
# ==========================================================
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://Sai_1415:6tarAzYUctiEpe68@cluster0.nfpedva.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")

db_client: Optional[AsyncIOMotorClient] = None
db: Optional[Any] = None

@app.on_event("startup")
async def startup_db_client():
    global db_client, db
    try:
        db_client = AsyncIOMotorClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        db = db_client.Cluster0
        await db.list_collection_names()
        print("✅ Connected to MongoDB Atlas successfully!")
    except Exception as e:
        db_client = None
        db = None
        print(f"❌ Could not connect to MongoDB Atlas: {e}")

@app.on_event("shutdown")
async def shutdown_db_client():
    global db_client
    if db_client:
        db_client.close()
        print("✅ Disconnected from MongoDB Atlas.")

# ==========================================================
# Authentication Setup
# ==========================================================
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    """Hash password safely (truncate if >72 bytes)."""
    password = password[:72]
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password (truncate input to 72 bytes)."""
    plain_password = plain_password[:72]
    return pwd_context.verify(plain_password, hashed_password)

class UserIn(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6)

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_email: str

@app.post("/api/auth/signup", status_code=201)
async def signup(user_data: UserIn):
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection failed.")
    existing = await db.users.find_one({"email": user_data.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered.")
    hashed_password = hash_password(user_data.password)
    await db.users.insert_one({"email": user_data.email, "hashed_password": hashed_password})
    return {"message": "User registered successfully. Please log in."}

@app.post("/api/auth/login", response_model=Token)
async def login(user_data: UserIn):
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection failed.")
    user = await db.users.find_one({"email": user_data.email})
    if not user or not verify_password(user_data.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Incorrect email or password.")
    token = f"fake_token_for_{user_data.email}"
    return Token(access_token=token, user_email=user_data.email)

# ==========================================================
# API Keys
# ==========================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyD3Zlhgi_ElXTxzZgmA1EqI9ECroDhmjPM")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "6ef08eee56814dae9d9dab20cce0cacb")

genai.configure(api_key=GEMINI_API_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
extraction_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)

# ==========================================================
# Trend Prediction
# ==========================================================
def run_trend_prediction(stock_symbol_or_name: str) -> str:
    stock_symbol = stock_symbol_or_name.upper().strip() + ".NS"
    try:
        df = yf.download(stock_symbol, start="2022-09-01", end=datetime.now(), progress=False)
        if df.empty:
            return f"❌ No data found for {stock_symbol_or_name}."
    except Exception as e:
        return f"❌ Error downloading data: {e}"

    features = ["Open", "High", "Low", "Close", "Volume"]
    data = df[features].fillna(method="ffill")

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values)
    time_step = 60
    train_len = int(np.ceil(len(scaled_data) * 0.95))
    train = scaled_data[:train_len]
    test = scaled_data[train_len - time_step:]

    def make_data(dataset, step):
        x, y = [], []
        for i in range(step, len(dataset)):
            x.append(dataset[i-step:i])
            y.append(dataset[i, 3])
        return np.array(x), np.array(y)

    x_train, y_train = make_data(train, time_step)
    x_test, y_test = make_data(test, time_step)

    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], len(features))),
        Dropout(0.2),
        LSTM(128, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[early_stop], verbose=0)

    pred = model.predict(x_test)
    dummy = np.zeros((len(pred), len(features)))
    dummy[:, 3] = pred.flatten()
    pred = scaler.inverse_transform(dummy)[:, 3]

    if len(pred) >= 5:
        start, end = pred[-5], pred[-1]
        diff = end - start
        if diff > 0:
            return f"📈 {stock_symbol_or_name} is showing an upward trend."
        elif diff < 0:
            return f"📉 {stock_symbol_or_name} is showing a downward trend."
        else:
            return f"➡ {stock_symbol_or_name} looks stable."
    return "❌ Not enough data."

# ==========================================================
# Sentiment Analysis
# ==========================================================
try:
    sentiment_model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    sentiment_tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    sentiment_pipeline = pipeline("sentiment-analysis", model=sentiment_model, tokenizer=sentiment_tokenizer)
except Exception as e:
    print(f"Error loading FinBERT: {e}")
    sentiment_pipeline = None

def fetch_news(query):
    url = f"https://newsapi.org/v2/everything?q={query}&language=en&pageSize=10&apiKey={NEWS_API_KEY}"
    try:
        data = requests.get(url).json()
        return [a["title"] + ". " + a.get("description", "") for a in data.get("articles", []) if a.get("description")]
    except Exception as e:
        print(f"News fetch error: {e}")
        return []

def run_sentiment_analysis(stock_name: str) -> str:
    if not sentiment_pipeline:
        return "❌ Sentiment model not loaded."
    news_list = fetch_news(stock_name)
    if not news_list:
        return f"No news for {stock_name}."
    results = [sentiment_pipeline(n)[0] for n in news_list]
    df = pd.DataFrame(results)
    avg_score = np.mean([r["score"] if r["label"] == "Positive" else -r["score"] for r in results])
    if avg_score > 0.2: mood = "Positive ✅"
    elif avg_score < -0.2: mood = "Negative ❌"
    else: mood = "Neutral 😐"
    return f"📊 Sentiment for {stock_name}: {mood} ({avg_score:.2f})"

# ==========================================================
# Document Upload
# ==========================================================
def analyze_document(content: str) -> str:
    if not content.strip():
        return "❌ Empty document."
    model = genai.GenerativeModel("gemini-2.5-flash")
    resp = model.generate_content(f"Summarize this financial document:\n{content}")
    return resp.text

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    if file.filename.endswith(".pdf"):
        doc = fitz.open(stream=content, filetype="pdf")
        text = "".join(page.get_text() for page in doc)
    else:
        text = content.decode("utf-8", errors="ignore")
    result = analyze_document(text)
    await db.uploads.insert_one({"filename": file.filename, "timestamp": datetime.now()})
    return {"title": "Document Analysis", "body": result}

# ==========================================================
# Dashboard
# ==========================================================
@app.get("/api/dashboard")
async def dashboard():
    total_q = await db.queries.count_documents({})
    uploads = await db.uploads.count_documents({})
    return {
        "totalQueries": total_q,
        "documentsAnalyzed": uploads,
        "status": "✅ Running fine"
    }

@app.get("/")
def root():
    return {"message": "✅ Financial AI Backend Running Successfully!"}
