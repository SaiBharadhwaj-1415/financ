from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from transformers import pipeline
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

# -------------------- App Setup --------------------
app = FastAPI(title="Financial AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- Database Setup --------------------
MONGO_URI = os.getenv(
    "MONGO_URI",
    "mongodb+srv://Sai_1415:6tarAzYUctiEpe68@cluster0.nfpedva.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
)

db_client: Optional[AsyncIOMotorClient] = None
db: Optional[AsyncIOMotorClient] = None

@app.on_event("startup")
async def startup_db_client():
    """Connect to MongoDB Atlas."""
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
    """Close the MongoDB connection."""
    global db_client
    if db_client:
        db_client.close()
        print("✅ Disconnected from MongoDB Atlas.")

# -------------------- Authentication --------------------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    """Hash password safely."""
    password = password[:72]
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify hashed password safely."""
    plain_password = plain_password[:72]
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception as e:
        print(f"⚠️ Password verification failed: {e}")
        return False

class UserIn(BaseModel):
    email: str = Field(..., example="test@example.com")
    password: str = Field(..., min_length=6)

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_email: str

@app.post("/api/auth/signup", status_code=201)
async def signup(user_data: UserIn):
    """Register a new user."""
    if db is None:
        return {"error": "Database connection failed."}

    try:
        existing_user = await db.users.find_one({"email": user_data.email.strip().lower()})
        if existing_user:
            return {"error": "Email already registered."}

        hashed_password = hash_password(user_data.password)
        new_user = {
            "email": user_data.email.strip().lower(),
            "hashed_password": hashed_password,
            "created_at": datetime.utcnow()
        }
        await db.users.insert_one(new_user)
        return {"message": "✅ User registered successfully. Please log in."}
    except Exception as e:
        print(f"Signup error: {e}")
        return {"error": "Signup failed. Please try again later."}

@app.post("/api/auth/login", response_model=Token)
async def login(user_data: UserIn):
    """Authenticate a user."""
    if db is None:
        return {"error": "Database connection failed."}

    try:
        user = await db.users.find_one({"email": user_data.email.strip().lower()})
        if not user or not verify_password(user_data.password, user["hashed_password"]):
            return {"error": "Incorrect email or password."}

        token = f"token_{user_data.email}_{int(datetime.utcnow().timestamp())}"
        return Token(access_token=token, user_email=user_data.email)
    except Exception as e:
        print(f"Login error: {e}")
        return {"error": "Login failed. Please try again later."}

# -------------------- LangChain Tools --------------------
@tool
def fetch_stock_price(symbol: str) -> str:
    """Fetch the latest stock price for a given symbol."""
    try:
        data = yf.Ticker(symbol)
        price = data.history(period="1d")["Close"].iloc[-1]
        return f"The current price of {symbol.upper()} is ${price:.2f}."
    except Exception as e:
        return f"Failed to fetch stock price: {e}"

@tool
def analyze_sentiment(text: str) -> str:
    """Analyze sentiment of a given text using Hugging Face pipeline."""
    try:
        sentiment_analyzer = pipeline("sentiment-analysis")
        result = sentiment_analyzer(text)[0]
        return f"Sentiment: {result['label']} (confidence {result['score']:.2f})"
    except Exception as e:
        return f"Sentiment analysis failed: {e}"

# -------------------- Google GenAI --------------------
genai.configure(api_key=os.getenv("GOOGLE_API_KEY", ""))

@app.post("/api/query")
async def query_ai(prompt: str = Form(...)):
    """Query Google Generative AI model for insights."""
    try:
        chat = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        response = chat.invoke(prompt)
        return {"response": response}
    except Exception as e:
        return {"error": f"AI query failed: {e}"}

# -------------------- Dashboard Endpoint --------------------
@app.get("/api/dashboard")
async def get_dashboard_stats():
    if db is None:
        return {"error": "Database not available."}

    # 1. Get total counts
    total_queries = await db.queries.count_documents({})
    documents_analyzed = await db.uploads.count_documents({})

    # 2. Positive sentiment count
    positive_sentiments = await db.queries.count_documents({"result": {"$regex": "positive", "$options": "i"}})

    # 3. Trend predictions count
    trend_predictions = await db.queries.count_documents({"mode": "trend"})

    # 4. Get recent queries and uploads
    queries_cursor = db.queries.find({}, {"_id": 0}).sort("timestamp", -1).limit(5)
    recent_queries = await queries_cursor.to_list(length=5)

    uploads_cursor = db.uploads.find({}, {"_id": 0}).sort("timestamp", -1).limit(5)
    recent_uploads = await uploads_cursor.to_list(length=5)

    all_recent = sorted(
        recent_queries + recent_uploads,
        key=lambda x: x.get("timestamp", datetime.min),
        reverse=True
    )

    recent = []
    for item in all_recent[:5]:
        if "query" in item:
            status_text = item.get("result", "")
            recent.append({
                "type": "Trend" if item.get("mode") == "trend" else "Sentiment",
                "label": item.get("query", ""),
                "status": status_text[:40] + "..." if len(status_text) > 40 else status_text,
                "color": (
                    "text-yellow-600" if item.get("mode") == "trend" else
                    "text-green-600" if "positive" in status_text.lower() else
                    "text-red-600"
                )
            })
        else:
            recent.append({
                "type": "Document",
                "label": item.get("filename", "Unknown File"),
                "status": "Summarized",
                "color": "text-indigo-600"
            })

    return {
        "totalQueries": total_queries,
        "positiveSentiment": f"{(positive_sentiments / total_queries * 100):.0f}%" if total_queries > 0 else "0%",
        "trendPredictions": f"{trend_predictions}/{total_queries}",
        "documentsAnalyzed": documents_analyzed,
        "recent": recent
    }

# -------------------- Health Check --------------------
@app.get("/")
async def root():
    """Simple health check endpoint."""
    return {"status": "Backend running successfully ✅"}
