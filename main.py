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

# -------------------- Health Check --------------------
@app.get("/")
async def root():
    """Simple health check endpoint."""
    return {"status": "Backend running successfully ✅"}
