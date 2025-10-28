from fastapi import FastAPI, UploadFile, File, Form
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
from motor.motor_asyncio import AsyncIOMotorClient  # MongoDB async client

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


# -------------------- Authentication Setup --------------------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


class UserIn(BaseModel):
    email: str = Field(..., example="test@example.com")
    password: str = Field(..., min_length=6)


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_email: str


# -------------------- FIXED SIGNUP & LOGIN --------------------
@app.post("/api/auth/signup", status_code=201)
async def signup(user_data: UserIn):
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
    if db is None:
        return {"error": "Database connection failed."}

    try:
        user = await db.users.find_one({"email": user_data.email.strip().lower()})
        if not user:
            return {"error": "User not found. Please sign up first."}

        if not verify_password(user_data.password, user["hashed_password"]):
            return {"error": "Incorrect password."}

        token = f"token_{user_data.email}_{int(datetime.utcnow().timestamp())}"
        return Token(access_token=token, user_email=user_data.email)
    except Exception as e:
        print(f"Login error: {e}")
        return {"error": "Login failed. Please try again later."}


# -------------------- API Keys --------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyD3Zlhgi_ElXTxzZgmA1EqI9ECroDhmjPM")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "6ef08eee56814dae9d9dab20cce0cacb")

genai.configure(api_key=GEMINI_API_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=GEMINI_API_KEY)
extraction_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1, google_api_key=GEMINI_API_KEY)

# -------------------- Trend Prediction --------------------
def run_trend_prediction(stock_symbol_or_name: str) -> str:
    stock_symbol = stock_symbol_or_name.upper().strip() + ".NS"
    print(f"📞 Running trend prediction for {stock_symbol_or_name} ({stock_symbol})")
    start_date_str = "2022-09-01"
    end_date = datetime.now()

    try:
        df = yf.download(stock_symbol, start=start_date_str, end=end_date, progress=False)
        if df.empty:
            return f"❌ No data returned for '{stock_symbol}'."
    except Exception as e:
        return f"❌ Error downloading data for {stock_symbol}: {e}"

    features = ["Open", "High", "Low", "Close", "Volume"]
    data = df[features]
    data.fillna(method="ffill", inplace=True)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values)
    time_step = 60
    training_data_len = int(np.ceil(len(scaled_data) * 0.95))
    train_data = scaled_data[:training_data_len]
    test_data = scaled_data[training_data_len - time_step:]

    def create_dataset(dataset, time_step):
        x, y = [], []
        for i in range(time_step, len(dataset)):
            x.append(dataset[i - time_step:i])
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
    model.compile(optimizer="adam", loss="mean_squared_error")
    early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    model.fit(x_train, y_train, batch_size=50, epochs=200, validation_split=0.2, callbacks=[early_stopping], verbose=0)

    predictions = model.predict(x_test)
    dummy_array = np.zeros((len(predictions), len(features)))
    dummy_array[:, 3] = predictions.flatten()
    predictions = scaler.inverse_transform(dummy_array)[:, 3]

    trend_window = 5
    if len(predictions) >= trend_window:
        start_price, end_price = predictions[-trend_window], predictions[-1]
        diff = end_price - start_price
        if diff > 0:
            return f"Based on the last {trend_window} days, {stock_symbol_or_name} is showing an 📈 Upward trend."
        elif diff < 0:
            return f"Based on the last {trend_window} days, {stock_symbol_or_name} is showing a 📉 Downward trend."
        else:
            return f"The price of {stock_symbol_or_name} is ↔ Stable."
    return f"❌ Not enough data for {stock_symbol_or_name}."


# -------------------- Sentiment Analysis --------------------
try:
    model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
except Exception as e:
    print(f"Error loading FinBERT: {e}")
    sentiment_pipeline = None


def fetch_news(query):
    url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&pageSize=10&apiKey={NEWS_API_KEY}"
    try:
        res = requests.get(url)
        res.raise_for_status()
        data = res.json()
        return [a["title"] + ". " + a.get("description", "") for a in data.get("articles", []) if a.get("description")]
    except Exception as e:
        print(f"Error fetching news: {e}")
        return []


def run_sentiment_analysis(stock_name: str) -> str:
    if not sentiment_pipeline:
        return "❌ Sentiment model not loaded."
    news_list = fetch_news(stock_name)
    if not news_list:
        return f"❌ No news articles found for '{stock_name}'."

    results = []
    for article in news_list:
        s = sentiment_pipeline(article)[0]
        results.append({"Sentiment": s["label"], "Confidence": round(s["score"], 2)})
    df = pd.DataFrame(results)
    label_to_score = {"Positive": 1, "Negative": -1}
    df["Weighted_Score"] = df.apply(lambda r: label_to_score.get(r["Sentiment"], 0) * r["Confidence"], axis=1)
    avg_score = df["Weighted_Score"].mean()

    if avg_score >= 0.6:
        overall = "Strongly Positive ✅"
    elif avg_score > 0.3:
        overall = "Moderately Positive 👍"
    elif avg_score > 0.1:
        overall = "Slightly Positive ☀"
    elif avg_score > -0.1:
        overall = "Mixed 😐"
    elif avg_score > -0.3:
        overall = "Slightly Negative 🌧"
    elif avg_score > -0.6:
        overall = "Moderately Negative ⚠"
    else:
        overall = "Strongly Negative ❌"

    return f"📊 Avg Sentiment Score for '{stock_name.title()}': {avg_score:.2f}\n🧠 Overall Market Sentiment: {overall}"


# -------------------- Document Upload --------------------
def analyze_document(document_content: str) -> str:
    if not document_content:
        return "❌ Please provide text content."
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(f"Summarize and analyze this document:\n{document_content}")
    return f"### Document Analysis Result:\n{response.text}"


# -------------------- Symbol Extraction --------------------
class StockSymbol(LangModel):
    stock_name: str = LangField(description="Clean, uppercase Indian stock ticker symbol (NSE or BSE).")


def extract_and_standardize_symbol(query: str) -> str:
    structured_llm = extraction_llm.with_structured_output(StockSymbol)
    try:
        result = structured_llm.invoke({"text": query})
        return result.stock_name.strip().upper()
    except Exception as e:
        print(f"Error extracting symbol: {e}")
        return ""


# -------------------- Tools --------------------
@tool
def sentiment_analysis_tool(stock_name: str) -> str:
    return run_sentiment_analysis(stock_name)


@tool
def trend_prediction_tool(stock_symbol: str) -> str:
    return run_trend_prediction(stock_symbol)


@tool
def upload_document_tool() -> str:
    return "Upload files using /api/upload."


tools = [sentiment_analysis_tool, trend_prediction_tool, upload_document_tool]
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a financial assistant that analyzes sentiment, predicts trends, or summarizes documents."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# -------------------- API Endpoints --------------------
class QueryRequest(BaseModel):
    query: str


@app.post("/api/query")
async def process_query(req: QueryRequest):
    if db is None:
        return {"title": "Error", "body": "Database not available."}

    user_query = req.query
    user_query_lower = user_query.lower()

    if any(word in user_query_lower for word in ["trend", "predict", "forecast", "lstm"]):
        mode = "trend"
        symbol = extract_and_standardize_symbol(user_query)
        if symbol:
            user_query = f"Predict the stock trend for {symbol}"
    elif any(word in user_query_lower for word in ["sentiment", "analyze", "analysis", "feeling", "emotion"]):
        mode = "sentiment"
    else:
        mode = "sentiment"

    result = agent_executor.invoke({"input": user_query})
    output_text = result["output"]

    query_log = {
        "query": user_query,
        "mode": mode,
        "result": output_text,
        "timestamp": datetime.now(),
    }
    await db.queries.insert_one(query_log)

    return {"title": "Response", "body": output_text}


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    if db is None:
        return {"title": "Error", "body": "Database not available."}

    content = await file.read()
    text = ""
    if file.filename.endswith(".pdf"):
        doc = fitz.open(stream=content, filetype="pdf")
        for page in doc:
            text += page.get_text()
    else:
        text = content.decode("utf-8", errors="ignore")

    if not text.strip():
        return {"title": "Error", "body": "❌ No readable text found."}

    result = analyze_document(text)
    upload_log = {"filename": file.filename, "timestamp": datetime.now()}
    await db.uploads.insert_one(upload_log)

    return {"title": "Document Analysis", "body": result}


@app.get("/api/dashboard")
async def get_dashboard_stats():
    if db is None:
        return {"error": "Database not available."}

    total_queries = await db.queries.count_documents({})
    documents_analyzed = await db.uploads.count_documents({})
    positive_sentiments = await db.queries.count_documents({"result": {"$regex": "positive", "$options": "i"}})
    trend_predictions = await db.queries.count_documents({"mode": "trend"})

    queries_cursor = db.queries.find({}, {"_id": 0}).sort("timestamp", -1).limit(5)
    recent_queries = await queries_cursor.to_list(length=5)

    uploads_cursor = db.uploads.find({}, {"_id": 0}).sort("timestamp", -1).limit(5)
    recent_uploads = await uploads_cursor.to_list(length=5)

    all_recent = sorted(
        recent_queries + recent_uploads,
        key=lambda x: x.get("timestamp", datetime.min),
        reverse=True,
    )

    recent = []
    for item in all_recent[:5]:
        if "query" in item:
            status_text = item["result"]
            recent.append({
                "type": "Trend" if item["mode"] == "trend" else "Sentiment",
                "label": item["query"],
                "status": status_text[:40] + "..." if len(status_text) > 40 else status_text,
                "color": (
                    "text-yellow-600" if item["mode"] == "trend" else
                    "text-green-600" if "positive" in status_text.lower() else
                    "text-red-600"
                )
            })
        else:
            recent.append({
                "type": "Document",
                "label": item["filename"],
                "status": "Summarized",
                "color": "text-indigo-600"
            })

    return {
        "totalQueries": total_queries,
        "positiveSentiment": f"{(positive_sentiments / total_queries * 100):.0f}%" if total_queries > 0 else "0%",
        "trendPredictions": f"{trend_predictions}/{total_queries}",
        "documentsAnalyzed": documents_analyzed,
        "recent": recent,
    }


@app.get("/")
def root():
    return {"message": "✅ Financial AI Backend Running Successfully!"}
