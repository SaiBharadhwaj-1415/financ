from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel
from services.trend import run_trend_prediction
from services.sentiment import run_sentiment_analysis
from services.document import analyze_document
from utils.stock_extractor import extract_and_standardize_symbol

router = APIRouter()

class QueryRequest(BaseModel):
    mode: str
    query: str

@router.post("/query")
async def handle_query(req: QueryRequest):
    mode = req.mode.lower()
    user_query = req.query.strip()
    
    if mode == "trend":
        symbol = extract_and_standardize_symbol(user_query)
        if not symbol:
            return {"title": "Error", "body": "Could not extract stock symbol from query."}
        result = run_trend_prediction(symbol)
        return {"title": f"Trend Prediction for {symbol}", "body": result}
    
    elif mode == "sentiment":
        result = run_sentiment_analysis(user_query)
        return {"title": f"Sentiment Analysis for {user_query}", "body": result}
    
    else:
        return {"title": "Error", "body": f"Unsupported mode: {mode}"}

@router.post("/upload-document")
async def upload_document(file: UploadFile = File(...)):
    content = await file.read()
    try:
        text_result = analyze_document(content, filename=file.filename)
        return {"title": f"Document Analysis: {file.filename}", "body": text_result}
    except Exception as e:
        return {"title": "Error", "body": str(e)}