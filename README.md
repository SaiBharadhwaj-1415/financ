---
title: Financ
emoji: 📈
colorFrom: gray
colorTo: yellow
sdk: docker
pinned: false
---

# FinRobot – AI Powered Financial Assistant

FinRobot is a full-stack web application that helps users analyze stock market trends, extract summaries from financial documents, and detect sentiment from financial news. The project includes a React frontend, FastAPI backend, MongoDB Atlas database, and an LSTM deep learning model trained on historical stock data.

## 📄 Project Report (PDF)
Download here → FINROBOTAn-OPEN-SOURCE-AI-AGENT-PLATFORM.pdf  
(If GitHub preview fails, click **Download** to open it)

## ✅ Technologies Used
- React JS  
- FastAPI (Python)  
- MongoDB Atlas  
- yFinance  
- LSTM (TensorFlow/Keras)  
- Transformers, NLP models  
- PyMuPDF, Tesseract OCR for document processing

## ✅ Project Structure
- Frontend → React UI  
- Backend → FastAPI + ML models  
- Database → MongoDB Atlas

## ✅ Frontend Setup
Install Node.js and npm  
Then run:  
cd frontend  
npm install  
npm start  
Frontend runs on: http://localhost:3000

## ✅ Backend Setup
Install Python 3.9+  
Create and activate virtual environment  
cd backend  
python -m venv venv  
venv\Scripts\activate (Windows)  
source venv/bin/activate (Mac/Linux)  
pip install -r requirements.txt  
uvicorn main:app --reload  
Backend runs on: http://127.0.0.1:8000

## ✅ MongoDB Setup
- Create cluster in MongoDB Atlas  
- Whitelist IP or allow all  
- Copy connection string  
- Create .env and add:  
MONGO_URI="your atlas connection url"

## ✅ Major Features
✅ Stock trend prediction using LSTM  
✅ Sentiment analysis of financial news  
✅ Upload PDF → get summary  
✅ Interactive UI  
✅ REST API communication

## ✅ Run Entire Project
Backend → uvicorn main:app --reload  
Frontend → npm start  
Open → http://localhost:3000

## ✅ Deployment
- Backend → Hugging Face Spaces  
- Frontend → Vercel

This project is developed for learning and research purposes.  
Reference: https://huggingface.co/docs/hub/spaces-config-reference
