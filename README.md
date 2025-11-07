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
➡️ **[Download Project Report](./FINROBOTAn-OPEN-SOURCE-AI-AGENT-PLATFORM.pdf)**
*(If GitHub preview fails, click download to open it.)*

## Technologies Used
- React JS
- FastAPI (Python)
- MongoDB Atlas
- yFinance
- LSTM (TensorFlow/Keras)
- Transformers and NLP models
- PyMuPDF, Tesseract OCR for document processing

## Project Structure
- **Frontend:** React UI
- **Backend:** FastAPI API + ML models
- **Database:** MongoDB Atlas for storing user data and information

## Frontend Setup
1. Install Node.js and npm
2. Run:
    cd frontend
    npm install
    npm start
Frontend runs on:
http://localhost:3000

## Backend Setup
1. Install Python 3.9+
2. Create and activate virtual environment
3. Install packages and run backend

Commands:
    cd backend
    python -m venv venv
    venv\Scripts\activate   (Windows)
    source venv/bin/activate  (Mac/Linux)
    pip install -r requirements.txt
    uvicorn main:app --reload
Backend runs on:
http://127.0.0.1:8000

## MongoDB Atlas Setup
- Create cluster in MongoDB Atlas
- Whitelist IP or allow 0.0.0.0
- Copy connection string
- Create `.env` file in backend:
    MONGO_URI="your atlas connection url"

## Major Features
✅ Predict stock trends using LSTM  
✅ Sentiment analysis of financial news and queries  
✅ Upload PDFs and receive summarized results  
✅ Interactive React interface  
✅ REST API communication between frontend and backend

## How to Run Entire Project
1. Start backend:
    uvicorn main:app --reload
2. Start frontend:
    npm start
3. Open in browser:
    http://localhost:3000

## Deployment
- Backend deployed on Hugging Face
- Frontend deployed on Vercel

This project is developed for learning and research purposes.
Check out the configuration reference:
https://huggingface.co/docs/hub/spaces-config-reference
