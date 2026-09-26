# app/main.py - COMPLETE WORKING VERSION

import time
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import start_http_server
from app.schemas import TransactionRequest, FraudResponse
from app.model_loader import model
from app.metrics import request_counter, latency_histogram, fraud_counter
from app.config import MODEL_VERSION

# CRITICAL: This creates the app instance that uvicorn looks for
app = FastAPI(
    title="Fraud Detection API",
    description="ML-powered fraud detection for financial transactions",
    version=MODEL_VERSION
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint
@app.get("/")
def root():
    return {
        "message": "Fraud Detection API Running",
        "version": MODEL_VERSION,
        "status": "operational"
    }

# Health check endpoint
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_version": MODEL_VERSION,
        "model_loaded": model is not None
    }

# Prediction endpoint
@app.post("/predict", response_model=FraudResponse)
async def predict(request: TransactionRequest):
    request_counter.inc()
    start_time = time.time()
    
    try:
        input_data = pd.DataFrame([[
            request.amount,
            request.transaction_hour,
            request.merchant_risk_score,
            request.user_age,
            request.account_tenure_days
        ]], columns=["amount", "transaction_hour", "merchant_risk_score", 
                     "user_age", "account_tenure_days"])
        
        prob = model.predict_proba(input_data)[0][1]
        prediction = 1 if prob >= 0.5 else 0
        
        if prediction == 1:
            fraud_counter.inc()
        
        latency = time.time() - start_time
        latency_histogram.observe(latency)
        
        return FraudResponse(
            fraud_probability=float(prob),
            is_fraud=prediction,
            model_version=MODEL_VERSION,
            latency_seconds=latency
        )
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")