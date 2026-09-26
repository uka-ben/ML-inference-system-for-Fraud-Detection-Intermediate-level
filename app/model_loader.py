# app/model_loader.py

import joblib
from app.config import MODEL_PATH

# -----------------------------
# LOAD MODEL AT STARTUP
# -----------------------------
# This runs when the module is imported (once at app startup)
# NOT per request - critical for performance!

print(f"📦 Loading model from: {MODEL_PATH}")

try:
    # joblib.load() deserializes the model
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully")
    
except FileNotFoundError:
    # Fail fast - better to crash at startup than fail during requests
    print(f"❌ Model not found at {MODEL_PATH}")
    raise  # Re-raise exception - container won't start

# model is now available for import in other files
# from app.model_loader import model