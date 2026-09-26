# app/config.py

import os

# -----------------------------
# MODEL PATH CONFIGURATION
# -----------------------------
# os.getenv() reads from environment variables
# This allows us to change the path without code changes
# In Docker/K8s, we can set MODEL_PATH=/app/models/prod-model.joblib
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
# First argument: environment variable name
# Second argument: default value if env var not set

# -----------------------------
# MODEL VERSION TRACKING
# -----------------------------
# Critical for production rollbacks and A/B testing
# If new model performs worse, we know which version to roll back to
MODEL_VERSION = "1.0.0"