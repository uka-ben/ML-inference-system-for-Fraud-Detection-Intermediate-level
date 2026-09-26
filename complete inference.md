this is all i want

🎓 END-TO-END FRAUD DETECTION SYSTEM — COMPLETE TEACHING SCRIPT
📋 SYSTEM OVERVIEW
This is a production-ready fraud detection system that demonstrates:

What We're Building:
ML Model: Trained on synthetic transaction data to detect fraudulent transactions

REST API: Serves predictions via FastAPI with input validation

Container: Docker packages the app for consistent deployment

Orchestration: Kubernetes manages scaling, health, and updates

Monitoring: Prometheus tracks metrics for observability

CI/CD: GitHub Actions automates testing and deployment to GHCR

Why This Matters:
Fraud detection is a critical real-world ML use case

Production readiness separates junior from senior engineers

End-to-end knowledge makes you invaluable to teams

📂 PROJECT STRUCTURE EXPLAINED
text
fraud-inference/
│
├── app/                    # Application code
│   ├── config.py           # Configuration management
│   ├── schemas.py          # Data validation
│   ├── model_loader.py     # Model loading logic
│   ├── metrics.py          # Prometheus metrics
│   └── main.py             # FastAPI application
│
├── models/                  # Trained models
│   └── model.joblib        # Serialized model
│
├── scripts/                 # Utility scripts
│   └── train.py            # Model training
│
├── k8s/                     # Kubernetes manifests
│   ├── deployment.yaml      # Pod deployment
│   ├── service.yaml         # Service exposure
│   ├── hpa.yaml             # Auto-scaling
│   ├── prometheus.yaml      # Monitoring
│   ├── alert-rules.yaml     # Alerting rules
│   └── network-policy.yaml  # Security policies
│
├── .github/workflows/       # CI/CD pipeline
│   └── deploy.yml          # GitHub Actions
│
├── Dockerfile               # Container definition
├── requirements.txt         # Python dependencies
└── example_request.json     # Sample API request
🔵 PHASE 1: CORE APPLICATION
LESSON 1 — TRAINING THE MODEL
File: scripts/train.py

What This File Does:
Creates synthetic fraud data (no real sensitive data needed)

Trains a Logistic Regression model (interpretable, fast)

Handles class imbalance (fraud is rare ~5%)

Saves model using Joblib (efficient for scikit-learn)

python
# scripts/train.py

# -----------------------------
# IMPORT LIBRARIES
# -----------------------------
import os                    # File handling - create directories, check paths
import joblib                # Save/load models efficiently (better than pickle)
import pandas as pd          # Data manipulation - DataFrames for ML
from sklearn.datasets import make_classification  # Create synthetic dataset
from sklearn.linear_model import LogisticRegression  # Fast, interpretable model
from sklearn.model_selection import train_test_split  # Split data for evaluation
from sklearn.metrics import classification_report  # Precision, recall, f1-score

# -----------------------------
# DEFINE FEATURE NAMES
# -----------------------------
# These are the inputs our model will use to make predictions
feature_names = [
    "amount",                 # Transaction amount - higher amounts might be riskier
    "transaction_hour",       # Hour of transaction (0-23) - fraud often happens at night
    "merchant_risk_score",    # Risk score of merchant (0-1) - some merchants are risky
    "user_age",               # Age of user - different demographics have different patterns
    "account_tenure_days"     # How long account has existed - new accounts are riskier
]

# -----------------------------
# GENERATE SYNTHETIC DATASET
# -----------------------------
# In real life, you'd use actual transaction data
# But for learning, we create realistic fake data

X, y = make_classification(
    n_samples=5000,           # 5000 transactions
    n_features=5,              # 5 features (matches our feature_names)
    n_informative=5,           # All features are useful (no junk features)
    n_redundant=0,             # No redundant features (simplifies learning)
    weights=[0.95, 0.05],      # 95% legitimate, 5% fraud (class imbalance!)
    random_state=42            # Set seed for reproducibility
)

# Convert to DataFrame with named columns for readability
X = pd.DataFrame(X, columns=feature_names)

# -----------------------------
# SPLIT INTO TRAIN/TEST
# -----------------------------
# We need to evaluate on data the model hasn't seen
# 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# TRAIN MODEL
# -----------------------------
# class_weight="balanced" is CRITICAL for fraud detection
# It automatically gives more importance to the minority class (fraud)
# Without this, the model would just predict "not fraud" for everything
model = LogisticRegression(class_weight="balanced")
model.fit(X_train, y_train)

# -----------------------------
# EVALUATE MODEL
# -----------------------------
# classification_report shows precision, recall, f1-score
# For fraud, recall (finding fraud) is often more important than precision
print("✅ Model Evaluation on Test Set:")
print(classification_report(y_test, model.predict(X_test)))

# -----------------------------
# SAVE MODEL
# -----------------------------
# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# joblib is more efficient than pickle for scikit-learn models
# It handles large numpy arrays better
joblib.dump(model, "models/model.joblib")
print("✅ Model saved to models/model.joblib")
🎯 Teaching Points:
Concept	Explanation
make_classification	Simulates structured data for ML demos when real data isn't available
class_weight='balanced'	Automatically adjusts for imbalanced datasets - crucial for fraud where positive class is rare
train_test_split	Separates data to evaluate how well model generalizes to unseen data
classification_report	Shows precision, recall, f1-score - better than accuracy for imbalanced problems
joblib.dump	Efficient serialization for scikit-learn models, handles numpy arrays better than pickle
📊 Why Class Imbalance Matters:
Fraud is rare (typically 0.1-5% of transactions)

A model that predicts "not fraud" 100% of the time would be 95%+ accurate but completely useless

class_weight='balanced' gives more importance to fraud samples during training

We care more about recall (finding fraud) than precision in many cases

LESSON 2 — CONFIGURATION LAYER
File: app/config.py

Purpose:
Central place to store configurable paths, versions, environment variables. Makes code portable across environments (dev, staging, prod).

python
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
🎯 Teaching Points:
Concept	Explanation
Environment variables	Allow containerized deployment without code changes - same image works everywhere
Default values	Fallback for local development so you don't need to set env vars
Versioning	Critical for A/B testing, canary deployments, and rollbacks in production
💡 Production Reality:
In real systems, you'd also store: database connection strings, API keys, feature flags

Tools like Kubernetes ConfigMaps or HashiCorp Vault manage these securely

Never hardcode secrets!

LESSON 3 — SCHEMA VALIDATION
File: app/schemas.py

Purpose:
Ensure API inputs are valid before processing. Automatically generates API documentation at /docs.

python
# app/schemas.py

from pydantic import BaseModel, Field, field_validator
# BaseModel: parent class for all schemas
# Field: adds validation rules and metadata
# field_validator: custom validation logic

# -----------------------------
# INPUT SCHEMA
# -----------------------------
# Validates data coming INTO our API
class TransactionRequest(BaseModel):
    # Each field uses Field() to add validation
    # ... means required (no default value)
    # gt=0 means greater than 0
    amount: float = Field(
        ..., 
        gt=0, 
        description="Transaction amount must be positive"
    )
    
    # ge=0 means greater than or equal to 0
    # le=23 means less than or equal to 23
    transaction_hour: int = Field(
        ..., 
        ge=0, le=23, 
        description="Hour of day (0-23)"
    )
    
    merchant_risk_score: float = Field(
        ..., 
        ge=0, le=1, 
        description="Merchant risk score (0-1)"
    )
    
    user_age: float = Field(
        ..., 
        gt=0, le=120, 
        description="User age in years"
    )
    
    account_tenure_days: float = Field(
        ..., 
        ge=0, 
        description="Days account has existed"
    )

    # Custom validator for additional logic
    # Runs after the built-in validators
    @field_validator("amount")
    def check_amount(cls, value):
        # You could add business logic here
        # e.g., flag amounts > $10,000 for review
        if value <= 0:
            raise ValueError("Amount must be positive")
        return value

# -----------------------------
# OUTPUT SCHEMA
# -----------------------------
# Validates data LEAVING our API
# Ensures we don't accidentally send bad data to clients
class FraudResponse(BaseModel):
    fraud_probability: float = Field(
        ..., ge=0, le=1,
        description="Probability of fraud (0-1)"
    )
    is_fraud: int = Field(
        ..., ge=0, le=1,
        description="Binary prediction: 1=fraud, 0=legit"
    )
    model_version: str
    latency_seconds: float
🎯 Teaching Points:
Concept	Explanation
BaseModel	Parent class for all validated schemas - provides automatic validation
Field(...)	Adds validation rules and documentation (ellipsis ... means required)
gt/ge/lt/le	Greater than, greater or equal, less than, less or equal - prevents invalid data
@field_validator	Custom validation logic for complex business rules
Auto-docs	FastAPI uses these schemas to generate /docs endpoint automatically
💡 Why Validation Matters:
Prevents crashes: Bad input won't reach your model

Security: Blocks injection attempts and invalid data

Documentation: Automatically shows expected format to API consumers

Frontend teams: Can see exactly what to send without asking

LESSON 4 — MODEL LOADER
File: app/model_loader.py

Purpose:
Load the ML model once at startup (not per request) for optimal performance.

python
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
🎯 Teaching Points:
Concept	Explanation
Module-level loading	Model loaded once when app starts, shared across all requests
Fail fast	If model missing, container won't start - good for production debugging
try/except	Graceful error handling with clear messages
Global variable	model becomes available for import elsewhere
💡 Performance Impact:
Loading per request: ~500ms + model load time = terrible performance

Loading once: ~5ms per request (just inference)

100x speed improvement!

LESSON 5 — METRICS
File: app/metrics.py

Purpose:
Define Prometheus metrics for monitoring system performance in production.

python
# app/metrics.py

from prometheus_client import Counter, Histogram
# Counter: monotonically increasing value (total requests)
# Histogram: distribution of values (latency percentiles)

# -----------------------------
# REQUEST COUNTER
# -----------------------------
# Tracks total number of requests over time
# Useful for traffic patterns and load analysis
request_counter = Counter(
    "fraud_request_count",           # Metric name - used in PromQL queries
    "Total number of fraud prediction requests"  # Description - shows in UI
)

# -----------------------------
# LATENCY HISTOGRAM
# -----------------------------
# Tracks response time distribution
# buckets define ranges for percentiles (P95, P99)
latency_histogram = Histogram(
    "fraud_prediction_latency_seconds",
    "Latency of fraud predictions in seconds",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1]  # Pre-defined buckets
    # This allows calculating percentiles without storing all values
)

# -----------------------------
# FRAUD DETECTION COUNTER
# -----------------------------
# Tracks how many frauds we've caught
# Sudden spike might indicate a fraud campaign
fraud_counter = Counter(
    "fraud_detected_count",
    "Number of transactions flagged as fraud"
)
🎯 Teaching Points:
Concept	Explanation
Counter	Monotonically increasing counter (total requests, total errors)
Histogram	Distribution of values (latency, request sizes) with buckets for percentiles
Buckets	Pre-defined ranges for calculating percentiles without storing all data points
Metric naming	Convention: application_metric_unit (e.g., fraud_latency_seconds)
💡 Why These Metrics Matter:
Request count: Track traffic patterns, plan capacity, detect drops

Latency: Monitor performance degradation before users complain

Fraud count: Alert on sudden spikes (possible attack or model issue)

Prometheus scrapes these every 15s for real-time dashboards

LESSON 6 — FASTAPI MAIN APP
File: app/main.py

Purpose:
The main API application handling all endpoints, business logic, and integrating all components.

python
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
🎯 Teaching Points:
Concept	Explanation
start_http_server(8001)	Exposes metrics for Prometheus scraping on separate port
@app.get("/health")	Kubernetes uses this for pod health checks (liveness/readiness)
predict_proba	Returns probability for each class - more info than just binary
response_model=FraudResponse	Validates output format matches schema
HTTPException	Structured error responses with proper status codes
async def	Allows concurrent request handling for better performance
💡 Production Features:
CORS: Enables frontend apps to call API from different domains

Error handling: Graceful failure with proper status codes and messages

Latency tracking: Monitor performance and set alerts

Structured logging: In production, use logging library instead of print

LESSON 7 — DOCKERFILE
File: Dockerfile

Purpose:
Containerize the application for consistent deployment anywhere (dev, staging, prod).

dockerfile
# Use lightweight Python image (slim version = smaller size)
FROM python:3.10-slim

# Set environment variables for optimal Python in containers
# PYTHONDONTWRITEBYTECODE: Don't create .pyc files (saves space)
# PYTHONUNBUFFERED: Print logs immediately (not buffered)
# PIP_NO_CACHE_DIR: Don't cache pip downloads (smaller image)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Set working directory inside container
WORKDIR /app

# Copy requirements FIRST (leverage Docker layer caching)
# Docker caches each layer - if requirements.txt doesn't change,
# it reuses the cached layer instead of reinstalling
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
# This layer only rebuilds when code changes
COPY . .

# Create non-root user for security (never run as root in production)
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Document which ports the container listens on
EXPOSE 8000  # API port
EXPOSE 8001  # Metrics port

# Command to run when container starts
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
🎯 Teaching Points:
Concept	Explanation
python:3.10-slim	Smaller image = faster builds, smaller attack surface
ENV variables	Optimize Python behavior for containers
Layer caching	Copy requirements first - dependencies only rebuild when requirements change
Non-root user	Security best practice - limits damage if container compromised
EXPOSE	Documentation of container ports (doesn't actually publish them)
CMD	Default command to run - can be overridden
💡 Docker Benefits:
Consistency: Same environment everywhere (dev, test, prod)

Isolation: No dependency conflicts between apps

Scalability: Easy to run multiple instances

CI/CD: Automate builds and deployments

LESSON 8 — KUBERNETES MANIFESTS
Deployment (k8s/deployment.yaml)
Purpose: Defines how pods should run, their resources, and health checks.

yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fraud-inference
  labels:
    app: fraud-inference
spec:
  replicas: 2  # Run 2 pods for high availability
  selector:
    matchLabels:
      app: fraud-inference
  template:
    metadata:
      labels:
        app: fraud-inference
      annotations:
        prometheus.io/scrape: "true"   # Tell Prometheus to scrape this pod
        prometheus.io/port: "8001"      # Metrics port
    spec:
      containers:
      - name: fraud-container
        image: fraud-api:latest  # Will be replaced in CI/CD
        imagePullPolicy: IfNotPresent  # Only pull if not present
        ports:
        - containerPort: 8000
          name: http
        - containerPort: 8001
          name: metrics
        
        # Resource requests and limits
        resources:
          requests:
            cpu: "250m"      # Request 0.25 CPU (guaranteed minimum)
            memory: "256Mi"  # Request 256MB RAM
          limits:
            cpu: "500m"      # Max 0.5 CPU (can't exceed)
            memory: "512Mi"  # Max 512MB RAM
        
        # Environment variables
        env:
        - name: MODEL_PATH
          value: "models/model.joblib"
        
        # Readiness probe - pod ready for traffic?
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5   # Wait 5s before first check
          periodSeconds: 5          # Check every 5s
        
        # Liveness probe - pod healthy? Restart if fails
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 15   # Wait 15s before first check
          periodSeconds: 20          # Check every 20s
Service (k8s/service.yaml)
Purpose: Exposes pods internally or externally with a stable endpoint.

yaml
apiVersion: v1
kind: Service
metadata:
  name: fraud-service
spec:
  type: LoadBalancer  # Exposes externally (cloud provider gives IP)
  selector:
    app: fraud-inference  # Routes to pods with this label
  ports:
  - name: http
    protocol: TCP
    port: 80          # External port clients connect to
    targetPort: 8000  # Container port receiving traffic
  - name: metrics
    protocol: TCP
    port: 8001
    targetPort: 8001
Horizontal Pod Autoscaler (k8s/hpa.yaml)
Purpose: Automatically scales pods based on CPU/memory metrics.

yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: fraud-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: fraud-inference
  minReplicas: 2      # Never go below 2 pods
  maxReplicas: 10     # Never go above 10 pods
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70  # Scale when CPU > 70%
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80   # Scale when memory > 80%
Prometheus (k8s/prometheus.yaml)
Purpose: Deploys Prometheus for metrics collection and monitoring.

yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s  # Collect metrics every 15s
    scrape_configs:
    - job_name: 'fraud-api'
      static_configs:
      - targets: ['fraud-service:8001']  # Where to scrape
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prometheus
spec:
  replicas: 1
  selector:
    matchLabels:
      app: prometheus
  template:
    metadata:
      labels:
        app: prometheus
    spec:
      containers:
      - name: prometheus
        image: prom/prometheus:latest
        ports:
        - containerPort: 9090  # Prometheus UI
        volumeMounts:
        - name: config
          mountPath: /etc/prometheus
      volumes:
      - name: config
        configMap:
          name: prometheus-config
🎯 Teaching Points:
Concept	Explanation
replicas	Number of pod instances for high availability and load distribution
resources.requests	Minimum guaranteed resources - scheduler uses this for placement
resources.limits	Maximum allowed resources - prevents noisy neighbor problems
readinessProbe	Checks if pod is ready to receive traffic (removes from service if failing)
livenessProbe	Checks if pod is alive (restarts if failing)
Service	Stable network endpoint that load balances across pods
HPA	Auto-scales based on metrics - handles traffic spikes automatically
ConfigMap	Inject configuration without rebuilding images
💡 Kubernetes Benefits:
Self-healing: Restarts failed containers automatically

Scaling: Manual or automatic based on load

Rolling updates: Zero-downtime deployments

Service discovery: Internal DNS for services

Load balancing: Distributes traffic across pods

LESSON 9 — REQUIREMENTS
File: requirements.txt

txt
# API Framework
fastapi==0.104.1          # Modern, fast web framework
uvicorn[standard]==0.24.0 # ASGI server for FastAPI

# Data Validation
pydantic==2.4.2          # Data validation using Python types

# ML & Data Processing
scikit-learn==1.3.2      # ML library with LogisticRegression
pandas==2.1.3            # Data manipulation (DataFrames)
joblib==1.3.2            # Efficient model serialization

# Monitoring
prometheus-client==0.17.1 # Prometheus metrics export
🎯 Version Pinning:
Pinning exact versions ensures reproducibility across environments

Use >= for flexibility in some cases (but risky)

Regularly update for security patches

LESSON 10 — EXAMPLE REQUEST
File: example_request.json

json
{
  "amount": 2500.75,
  "transaction_hour": 23,
  "merchant_risk_score": 0.92,
  "user_age": 45,
  "account_tenure_days": 30
}
Test commands:

bash
# Test prediction endpoint
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @example_request.json

# Expected response format
{
  "fraud_probability": 0.87,
  "is_fraud": 1,
  "model_version": "1.0.0",
  "latency_seconds": 0.023
}

# Check Prometheus metrics
curl http://localhost:8001/metrics
LESSON 11 — RUN LOCALLY
bash
# 1. Create virtual environment (isolates dependencies)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train model
python scripts/train.py

# 4. Run API locally (--reload auto-restarts on code changes)
uvicorn app.main:app --reload --port 8000

# 5. Test in another terminal
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -d @example_request.json -H "Content-Type: application/json"
🔵 PHASE 2: PRODUCTION DEPLOYMENT WITH GHCR
LESSON 12 — GHCR SETUP & DEPLOYMENT
Why GHCR Instead of Docker Hub or ECR?
Free for public repositories

Integrated with GitHub (same credentials)

Fast global CDN

Perfect for CI/CD with GitHub Actions

bash
# ----------------------------------------
# STEP 1: CREATE GITHUB TOKEN
# ----------------------------------------
# Go to: GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
# Scopes needed: write:packages, read:packages, delete:packages
# Save token securely!

# ----------------------------------------
# STEP 2: LOGIN TO GHCR
# ----------------------------------------
echo $GITHUB_TOKEN | docker login ghcr.io -u $GITHUB_USERNAME --password-stdin
# This creates an entry in ~/.docker/config.json

# ----------------------------------------
# STEP 3: BUILD AND TAG IMAGE
# ----------------------------------------
# Build the Docker image
docker build -t fraud-api:latest .

# Tag for GHCR (format: ghcr.io/OWNER/REPO/IMAGE:TAG)
docker tag fraud-api:latest ghcr.io/$GITHUB_USERNAME/fraud-detection/fraud-api:latest

# ----------------------------------------
# STEP 4: PUSH TO GHCR
# ----------------------------------------
docker push ghcr.io/$GITHUB_USERNAME/fraud-detection/fraud-api:latest
# Image is now stored in GitHub's registry

# ----------------------------------------
# STEP 5: UPDATE KUBERNETES DEPLOYMENT
# ----------------------------------------
# Replace image in deployment.yaml with GHCR URL
sed -i "s|image:.*|image: ghcr.io/$GITHUB_USERNAME/fraud-detection/fraud-api:latest|g" k8s/deployment.yaml

# Apply to Kubernetes cluster
kubectl apply -f k8s/
🎯 Teaching Points:
Concept	Explanation
GHCR	GitHub Container Registry - free, integrated with GitHub
Image tagging	Format: registry/owner/repo/image:tag
docker login	Authenticates with registry using token
sed command	Stream editor - updates file in place (useful in CI/CD)
LESSON 13 — CI/CD PIPELINE WITH GITHUB ACTIONS
File: .github/workflows/deploy.yml

Purpose:
Automatically build, test, and deploy whenever code is pushed to main branch.

yaml
# .github/workflows/deploy.yml
name: Deploy to Production

# Trigger on pushes to main branch
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    # Grant permissions for GHCR
    permissions:
      contents: read
      packages: write  # Required for GHCR push
    
    steps:
      # Step 1: Check out the code
      - name: Checkout code
        uses: actions/checkout@v3
      
      # Step 2: Train and validate model
      - name: Train model
        run: python scripts/train.py
        # In real production, you'd also run tests here
      
      # Step 3: Login to GHCR
      - name: Login to GHCR
        run: echo "${{ secrets.GITHUB_TOKEN }}" | docker login ghcr.io -u ${{ github.actor }} --password-stdin
        # GITHUB_TOKEN is automatically provided by GitHub Actions
      
      # Step 4: Build and push multiple tags
      - name: Build and push
        run: |
          IMAGE=ghcr.io/${{ github.repository }}/fraud-api
          # Tag with 'latest', commit SHA, and date
          docker build -t $IMAGE:latest \
                      -t $IMAGE:${{ github.sha }} \
                      -t $IMAGE:$(date +%Y%m%d) .
          docker push $IMAGE:latest
          docker push $IMAGE:${{ github.sha }}
          docker push $IMAGE:$(date +%Y%m%d)
      
      # Step 5: Deploy to Kubernetes
      - name: Deploy to K8s
        run: |
          # Update deployment with specific commit SHA
          sed -i "s|image:.*|image: ghcr.io/${{ github.repository }}/fraud-api:${{ github.sha }}|g" k8s/deployment.yaml
          # Apply all manifests
          kubectl apply -f k8s/
          # Wait for rollout to complete
          kubectl rollout status deployment/fraud-inference --timeout=5m
🎯 Teaching Points:
Concept	Explanation
GitHub Actions	CI/CD built into GitHub - no separate CI server needed
on: [push]	Trigger workflow on code push
permissions	Grant workflow permissions (like pushing to GHCR)
secrets.GITHUB_TOKEN	Auto-generated token for GitHub API access
Multiple tags	Tag with latest, commit SHA, and date for traceability
rollout status	Wait for deployment to complete successfully
💡 CI/CD Benefits:
Automated: No manual steps, reduces human error

Consistent: Same process every time

Traceable: Every deployment linked to a commit

Fast: Catch issues immediately

LESSON 14 — PRODUCTION ENHANCEMENTS
Alerting Rules (k8s/alert-rules.yaml)
Purpose: Define when Prometheus should send alerts.

yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: fraud-alerts
spec:
  groups:
  - name: fraud-alerts
    rules:
    # Alert when error rate exceeds 5% for 5 minutes
    - alert: HighErrorRate
      expr: rate(fraud_request_count{status="500"}[5m]) > 0.05
      for: 5m  # Must be true for 5 minutes before alerting
      annotations:
        summary: "Error rate > 5%"
        description: "Error rate is {{ $value }}% for 5 minutes"
    
    # Alert when P95 latency exceeds 500ms
    - alert: HighLatency
      expr: histogram_quantile(0.95, rate(fraud_prediction_latency_seconds_bucket[5m])) > 0.5
      for: 5m
      annotations:
        summary: "P95 latency > 500ms"
        description: "P95 latency is {{ $value }}s"
    
    # Alert on fraud rate spike (potential attack)
    - alert: FraudRateSpike
      expr: (rate(fraud_detected_count[5m]) / rate(fraud_request_count[5m])) > 0.2
      for: 5m
      annotations:
        summary: "Fraud rate spike detected"
        description: "Fraud rate is {{ $value }}%"
Model Versioning Strategy
bash
# ----------------------------------------
# VERSION TAGGING STRATEGY
# ----------------------------------------

# Semantic version (v1.2.3) - for releases
docker tag fraud-api:latest ghcr.io/$USER/fraud-detection/fraud-api:v1.2.3
docker push ghcr.io/$USER/fraud-detection/fraud-api:v1.2.3

# Git commit SHA - for traceability
docker tag fraud-api:latest ghcr.io/$USER/fraud-detection/fraud-api:git-abc1234
docker push ghcr.io/$USER/fraud-detection/fraud-api:git-abc1234

# Date tag - for cleanup policies
docker tag fraud-api:latest ghcr.io/$USER/fraud-detection/fraud-api:2024-01-15
docker push ghcr.io/$USER/fraud-detection/fraud-api:2024-01-15

# Deploy specific version
kubectl set image deployment/fraud-inference \
  fraud-container=ghcr.io/$USER/fraud-detection/fraud-api:v1.2.3
Network Security (k8s/network-policy.yaml)
Purpose: Restrict traffic to/from pods for security.

yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: fraud-network-policy
spec:
  podSelector:
    matchLabels:
      app: fraud-inference
  policyTypes:
  - Ingress  # Incoming traffic rules
  - Egress   # Outgoing traffic rules
  ingress:
  # Allow only from ingress controller
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  egress:
  # Allow only to monitoring
  - to:
    - namespaceSelector:
        matchLabels:
          name: monitoring
    ports:
    - protocol: TCP
      port: 9090
LESSON 15 — MONITORING QUERIES
Purpose: PromQL queries to monitor system health in Grafana.

promql
# ----------------------------------------
# KEY METRICS FOR DASHBOARD
# ----------------------------------------

# Request rate (last 5 minutes)
# Shows traffic patterns
rate(fraud_request_count[5m])

# P95 latency (last 5 minutes)
# User experience metric - 95% of requests faster than this
histogram_quantile(0.95, rate(fraud_prediction_latency_seconds_bucket[5m]))

# Fraud rate percentage
# What % of transactions are flagged as fraud
(rate(fraud_detected_count[5m]) / rate(fraud_request_count[5m])) * 100

# Error rate percentage
# API errors - should be near 0
(rate(fraud_request_count{status="500"}[5m]) / rate(fraud_request_count[5m])) * 100

# Current pod count
# How many instances are running
count(kube_pod_status_ready{namespace="default", condition="true"})

# CPU usage per pod
# For resource planning
avg(rate(container_cpu_usage_seconds_total{namespace="default"}[5m])) by (pod)

# Memory usage per pod
sum(container_memory_usage_bytes{namespace="default"}) by (pod)
LESSON 16 — TROUBLESHOOTING GUIDE
bash
# ----------------------------------------
# COMMON ISSUES & SOLUTIONS
# ----------------------------------------

# Check pod logs (see what's happening inside)
kubectl logs -l app=fraud-inference --tail=50

# Check pod events (why pod isn't starting)
kubectl get events --sort-by='.lastTimestamp' | tail -10

# Test GHCR pull manually (verify image exists)
docker pull ghcr.io/$USER/fraud-detection/fraud-api:latest

# Rollback to previous version (when new version fails)
kubectl rollout undo deployment/fraud-inference

# Check resource usage (is pod hitting limits?)
kubectl top pods

# Port-forward Prometheus UI (check metrics are coming)
kubectl port-forward service/prometheus 9090:9090

# Check if model exists in container (build issue?)
kubectl exec deploy/fraud-inference -- ls -la models/

# Debug image pull issues (authentication problem?)
kubectl describe pod -l app=fraud-inference | grep -A10 "Events"

# Scale manually (handle sudden traffic)
kubectl scale deployment fraud-inference --replicas=5
LESSON 17 — PRODUCTION CHECKLIST
text
✅ PRE-DEPLOYMENT CHECKS
☐ GHCR token with write:packages scope
☐ Image tagged with version + git SHA
☐ Vulnerability scan passed (Trivy/Snyk)
☐ Model performance validated (precision/recall)
☐ Resource limits configured appropriately
☐ Health probes enabled and tested

✅ DEPLOYMENT CHECKS
☐ Rolling update strategy configured
☐ Image pull secret created (if private repo)
☐ Environment variables set correctly
☐ Secrets applied (API keys, etc.)
☐ Network policies set for security

✅ POST-DEPLOYMENT CHECKS
☐ Smoke tests passed (basic functionality)
☐ Metrics scraping verified in Prometheus
☐ Alerts configured and tested
☐ Logs shipping to central system
☐ Rollback plan documented

✅ SLA TARGETS (Service Level Agreements)
☐ P99 latency < 500ms
☐ Error rate < 1%
☐ GHCR pull success > 99%
☐ Availability > 99.9% ( < 8.76 hours downtime/year)
LESSON 18 — QUICK COMMANDS REFERENCE
bash
# ----------------------------------------
# DEVELOPMENT COMMANDS
# ----------------------------------------

# Train model
python scripts/train.py

# Run API locally with auto-reload
uvicorn app.main:app --reload --port 8000

# Test health endpoint
curl http://localhost:8000/health

# Test prediction
curl -X POST http://localhost:8000/predict -d @example_request.json -H "Content-Type: application/json"

# ----------------------------------------
# DOCKER COMMANDS
# ----------------------------------------

# Build image
docker build -t fraud-api:latest .

# Run container locally
docker run -p 8000:8000 -p 8001:8001 fraud-api:latest

# Tag for GHCR
docker tag fraud-api:latest ghcr.io/$USER/fraud-detection/fraud-api:latest

# Push to GHCR
docker push ghcr.io/$USER/fraud-detection/fraud-api:latest

# ----------------------------------------
# KUBERNETES COMMANDS
# ----------------------------------------

# Deploy all resources
kubectl apply -f k8s/

# Get status of all resources
kubectl get pods,svc,hpa

# Scale deployment
kubectl scale deployment fraud-inference --replicas=5

# Stream logs from all pods
kubectl logs -l app=fraud-inference -f

# Rollback to previous version
kubectl rollout undo deployment/fraud-inference

# Port-forward service to local machine
kubectl port-forward service/fraud-service 8000:80

# Exec into pod for debugging
kubectl exec -it deploy/fraud-inference -- /bin/bash

# ----------------------------------------
# GHCR COMMANDS
# ----------------------------------------

# List image versions via API
curl -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/user/packages/container/fraud-api/versions

# Delete old images (cleanup)
# Install ghcr-cleanup tool or use API



