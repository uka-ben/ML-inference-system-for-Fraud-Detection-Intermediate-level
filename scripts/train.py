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