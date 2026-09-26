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