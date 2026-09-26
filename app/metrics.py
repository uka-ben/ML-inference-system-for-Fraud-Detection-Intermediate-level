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