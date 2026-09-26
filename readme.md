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