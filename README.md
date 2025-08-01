# Project README
# MLOps Pipeline - Iris Classification

This repo demonstrates an end-to-end MLOps workflow using:
- Iris dataset
- MLflow for experiment tracking
- FastAPI for serving predictions
- Docker for containerization
- GitHub Actions for CI/CD
- SQLite for logging

## 🚀 Run Locally

```bash
docker build -t iris-api .
docker run -p 8000:8000 iris-api
