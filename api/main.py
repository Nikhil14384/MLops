from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import PlainTextResponse
import joblib
import pandas as pd
import sqlite3
import time
from prometheus_client import Counter, generate_latest

app = FastAPI()
model = joblib.load("models/best_model.pkl")

# Pydantic input schema
class IrisInput(BaseModel):
    sepal_len: float
    sepal_wid: float
    petal_len: float
    petal_wid: float

# Prometheus counter
PREDICTION_COUNT = Counter("prediction_requests_total", "Total prediction requests")

# SQLite logging
def log_prediction(data, prediction):
    conn = sqlite3.connect("logs/predictions.db")
    cursor = conn.cursor()
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS predictions (timestamp TEXT, sepal_len REAL, sepal_wid REAL, petal_len REAL, petal_wid REAL, prediction INTEGER)"
    )
    cursor.execute(
        "INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?)",
        (time.strftime("%Y-%m-%d %H:%M:%S"), *data, prediction)
    )
    conn.commit()
    conn.close()

@app.post("/predict")
def predict(input: IrisInput):
    try:
        features = [input.sepal_len, input.sepal_wid, input.petal_len, input.petal_wid]
        df = pd.DataFrame([features], columns=["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"])
        prediction = int(model.predict(df)[0])
        log_prediction(features, prediction)
        PREDICTION_COUNT.inc()
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest(), media_type="text/plain")
