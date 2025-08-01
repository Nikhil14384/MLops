from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.predict import load_model, predict
import sqlite3
import time

app = FastAPI()
model = load_model()

# SQLite logging
def log_prediction(data, result):
    conn = sqlite3.connect("logs/predictions.db")
    cursor = conn.cursor()
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS predictions (timestamp TEXT, sepal_len REAL, sepal_wid REAL, petal_len REAL, petal_wid REAL, prediction INTEGER)"
    )
    cursor.execute(
        "INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?)",
        (time.strftime("%Y-%m-%d %H:%M:%S"), *data, result)
    )
    conn.commit()
    conn.close()

class IrisInput(BaseModel):
    sepal_len: float
    sepal_wid: float
    petal_len: float
    petal_wid: float

@app.post("/predict")
def get_prediction(input: IrisInput):
    try:
        data = [input.sepal_len, input.sepal_wid, input.petal_len, input.petal_wid]
        result = predict(model, data)
        log_prediction(data, result)
        return {"prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
