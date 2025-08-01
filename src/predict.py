import pandas as pd
import joblib

def load_model(path="models/best_model.pkl"):
    return joblib.load(path)

def predict(model, input_data: list):
    df = pd.DataFrame([input_data], columns=["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"])
    prediction = model.predict(df)
    return int(prediction[0])
