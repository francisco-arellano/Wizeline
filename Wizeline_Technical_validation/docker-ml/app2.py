from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np

app = FastAPI()

# Load the trained model
model = joblib.load("model.pkl")

@app.get("/")
def read_root():
    return {"message": "Welcome to the ML Model API"}

@app.post("/predict/")
def predict():
    to_csv = pd.read_csv("new_prediction.xlsx")
    features = np.array(to_csv).reshape(1, -1)

    y_pred = model.predict(features)

    return y_pred