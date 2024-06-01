import joblib
import pandas as pd
import numpy as np


# Load the trained model
model = joblib.load("model.pkl")


def predict():
    to_csv = pd.read_csv("new_prediction.csv")
    features = np.array(to_csv).reshape(1, -1)

    y_pred = model.predict(features)

    return int(y_pred)

answer = predict()

if answer == 1:
    print(f"\nThe prediction is: {answer}, the patient does not have appendicitis\n")
else:
    print(f"\nThe prediction is: {answer}, the patient has appendicitis\n")