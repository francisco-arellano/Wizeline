import yaml
import joblib
import numpy as np
from data import get_data
from preprocessing import data_cleaning, prepare_pipeline
from model import create_pipeline, train_model
from evaluation import get_metrics
import os

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

np.random.seed(params["base"]["random_seed"])

get_data(params)
data_cleaning(params)
prepare_pipeline(params)

X_train, X_test, y_train, y_test = create_pipeline(params)
model = train_model(params, X_train, y_train)

joblib.dump(model, os.path.join('../models', 'model.pkl'))

y_pred = model.predict(X_test)

mae, msq, rmse, r2, acc, cr = get_metrics(y_test, y_pred)

print(f"\nThe model's accuracy is: {acc*100}%")

print('\nMean Absolute Error:', mae)
print('Mean Squared Error:', msq)
print('Mean Root Squared Error:', rmse)
print('R2 Score: ', r2)
print('\n', cr)
