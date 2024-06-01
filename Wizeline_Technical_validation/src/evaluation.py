import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import sys

def get_metrics(y_test, y_pred):
    # Evaluating Model's Performance
    mae = mean_absolute_error(y_test, y_pred)
    msq = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    cr = classification_report(y_test, y_pred)

    with open(os.path.join('../metrics', 'evaluation_metrics.json'), 'w') as f:
        f.write(f'{{"ACC": {acc}, "RMSE": {rmse}, "MSE":{msq} "MAE": {mae}, "R2": {r2}}}')

    return mae, msq, rmse, r2, acc, cr

if __name__ == "__main__":

    data_path = sys.argv[1]

    with open(data_path, "r") as f:
        params = yaml.safe_load(f)

    X_test = pd.read_csv(params["model"]["x_test"])
    y_test = pd.read_csv(params["model"]["y_test"])
    model = joblib.load(params["model"]["path"])
    y_pred = model.predict(X_test)

    mae, msq, rmse, r2, acc, cr = get_metrics(y_test, y_pred)

    print(f"\nThe model's accuracy is: {acc*100}%")

    print('\nMean Absolute Error:', mae)
    print('Mean Squared Error:', msq)
    print('Mean Root Squared Error:', rmse)
    print('R2 Score: ', r2)
    print('\n', cr)
