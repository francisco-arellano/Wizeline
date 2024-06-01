import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import os
import sys

def create_pipeline(params):

    X_scaled = pd.read_csv(params["pre_process"]["pipeline_features"])
    y = pd.read_csv(params["pre_process"]["pipeline_targets"])

    """Prepare data for training and evaluation."""
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, shuffle=True)

    X_train.to_csv(params["model"]["x_train"], index=False)
    X_test.to_csv(params["model"]["x_test"], index=False)
    y_train.to_csv(params["model"]["y_train"], index=False)
    y_test.to_csv(params["model"]["y_test"], index=False)

    return X_train, X_test, y_train, y_test

def train_model(params, X_train, y_train):

    # Instantiating LogisticRegression() Model
    lr = LogisticRegression(
        penalty= params["model"]["penalty"],
        dual= params["model"]["dual"],
        tol= params["model"]["tol"],
        C= params["model"]["C"],
        fit_intercept= params["model"]["fit_intercept"],
        intercept_scaling= params["model"]["intercept_scaling"],
        solver= params["model"]["solver"],
        max_iter= params["model"]["max_iter"]
    )

    # Training/Fitting the Model
    model = lr.fit(X_train, y_train.values.ravel())

    return model

if __name__ == '__main__':

    data_path = sys.argv[1]

    with open(data_path, "r") as f:
        params = yaml.safe_load(f)

    X_train, X_test, y_train, y_test = create_pipeline(params)
    model = train_model(params, X_train, y_train)

    print(X_train, X_test, y_train, y_test)

    joblib.dump(model, os.path.join('../models', 'model.pkl'))
