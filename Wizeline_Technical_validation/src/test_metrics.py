import unittest
from unittest import TextTestRunner
import yaml
import joblib
from evaluation import get_metrics
import pandas as pd

data_path = "params.yaml"

with open(data_path, "r") as f:
    params = yaml.safe_load(f)

# Working tests
class Test_correct_evaluation(unittest.TestCase):
    """Class tests all working methods in the customer class"""
    def test_metrics(self):
        """Tests the customer creation method"""
        X_test = pd.read_csv(params["model"]["x_test"])
        y_test = pd.read_csv(params["model"]["y_test"])
        model = joblib.load(params["model"]["path"])
        y_pred = model.predict(X_test)

        mae, msq, rmse, r2, acc, cr = get_metrics(y_test, y_pred)

        self.assertGreater(acc, 0.80)


# Check if the script is run as the main program.
if __name__ == '__main__':
    # Run the test cases using 'unittest.main()'.
    result = unittest.main()
