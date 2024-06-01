"""Module provides a way to test the methods used in CustomerList class."""
import unittest
import yaml
from model import create_pipeline, train_model
import pandas as pd


data_path = "params.yaml"

with open(data_path, "r") as f:
    params = yaml.safe_load(f)

# Working tests
class Test_correct_data_size(unittest.TestCase):
    """Class tests all working methods in the customer class"""
    def test_pipeline_creation(self):
        """Tests the customer creation method"""
        X_train, X_test, y_train, y_test = create_pipeline(params)

        self.assertEqual(create_pipeline(params)[0].size, X_train.size)

class Test_correct_model(unittest.TestCase):
    """Class tests all working methods in the customer class"""
    def test_pipeline_creation(self):
        """Tests the customer creation method"""
        X_train, X_test, y_train, y_test = create_pipeline(params)

        self.assertEqual(str(train_model(params, X_train, y_train)), "LogisticRegression()")

# Check if the script is run as the main program.
if __name__ == '__main__':
    # Run the test cases using 'unittest.main()'.
    unittest.main()