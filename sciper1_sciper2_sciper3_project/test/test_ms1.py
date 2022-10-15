"""
Module for auto-testing student projects.
This is based on the file from Francois Fleuret's
"Deep Learning Course": https://fleuret.org/dlc/.

This is the Milestone 1 version.


Note to students:

1. Run this script with the command:
python test_ms1.py -p path_to_project_folder

2. More tests will be present in the final test file.
   This release version is for you to verify that your 
   project is compatible with our grading system.
"""

import re
import sys
import os
import unittest
import importlib
from pathlib import Path

import numpy as np


class HidePrints:
    """Disable normal printing for calling student code."""
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
no_print = HidePrints  # alias


class DummyMethod:
    """This is a dummy classification method used to test the cross-validation."""

    def __init__(self, *args, **kwargs):
        self.task_kind = 'classification'
        self.set_arguments(*args, **kwargs)

    def set_arguments(self, *args, **kwargs):
        self.dummy_arg = int(kwargs.get("dummy_arg", 1))

    def fit(self, training_data, training_labels):
        self.w = int(np.median(training_labels) / 2 * self.dummy_arg)
        return self.predict(training_data)

    def predict(self, test_data):
        pred_labels = np.full((test_data.shape[0],), self.w, int)
        return pred_labels


class TestProject(unittest.TestCase):

    @staticmethod
    def title(msg):
        print(f"\n==============\n> {msg} ...")
    

    def test_1_folder_structure(self):
        self.title("Testing folder structure")
        self.assertTrue(project_path.exists(), f"No folder found at {project_path}")

        # Main files
        for file in ["__init__.py", "data.py", "main.py", "metrics.py", "utils.py",
                     project_path.name[:-7] + "report.pdf"]:
            with self.subTest(f"Checking file {file}"):
                self.assertTrue((project_path / file).exists(), f"No file {file} found at {project_path}")
        
        # Methods
        method_path = project_path / "methods"
        self.assertTrue(method_path.exists(), f"{method_path} not found")
        for file in ["__init__.py", "dummy_methods.py",
                     "linear_regression.py", "logistic_regression.py", "cross_validation.py"]:
            with self.subTest(f"Checking file methods/{file}"):
                self.assertTrue((method_path / file).exists(), f"No file {file} found at {method_path}")

    
    def _import_and_test(self, name, class_name, *args, **kwargs):
        # Code structure
        module = importlib.import_module(f"methods.{name}")
        method = module.__getattribute__(class_name)(*args, **kwargs)
        for fn in ["set_arguments", "fit", "predict"]:
            _ = method.__getattribute__(fn)
        
        # Functions inputs and outputs
        N, D, D_r = 10, 3, 2
        training_data = np.random.rand(N, D)
        if method.task_kind == 'classification':
            training_labels = np.random.randint(0, D, N)
        elif method.task_kind == 'regression':
            training_labels = np.random.rand(N, D_r)
        test_data = np.random.rand(N, D)
        with no_print():
            pred_labels = method.fit(training_data, training_labels)
        self.assertIsInstance(pred_labels, np.ndarray, f"{name}.{class_name}.fit() should output an array, not {type(pred_labels)}")
        self.assertEqual(pred_labels.shape, training_labels.shape, f"{name}.{class_name}.fit() output has wrong shape ({pred_labels.shape} != {training_labels.shape})")
        with no_print():
            pred_labels = method.predict(test_data)
        self.assertIsInstance(pred_labels, np.ndarray, f"{name}.{class_name}.predict() should output an array, not {type(pred_labels)}")
        self.assertEqual(pred_labels.shape, training_labels.shape, f"{name}.{class_name}.predict() output has wrong shape ({pred_labels.shape} != {training_labels.shape})")
        
        return method
    

    def test_2_dummy_methods(self):
        self.title("Testing dummy methods")

        _ = self._import_and_test("dummy_methods", "DummyClassifier")
        _ = self._import_and_test("dummy_methods", "DummyRegressor")
    

    def test_3a_linear_regression(self):
        self.title("Testing linear regression")

        linear_regression = self._import_and_test("linear_regression", "LinearRegression",
                                                  lmda=0.)

        # Test on easy dummy data
        N = 20
        training_data = np.linspace(-1, 1, N)[:,None]
        training_labels = 2 * training_data
        test_data = np.random.rand(N, 1) * 2 - 1
        test_labels = 2 * test_data
        with no_print():
            pred_labels_train = linear_regression.fit(training_data, training_labels)
            pred_labels_test = linear_regression.predict(test_data)
        self.assertTrue(np.isclose(pred_labels_train, training_labels).all(), f"LinearRegression.fit() is not working on dummy data")
        self.assertTrue(np.isclose(pred_labels_test, test_labels).all(), f"LinearRegression.predict() is not working on dummy data")
    

    def test_3b_logistic_regression(self):
        self.title("Testing logistic regression")

        logistic_regression = self._import_and_test("logistic_regression", "LogisticRegression",
                                                    lr=1e-3, max_iters=500)

        # Test on easy dummy data
        N = 20
        training_data = np.concatenate([
            np.linspace(-5, -0.25, N//2)[:,None],
            np.linspace(0.25, 5, N//2)[:,None]
        ], axis=0)
        training_labels = (training_data[:,0] > 0.).astype(int)
        test_data = np.array([-10., -5., -1., 1., 5., 10.])[:, None]
        test_labels = (test_data[:,0] > 0.).astype(int)
        with no_print():
            pred_labels_train = logistic_regression.fit(training_data, training_labels)
            pred_labels_test = logistic_regression.predict(test_data)
        self.assertTrue((pred_labels_train == training_labels).all(), f"LogisticRegression.fit() is not working on dummy data")
        self.assertTrue((pred_labels_test == test_labels).all(), f"LogisticRegression.predict() is not working on dummy data")
        

    def test_3c_cross_validation(self):
        self.title("Testing cross-validation")

        # Code structure
        module = importlib.import_module("methods.cross_validation")
        splitting_fn = module.__getattribute__("splitting_fn")
        cross_validation = module.__getattribute__("cross_validation")
        
        # Functions inputs and outputs
        N, D, k_fold = 20, 3, 4
        fold_size = N//k_fold
        data, labels, indices = np.arange(N*D).reshape(N, D), np.full((N,), 2, int), np.random.permutation(N)
        search_arg_name, search_args = "dummy_arg", [1, 2, 3]
        method_obj = DummyMethod()
        with no_print():
            train_data, train_label, val_data, val_label = splitting_fn(data, labels, indices, fold_size, 2)
        for i, (arr, shape) in enumerate(zip([train_data, train_label, val_data, val_label], 
                                             [(N-fold_size, D), (N-fold_size,), (fold_size, D), (fold_size,)])):
            with self.subTest(f"Checking output no {i}"):
                self.assertIsInstance(arr, np.ndarray, f"cross_validation.splitting_fn() should output arrays, not {type(arr)}")
                self.assertEqual(arr.shape, shape, f"cross_validation.splitting_fn() output has wrong shape ({arr.shape} != {shape})")
        with no_print():
            best_hyperparam, best_acc = cross_validation(method_obj, search_arg_name, search_args, data, labels, k_fold)
        self.assertIsInstance(best_hyperparam, (int, float), f"cross_validation.cross_validation() output `best_hyperparam` should be float or int, not {type(best_hyperparam)}")
        self.assertIsInstance(best_acc, float, f"cross_validation.cross_validation() output `best_acc` should be float, not {type(best_acc)}")

        # Test on easy dummy data
        self.assertTrue(all([x not in train_data for x in val_data]), f"cross_validation.splitting_fn() is not working on dummy data")
        self.assertEqual(best_hyperparam, 2, f"cross_validation.cross_validation() is not working on dummy data")
        self.assertTrue(np.isclose(best_acc, 1.), f"cross_validation.cross_validation() is not working on dummy data")


def warn(msg):
    print(f"\33[33m/!\\ Warning: {msg}\33[39m")


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-p', '--project-path', help='Path to the project folder', required=True)
    args = parser.parse_args()
    
    project_path = Path(args.project_path)

    if re.match(r'^((\d{6})_){3}project$', project_path.name) is None:
        warn("Project folder name must be in the form 'XXXXXX_XXXXXX_XXXXXX_project'")

    sys.path.insert(0, args.project_path)
    unittest.main(argv=[''], verbosity=0)