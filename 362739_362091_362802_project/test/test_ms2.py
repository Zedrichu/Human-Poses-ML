"""
Module for auto-testing student projects.
This is based on the file from Francois Fleuret's
"Deep Learning Course": https://fleuret.org/dlc/.

This is the Milestone 2 version.


Note to students:

1. Run this script with the command:
python test_ms2.py -p path_to_project_folder

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
import torch
from torch.utils.data import TensorDataset, DataLoader


class HidePrints:
    """Disable normal printing for calling student code."""
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

class NoHidePrints:
    """Don't disable normal printing for calling student code."""
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class TestProject(unittest.TestCase):

    @staticmethod
    def title(msg):
        print(f"\n==============\n> {msg} ...")
    

    def test_1_folder_structure(self):
        self.title("Testing folder structure")
        self.assertTrue(project_path.exists(), f"No folder found at {project_path}")

        # Main files
        for file in ["__init__.py", "data.py", "main.py", "metrics.py", "utils.py"]:
            with self.subTest(f"Checking file {file}"):
                self.assertTrue((project_path / file).exists(), f"No file {file} found at {project_path}")
        with self.subTest(f"Checking report file"):
            report_name = (project_path.absolute().name[:-7] + "report.pdf", "report.pdf")
            self.assertTrue((project_path / report_name[0]).exists() or (project_path / report_name[1]).exists(), 
                            f"No report found at {project_path}")
        
        # Methods
        method_path = project_path / "methods"
        self.assertTrue(method_path.exists(), f"{method_path} not found")
        for file in ["__init__.py", "dummy_methods.py",
                     "pca.py", "knn.py", "deep_network.py"]:
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
    

    def test_4a_pca(self):
        self.title("Testing PCA")

        # Code structure
        module = importlib.import_module("methods.pca")
        pca = module.__getattribute__("PCA")()
        for fn in ("find_principal_components", "reduce_dimension"):
            _ = pca.__getattribute__(fn)
        
        # Functions inputs and outputs
        N, D, d = 10, 5, 2
        pca = module.__getattribute__("PCA")(d)
        data = np.random.rand(N, D)
        with no_print():
            exvar = pca.find_principal_components(data)
        self.assertIsInstance(exvar, float, f"pca.PCA.find_principal_components() should output a float, not {type(exvar)}")
        self.assertIsInstance(pca.mean, np.ndarray, f"pca.PCA.mean should be an array, not {type(pca.mean)}")
        self.assertEqual(pca.mean.shape, (D,), f"pca.PCA.mean has wrong shape ({pca.mean.shape} != {(D,)})")
        self.assertIsInstance(pca.W, np.ndarray, f"pca.PCA.W should be an array, not {type(pca.W)}")
        self.assertEqual(pca.W.shape, (D, d), f"pca.PCA.W has wrong shape ({pca.W.shape} != {(D, d)})")
        with no_print():
            Y = pca.reduce_dimension(data)
        self.assertIsInstance(Y, np.ndarray, f"pca.PCA.reduce_dimension() should output an array, not {type(Y)}")
        self.assertEqual(Y.shape, (N, d), f"pca.PCA.reduce_dimension() output has wrong shape ({Y.shape} != {(N, d)})")

        # Test on easy dummy data
        N, D, d = 10, 2, 1
        pca = module.__getattribute__("PCA")(d)
        data = np.array([[2.77, 1.67], [1.96, 1.26], [ 0.67, 0.51], [0.99, 1.17], [-0.51, 0.21],
                         [0.12, 0.35], [ 2.46, 1.66], [2.05, 1.52], [1.51, 1.37], [ 2.09, 1.47]])
        proj = np.array([-1.46, -0.55, 0.94, 0.35, 2.12, 1.50, -1.18, -0.75, -0.20, -0.76]).reshape(N, d)
        with no_print():
            exvar = pca.find_principal_components(data)
            Y = pca.reduce_dimension(data)
        self.assertGreater(exvar, 95., f"pca.PCA.find_principal_components() is not working on dummy data")
        self.assertLess(np.linalg.norm(pca.mean - [1.41, 1.12]), 0.01, f"pca.PCA.find_principal_components() is not working on dummy data")
        self.assertLess(np.linalg.norm(pca.W - [[-0.89], [-0.45]]), 0.01, f"pca.PCA.find_principal_components() is not working on dummy data")
        self.assertLess(np.abs(Y - proj).max(), 0.01, f"pca.PCA.reduce_dimension() is not working on dummy data")
    

    def test_4b_knn(self):
        self.title("Testing kNN")

        knn = self._import_and_test("knn", "KNN", k=1)

        # Test on easy dummy data
        training_data = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
        training_labels = np.array([0, 1, 2, 3])
        test_data = np.array([[0., 0.1], [1.2, -0.2], [0.1, 0.9], [20., 20.]])
        test_labels = np.array([0, 1, 2, 3])
        with no_print():
            pred_labels_train = knn.fit(training_data, training_labels)
            pred_labels_test = knn.predict(test_data)
        self.assertTrue((pred_labels_train == training_labels).all(), f"KNN.fit() is not working on dummy data")
        self.assertTrue((pred_labels_test == test_labels).all(), f"KNN.predict() is not working on dummy data")
    

    def test_4c_deep_network(self):
        self.title("Testing deep-network")

        # For dummy data
        D, C = 10, 4
        lr, epochs = 0.01, 2

        # Code structure
        module = importlib.import_module("methods.deep_network")
        simple_network = module.__getattribute__("SimpleNetwork")(D, C)
        trainer = module.__getattribute__("Trainer")(simple_network, lr, epochs)
        for fn in ["train_all", "train_one_epoch", "eval"]:
            _ = trainer.__getattribute__(fn)
        
        # Functions inputs/outputs
        N, bs = 50, 8
        train_dataset = TensorDataset(torch.randn(N, D), torch.randn(N, 1), torch.randint(0, C, (N,)))
        val_dataset = TensorDataset(torch.randn(N, D), torch.randn(N, 1), torch.randint(0, C, (N,)))
        dataloader_train = DataLoader(train_dataset, batch_size=bs, shuffle=True)
        dataloader_val = DataLoader(val_dataset, batch_size=bs, shuffle=False)
        with no_print():
            x, _, _ = next(iter(dataloader_train))
            output_class = simple_network(x)
        self.assertIsInstance(output_class, torch.Tensor, f"deep_network.SimpleNetwork.fit() should output a tensor, not {type(output_class)}")
        self.assertEqual(output_class.shape, (bs, C), f"deep_network.SimpleNetwork.fit() output has wrong shape ({output_class.shape} != {(bs, C)})")
        with no_print():
            trainer.train_all(dataloader_train, dataloader_val)
            results_class = trainer.eval(dataloader_val)
        self.assertIsInstance(results_class, torch.Tensor, f"deep_network.Trainer.eval() should output a tensor, not {type(results_class)}")
        self.assertEqual(results_class.shape, (N,), f"deep_network.Trainer.eval() output has wrong shape ({results_class.shape} != {(N,)})")


def warn(msg):
    print(f"\33[33m/!\\ Warning: {msg}\33[39m")


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-p', '--project-path', help='Path to the project folder', required=True)
    parser.add_argument('--no-hide', action='store_true', help='Enable printing from the student code')
    args = parser.parse_args()
    
    project_path = Path(args.project_path)

    dir_name = project_path.absolute().name
    if re.match(r'^((\d{6})_){3}project$', dir_name) is None:
        warn("Project folder name must be in the form 'XXXXXX_XXXXXX_XXXXXX_project'")

    if args.no_hide:
        no_print = NoHidePrints
    else:
        no_print = HidePrints

    sys.path.insert(0, args.project_path)
    unittest.main(argv=[''], verbosity=0)