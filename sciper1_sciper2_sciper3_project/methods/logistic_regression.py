import numpy as np
import sys
sys.path.append('..')
from utils import label_to_onehot


class LogisticRegression(object):
    """
        LogisticRegression classifier object.
        Feel free to add more functions to this class if you need.
        But make sure that __init__, set_arguments, fit and predict work correctly.
    """

    def __init__(self, *args, **kwargs):
        """
            Initialize the task_kind (see dummy_methods.py)
            and call set_arguments function of this class.
        """
        self.task_kind = 'classification'
        self.set_arguments(*args, **kwargs)

    def set_arguments(self, *args, **kwargs):
        """
            args and kwargs are super easy to use! See dummy_methods.py
            The LogisticRegression class should have variables defining the learning rate (lr)
            and the number of max iterations (max_iters)
            You can either pass these as args or kwargs.
        """
        
        if "lr" in kwargs and "max_iters" in kwargs:
            self.lr = kwargs["lr"]
            self.max_iters = kwargs["max_iters"]
        elif len(args) > 0:
            self.lr = args[0]
            self.max_iters = args[1]
        else: 
            self.lr = 1
            self.max_iters = 1

    def sigmoid(t):
        """ Sigmoid function
        
        Args:
            t (np.array): Input data of shape (N, )
            
        Returns:
            res (np.array): Probabilites of shape (N, ), where each value is in [0, 1].
        """
        res = 1 / (1 + np.exp(-t))
        return res

    def classify(self, data, w):
        """
        Classifies data into 2 classes.

        Args:
            data (np.array): Dataset of shape (N, D)

        Returns:
            (np.array): Predictions of labels (N, )
        """
        pred = self.sigmoid(data @ w)
        return int(pred[pred >= 0.5])

    def accuracy_fn(labels_true, labels_pred):
        """
        Computes accuracy of current model parameters.

        Args:
            labels_true (np.array): True labels of shape (N,)
            labels_pred (np.array): Predicted labels of shape (N,)
        
        Returns:
            acc (float): Accuracy [0..1]
        """
        acc = np.sum(labels_true == labels_pred) / labels_pred.shape[0] 
        return acc

    def fit(self, training_data, training_labels, max_iters=10, lr=0.001):
        """
            Trains the model, returns predicted labels for training data.
            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): regression target of shape (N,regression_target_size)
            Returns:
                pred_labels (np.array): target of shape (N,regression_target_size)
        """
        w = np.random.normal(0.0, 0.1, training_data.shape[0])
        for _ in range(max_iters):
            # Compute the updated gradient
            grad = gradient(training_data, training_labels, w)
            # Update the weights
            w -= lr * grad

            predictions = self.classify(training_data, w)
            #check accurancy and break if 100%
            if self.accuracy_fn(predictions, training_labels) == 1:
                break
        self.W = w
        pred_labels = self.classify(training_data, w)
        return pred_labels

    def gradient(self, W, training_data, traning_labels):
        grad = traning_data.T @ (self.sigmoid(training_data @ W) - training_labels)
        return grad

    def predict(self, test_data):
        """
            Runs prediction on the test data.
            
            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                test_labels (np.array): labels of shape (N,)
        """   
        pred_labels = classify(test_data, self.W)
        return pred_labels
