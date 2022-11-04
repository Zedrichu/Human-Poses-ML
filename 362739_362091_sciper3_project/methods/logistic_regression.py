import numpy as np
import sys
sys.path.append('..')
from utils import label_to_onehot, onehot_to_label


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
            if len(args) > 1:
                self.max_iters = args[1]
        else: 
            self.lr = 0.01
            self.max_iters = 1000

    def f_softmax(self, data, w):
        """ Softmax function
        
        Args:
            data (np.array): Input data of shape (N, D)
            w (np.array): Weights of shape (D, C) where C is # of classes
            
        Returns:
            res (np.array): Probabilites of shape (N, C), where each value is in 
                range [0, 1] and each row sums to 1.
        """
        res = np.exp(data @ w) / np.sum(np.exp(data @ w), axis=1, keepdims=True)
        return res

    def classify(self, data, w):
        """ Classification function for multi class logistic regression. 
        
        Args:
            data (np.array): Dataset of shape (N, D).
            w (np.array): Weights of logistic regression model of shape (D, C)
        Returns:
            predictions (np.array): Label assignments of data of shape (N, ) (NOT one-hot!).
        """
        y_hat = self.f_softmax(data, w)
        predictions = np.argmax(y_hat, axis=1)
        return predictions

    def accuracy_fn(self, labels_true, labels_pred):
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

    def fit(self, training_data, training_labels, max_iters=100, lr=0.01):
        """
            Trains the model, returns predicted labels for training data.
            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): regression target of shape (N,)
            Returns:
                pred_labels (np.array): target of shape (N,)
        """
        k = np.unique(training_labels).size
        w = np.random.normal(0, 0.1, [training_data.shape[1], k])
        for _ in range(max_iters):
            # Compute the updated gradient
            grad = self.gradient(training_data, label_to_onehot(training_labels), w)
            # Update the weights
            w -= lr * grad

            predictions = self.classify(training_data, w)
            #check accurancy and break if 100%
            if self.accuracy_fn(training_labels, predictions) == 1:
                break
        self.W = w
        pred_labels = self.classify(training_data, w)
        return pred_labels

    def gradient(self, training_data, training_labels, W):
        """ Gradient function for multi class logistic regression
    
        Args:
            data (np.array): Input data of shape (N, D)
            labels (np.array): Labels of shape  (N, C)  (in one-hot representation)
            w (np.array): Weights of shape (D, C)
            
        Returns:
            grad_w (np.array): Gradients of shape (D, C)
        """
        soft = self.f_softmax(training_data, W)
        grad = training_data.T @ (soft - training_labels)
        return grad

    def predict(self, test_data):
        """
            Runs prediction on the test data.
            
            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                test_labels (np.array): labels of shape (N,)
        """   
        pred_labels = self.classify(test_data, self.W)
        return pred_labels