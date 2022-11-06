import numpy as np
import sys
import time
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
        
        if "lr" in kwargs:
            self.lr = kwargs["lr"]
        elif len(args) > 0:
            self.lr = args[0]
        else: 
            self.lr = 1e-4

        if "max_iters" in kwargs:
            self.max_iters = kwargs["max_iters"]
        elif len(args) > 1:
            self.max_iters = args[1]
        else:
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

    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): regression target of shape (N,)
            Returns:
                pred_labels (np.array): target of shape (N,)
        """
        LOG_PERIOD = 100
        print("Started Logistic Regression training with learning rate {} and max iterations {}\n".format(self.lr, self.max_iters))

        start = time.time()
        k = np.unique(training_labels).size
        weight = np.random.normal(0, 0.1, [training_data.shape[1], k])
        acc = 0
        for it in range(self.max_iters):
            # Compute the updated gradient
            grad = self.gradient(training_data, label_to_onehot(training_labels), weight)
            # Update the weights
            weight -= self.lr * grad

            predictions = self.classify(training_data, weight)
            temp = self.accuracy_fn(training_labels, predictions) 
            # check accurancy improvement, break if no change
            if (abs(temp-acc) < 1e-25):
                break
            else:
                acc = temp
            #check accurancy and break if 100%
            if it % LOG_PERIOD == 0:
                print("Training accuracy at iteration {} is {}".format(it, acc))
            if acc == 1:
                break
        
        self.W = weight
        pred_labels = self.classify(training_data, weight)
        print("Final accuracy after training is {}\n".format(self.accuracy_fn(training_labels, pred_labels)))
        end = time.time()
        print("Runtime of Logistic Regression training: {} sec\n".format(str(end-start)))
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