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

    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): regression target of shape (N,regression_target_size)
            Returns:
                pred_labels (np.array): target of shape (N,regression_target_size)
        """
        
        return pred_labels

    def predict(self, test_data):
        """
            Runs prediction on the test data.
            
            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                test_labels (np.array): labels of shape (N,)
        """   
        ##
        ###
        #### YOUR CODE HERE! 
        ###
        ##

        return pred_labels
