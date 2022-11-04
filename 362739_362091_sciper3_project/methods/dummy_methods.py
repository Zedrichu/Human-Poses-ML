import numpy as np

class DummyClassifier(object):
    """
        This method is a dummy method! It returns a random label for classification.
    """
    def __init__(self, *args, **kwargs):
        """
            The task_kind determines whether the task is classification or regression. 
            You need to set this correctly for the methods you write!
            Should call the set_arguments method and pass the args and kwargs there.
            More about this in the set_arguments function definition below.
        """
        self.task_kind = 'classification'
        self.set_arguments(*args, **kwargs)

    def set_arguments(self, *args, **kwargs):
        """
            args and kwargs are super easy to use! 
            *   The parameter you pass is an "arg" if you don't specify the parameter name. 
                It is stored in a list of args.
                For example, if you create DummyClassifier like this:
                    method_obj = DummyClassifier(1e-5)
                Then args is a list with one element: 
                    [1e-5]
            *   The parameter you pass is a "kwarg" if you specify the parameter name. 
                If you create DummyClassifier like this:
                    method_obj = DummyClassifier(dummy_arg=1e-5)
                Now kwargs is a dictionary:
                    {"dummy_arg": 1e-5}
        """

        # first checks if "dummy_arg" was passed as a kwarg.
        if "dummy_arg" in kwargs:
            self.dummy_arg = kwargs["dummy_arg"]
        # if not, then check if args is a list with size bigger than 0.
        elif len(args) >0 :
            self.dummy_arg = args[0]
        # if there were no args or kwargs passed, we set the dummy_arg to 1 (default value).
        else:
            self.dummy_arg = 1

    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            In the case of the DummyClassifier, this method will return 
            random predicted labels.
            
            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): labels of shape (N,)
            Returns:
                pred_labels (np.array): labels of shape (N,)
        """

        self.D, self.C = training_data.shape[1], int(np.max(training_labels)+1)
        pred_labels = np.random.randint(low=0, high=self.C, size=training_data.shape[0])
        return pred_labels
                               
    def predict(self, test_data):
        """
            Runs prediction on the test data.
            In the case of the DummyClassifier, this method will return 
            random predicted labels.
            
            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                pred_labels (np.array): labels of shape (N,)
        """
        pred_labels = np.random.randint(low=0, high=self.C, size=test_data.shape[0])
        return pred_labels

class DummyRegressor(object):
    """
        This method is a dummy method for regression!
    """

    def __init__(self, *args, **kwargs):
        self.task_kind = 'regression'
        self.set_arguments(*args, **kwargs)

    def set_arguments(self, *args, **kwargs):
        # first checks if "dummy_arg" was passed as a kwarg.
        if "dummy_arg" in kwargs:
            self.dummy_arg = kwargs["dummy_arg"]
        # if not, then check if args is a list with size bigger than 0.
        elif len(args) >0 :
            self.dummy_arg = args[0]
        # if there were no args or kwargs passed, we set the dummy_arg to 1 (default value).
        else:
            self.dummy_arg = 1

    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            In the case of the dummy_predictor, this method will return 
            random predicted regression targets.
            
            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): regression target of shape (N,regression_target_size)
            Returns:
                pred_regression_targets (np.array): target of shape (N,regression_target_size)
        """
        self.w = np.random.rand(training_data.shape[-1],training_labels.shape[-1])
        pred_regression_targets = training_data@self.w
        return pred_regression_targets
                               
    def predict(self, test_data):  
        """
            Runs prediction on the test data.
            In the case of the dummy_regression, this method will return 
            random predicted target values.
            
            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                pred_regression_targets (np.array): labels of shape (N,regression_target_size)
        """      
        pred_regression_targets = test_data@self.w
        return pred_regression_targets