import numpy as np

class KNN(object):
    """
        kNN classifier object.
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
            The KNN class should have a variable defining the number of neighbours (k).
            You can either pass this as an arg or a kwarg.
        """
        
        if "k" in kwargs:
            self.k = kwargs["k"]
        elif len(args) > 0:
            self.k = args[0]
        else:
            self.k = 1    
        
    def euclidean_dist(self, sample, training_samples):
        """
            Function to compute the Euclidean distance between a single sample
            vector and all training samples.
            
            Inputs:
                sample: shape (D,)
                training_samples: shape (NxD) 
            Outputs:
                return distance vector of length N
        """
        return np.sqrt(np.sum((sample-training_samples)**2, axis = 1))

    def find_k_nearest_neighbors(self, k, distances):
        """
            Find the indices of the k smallest distances from the list
        """
        indices = np.argsort(distances)[0:k]
        return indices

    def predict_label(self, neighbor_labels):
        """
            Return the most frequent label in the input
        """
        label = np.argmax(np.bincount(neighbor_labels))
        return label

    # def choose_random_sample(data):
    #     """
    #         Randomly chose a single datapoint from given matrix

    #         Input:
    #             data: shape (NxD)
    #         Output:
    #             return a randomly chosen index of length (D,)
    #     """
    #     index = np.random.randint(data.shape[0])
    #     return index

    def kNN_sample(self, sample):
        distances = self.euclidean_dist(sample, self.training_data)
        indices = self.find_k_nearest_neighbors(self.k, distances)
        neighbor_labels = self.training_labels[indices]
        best_label = self.predict_label(neighbor_labels)
        return best_label
        
    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Hint: Since KNN does not really have parameters to train, you can try saving the training_data
            and training_labels as part of the class. This way, when you call the "predict" function
            with the test_data, you will have already stored the training_data and training_labels
            in the object.
            
            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): labels of shape (N,)
            Returns:
                pred_labels (np.array): labels of shape (N,)
        """
        print(f'Started KNN training with hyperparameter k={self.k}...')

        self.training_data = training_data
        self.training_labels = training_labels
        
        pred_labels = np.array([self.kNN_sample(x) for x in training_data])        
        return pred_labels
                               
    def predict(self, test_data):
        """
            Runs prediction on the test data.
            
            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                test_labels (np.array): labels of shape (N,)
        """
        test_labels = np.array([self.kNN_sample(x) for x in test_data])
        return test_labels