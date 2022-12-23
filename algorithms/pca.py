import numpy as np
import time

class PCA(object):
    """
        PCA dimensionality reduction object.
        Feel free to add more functions to this class if you need.
        But make sure that __init__, find_principal_components, and reduce_dimension work correctly.
    """
    def __init__(self, *args, **kwargs):
        """
            You don't need to initialize the task kind for PCA.
            Call set_arguments function of this class.
        """
        self.set_arguments(*args, **kwargs)
        # the mean of the training data (will be computed from the training data and saved to this variable)
        self.mean = None 
        # the principal components (will be computed from the training data and saved to this variable)
        self.W = None

    def set_arguments(self, *args, **kwargs):
        """
            args and kwargs are super easy to use! See dummy_methods.py
            The PCA class should have a variable defining the number of dimensions (d).
            You can either pass this as an arg or a kwarg.
        """
        if "d" in kwargs:
            self.d = kwargs["d"]
        elif len(args) > 0:
            self.d = args[0]
        else:
            self.d = 1

    def find_principal_components(self, training_data):
        """
            Finds the principal components of the training data. Returns the explained variance in percentage.
            IMPORTANT: 
            This function should save the mean of the training data and the principal components as
            self.mean and self.W, respectively.

            Arguments:
                training_data (np.array): training data of shape (N,D)
            Returns:
                exvar (float): explained variance
        """
        start = time.time()

        # Compute the mean of data
        self.mean = np.mean(training_data, axis=0)
        
        # Center the data with the mean
        X_tilde = training_data - self.mean

        # Create the covariance matrix
        C = X_tilde.T @ X_tilde * 1/training_data.shape[0] # DxN dot NxD = DxD
        
        # Compute the eigenvectors and eigenvalues. Hint: use np.linalg.eigh
        eigvals, eigvecs = np.linalg.eigh(C) # symmetric matrix C
        
        # Choose the top d eigenvalues and corresponding eigenvectors. Sort the eigenvalues( with corresponding eigenvectors )
        # in decreasing order first.
        idx = np.argsort(-eigvals)[:self.d] # argsort elements in decreasing order
        # select d greatest eigenvalue indices

        # Create matrix W and the corresponding eigenvalues
        self.W = eigvecs[:, idx] # d corresponding eigenvectors
        eg = eigvals[idx] # d greatest eigenvalues
        
        # Compute the explained variance
        exvar = np.sum(eg) / np.sum(eigvals) * 100

        end = time.time()
    
        print(f'Pre-PCA Runtime |> {end-start}s')
        return exvar

    def reduce_dimension(self, data):
        """
            Reduce the dimensions of the data, using the previously computed
            self.mean and self.W. 

            Arguments:
                data (np.array): data of shape (N,D)
            Returns:
                data_reduced (float): reduced data of shape (N,d)
        """
        start = time.time()
        centered_data = data - self.mean[0:data.shape[1]]
        data_reduced = centered_data @ self.W[0:data.shape[1]]
        end = time.time()
        print(f'PCA Reduction Runtime |> {end-start}s')
        return data_reduced
        

