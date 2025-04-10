import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class PCA(object):

    def __init__(self):
        self.U = None
        self.S = None
        self.V = None

    def fit(self, X: np.ndarray) ->None:
        """		
        Decompose dataset into principal components by finding the singular value decomposition of the centered dataset X
        You may use the numpy.linalg.svd function
        Don't return anything. You can directly set self.U, self.S and self.V declared in __init__ with
        corresponding values from PCA. See the docstrings below for the expected shapes of U, S, and V transpose
        
        Hint: np.linalg.svd by default returns the transpose of V
              Make sure you remember to first center your data by subtracting the mean of each feature.
        
        Args:
            X: (N,D) numpy array corresponding to a dataset
        
        Return:
            None
        
        Set:
            self.U: (N, min(N,D)) numpy array
            self.S: (min(N,D), ) numpy array
            self.V: (min(N,D), D) numpy array
        """
        X_centered = X - np.mean(X, axis=0)
        U, S, V = np.linalg.svd(X_centered, full_matrices=False)
        
        self.U = U
        self.S = S 
        self.V = V 

    def transform(self, data: np.ndarray, K: int=2) ->np.ndarray:
        """		
        Transform data to reduce the number of features such that final data (X_new) has K features (columns)
        by projecting onto the principal components.
        Utilize class members that were previously set in fit() method.
        
        Args:
            data: (N,D) numpy array corresponding to a dataset
            K: int value for number of columns to be kept
        
        Return:
            X_new: (N,K) numpy array corresponding to data obtained by applying PCA on data
        
        Hint: Make sure you remember to first center your data by subtracting the mean of each feature.
        """
        data_centered = data - np.mean(data, axis=0) # A
        X_new = data_centered @ self.V.T[:, :K]
        
        return X_new
        

    def transform_rv(self, data: np.ndarray, retained_variance: float=0.99
        ) ->np.ndarray:
        """		
        Transform data to reduce the number of features such that the retained variance given by retained_variance is kept
        in X_new with K features
        Utilize self.U, self.S and self.V that were set in fit() method.
        
        Args:
            data: (N,D) numpy array corresponding to a dataset
            retained_variance: float value for amount of variance to be retained
        
        Return:
            X_new: (N,K) numpy array corresponding to data obtained by applying PCA on data, where K is the number of columns
                   to be kept to ensure retained variance value is retained_variance
        
        Hint: Make sure you remember to first center your data by subtracting the mean of each feature.
        """
        data_centered = data - np.mean(data, axis=0) 
        S_squared = np.square(self.S) # covariance squared
        total_variance = np.sum(S_squared)
        cumsum = np.cumsum(S_squared)
        cumsum_variance = cumsum / total_variance
        
        K = None
        for k in range(len(cumsum_variance)):
            if cumsum_variance[k] >= retained_variance:
                K = k + 1
                break
            
        X_new = data_centered @ self.V.T[:, :K]
            
        return X_new

    def get_V(self) ->np.ndarray:
        """		
        Getter function for value of V
        """
        return self.V

    def visualize(self, X: np.ndarray, y: np.ndarray, fig_title) ->None:
        """		
        You have to plot three different scatterplots (2D and 3D for strongest two features and 2D for two random features) for this function.
        For plotting the 2D scatterplots, use your PCA implementation to reduce the dataset to only 2 (strongest and later random) features.
        You'll need to run PCA on the dataset and then transform it so that the new dataset only has 2 features.
        Create a scatter plot of the reduced data set and differentiate points that have different true labels using color using matplotlib.
        
        Args:
            xtrain: (N,D) numpy array, where N is number of instances and D is the dimensionality of each instance
            ytrain: (N,) numpy array, the true labels
        
        Return: None
        """
        self.fit(X)
        
        # K = 2 nonrandom
        X_new_2 = self.transform(X, K=2)
        plt.scatter(X_new_2[:, 0], X_new_2[:, 1], c=y, cmap='viridis')
        plt.xlabel("Feature 1 (PCA 1)")
        plt.ylabel("Feature 2 (PCA 2)")  
        plt.title(fig_title + " - 2D Nonrandom")
        plt.show()
        
        # K = 3 nonrandom
        X_new_3 = self.transform(X, K=3)
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(X_new_3[:, 0], X_new_3[:, 1], X_new_3[:, 2], c=y, cmap='viridis')
        ax.set_xlabel("Feature 1 (PCA 1)")
        ax.set_ylabel("Feature 2 (PCA 2)")
        ax.set_zlabel("Feature 3 (PCA 3)")
        ax.set_title(fig_title + " - 3D Nonrandom")
        plt.show()
        
        # K = 2 random
        _, D = X.shape
        k = 2
        random_features = np.random.choice(D, k, replace=False)
        X_new_2_rand = X[:, random_features]
        plt.scatter(X_new_2_rand[:, 0], X_new_2_rand[:, 1], c=y, cmap='viridis')
        plt.xlabel("Feature 1 (PCA 1)")
        plt.ylabel("Feature 2 (PCA 2)")
        plt.title(fig_title + " - 2D Random")
        plt.show()
