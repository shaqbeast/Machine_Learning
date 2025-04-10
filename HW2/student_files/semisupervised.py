"""
File: semisupervised.py
Project: autograder_test_files
File Created: September 2020
Author: Shalini Chaudhuri (you@you.you)
Updated: September 2022, Arjun Agarwal
"""
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from tqdm import tqdm
SIGMA_CONST = 1e-06
LOG_CONST = 1e-32


def complete_(data):
    """	
    Args:
        data: N x (D+1) numpy array where the last column is the labels
    Return:
        labeled_complete: n x (D+1) array (n <= N) where values contain both complete features and labels
    """
    labeled_complete = []
    N, _ = data.shape
    for i in range(N):
        if np.any(np.isnan(data[i])):
            continue
        else:
            labeled_complete.append(data[i])
    
    return np.array(labeled_complete)


def incomplete_(data):
    """	
    Args:
        data: N x (D+1) numpy array where the last column is the labels
    Return:
        labeled_incomplete: n x (D+1) array (n <= N) where values contain incomplete features but complete labels
    """
    labeled_incomplete = []
    N, D = data.shape
    D -= 1
    D_indices = np.arange(D)
    for i in range(N):
        if np.any(np.isnan(data[i, D_indices])):
            labeled_incomplete.append(data[i])
    
    return np.array(labeled_incomplete)


def unlabeled_(data):
    """	
    Args:
        data: N x (D+1) numpy array where the last column is the labels
    Return:
        unlabeled_complete: n x (D+1) array (n <= N) where values contain complete features but incomplete labels
    """
    unlabeled_complete = []
    N, D = data.shape
    D -= 1
    for i in range(N):
        if np.any(np.isnan(data[i, D])):
            unlabeled_complete.append(data[i])
    
    return np.array(unlabeled_complete)


class CleanData(object):

    def __init__(self):
        pass

    def pairwise_dist(self, x, y):
        """		
        Args:
            x: N x D numpy array
            y: M x D numpy array
        Return:
            dist: N x M array, where dist[i, j] is the euclidean distance between
            x[i, :] and y[j, :]
        """
        x_squared_arr = np.square(x)
        x_squared = np.sum(x_squared_arr, axis=1, keepdims=True)
        
        y_squared_arr = np.square(y)
        y_squared = np.sum(y_squared_arr, axis=1, keepdims=True).T
        
        x_and_y_dot_product = x @ y.T
        
        dist_squared = x_squared + y_squared - (2.0 * x_and_y_dot_product)
        dist = np.sqrt(np.maximum(dist_squared, 0))
        
        return dist 


    def __call__(self, incomplete_points, complete_points, K, **kwargs):
        """		
        Function to clean or "fill in" NaN values in incomplete data points based on
        the average value for that feature for the K-nearest neighbors in the complete data points.
        
        Args:
            incomplete_points: N_incomplete x (D+1) numpy array, the incomplete labeled observations
            complete_points:   N_complete   x (D+1) numpy array, the complete labeled observations
            K: integer, corresponding to the number of nearest neighbors you want to base your calculation on
            kwargs: any other args you want
        Return:
            clean_points: (N_complete + N_incomplete) x (D+1) numpy array, containing both the complete points and recently filled points
        
        Notes:
            (1) The first D columns are features, and the last column is the class label
            (2) There may be more than just 2 class labels in the data (e.g. labels could be 0,1,2 or 0,1,2,...,M)
            (3) There will be at most 1 missing feature value in each incomplete data point (e.g. no points will have more than one NaN value)
            (4) You want to find the k-nearest neighbors, from the complete dataset, with the same class labels;
            (5) There may be missing values in any of the features. It might be more convenient to address each feature at a time.
            (6) Do NOT use a for-loop over N_incomplete; you MAY use a for-loop over the M labels and the D features (e.g. omit one feature at a time)
            (7) You do not need to order the rows of the return array clean_points in any specific manner
        """
        # Step 1: identify the unique class labels
        _, D = incomplete_points.shape
        D -= 2
        incomplete_class_labels = np.unique(incomplete_points[:, D + 1])
        complete_class_labels = np.unique(complete_points[:, D + 1])
        
        # Step 2: process each label separately to create the subset 
        for M in incomplete_class_labels:
            incomplete_subset_class = incomplete_points[incomplete_points[:, D + 1] == M] # gets a subset of a specific class label - incomplete
            complete_subset_class = complete_points[complete_points[:, D + 1] == M] # gets a subset of a specifc class label - complete
            for d in range(D):
                incomplete_wo_feature = np.delete(incomplete_subset_class, d, axis=1) # take out the feature - incomplete
                complete_wo_feature = np.delete(complete_subset_class, d, axis=1) # take out the feature - complete
                dist = self.pairwise_dist(incomplete_wo_feature, complete_wo_feature) # get pairwise distances
                
                k_nearest_neighbors_indices = np.argsort(dist, axis=1)[:, :K] # gets the indices for the k nearest neighbor                
                k_nearest_neighbors = complete_subset_class[k_nearest_neighbors_indices] # our k clusters
                feature_avg = np.nanmean(k_nearest_neighbors[:, :, d], axis=1) # avg value for that feature
                condition = np.isnan(incomplete_points[:, d]) & (incomplete_points[:, D + 1] == M) # choosing the rows in incomplete_points that contain the specific feature we're looking at and the correct class value
                incomplete_points[condition, d] = feature_avg[0]
                
        clean_points = np.vstack((complete_points, incomplete_points))
                 
        return clean_points


def median_clean_data(data):
    """	
    Args:
        data: N x (D+1) numpy array where only last column is guaranteed non-NaN values and is the labels
    Return:
        median_clean: N x (D+1) numpy array where each NaN value in data has been replaced by the median feature value
    Notes:
        (1) When taking the median of any feature, do not count the NaN value
        (2) Return all values to max one decimal point
        (3) The labels column will never have NaN values
    """
    median_clean = data
    N, D = data.shape
    D -= 1 
    
    for d in range(D):
        rows_of_feature = data[:, d]
        filtered_rows = np.delete(rows_of_feature, np.where(np.isnan(rows_of_feature)), axis=0)
        sorted_rows = np.sort(filtered_rows)
        median = np.median(sorted_rows)
        condition = np.isnan(data[:, d])
        median_clean[condition, d] = median

    return median_clean

class SemiSupervised(object):

    def __init__(self):
        pass

    def softmax(self, logit):
        """		
        Args:
            logit: N x D numpy array
        Return:
            prob: N x D numpy array where softmax has been applied row-wise to input logit
        """
        max_value_row = np.max(logit, axis=1, keepdims=True) # an N x 1 numpy array that has the max values in each row
        subtracted_logit = logit - max_value_row # use broadcasting to subtract each value in the rows from max 

        prob = np.exp(subtracted_logit) / np.sum(np.exp(subtracted_logit), axis=1, keepdims=True)

        return prob

    def logsumexp(self, logit):
        """		
        Args:
            logit: N x D numpy array
        Return:
            s: N x 1 array where s[i,0] = logsumexp(logit[i,:])
        """
        max_value_row = np.max(logit, axis=1, keepdims=True)
        subtracted_logit = logit - max_value_row
        exp_logit = np.exp(subtracted_logit)
  
        s = np.log(np.sum(exp_logit, axis=1, keepdims=True) + LOG_CONST)
        s += max_value_row

        return s 

    def normalPDF(self, logit, mu_i, sigma_i):
        """		
        Args:
            logit: N x D numpy array
            mu_i: 1xD numpy array, the center for the ith gaussian.
            sigma_i: 1xDxD numpy array, the covariance matrix of the ith gaussian.
        Return:
            pdf: 1xN numpy array, the probability distribution of N data for the ith gaussian
        
        Hint:
            np.diagonal() should be handy.
        """
        D = mu_i.shape[1]  # dimensionality
        N = logit.shape[0]  # number of data points

        sigma_inv = np.linalg.inv(sigma_i)
        sigma_det = np.linalg.det(sigma_i)
  
        diff = logit - mu_i  # N x D
        maha_dist = np.sum(diff @ sigma_inv * diff, axis=1)  # Mahalanobis distance, N x 1

        norm_factor = np.sqrt((2 * np.pi) ** D * sigma_det)  # normalization factor
        pdf = np.exp(-0.5 * maha_dist) / norm_factor  # N x 1
        
        return pdf


    def _init_components(self, points, K, **kwargs):
        """		
        Args:
            points: Nx(D+1) numpy array, the observations
            K: number of components
            kwargs: any other args you want
        Return:
            pi: numpy array of length K; contains the prior probabilities of each class k
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
        
        Hint:
            1. Given that the data is labeled, what's the best estimate for pi?
            2. Using the labels, you can look at individual clusters and estimate the best value for mu, sigma
        """
        N, D_plus_1 = points.shape
        D = D_plus_1 - 1  # Number of features (excluding the class label)
        
        # Extract the labels (class values) from the last column
        labels = points[:, D]

        # Pi
        pi = []
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        for label in unique_labels:
            pi.append(label_counts[np.where(unique_labels == label)[0][0]] / N)
        pi = np.array(pi)

        # Mu
        mu = []
        for label in unique_labels:
            # Get all points with the current label
            class_points = points[labels == label, :D]
            # Calculate the mean of the features for this class
            mu.append(np.mean(class_points, axis=0))
        mu = np.array(mu)

        # Sigma
        sigma = np.zeros((K, D, D))  # Initialize the covariance matrices
        for label in unique_labels:
            class_points = points[labels == label, :D]
            for d in range(D):
                variance = np.var(class_points[:, d])
                sigma[label, d, d] = variance  # Fill the diagonal for feature d

        return pi, mu, sigma

    def _ll_joint(self, points, pi, mu, sigma, **kwargs):
        """		
        Args:
            points: NxD numpy array, the observations
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
        Return:
            ll(log-likelihood): NxK array, where ll(i, j) = log pi(j) + log NormalPDF(points_i | mu[j], sigma[j])
        """
        ll = np.zeros((points.shape[0], len(pi)))
        for k in range(len(pi)):
            log_pi_k = np.log(pi[k] + LOG_CONST)
            log_pdf_k = np.log(self.normalPDF(points, mu[k], sigma[k]) + LOG_CONST)
            ll[:, k] = log_pi_k + log_pdf_k
        return ll

    def _E_step(self, points, pi, mu, sigma, **kwargs):
        """		
        Args:
            points: NxD numpy array, the observations
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
        Return:
            gamma: NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
        
        Hint: You should be able to do this with just a few lines of code by using _ll_joint() and softmax() defined above.
        """
        ll = self._ll_joint(points, pi, mu, sigma)
        gamma = self.softmax(ll)
        return gamma
        

    def _M_step(self, points, gamma, **kwargs):
        """		
        Args:
            points: NxD numpy array, the observations
            gamma: NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
        Return:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
        
        Hint:  There are formulas in the slide.
        """
        N, D = points.shape
        K = gamma.shape[1]
        pi = np.sum(gamma, axis=0) / N
        mu = (gamma.T @ points) / np.sum(gamma, axis=0)[:, None]
        sigma = np.zeros((K, D, D))
        for k in range(K):
            diff = points - mu[k]
            sigma[k] = (gamma[:, k][:, None] * diff).T @ diff / np.sum(gamma[:, k])
        return pi, mu, sigma

    def __call__(self, points, K, max_iters=100, abs_tol=1e-16, rel_tol=
        1e-16, **kwargs):
        """		
        Args:
            points: N x (D+1) numpy array, where
                - N is # points,
                - D is the number of features,
                - the last column is the point labels (when available) or NaN for unlabeled points
            K: integer, number of clusters
            max_iters: maximum number of iterations
            abs_tol: convergence criteria w.r.t absolute change of loss
            rel_tol: convergence criteria w.r.t relative change of loss
            kwargs: any additional arguments you want
        Return:
            pi, mu, sigma: (1xK np array, KxD numpy array, KxDxD numpy array)
        
        Hint: Look at Table 1 in the paper
        """
        pi, mu, sigma = self._init_components(points, K)
        prev_loss = None
        for _ in range(max_iters):
            gamma = self._E_step(points[:, :-1], pi, mu, sigma)
            pi, mu, sigma = self._M_step(points[:, :-1], gamma)
            loss = np.sum(self.logsumexp(self._ll_joint(points[:, :-1], pi, mu, sigma)))
            if prev_loss is not None and abs(loss - prev_loss) < abs_tol:
                break
            prev_loss = loss
        return pi, mu, sigma


class ComparePerformance(object):

    def __init__(self):
        pass

    @staticmethod
    def accuracy_semi_supervised(training_data, validation_data, K: int
        ) ->float:
        """
        Train a classification model using your SemiSupervised object on the training_data.
        Classify the validation_data using the trained model
        Return the accuracy score of the model's predicted classification of the validation_data

        Args:
            training_data: N_t x (D+1) numpy array, where
                - N_t is the number of data points in the training set,
                - D is the number of features, and
                - the last column represents the labels (when available) or a flag that allows you to separate the unlabeled data.
            validation_data: N_v x(D+1) numpy array, where
                - N_v is the number of data points in the validation set,
                - D is the number of features, and
                - the last column are the labels
            K: integer, number of clusters for SemiSupervised object
        Return:
            accuracy: floating number

        Note: validation_data will NOT include any unlabeled points
        """
        pi, mu, sigma = SemiSupervised()(training_data, K)
        classification_probs = SemiSupervised()._E_step(validation_data[:,
            :-1], pi, mu, sigma)
        classification = np.argmax(classification_probs, axis=1)
        semi_supervised_score = accuracy_score(validation_data[:, -1],
            classification)
        return semi_supervised_score

    @staticmethod
    def accuracy_GNB(training_data, validation_data) ->float:
        """
        Train a Gaussion Naive Bayes classification model (sklearn implementation) on the training_data.
        Classify the validation_data using the trained model
        Return the accuracy score of the model's predicted classification of the validation_data

        Args:
            training_data: N_t x (D+1) numpy array, where
                - N is the number of data points in the training set,
                - D is the number of features, and
                - the last column represents the labels
            validation_data: N_v x (D+1) numpy array, where
                - N_v is the number of data points in the validation set,
                - D is the number of features, and
                - the last column are the labels
        Return:
            accuracy: floating number

        Note: both training_data and validation_data will NOT include any unlabeled points
        """
        gnb_model = GaussianNB()
        gnb_model.fit(training_data[:, :-1], training_data[:, -1])
        gnb_score = gnb_model.score(validation_data[:, :-1],
            validation_data[:, -1])
        return gnb_score

    @staticmethod
    def accuracy_comparison():
        all_data = np.loadtxt('data/data.csv', delimiter=',')
        labeled_complete = complete_(all_data)
        labeled_incomplete = incomplete_(all_data)
        unlabeled = unlabeled_(all_data)
        cleaned_data = CleanData()(labeled_incomplete, labeled_complete, 10)
        cleaned_and_unlabeled = np.concatenate((cleaned_data, unlabeled), 0)
        labeled_data = np.concatenate((labeled_complete, labeled_incomplete), 0
            )
        median_cleaned_data = median_clean_data(labeled_data)
        print(f'All Data shape:                 {all_data.shape}')
        print(f'Labeled Complete shape:         {labeled_complete.shape}')
        print(f'Labeled Incomplete shape:       {labeled_incomplete.shape}')
        print(f'Labeled shape:                  {labeled_data.shape}')
        print(f'Unlabeled shape:                {unlabeled.shape}')
        print(f'Cleaned data shape:             {cleaned_data.shape}')
        print(f'Cleaned + Unlabeled data shape: {cleaned_and_unlabeled.shape}')
        validation = np.loadtxt('data/validation.csv', delimiter=',')
        accuracy_complete_data_only = ComparePerformance.accuracy_GNB(
            labeled_complete, validation)
        accuracy_cleaned_data = ComparePerformance.accuracy_GNB(cleaned_data,
            validation)
        accuracy_median_cleaned_data = ComparePerformance.accuracy_GNB(
            median_cleaned_data, validation)
        accuracy_semi_supervised = ComparePerformance.accuracy_semi_supervised(
            cleaned_and_unlabeled, validation, 2)
        print('===COMPARISON===')
        print(
            f'Supervised with only complete data, GNB Accuracy: {np.round(100.0 * accuracy_complete_data_only, 3)}%'
            )
        print(
            f'Supervised with KNN clean data, GNB Accuracy:     {np.round(100.0 * accuracy_cleaned_data, 3)}%'
            )
        print(
            f'Supervised with Median clean data, GNB Accuracy:    {np.round(100.0 * accuracy_median_cleaned_data, 3)}%'
            )
        print(
            f'SemiSupervised Accuracy:                          {np.round(100.0 * accuracy_semi_supervised, 3)}%'
            )
