"""
File: kmeans.py
Project: Downloads
File Created: Feb 2021
Author: Rohit Das
"""
import numpy as np

class KMeans(object):

    def __init__(self, points, k, init='random', max_iters=10000, rel_tol=1e-05
        ):
        """
        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            init : how to initial the centers
            max_iters: maximum number of iterations (Hint: You could change it when debugging)
            rel_tol: convergence criteria with respect to relative change of loss (number between 0 and 1)
        Return:
            none
        """
        self.points = points
        self.K = k
        if init == 'random':
            self.centers = self.init_centers()
        else:
            self.centers = self.kmpp_init()
        self.assignments = None
        self.loss = 0.0
        self.rel_tol = rel_tol
        self.max_iters = max_iters

    def init_centers(self):
        """		
            Initialize the centers randomly
        Return:
            self.centers : K x D numpy array, the centers.
        Hint: Please initialize centers by randomly sampling points from 1the dataset in case the autograder fails.
        """
        N, _ = self.points.shape
        center_indices = np.random.choice(N, self.K, replace=True)  # out of N points, randomly select K unique indices
        self.centers = self.points[center_indices]
        return self.centers
            

    def kmpp_init(self):
        """		
            Use the intuition that points further away from each other will probably be better initial centers.
            To complete this method, refer to the steps outlined below:.
            1. Sample 1% of the points from dataset, uniformly at random (UAR) and without replacement.
            This sample will be the dataset the remainder of the algorithm uses to minimize initialization overhead.
            2. From the above sample, select only one random point to be the first cluster center.
            3. For each point in the sampled dataset, find the nearest cluster center and record the squared distance to get there.
            4. Examine all the squared distances and take the point with the maximum squared distance as a new cluster center.
            In other words, we will choose the next center based on the maximum of the minimum calculated distance
            instead of sampling randomly like in step 2. You may break ties arbitrarily.
            5. Repeat 3-4 until all k-centers have been assigned. You may use a loop over K to keep track of the data in each cluster.
        Return:
            self.centers : K x D numpy array, the centers.
        Hint:
            You could use functions like np.vstack() here.
        """
        
        # check method to make sure it works 
        
        # Step 1
        N, _ = self.points.shape
        one_percent_value = int(0.01 * N)
        one_percent_indices = np.random.choice(N, one_percent_value, replace=False)
        one_percent_points = self.points[one_percent_indices]
        
        # Step 2
        one_percent_rows, _ = one_percent_points.shape
        random_index = np.random.randint(0, one_percent_value)
        first_cluster_center = one_percent_points[random_index]
        cluster_centers = first_cluster_center.reshape(1, -1)
        
        # Step 5
        for _ in range(self.K - 1):
            # Step 3 
            distance = pairwise_dist(one_percent_points, cluster_centers) # get the distances between the sampled points and the cluster centers
            min_distance_values = np.min(distance, axis=1) # get the minimum distance values in distance 
            squared_distances = np.square(min_distance_values)  # square the distance

            # Step 4 
            max_index = np.argmax(squared_distances) 
            new_cluster_center = one_percent_points[max_index].reshape(1, -1) # contains the max_row
            cluster_centers = np.vstack([cluster_centers, new_cluster_center])
        
        self.centers = cluster_centers
        return self.centers
        
        

    def update_assignment(self):
        """		
            Update the membership of each point based on the closest center
        Return:
            self.assignments : numpy array of length N, the cluster assignment for each point
        Hint: Do not use loops for the update_assignment function
        Hint: You could call pairwise_dist() function
        Hint: In case the np.sqrt() function is giving an error in the pairwise_dist() function, you can use the squared distances directly for comparison.
        """
        distances = pairwise_dist(self.points, self.centers)
        self.assignments = np.argmin(distances, axis=1)
        
        return self.assignments

    def update_centers(self):
        """		
            update the cluster centers
        Return:
            self.centers: new centers, a new K x D numpy array of float dtype, where K is the number of clusters, and D is the dimension.
        
        HINT: Points may be integer, but the centers should not have to be. Watch out for dtype casting!
        HINT: If there is an empty cluster then it won't have a cluster center, in that case the number of rows in self.centers can be less than K.
        """
        _, D = self.points.shape 
        new_centers = np.zeros((self.K, D), dtype=float)
        
        # compute mean of each cluster
        for k in range(self.K):
            indices = np.where(self.assignments == k)
            points_in_cluster = self.points[indices]
            if points_in_cluster.shape[0] > 0: # if rows is greater than 0
                new_cluster_center = np.mean(points_in_cluster, axis=0) # column-wise average
                new_centers[k] = new_cluster_center
        
        self.centers = new_centers
        
        return self.centers
            
        

    def get_loss(self):
        """		
            The loss will be defined as the sum of the squared distances between each point and it's respective center.
        Return:
            self.loss: a single float number, which is the objective function of KMeans.
        """
        
        squared_distances = np.square(pairwise_dist(self.points, self.centers))
        rows_indices = np.arange(len(squared_distances)) # gets row indices of squared distances
        self.loss = np.sum(squared_distances[rows_indices, self.assignments.astype(int)]) # finds the rows and their respective cluster assignemnts (very powerful; you can access an array of indices rather than just one)
        
        return self.loss
        

    def train(self):
        """		
            Train KMeans to cluster the data:
                0. Recall that centers have already been initialized in __init__
                1. Update the cluster assignment for each point
                2. Update the cluster centers based on the new assignments from Step 1
                3. Check to make sure there is no mean without a cluster,
                   i.e. no cluster center without any points assigned to it.
                   - In the event of a cluster with no points assigned,
                     pick a random point in the dataset to be the new center and
                     update your cluster assignment accordingly.
                4. Calculate the loss and check if the model has converged to break the loop early.
                   - The convergence criteria is measured by whether the percentage difference
                     in loss compared to the previous iteration is less than the given
                     relative tolerance threshold (self.rel_tol).
                   - Relative tolerance threshold (self.rel_tol) is a number between 0 and 1.
                5. Iterate through steps 1 to 4 max_iters times. Avoid infinite looping!
        
        Return:
            self.centers: K x D numpy array, the centers
            self.assignments: Nx1 int numpy array
            self.loss: final loss value of the objective function of KMeans.
        
        HINT: Do not loop over all the points in every iteration. This may result in time out errors
        HINT: Make sure to care of empty clusters. If there is an empty cluster the number of rows in self.centers can be less than K.
        """
        
        # Step 5 
        loss_values = np.empty(0)
        for i in range(self.max_iters):
            # Step 1 
            self.assignments = self.update_assignment()
            
            # Step 2 
            new_centers = self.update_centers()
            self.centers = new_centers 
            
            # Step 3 
            possible_cluster_indices = np.arange(len(self.centers)) # grabs the range of the indices for the clusters 
            missing_values = np.setdiff1d(possible_cluster_indices, self.assignments) # finds the specific indices of which clusters don't have points assigned to it
            N, _ = self.points.shape
            new_center_indices = np.random.choice(N, len(missing_values), replace=False)  # randomly selects new values for centers based on how many missing values we have
            self.centers[missing_values] = self.points[new_center_indices] 
            
            # Step 4 
            loss = self.get_loss()
            loss_values = np.append(loss_values, loss)
            if i > 0:
                current_loss = loss_values[i]
                previous_loss = loss_values[i - 1]
                percentage_diff = abs(current_loss - previous_loss) / ((current_loss + previous_loss) / 2)
                if percentage_diff < self.rel_tol:
                    break
        
        return (self.centers, self.assignments, self.loss) 
                    
        
        

def pairwise_dist(x, y):
    """	
    Args:
        x: N x D numpy array
        y: M x D numpy array
    Return:
            dist: N x M array, where dist2[i, j] is the euclidean distance between
            x[i, :] and y[j, :]
    
    HINT: Do not use loops for the pairwise_distance function
    """
    
    # Why do I have to transpose y at all? 
    x_squared_arr = np.square(x)
    x_squared = np.sum(x_squared_arr, axis=1, keepdims=True)
    
    y_squared_arr = np.square(y)
    y_squared = np.sum(y_squared_arr, axis=1, keepdims=True).T
    
    x_and_y_dot_product = x @ y.T
    
    dist_squared = x_squared + y_squared - (2.0 * x_and_y_dot_product)
    dist = np.sqrt(np.maximum(dist_squared, 0))
    
    return dist 


def fowlkes_mallow(xGroundTruth, xPredicted):
    """	
    Args:
        xPredicted : list of length N where N = no. of test samples
        xGroundTruth: list of length N where N = no. of test samples
    Return:
        fowlkes-mallow value: final coefficient value of type np.float64
    
    HINT: You can use loops for this function.
    HINT: The idea is to make the comparison of Predicted and Ground truth in pairs.
        1. Choose a pair of points from the Prediction.
        2. Compare the prediction pair pattern with the ground truth pair.
        3. Based on the analysis, we can figure out whether it's a TP/FP/FN/FP.
        4. Then calculate fowlkes-mallow value
    """
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    pair = 2
    
    for i in range(len(xPredicted)):
        for j in range(i + 1, len(xPredicted)):
            pair_indices = [i, j]
            # find the random indices for the pairs            
            pair_predicted = []
            pair_ground = []
            for k in range(len(pair_indices)):
                pair_predicted.append(xPredicted[pair_indices[k]])
                pair_ground.append(xGroundTruth[pair_indices[k]])
            
            if pair_predicted[0] == pair_predicted[1] and pair_ground[0] == pair_ground[1]:
                TP += 1 
            elif pair_predicted[0] == pair_predicted[1] and pair_ground[0] != pair_ground[1]:
                FP += 1
            elif pair_predicted[0] != pair_predicted[1] and pair_ground[0] == pair_ground[1]:      
                FN += 1
            else:
                TN += 1  
    
    FM = TP / (np.sqrt((TP + FN) * (TP + FP)))
    
    return FM 
        
        
    
    
