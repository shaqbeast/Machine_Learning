import numpy as np


class HierarchicalClustering(object):

    def __init__(self, points: np.ndarray):
        self.N, self.D = points.shape
        self.points = points
        self.current_iteration = 0
        self.distances, self.cluster_ids = self.create_distances(points)
        self.clustering = np.zeros((self.N - 1, 4))
        self.cluster_sizes = np.zeros(self.N * 2 - 1)
        self.cluster_sizes[:self.N] = 1

    def create_distances(self, points: np.ndarray) ->np.ndarray:
        """		
        Create the pairwise distance matrix and index map, given points.
            The distance between a cluster and itself should be np.inf
        Args:
            points: N x D numpy array where N is the number of points
        Return:
            distances: N x N numpy array where distances[i][j] is the euclidean distance between points[i, :] and points[j, :].
                       distances[i, i] should always be np.inf in order to calculate the closest clusters more easily
            cluster_ids: (N,) numpy array where index_array[i] gives the cluster id of the i-th column
                         and i-th row of distances. Initially, each point i is assigned cluster id i
        """
        # Distances
        distances = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(i, self.N):
                if i == j: # distances[i, i]
                    distances[i, j] = np.inf
                else:
                    euclidean_distance = np.linalg.norm([points[i] - points[j]])
                    distances[i, j] = euclidean_distance
                    distances[j, i] = euclidean_distance # distances is symmetrical so we can populate this portion as well                  

        # Cluster_IDs
        cluster_ids = np.arange(self.N)
        
        return (distances, cluster_ids)
    def iterate(self):
        """		
        Performs one iteration of the algorithm
            1. Find the two closest clusters using self.distances (if there are multiple minimums, use the first occurence in flattened array)
            2. Replace first cluster's row and col with the newly combined cluster distances in self.distances,
               ensuring distances[i, i] is still np.inf
            3. Delete second cluster's row and col in self.distances
            4. Update self.cluster_ids where new cluster's id should be self.N + self.current_iteration,
               see definition in `create_distances` for more details
            5. Update self.cluster_sizes, where self.cluster_sizes[i] contains the number of points with cluster id i
            6. Update self.clustering, where
               self.clustering[self.current_iteration] = [first cluster id, second cluster id, distance between first and second clusters, size of new cluster]
            7. Update current_iteration
        Hint:
        You'll need to update self.distances, self.cluster_ids, self.cluster_sizes, self.clustering, and self.current_iteration
        
        While self.distances and self.cluster_ids only keeps information about the current clusters,
            self.cluster_sizes keep track of sizes for all clusters
        """
        # Step 1 
        if self.distances.shape[0] != 1:
            distances_flattened = self.distances.flatten() # 1D array of distances
            min_index_1D = np.argmin(distances_flattened)
            min_value = distances_flattened[min_index_1D]

            # Step 2 
            min_index_2D = np.unravel_index(min_index_1D, self.distances.shape) # finds the index of the 1st min value in 2D array
            first, second = min_index_2D # first = row(1st cluster), second = col(2nd cluster)
            for i in range(self.distances.shape[0]):
                if first == i: 
                    continue # let inf stay inf
                else:
                    self.distances[first, i] = min(self.distances[i, first], self.distances[i, second]) # this could def be problematic
                    self.distances[i, first] = self.distances[first, i]
            first_cluster_combined = self.cluster_ids[first]
            second_cluster_combined = self.cluster_ids[second]
            
            # Step 3 
            self.distances = np.delete(self.distances, second, axis=0)
            self.distances = np.delete(self.distances, second, axis=1)
            
            # Step 4
            self.cluster_ids = np.delete(self.cluster_ids, second)
            self.cluster_ids[first] = self.N + self.current_iteration
            
            # Step 5 
            self.cluster_sizes[self.N + self.current_iteration] += (self.cluster_sizes[first_cluster_combined] + self.cluster_sizes[second_cluster_combined])
            
            # Step 6 
            self.clustering[self.current_iteration] = [first_cluster_combined, second_cluster_combined, min_value, self.cluster_sizes[self.N + self.current_iteration]]
            
            # Step 7
            self.current_iteration += 1
        



    def fit(self):
        """		
        Fits the model on the dataset by calling `iterate`.
        Each call of `iterate` should combine two clusters, logging what was combined in self.clustering
        
        Return:
            self.clustering, where self.clustering[iteration_index] = [i, j, distance between i and j, size of new cluster]
        """
        for i in range(100):
            self.iterate()
        
        return self.clustering
