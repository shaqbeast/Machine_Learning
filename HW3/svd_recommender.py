import math
from typing import List, Tuple
import numpy as np
import pandas as pd
from eigenfaces import Eigenfaces


class SVDRecommender(object):

    def __init__(self) ->None:
        """
        Initialize with EigenFaces object for SVD purposes
        """
        self.eigenfaces = Eigenfaces()

    def load_movie_data(self, filepath: str='./data/movies.csv') ->None:
        """
        PROVIDED TO STUDENTS:
        Load movie data and create mappings from movie name to movie ID and vice versa
        """
        movies_df = pd.read_csv(filepath)
        self.movie_names_dict = dict(zip(movies_df.movieId, movies_df.title))
        self.movie_id_dict = dict(zip(movies_df.title, movies_df.movieId))

    def load_ratings_datasets(self, train_filepath: str=
        './data/ratings_train.csv', test_filepath: str=
        './data/ratings_test.csv') ->Tuple[pd.DataFrame, pd.DataFrame]:
        """
        PROVIDED TO STUDENTS: Load train and test user-movie ratings datasets
        """
        train = pd.read_csv(train_filepath)
        test = pd.read_csv(test_filepath)
        return train, test

    def get_movie_name_by_id(self, movie_id: int) ->str:
        """
        PROVIDED TO STUDENTS: Get movie name for corresponding movie id
        """
        return self.movie_names_dict[movie_id]

    def get_movie_id_by_name(self, movie_name: str) ->int:
        """
        PROVIDED TO STUDENTS: Get movie id for corresponding movie name
        """
        return self.movie_id_dict[movie_name]

    def recommender_svd(self, R: np.ndarray, k: int) ->Tuple[np.ndarray, np
        .ndarray]:
        """		
        Given the matrix of Ratings (R) and number of features (k), build the singular
        value decomposition of R with numpy's SVD and use the compress method that you
        implemented in eigenfaces.py to build reduced feature matrices U_k and V_k.
        
        Args:
            R: (NxM) numpy array the train dataset upon which we'll try to predict / fill in missing predictions
            k: (int) number of important features we would like to use for our prediction
        Return:
            U_k: (Nxk) numpy array containing k features for each user
            V_k: (kXM) numpy array containing k features for each movie
        """
        U, S, V = np.linalg.svd(R, full_matrices=False)
        U_compressed, S_compressed, V_compressed = self.eigenfaces.compress(U=U, S=S, V=V, k=k)
        U_k = U_compressed * np.sqrt(S_compressed)
        V_k = (np.sqrt(S_compressed) * V_compressed.T).T 
        
        return (U_k, V_k)
        

    def predict(self, R: np.ndarray, U_k: np.ndarray, V_k: np.ndarray,
        users_index: dict, movies_index: dict, user_id: int, movies_pool:
        list, top_n: int=3) ->np.ndarray:
        """		
        Given a user specified by `user_id`, recommend the `top_n` movies that the user would want to watch among a list of movies in `movies_pool`.
        Use the compressed SVD user matrix `U_k` and movie matrix `V_k` in your prediction.
        
        Args:
            R: (NxM) numpy array the train dataset containing only the given user-movie ratings
            U_k: (Nxk) numpy array containing k features for each user
            V_k: (kXM) numpy array containing k features for each movie
            users_index: (N,) dictionary containing a mapping from actual `userId` to the index of the user in R (or) U_k
            movies_index: (M,) dictionary containing a mapping from actual `movieId` to the movie of the user in R (or) V_k
            user_id: (str) the user we want to recommend movies for
            movies_pool: List(str) numpy array of movie_names from which we want to select the `top_n` recommended movies
            top_n: (int) number of movies to recommend
        
        Return:
            recommendation: (top_n,) numpy array of movies the user with user_id would be
                            most interested in watching next and hasn't watched yet.
                            Must be a subset of `movies_pool`
        
        Hints:
            1. You can use R to filter out movies already watched (or rated) by the user
            2. Utilize method `get_movie_id_by_name()` defined above to convert movie names to Id
            3. Utilize dictionaries `users_index` and `movies_index` to map between userId, movieId to their
                corresponding indices in R (or U_k, V_k)
        """
        # Step 1: Find the movies that the user has already watched using R matrix
        # Step 2: Remove those specific movies from the movie_pool
        # Step 3: For every remaining movie in movie_pool, get the movie_id and map it to get the index using movies_index in V_k
        # Step 4: Get what the user likes within U_k by mapping the user's Id to the index in U_k 
        # Step 5: Compute dot product for each of the remaining movies m_j with u_i and find the top_n movies 
        
        # Step 1
        user_index = users_index[user_id] # grabs the user_index
        user_movies = R[user_index] # grabs the movies the user has watched
        total_movies = np.isnan(user_movies) # finds the movies that user has and hasn't watched (True/False np.arr)
        
        movies_watched_indices = np.where(total_movies == False)[0] # grabs the movie indices the user has watched
        movies_watched = []
        for id, index in movies_index.items():
            if index in movies_watched_indices:
                movies_watched.append(id)
        for i in range(len(movies_watched)):
            movies_watched[i] = self.get_movie_name_by_id(movies_watched[i]) # convert the movie_id with the movie name
        
        # Step 2
        movies_watched_pool = np.isin(movies_pool, movies_watched) # finding which movies the user has watched within the pool 
        movies_watched_pool_indices = np.where(movies_watched_pool == False)[0] # finds the indices for which movies the user HASN'T watched
        movies_pool = movies_pool[movies_watched_pool_indices] # sets movie_pool to these movies the user hasn't watched
        
        # Step 3 
        movies_pool_id = [] # List(int) version of movies_pool
        for movie_name in movies_pool:
            movies_pool_id.append(self.get_movie_id_by_name(movie_name))
        movies_pool_id = np.array(movies_pool_id) # convert to np.array
        vectorized_lookup = np.vectorize(movies_index.get) # create a vectorized function for get
        movies_pool_indices = vectorized_lookup(movies_pool_id)
        
        # Step 4
        user_features = U_k[user_index] # 1 x K
        
        # Step 5
        user_movie_preferences = user_features @ V_k[:, movies_pool_indices] # V_k = K X M
        top_n_indices = np.argsort(user_movie_preferences)[-top_n:] # picks the top_n largest movie value indices
        recommendation = movies_pool[top_n_indices][::-1] # reverse to recommend highest value movie first
        
        return recommendation        

    def create_ratings_matrix(self, ratings_df: pd.DataFrame) ->Tuple[np.
        ndarray, dict, dict]:
        """
        FUNCTION PROVIDED TO STUDENTS

        Given the pandas dataframe of ratings for every user-movie pair,
        this method returns the data in the form a N*M matrix where,
        M[i][j] is the rating provided by user:(i) for movie:(j).

        Args:
            ratings_df: (pd.DataFrame) containing (userId, movieId, rating)
        """
        userList = ratings_df.iloc[:, 0].tolist()
        movieList = ratings_df.iloc[:, 1].tolist()
        ratingList = ratings_df.iloc[:, 2].tolist()
        users = list(set(ratings_df.iloc[:, 0]))
        movies = list(set(ratings_df.iloc[:, 1]))
        users_index = {users[i]: i for i in range(len(users))}
        pd_dict = {movie: [np.nan for i in range(len(users))] for movie in
            movies}
        for i in range(0, len(ratings_df)):
            movie = movieList[i]
            user = userList[i]
            rating = ratingList[i]
            pd_dict[movie][users_index[user]] = rating
        X = pd.DataFrame(pd_dict)
        X.index = users
        itemcols = list(X.columns)
        movies_index = {itemcols[i]: i for i in range(len(itemcols))}
        return np.array(X), users_index, movies_index
    
    
