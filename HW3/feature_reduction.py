import math
import sys
from typing import List
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


class FeatureReduction(object):

    def __init__(self):
        pass

    @staticmethod
    def forward_selection(data: pd.DataFrame, target: pd.Series,
        significance_level: float=0.1) ->dict:
        """		
        Args:
            data: (pandas data frame) contains the features
            target: (pandas series) represents target values to search to generate significant features
            significance_level: (list) threshold to reject the null hypothesis
        Return:
            best_features: (list) contains significant features. Each feature name is a string.
        Hint:
            Forward Selection Steps:
            1. Start with an empty list of selected features (null model)
            2. For each feature NOT yet included in the selected features:
                - Fit a simple regression model using the the selected features AND the feature under consideration
                - Use sm.OLS() with sm.add_constant() to add a bias constant to your features when fitting the model
                - Record the p-value of the feature under consideration
                - Use OLSResults.pvalues to access the pvalues of each feature in a fitted model
            3. Find the feature with the minimum p-value.
                - If the feature's p-value < significance level, add the feature to the selected features and repeat from Step 2.
                - Otherwise, stop and return the selected features
        
            - You can access the feature names using data.columns.tolist().
            - You can index into a Pandas dataframe using multiple column names at once in a list.
        """
        
        # Step 1
        selected_features = list()
        
        # Step 2
        remaining_features = data.columns.to_list()
        while remaining_features: # while there is still features remaining in the column
            model_p_values = dict()
            for feature in remaining_features:
                X = data[selected_features + [feature]] # testing the feature with the selected features
                X = sm.add_constant(X)
                model = sm.OLS(target, X).fit()
                p_values = model.pvalues
                model_p_values[feature] = p_values[feature]
            
            # Step 3
            min_feature = min(model_p_values, key=model_p_values.get) # get the feature with min p-value
            min_p_value = model_p_values[min_feature] # find p-value
            if min_p_value < significance_level:
                selected_features.append(min_feature)
                remaining_features.remove(min_feature)
            else:
                break
            
        return selected_features

        
        
                
                
                
                


    @staticmethod
    def backward_elimination(data: pd.DataFrame, target: pd.Series,
        significance_level: float=0.1) ->dict:
        """		
        Args:
            data: (pandas data frame) contains the features
            target: (pandas series) represents target values to search to generate significant features
            significance_level: (list) threshold to reject the null hypothesis
        Return:
            best_features: (float) contains significant features. Each feature name is a string.
        Hint:
            Backward Elimination Steps:
            1. Start with a full list of ALL features as selected features.
            2. Fit a simple regression model using the selected features
                - Use sm.OLS() with sm.add_constant() to add a bias constant to your features when fitting the model
            3. Find the feature with the maximum p-value.
                - Use OLSResults.pvalues to access the pvalues of each feature in a fitted model
                - If the feature's p-value >= significance level, REMOVE the feature to the selected features and repeat from Step 2.
                - Otherwise, stop and return the selected features.
        
            - You can access the feature names using data.columns.tolist().
            - You can index into a Pandas dataframe using multiple column names at once in a list.
        """
        # Step 1 
        selected_features = data.columns.to_list()
        
        # Step 2
        while selected_features:
            X = data[selected_features]
            X = sm.add_constant(X)
            model = sm.OLS(target, X).fit()
            p_values = model.pvalues
            
            # Step 3
            max_p_value = p_values.max()
            max_feature = p_values.idxmax()
            if max_p_value >= significance_level:
                selected_features.remove(max_feature)
            else:
                break
            
        return selected_features

    def evaluate_features(data: pd.DataFrame, y: pd.Series, features: list
        ) ->None:
        """
        PROVIDED TO STUDENTS

        Performs linear regression on the dataset only using the features discovered by feature reduction for each significance level.

        Args:
            data: (pandas data frame) contains the feature matrix
            y: (pandas series) output labels
            features: (python list) contains significant features. Each feature name is a string
        """
        print(f'Significant Features: {features}')
        data_curr_features = data[features]
        x_train, x_test, y_train, y_test = train_test_split(data_curr_features,
            y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = math.sqrt(mse)
        print(f'RMSE: {rmse}')
        print()
