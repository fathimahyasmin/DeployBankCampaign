from scipy.stats.mstats import winsorize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KBinsDiscretizer
import pandas as pd
import numpy as np


class Cleaning(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        
        # HANDLING OUTLIERS IN NUMERICAL COLUMNS
        # --------------------------------------
        # handling ouliers in balance column
        X['balance'] = winsorize(X['balance'], limits=(0.05, 0.05))
        X['duration'] = winsorize(X['duration'], limits=(0.05, 0.05))
        X['previous'] = winsorize(X['previous'], limits=(0.05, 0.05))

        # handling outliers in pdays column
        pdays_bins = [-2, -1, 90, 180, 270, 360, np.inf]
        pdays_labels = ['no prior contact', '0-3 months', '3-6 months', '6-9 months', '9-12 months', 'over a year']
        X['pdays'] = pd.cut(X['pdays'], bins=pdays_bins, labels=pdays_labels, right=False)

        # handling outliers in age column
        age_bins = [0, 14, 24, 54, 64, np.inf]
        age_labels = ['children', 'early working age', 'prime working age', 'mature working age', 'elderly']
        X['age'] = pd.cut(X['age'], bins=age_bins, labels=age_labels, right=False)

        # handling outliers in campaign column
        equal_freq = KBinsDiscretizer(n_bins = 5, encode = 'ordinal', strategy='quantile')
        equal_freq.set_output(transform='pandas')
        equal_freq.fit(X[['campaign']])
        # X['campaign'] = pd.qcut(X['campaign'], q=4, labels=['low frequency', 'medium frequency', 'high frequency', 'very high frequency'])

        # HANDLE PROBLEMS IN CATEGORICAL COLUMNS
        # ---------------------------------------

        # reduce cardinality and handle unknown in 'job' column
        X['job'].replace({"entrepreneur" : "self-employed",
                          "student": "unemployed",
                          "retired": "unemployed",   
                          "unknown" : "unemployed",
                          "housemaid": "services"},
                          inplace= True)
        
        # replace value month
        X['month'].replace({"jan": 1,
                            "feb": 2,
                            "mar": 3,
                            "apr": 4,
                            "may": 5,
                            "jun": 6,
                            "jul": 7,
                            "aug": 8,
                            "sep": 9,
                            "oct": 10,
                            "nov": 11,
                            "dec": 12},inplace= True)
        
        # replace value for ordinal encoding
        X['poutcome'].replace({'unknown': 0, 
                               'other': 1, 
                               'failure': 2, 
                               'success': 3}, inplace = True)
        
        # replace value for ordinal encoding
        X['education'].replace({'unknown': 0, 
                               'primary': 1, 
                               'secondary': 2, 
                               'tertiary': 3}, inplace = True)

        return X.to_numpy()