'''
purpose_: Custom Model Logistic Regression
author_: Sanjay Seetharam
status_: development
version_: 1.0
'''

from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import pandas as pd
from numpy.linalg import inv

class LogisticRegression(BaseEstimator, ClassifierMixin):
    '''Perform Simple Logistic Regression
    
    Parameters
    ----------
    columns_to_be_applied: feature
    
    Returns
    -------
    Array
        Probability > 0.5 = 1 ; 0
    
    '''

    def __init__(self, columns_to_be_applied) -> object:
       
        # Check if the Argument passed is List
        if not isinstance(columns_to_be_applied, list):
            raise ValueError("The LogisticRegression is expecting a list of strings with" 
                                "the column names to be applied.")

        # Check if the Argument contains string
        if not all([isinstance(item, str) for item in columns_to_be_applied]):
            raise ValueError(
                "The LogisticRegression is expecting a list of strings 'columns_to_be_applied'."
                "At least one of the items was not a string."
            )          
            
        self.columns_to_be_applied = columns_to_be_applied # Features
        self.w = None  # Weight
        
    @staticmethod
    def check_if_argument_is_pd_df(X):
        '''Check if Argument is Pandas DataFrame

        Parameters
        ---------
        X: array-like of shape (n_samples, n_features)
            Input Samples

        Returns
        ------
        Boolean;
            If False, raise error with a message

        '''
        if not isinstance(X, pd.DataFrame):
            raise RuntimeError("The argument expected should be pandas dataframe.")
        
    @staticmethod
    def input_df_checks(X, columns_to_be_applied, is_fitting = True):
        '''Perform data checks

        Parameters
        ----------
        X: (array-like, sparse matrix) of shape (n_samples, n_features)
            Input Sample

        columns_to_be_applied: array
            Feature

        is_fitting: boolean
            if true; "fitting"
            else; "transforming"

        '''
        LogisticRegression.check_if_argument_is_pd_df(X)

        # Check if the columns exist inside the pandas dataframe
        if not all([item in X.columns for item in columns_to_be_applied]):
            raise RuntimeError(
                f"The dataframe passed for {'fitting' if is_fitting else 'transforming'}"
                "does not contain all the required columns specified in the"
                "initialisation of the object."
            )

        # Check if the columns contain numeric values:
        if not (
            X[columns_to_be_applied].shape[1] == 
            X[columns_to_be_applied].select_dtypes(include = np.number).shape[1]
        ):
            raise RuntimeError(
                "The required columns for scaling are not numeric!"
            )                        

    def fit(self, X, y = None):
        
        '''Fit model according to the sample

        Parameters
        ----------
        X: (array-like, sparse matrix) of shape (n_samples, n_features)
            Input Sample

        y: None
            Ignored

        Returns
        -------
        object;
            Fitted estimator
        '''        

        # Check if dataframe has NaN values
        self.input_df_checks(X, self.columns_to_be_applied, False)
        if X[self.columns_to_be_applied].isnull().values.any():
            raise ValueError("in the required columns there are nan values!")
        
        # Check if target is numpy array
        if not isinstance(y, np.ndarray):
            raise RuntimeError("The label y was expected to be numpy array.")
        
        # Check if target is shape 1D array
        if not (y.ndim == 1):
            raise RuntimeError("The label y should be 1D.")
        
        # Check if target has real numbers
        if not all(np.isreal(y)):
            raise ValueError(
                "The label was expected to be a real number. The array passsed has some values that are not real."
            )
        
        X = X[self.columns_to_be_applied].values
        X = np.c_[np.ones(X.shape[0]), X]
        self.w = np.dot(np.dot(inv(np.dot(X.T, X)), X.T), y)

    def sigmoid(self, x):
        ''' Calculate numerically stable sigmoid curve
        
        Parameters
        ----------
        x: (number | array);
            input value/values
            
        Returns
        -------
            Number | Array
        '''
        return 1 / (1 + np.exp(-x))
    
    def predict(self, X):  
        """ determine probability of the given data samples
        
        Parameters
        ----------
        X: (array-like, sparse matrix) of shape (n_samples, n_features)
            Input Sample
            
        Returns
        -------
        Array;
            The probability of this number to be of class 1 (if prob >= 0.5 class=1)
        
        """
        if self.w is None:
            raise RuntimeError("You need to fit before you predict.")
        self.input_df_checks(X, self.columns_to_be_applied, False)
        X = X[self.columns_to_be_applied].values
        X = np.c_[np.ones(X.shape[0]), X]
        
        # Compute sigmoid and permute 1 with probability more than 50%
        return np.array([1 if i > 0.5 else 0 for i in self.sigmoid(np.dot(X, self.w))])        
        
