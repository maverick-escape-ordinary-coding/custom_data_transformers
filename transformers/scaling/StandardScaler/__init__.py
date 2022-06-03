'''
purpose_: Custom Transformer Standard Scaler
author_: Sanjay Seetharam
status_: development
version_: 1.0
'''

from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd
import numpy as np

class StandardScaler(BaseEstimator, TransformerMixin):
    '''Perform Z-score normalisation

    Parameters
    ----------
    columns_to_be_applied: Column to Standardise

    Returns
    -------
    standardised array
    '''

    def __init__(self, columns_to_be_applied, ignore_nan = False) -> object:

        # Check if the Argument passed is List
        if not isinstance(columns_to_be_applied, list):
            raise ValueError("The StandardScaler is expecting a list of strings with" 
                             "the column names to be applied.")

        # Check if the Argument contains string
        if not all([isinstance(item, str) for item in columns_to_be_applied]):
            raise ValueError(
                "The StandardScaler is expecting a list of strings 'columns_to_be_applied'."
                "At least one of the items was not a string."
            )

        # Check condition boolean for ignore_nan
        if not isinstance(ignore_nan, bool):
            raise ValueError("The StandardScaler is expecting the "
                             "argument 'ignore_nan' to be boolean.")

        self.columns_to_be_applied = columns_to_be_applied # Features
        self.mean_value_df = None # Mean of Vector
        self.sd_value_df = None # Standard Deviation of Vector
        self.ignore_nan = ignore_nan # Condition for NaN
        
    @staticmethod
    def __standard_scaler(mean_value, sd_value) -> float:
        '''feature standardisation

        Parameters
        ----------
        mean_value: float;
            mean of feature vector

        sd_value: float;
            standard deviation of feature vector

        Returns
        -------
        scaler: float;
            standardised value

        '''
        def scaler(x):
            '''
            standardised_value =
            feature_vector - mean(feature_vector) / standard_deviation(feature_vector)
            '''
            return (x - mean_value) / sd_value
        return scaler

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

    def fit_transform(self, X, y = None, **fit_params):
        ''' Fit to data, then transform it

        Parameters
        ----------
        X: (array-like, sparse matrix) of shape (n_samples, n_features)
            Input Sample

        y: array-like of shape (n_samples) or (n_samples, n_outputs), default = None
            Target Values. Return None for unsupervised 

        **fit_params: dict
            Additional fitting parameters

        Returns
        -------
        Transformed Array

        '''
        self.fit(X)
        return self.transform(X)

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
        StandardScaler.check_if_argument_is_pd_df(X)

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
        '''Compute mean and standard deviation to be used for later standardising

        Parameters
        ----------
        X: (array-like, sparse matrix) of shape (n_samples, n_features)
            Input Sample

        y: None
            Ignored

        Returns
        -------
        object;
            Fitted Scaler
        '''
        self.input_df_checks(X, self.columns_to_be_applied, False)
        if (not self.ignore_nan) and X[self.columns_to_be_applied].isnull().values.any():
            raise ValueError("in the required columns there are nan values!")

        self.mean_value_df = {}
        self.sd_value_df = {}

        for col in self.columns_to_be_applied:
            self.mean_value_df[col] = X.loc[X[col].notna(), [col]].mean()
            self.sd_value_df[col] = X.loc[X[col].notna(), [col]].std()

    def transform(self, X, y = None):
        '''Standardise features of X

        Parameters
        ----------
        X: (array-like, sparse matrix) of shape (n_samples, n_features)
            Input Sample

        y: None
            Ignored

        Returns
        -------
        X: (array-like, sparse matrix)
            Transformed data
        '''

        self.input_df_checks(X, self.columns_to_be_applied, False)
        for col in self.columns_to_be_applied:
            X.loc[:, [col]] = X[col].apply(
                StandardScaler.__standard_scaler(
                    self.mean_value_df[col],
                    self.sd_value_df[col]
                )
            )

        return X
        