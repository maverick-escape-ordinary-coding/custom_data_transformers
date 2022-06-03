from sklearn.base import BaseEstimator, RegressorMixin

import pandas as pd
import numpy as np
from numpy.linalg import inv


# Notes for the following model: https://www.stat.cmu.edu/~cshalizi/mreg/15/lectures/13/lecture-13.pdf
# Detailed sources: http://www.dcs.gla.ac.uk/~srogers/firstcourseml/
from sklearn.utils.multiclass import is_multilabel


class LinearRegression(BaseEstimator, RegressorMixin):

    def __init__(self, columns_to_be_applied):
        # Check if the argument passed is list.
        if not isinstance(columns_to_be_applied, list):
            raise ValueError("The LinearRegression is expecting a list of strings with the column names to be applied.")

        if not all([isinstance(item, str) for item in columns_to_be_applied]):
            raise ValueError(
                "The LinearRegression is expecting a list of strings 'columns_to_be_applied'. At least one of the items was not a str.")

        self.columns_to_be_applied = columns_to_be_applied # feature
        self.w = None # weights

    @staticmethod
    def check_if_argument_is_pd_df(X):
        if not isinstance(X, pd.DataFrame):
            raise RuntimeError("The argument expected should be pandas df.")

    @staticmethod
    def input_df_checks(X, columns_to_be_applied, is_fitting=True):
        LinearRegression.check_if_argument_is_pd_df(X)
        # Check if the columns exist inside the pandas df.
        if not all([item in X.columns for item in columns_to_be_applied]):
            raise RuntimeError(
                "The dataframe passed for {} does not contain all the required columns specified in the "
                "initialization of the object.".format("fitting" if is_fitting else "transforming"))

        # Check if the columns contain numeric values:
        if not (
                X[columns_to_be_applied].shape[1]
                ==
                X[columns_to_be_applied].select_dtypes(include=np.number).shape[1]
        ):
            raise RuntimeError(
                "The required columns for LinearRegression are not numeric!"
            )

    def fit(self, X, y = None):
        self.input_df_checks(X, self.columns_to_be_applied)

        if X[self.columns_to_be_applied].isnull().values.any():
            raise ValueError("In the required columns there are nan values!")

        if not isinstance(y, np.ndarray):
            raise RuntimeError("The label y was expected to be a numpy array.")

        if not (y.ndim == 1):
            raise RuntimeError("The label y should be 1D.")

        if not all(np.isreal(y)):
            raise ValueError(
                "The label was expected to be a real number. The array passed has some values that are not real."
            )
        X = X[self.columns_to_be_applied].values
        X = np.c_[np.ones(X.shape[0]), X]
        self.w = np.dot(np.dot(inv(np.dot(X.T, X)), X.T), y)

    def predict(self, X):
        if self.w is None:
            raise RuntimeError("You need to fit before you predict.")
        self.input_df_checks(X, self.columns_to_be_applied, False)
        X = X[self.columns_to_be_applied].values
        X = np.c_[np.ones(X.shape[0]), X]
        return np.dot(self.w, X.T)
