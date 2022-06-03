from sklearn.base import TransformerMixin, BaseEstimator

import pandas as pd
import numpy as np


# This code is not meant for production.
# Is there any change you would suggest us making?
class MinMaxScaler(BaseEstimator, TransformerMixin):

    def __init__(self, columns_to_be_applied, ignore_nan=False):

        # Check if the argument passed is list.
        if not isinstance(columns_to_be_applied, list):
            raise ValueError("The MinMaxScaler is expecting a list of strings with the column names to be applied.")

        if not all([isinstance(item, str) for item in columns_to_be_applied]):
            raise ValueError(
                "The MinMaxScaler is expecting a list of strings 'columns_to_be_applied'. At least one of the items was not a str.")

        if not isinstance(ignore_nan, bool):
            raise ValueError("The MinMaxScaler is expecting the argument 'ignore_nan' to be boolean.")

        self.columns_to_be_applied = columns_to_be_applied
        self.min_value_df = None
        self.max_value_df = None
        self.ignore_nan = ignore_nan

    @staticmethod
    def __min_max_scaler(min_value, max_value):
        def scaler(x):
            return (x - min_value) / (max_value - min_value)
        return scaler

    @staticmethod
    def check_if_argument_is_pd_df(X):
        if not isinstance(X, pd.DataFrame):
            raise RuntimeError("The argument expected should be pandas df.")

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X)
        return self.transform(X)

    @staticmethod
    def input_df_checks(X, columns_to_be_applied, is_fitting=True):
        MinMaxScaler.check_if_argument_is_pd_df(X)
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
                "The required columns for scaling are not numeric!"
            )

    def fit(self, X, y=None):
        self.input_df_checks(X, self.columns_to_be_applied)

        if (not self.ignore_nan) and X[self.columns_to_be_applied].isnull().values.any():
            raise ValueError("In the required columns there are nan values!")

        self.min_value_df = {}
        self.max_value_df = {}
        for col in self.columns_to_be_applied:
            self.min_value_df[col] = X.loc[X[col].notna(), [col]].min()
            self.max_value_df[col] = X.loc[X[col].notna(), [col]].max()

    def transform(self, X, y=None):
        self.input_df_checks(X, self.columns_to_be_applied, False)
        for col in self.columns_to_be_applied:
            X.loc[:, [col]] = X[col].apply(
                MinMaxScaler.__min_max_scaler(
                    self.min_value_df[col],
                    self.max_value_df[col]
                )
            )
        return X
