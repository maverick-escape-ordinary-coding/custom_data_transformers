from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.stats import bernoulli
from sklearn.utils.multiclass import is_multilabel
import numpy as np
import pandas as pd


class BernoulliEstimator(BaseEstimator, ClassifierMixin):

    def __init__(self):
        self.bern_prob = None

    def fit(self, X, y=None):

        if not isinstance(y, np.ndarray):
            raise RuntimeError("The label y was expected to be a numpy array 1D.")

        if not (y.ndim == 1):
            raise RuntimeError("The label y should be 1D.")

        if not is_multilabel(y) and np.array_equal(y, y.astype(bool)):
            self.bern_prob = np.sum(np.where(y == 1, 1, 0)) / y.shape[0]
        else:
            raise RuntimeError("The label y should be binary with 1s and 0s!")

    def predict(self, X):
        if self.bern_prob is None:
            raise RuntimeError("You must train the classifier before predicting data!")

        if not isinstance(X, pd.DataFrame):
            raise RuntimeError("The X should be a pandas dataframe!")

        # return as prediction a sample from the Bernoulli dist, the sample has as size the number of rows of the X
        return bernoulli.rvs(self.bern_prob, size=X.shape[0])
