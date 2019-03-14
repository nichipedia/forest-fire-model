from sklearn.preprocessing import PolynomialFeatures
import numpy as np


def genPolyFeatures(X, degree=1):
    ones = np.ones((len(X[:,-1]), 1))
    col = np.array(X[:,3], ndmin=2).T
    col1 = np.array(X[:,-1], ndmin=2).T
    sqr = np.power(col, degree)
    ab = col*col1
    X = np.concatenate((ones, X), 1)
    if degree == 1:
        return X
    X = np.concatenate((X, ab), 1)
    X = np.concatenate((X, sqr), 1)
    return X
