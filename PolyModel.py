import numpy as np
from numpy.linalg import pinv
from sklearn.preprocessing import PolynomialFeatures

class PolyModel:

    def __init__(self, X, Y):
        self.B = pinv(X.T.dot(X)).dot(X.T.dot(Y))

    def predict(X):
        yhat = X.dot(self.B)
        return yhat
