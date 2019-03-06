import numpy as np
from numpy.linalg import pinv

class PolyModel:

    def __init__(self, X, Y):
        self.B = pinv(X.T.dot(X)).dot(X.T.dot(Y))

    def predict(self, X):
        yhat = X.dot(self.B)
        return yhat
