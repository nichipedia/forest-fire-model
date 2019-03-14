import numpy as np
from numpy.linalg import pinv

class PolyModel:

    def __init__(self, X, Y, lamda):
        rows = len(X[-1,:])
        self.B = pinv(X.T.dot(X) + np.identity(rows)*lamda).dot(X.T.dot(Y))

    def predict(self, X):
        yhat = X.dot(self.B)
        return yhat
