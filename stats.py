import numpy as np
import math
from sklearn.preprocessing import PolynomialFeatures
from PolyModel import PolyModel

def mse(model, X, Y):
    N = len(X)
    total = 0
    for i in range(N):
       total = total + math.fabs(Y[i] - model.predict(X[i]))
    mse = total/N
    return mse

def rmse(model, X, Y):
    N = len(X)
    total = 0
    for i in range(N):
        temp = (Y[i] - model.predict(X[i]))
        temp = math.pow(temp, 2)
        total = total + temp
    total = total/N
    rmse = math.sqrt(total)
    return rmse

def kFoldCrossValidation(xFolds):
    features = ['X', 'Y', 'month', 'day', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']
    area = ['area']
    N = len(xFolds)
    mseList = []
    rmseList = []
    for i in range(N):
        xTest = xFolds[i][features].values
        yTest = xFolds[i][area].values
        x = None
        y = None
        for j in range(N):
            if j != i:
                if x is None:
                    x = xFolds[i][features].values
                    y = xFolds[i][area].values
                else:
                    np.append(x, xFolds[i][features].values, axis=0)
                    np.append(y, xFolds[i][area].values, axis=0)
                    #x.append(xFolds[i][features].values, axis=0)
                    #y.append(xFolds[i][area].values, axis=0)
        model = PolyModel(x, y)
        mseList.append(mse(model, xTest, yTest))
        rmseList.append(rmse(model, xTest, yTest))
    return mseList, rmseList

