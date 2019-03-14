import numpy as np
import math
from sklearn.preprocessing import PolynomialFeatures
from PolyModel import PolyModel
from FeatureGenerator import genPolyFeatures

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

def kFoldCrossValidation(xFolds, features, degree=1, lamda=0):
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
        x = genPolyFeatures(x, degree)
        xTest = genPolyFeatures(xTest, degree)
        model = PolyModel(x, y, lamda)
        mseList.append(mse(model, xTest, yTest))
        rmseList.append(rmse(model, xTest, yTest))
    return mseList, rmseList

