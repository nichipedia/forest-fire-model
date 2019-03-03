import numpy as np
import math

def mse(model, X, Y):
    N = len(X)
    total = 0
    for i in range(N):
       total = total + (Y[i] - model.predict(X[i]))
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

def kFoldCrossValidation(xFolds, yFolds):
    N = len(xFolds)
    mseList = []
    rmseList = []
    for i in range(N):
        xTest = xFolds[i]
        yTest = yFolds[i]
        x = Null
        y = Null
        for j in range(N):
            if j != i:
                if x is Null:
                    x = xFolds[i]
                    y = yFolds[i]
                else:
                    x.concatenate(xFolds[i], axis=0)
                    y.concatenate(yFolds[i], axis=0)
        model = Model(x, y)
        mseList.append(mse(model, xTest, yTest))
        rmseList.append(rmse(model, xTest, yTest))
    return mseList, rmseList

