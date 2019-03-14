import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from PolyModel import PolyModel
from data_tools import preprocess_data
from stats import kFoldCrossValidation
from data_tools import getKDataFolds

allFeatures = ['X', 'Y', 'month', 'day', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']
df, pcc = preprocess_data()
topFeatures = pcc[:5].index.values

folds, top = getKDataFolds(10)
print('FeatureSet, Order, STD MAE, STD RMSE, MEAN MAE, MEAN RMSE')
for i in range(1,7):
    mae, rmse = kFoldCrossValidation(folds, allFeatures, degree=i)
    stdMae = np.std(mae)
    meanMae = np.mean(mae)
    stdRmse = np.std(rmse)
    meanRmse = np.mean(mae)
    print('12, {}, {}, {}, {}, {}'.format(i, stdMae, stdRmse, meanMae, meanRmse))
    mae, rmse = kFoldCrossValidation(top, topFeatures, degree=i)
    stdMae = np.std(mae)
    meanMae = np.mean(mae)
    stdRmse = np.std(rmse)
    meanRmse = np.mean(mae)
    print('5, {}, {}, {}, {}, {}'.format(i, stdMae, stdRmse, meanMae, meanRmse))


errors = [0.00000005, 1.000005, 2.0004, 1, 2]
maeAll = []
rmseAll = []
maeTop = []
rmseTop = []
for error in errors:
    mae, rmse = kFoldCrossValidation(folds, topFeatures, degree=i, lamda=error)
    maeMean = np.mean(mae)
    maeAll.append(maeMean)
    rmseMean = np.mean(rmse)
    rmseAll.append(rmseMean)
    mae, rmse = kFoldCrossValidation(top, topFeatures, degree=i, lamda=error)
    maeMean = np.mean(mae)
    maeTop.append(maeMean)
    rmseMean = np.mean(rmse)
    rmseTop.append(rmseMean)


