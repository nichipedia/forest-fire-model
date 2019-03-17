import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from PolyModel import PolyModel
from data_tools import preprocess_data
from stats import kFoldCrossValidation
from data_tools import getKDataFolds
import matplotlib.pyplot as plt

def plotter(x, y, xTitle, yTitle):
    title = '{}_VS_{}'.format(yTitle, xTitle)
    plt.title(title)
    plt.xlabel(xTitle)
    plt.ylabel(yTitle)
    plt.scatter(x,y,c='blue')
    plt.savefig('./plots/{}.png'.format(title))
    plt.clf()

allFeatures = ['X', 'Y', 'month', 'day', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']
df, pcc = preprocess_data()
topFeatures = pcc[:5].index.values

x = [1,2,4,5,6]
yMaeAll = []
yRmseAll = []
yMaeTop = []
yRmseTop = []
folds, top = getKDataFolds(10)
print('FeatureSet, Order, STD MAE, STD RMSE, MEAN MAE, MEAN RMSE')
for i in range(1,7):
    mae, rmse = kFoldCrossValidation(folds, allFeatures, degree=i)
    stdMae = np.std(mae)
    meanMae = np.mean(mae)
    stdRmse = np.std(rmse)
    meanRmse = np.mean(mae)
    if i != 3:
        yMaeAll.append(meanMae)
        yRmseAll.append(meanRmse)
    print('12, {}, {}, {}, {}, {}'.format(i, stdMae, stdRmse, meanMae, meanRmse))
    mae, rmse = kFoldCrossValidation(top, topFeatures, degree=i)
    stdMae = np.std(mae)
    meanMae = np.mean(mae)
    stdRmse = np.std(rmse)
    meanRmse = np.mean(mae)
    print('5, {}, {}, {}, {}, {}'.format(i, stdMae, stdRmse, meanMae, meanRmse))
    if i != 3:
        yMaeTop.append(meanMae)
        yRmseTop.append(meanRmse)

plotter(x, yMaeAll, 'Order', 'MAEAll')
plotter(x, yRmseAll, 'Order', 'RMSEAll')
plotter(x, yMaeTop, 'Order', 'MAETop')
plotter(x, yMaeTop, 'Order', 'RMSETop')


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

plotter(np.log(errors), np.log(maeAll), 'lnLamda', 'MAEAll')
plotter(np.log(errors), np.log(rmseAll), 'lnLamda', 'RMSEAll')
plotter(np.log(errors), np.log(maeTop), 'lnLamda', 'MAETop')
plotter(np.log(errors), np.log(rmseTop), 'lnLamda', 'RMSETop')
