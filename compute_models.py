import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from PolyModel import PolyModel
from data_tools import preprocess_data
from stats import kFoldCrossValidation
from data_tools import getKDataFolds

features = ['X', 'Y', 'month', 'day', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']
area = ['area']

pf1 = PolynomialFeatures(degree=1) # Creates Polynomial Features of degree 1
pf2 = PolynomialFeatures(degree=2) # Creates Polynomial Features of degree 2
pf4 = PolynomialFeatures(degree=4) # Creates Polynomial Features of degree 4
pfYC = PolynomialFeatures(degree=3) # Creates Polynomial Features of degree 5
pf6 = PolynomialFeatures(degree=6) # Creates Polynomial Features of degree 6


df, pcc = preprocess_data()




X = df[features].values

#X1 = pf1.fit_transform(X)
#X2 = pf2.fit_transform(X)
#X3 = pfYC.fit_transform(X)
#X4 = pf4.fit_transform(X)
#X6 = pf6.fit_transform(X)

Y = df[area].values

folds, top = getKDataFolds(10)
for i in range(1,7):
    mer, bler = kFoldCrossValidation(folds, i)
    print(mer)


#model2 = PolyModel(X2, Y)
#model3 = PolyModel(X3, Y)
#model4 = PolyModel(X4, Y)
#model6 = PolyModel(X6, Y)


