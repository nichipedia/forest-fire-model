from PolyModel import PolyModel
import pandas as pd
import numpy as np

X = np.loadtxt('../HousingData/X.txt')
Y = np.loadtxt('../HousingData/Y.txt')

Q = np.matrix([1, 5, 3, 25000])

model = PolyModel(X, Y)

print(model.predict(Q))
