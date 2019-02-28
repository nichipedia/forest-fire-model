import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_tools import preprocess_data

data, pcc = preprocess_data(False)
top5 = pcc.head().index.values
y = data['area'].values

for label in top5:
    x = data[label].values
    title = label + '.VS.' + 'area'
    plt.title(title)
    plt.xlabel(label)
    plt.ylabel('area')
    plt.scatter(x,y,c='blue')
    plt.savefig('./plots/' + title + '.png')
    plt.clf()
