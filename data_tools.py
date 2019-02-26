import numpy as np
import pandas as pd
from sklearn import preprocessing
from scipy.stats.stats import pearsonr

months = {
      'jan': 1,
      'feb': 2,
      'mar': 3,
      'apr': 4,
      'may': 5,
      'jun': 6,
      'jul': 7,
      'aug': 8,
      'sep': 9,
      'oct': 10,
      'nov': 11,
      'dec': 12
}

days = {
        'mon': 1,
        'tue': 2,
        'wed': 3,
        'thu': 4,
        'fri': 5,
        'sat': 6,
        'sun': 0
}

rows = {
        0: 'X',
        1: 'Y',
        2: 'month',
        3: 'day',
        4: 'FFMC',
        5: 'DMC',
        6: 'DC',
        7: 'ISI',
        8: 'temp',
        9: 'RH',
        10: 'wind',
        11: 'rain',
        12: 'area'
}

def preprocess_data():
    df = pd.read_csv('./data.csv') # Read CSV into a data frame
    df.month = df.month.map(months) # Map the month strings to numbers
    df.day = df.day.map(days) # Map the day strings to numbers
    x = df.values
    normalizer = preprocessing.MinMaxScaler()
    x_scaled = normalizer.fit_transform(x) # Normalize all of the data
    df = pd.DataFrame(x_scaled, columns=df.columns)
    x = np.array(df.month)
    y = df.area.values
    temp = []
    for col in df:
        if col != 'area':
            x = df[col].values
            coef = pearsonr(x, y) # Calculate the Pearson Coefficent
            temp.append(coef[0])

    top = pd.DataFrame(temp, columns=['pcc'])
    top.rename(index=rows, inplace=True)
    top = top.sort_values(['pcc'], ascending=False) # Sort the PCC values so the top 5 are easy to get
    return df, top # Return this as a tuple for use else where!
