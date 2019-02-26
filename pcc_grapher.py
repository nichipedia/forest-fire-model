import numpy as np
import pandas as pd
from data_tools import preprocess_data




df, topPCC = preprocess_data()


print(topPCC.head())
