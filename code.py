import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

dataset = pd.read_csv('winequality-red.csv')

ind = dataset.iloc[:, [7, 10]].values
dep = dataset.iloc[:, 11].values


randomForestRegressor = RandomForestRegressor(n_estimators=10, random_state=0)
randomForestRegressor.fit(ind, dep)
