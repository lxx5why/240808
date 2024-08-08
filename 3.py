import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Binarizer
from sklearn.linear_model import LinearRegression

header = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv('./data/pima-indians-diabetes.data.csv', names =header)

array = data.values
X = array[:, 0:8]
Y = array[:, 8]
print(X.shape, Y.shape)

# scaler = StandardScaler()
# rescaled_X = scaler.fit_transform(X)
# print(rescaled_X)

scaler = MinMaxScaler(feature_range=(0,1))
rescaled_X = scaler.fit_transform(X)
print(rescaled_X)
model = LinearRegression()
model.fit(X, Y)

predicted_Y = model.predict(X)
y = (predicted_Y > 0.5).astype(int)
print(y)

# scaler = Binarizer(threshold=0.5)
# rescaled_X = scaler.fit_transform(X)
# print(rescaled_X)

