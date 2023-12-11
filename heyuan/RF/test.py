from  scipy.io import loadmat
train_x = loadmat('../utils/F001_window_size_trainX.mat')
train_y = loadmat('../utils/F001_window_size_trainY.mat')
test_x = loadmat('../utils/F001_window_size_testX.mat')
test_y = loadmat('../utils/F001_window_size_testY.mat')
# print(test_x)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from scipy import interpolate
import scipy.io as sio
import numpy as np
min_max_scaler = preprocessing.MinMaxScaler()

# Import dataset
RUL_F001 = np.loadtxt('../ARIMA/RUL_FD001.txt')
train_F001 = np.loadtxt('../ARIMA/train_FD001.txt')
test_F001 = np.loadtxt('../ARIMA/test_FD001.txt')
train_F001[:, 2:] = min_max_scaler.fit_transform(train_F001[:, 2:])
test_F001[:, 2:] = min_max_scaler.transform(test_F001[:, 2:])
train_01_nor = train_F001
test_01_nor = test_F001

# Delete worthless sensors
# train_01_nor = np.delete(train_01_nor, [5, 9, 10, 14, 20, 22, 23], axis=1)
# test_01_nor = np.delete(test_01_nor, [5, 9, 10, 14, 20, 22, 23], axis=1)
train_01_nor = np.delete(train_01_nor, [2, 3, 4, 5, 9, 10, 14, 20, 22, 23], axis=1)
test_01_nor = np.delete(test_01_nor, [2, 3, 4, 5, 9, 10, 14, 20, 22, 23], axis=1)

# parameters of data process
RUL_max = 125.0
window_Size = 40


