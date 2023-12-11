import numpy as np
import pandas as pd
import seaborn as sns
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import QuantileTransformer , PowerTransformer
import warnings
warnings.filterwarnings('ignore')

# get file names
train_file = "train_FD001.txt"
test_file = "test_FD001.txt"
RUL_file = "RUL_FD001.txt"

#load file and set columns name
df = pd.read_csv(train_file,sep=" ",header=None)
df.drop(columns=[26,27],inplace=True)
columns = ["Section-{}".format(i)  for i in range(26)]
df.columns = columns


print(df.describe())

# name of each part
MachineID_name = ["Section-0"]
RUL_name = ["Section-1"]
OS_name = ["Section-{}".format(i) for i in range(2,5)]
Sensor_name = ["Section-{}".format(i) for i in range(5,26)]

# data of each part
MachineID_data = df[MachineID_name]
RUL_data = df[RUL_name]
OS_data = df[OS_name]
Sensor_data = df[Sensor_name]

# get each unit cycle
MachineID_series = df["Section-0"]
RUL_series = df["Section-1"]
grp = RUL_data.groupby(MachineID_series)
max_cycles = np.array([max(grp.get_group(i)["Section-1"]) for i in MachineID_series.unique()])


# one can plot origin data map here


#  then delete constant values   [20631 17]
df.drop(columns=["Section-0", # unit number
                "Section-4", # Operatinal Setting
                "Section-5", # Sensor data
                "Section-9", # Sensor data
                "Section-10", # Sensor data
                "Section-14",# Sensor data
                "Section-20",# Sensor data
                "Section-22",# Sensor data
                "Section-23"] , inplace=True)

# get rul info
rul_lst = [j  for i in MachineID_series.unique() for j in np.array(grp.get_group(i)[::-1]["Section-1"])]
rul_col = pd.DataFrame({"rul":rul_lst})


# get test data and rul value
df_test = pd.read_csv(test_file, sep=" ",header=None)
df_test.drop(columns=[26,27],inplace=True)
df_test.columns = columns


df_test_rul = pd.read_csv(RUL_file, names=['rul'])

print(df_test.describe())
RUL_name = ["Section-1"]
RUL_data = df_test[RUL_name]
MachineID_series = df_test["Section-0"]
grp = RUL_data.groupby(MachineID_series)
max_cycles_test = np.array([max(grp.get_group(i)["Section-1"]) for i in MachineID_series.unique()])

df_test.drop(df_test[["Section-0",
                "Section-4", # Operatinal Setting
                "Section-5", # Sensor data
                "Section-9", # Sensor data
                "Section-10", # Sensor data
                "Section-14",# Sensor data
                "Section-20",# Sensor data
                "Section-22",# Sensor data
                "Section-23"]], axis=1 , inplace=True)


# one can normalization origin data here
minmaxscaler_test = MinMaxScaler(feature_range=(0, 1))
df_test_norm = minmaxscaler_test.fit_transform(df_test)
df_test_norm = pd.DataFrame(df_test_norm)
df_test_norm = np.array(df_test_norm)

minmaxscaler = MinMaxScaler(feature_range=(0, 1))
df_norm = minmaxscaler.fit_transform(df)
df_norm = pd.DataFrame(df_norm)
df_norm = np.array(df_norm)


# map data to gauss distribution
# pt = PowerTransformer()
# df_test = pt.fit_transform(df_test)
# df_test=np.nan_to_num(df_test)

# get test data
test_data = []
i = 0
count = 0
while i < len(df_test_norm):
    temp = []
    j = int(max_cycles_test[count]) # get each unit cycle
    count = count+1
    #print(j)
    if j == 0:
        break
    while j!=0:
        temp.append(df_test_norm[i])
        i=i+1
        j=j-1
    test_data.append(temp)


# VAR and predict
predictions = []
for i in range(len(test_data)):
    test_model = VAR(test_data[i][0:len(test_data[i])-1])
    test_model_fit = test_model.fit()
    test_pred = test_model_fit.forecast(test_model_fit.y, steps=1)
    predictions.append(test_pred)

print(predictions)