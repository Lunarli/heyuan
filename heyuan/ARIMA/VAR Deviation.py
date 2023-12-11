#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import math
import matplotlib.pyplot as plt
from matplotlib import cm

import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.vector_ar.var_model import VAR

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import QuantileTransformer , PowerTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

# from tensorflow.keras import optimizers
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import TimeDistributed, Flatten
# from tensorflow.keras.layers.core import Dense, Dropout, Activation
# from tensorflow.keras.layers.recurrent import LSTM
# from sklearn.metrictensorflow.s import mean_squared_error

import warnings 
warnings.filterwarnings('ignore')


cmap = cm.get_cmap('Spectral') # Colour map (there are many others)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score

# from keras.models import load_model

# <a id='data_analysis'>1. Data analysis</a>

train_file = "train_FD001.txt" 
test_file = "test_FD001.txt"
RUL_file = "RUL_FD001.txt"

df = pd.read_csv(train_file,sep=" ",header=None)
df.head()

#columns = ['unit_number','time_in_cycles','setting_1','setting_2','TRA','T2','T24','T30','T50','P2','P15','P30','Nf',
#           'Nc','epr','Ps3 0','phi','NRf','NRc','BPR','farB','htBleed','Nf_dmd','PCNfR_dmd','W31','W32' ]
#delete NaN values
df.drop(columns=[26,27],inplace=True)
columns = ["Section-{}".format(i)  for i in range(26)]
df.columns = columns
df.head()

#### Dataset statistics  for each parameter

df.describe()

# Names 
MachineID_name = ["Section-0"]
RUL_name = ["Section-1"]
OS_name = ["Section-{}".format(i) for i in range(2,5)]
Sensor_name = ["Section-{}".format(i) for i in range(5,26)]

# Data in pandas DataFrame
MachineID_data = df[MachineID_name]
RUL_data = df[RUL_name]
OS_data = df[OS_name]
Sensor_data = df[Sensor_name]

# Data in pandas Series
MachineID_series = df["Section-0"]
RUL_series = df["Section-1"]

grp = RUL_data.groupby(MachineID_series)
max_cycles = np.array([max(grp.get_group(i)["Section-1"]) for i in MachineID_series.unique()]) # every unit cycle
print("Max Life >> ",max(max_cycles))
print("Mean Life >> ",np.mean(max_cycles))
print("Min Life >> ",min(max_cycles))

# for i in range(26):
#     print(str(i))
#     print(df['Section-'+str(i)])

#delete columns with constant values that do not carry information about the state of the unit
#data = pd.concat([RUL_data,OS_data,Sensor_data], axis=1)
df.drop(columns=["Section-0", # unit number
                "Section-4", # Operatinal Setting
                "Section-5", # Sensor data
                "Section-9", # Sensor data
                "Section-10", # Sensor data
                "Section-14",# Sensor data
                "Section-20",# Sensor data
                "Section-22",# Sensor data
                "Section-23"] , inplace=True)

df.head()  # [20631 17]  2 operation settings + 15 sensor data


# get remaining life
def RUL_df():
    rul_lst = [j  for i in MachineID_series.unique() for j in np.array(grp.get_group(i)[::-1]["Section-1"])]
    rul_col = pd.DataFrame({"rul":rul_lst})
    return rul_col

RUL_df().head()

def create_dataset(X, look_back=5):
    data = []
    for i in range(len(X)-look_back-1):
        data.append(X[i:(i+look_back)])
    return np.array(data)


df_test = pd.read_csv(test_file, sep=" ",header=None)
df_test.drop(columns=[26,27],inplace=True)
df_test.columns = columns
df_test.head()

df_test_rul = pd.read_csv(RUL_file, names=['rul'])
df_test_rul.head()
print(df_test.describe())
RUL_name = ["Section-1"]
RUL_data = df_test[RUL_name]
MachineID_series = df_test["Section-0"]
grp = RUL_data.groupby(MachineID_series)
max_cycles = np.array([max(grp.get_group(i)["Section-1"]) for i in MachineID_series.unique()])

print(max_cycles)

df_test.drop(df_test[["Section-0",
                "Section-4", # Operatinal Setting
                "Section-5", # Sensor data
                "Section-9", # Sensor data
                "Section-10", # Sensor data
                "Section-14",# Sensor data
                "Section-20",# Sensor data
                "Section-22",# Sensor data
                "Section-23"]], axis=1 , inplace=True)
#df_test = df_test.groupby(["Section-0"])
#print(df_test)

# normalization
gen = MinMaxScaler(feature_range=(0, 1))
df_test = gen.fit_transform(df_test)
df_test = pd.DataFrame(df_test)
#df_test = df_test.rolling(20).mean()
pt = PowerTransformer()   # map data to gauss distribution
df_test = pt.fit_transform(df_test)
df_test=np.nan_to_num(df_test)

test_data = []
i = 0
count = 0
while i < len(df_test):
    temp = []
    j = int(max_cycles[count])
    count = count+1
    #print(j)
    if j == 0:
        break
    while j!=0:
        temp.append(df_test[i])
        i=i+1
        j=j-1
    test_data.append(temp)

print(len(test_data))

y_new = [] # 100 None 17
for i in range(len(test_data)):
    y_new.append(pd.DataFrame(test_data[i]))

y_new[0]


# In[621]:


for y in y_new:
    print(len(y))


# In[ ]:





# # Lookback = 1

# In[622]:

# prediction last cycle of each test unit
predictions = []
for i in range(len(y_new)):
    test_model = VAR(y_new[i][0:len(y_new[i])-1])
    test_model_fit = test_model.fit()
    test_pred = test_model_fit.forecast(test_model_fit.y, steps=1)
    predictions.append(test_pred)


# In[623]:


predictions = np.array(predictions)

predictions.shape


# In[624]:


actual = []
for i in range(len(y_new)):
    actual.append(y_new[i].iloc[-1])


# In[625]:


actual=np.array(actual)


# In[626]:


actual = actual.reshape((100,17))
predictions = predictions.reshape((100,17))


# In[627]:


print(actual.shape,predictions.shape)


# In[628]:


compare=2
df_actual,df_pred = pd.DataFrame(),pd.DataFrame()


# In[629]:


difference = []
for i in range(compare):
    df_actual[i+1],df_pred[i+1] = actual[i],predictions[i]
    difference.append(abs(actual[i]-predictions[i]))
# df_actual,df_pred = df_actual.T,df_pred.T


# In[630]:


difference


# In[631]:


df_diff =  pd.DataFrame(difference)
df_diff = df_diff.T   # [17 2]


# In[632]:


columns=[]
for i in range(compare):
    columns.append('Engine '+str(i+1))
df_pred.columns=columns
df_actual.columns=columns
df_diff.columns=columns


# In[633]:


df_pred['Features']=np.arange(1,18)


# In[634]:


df_actual['Features']=np.arange(1,18)


# In[635]:


df_diff['Features']=np.arange(1,18)


# In[636]:


df_diff


# In[637]:


# display(df_pred,df_actual)


# In[638]:


df_diff.describe()


# In[639]:


df_pred=df_pred.drop(['Features'], axis=1)
df_actual=df_actual.drop(['Features'], axis=1)
df_diff=df_diff.drop(['Features'], axis=1)


# In[640]:


# display(df_pred,df_actual)


# In[641]:


sns.set_theme(style="whitegrid")
a4_dims = (9,5)
fig, ax = plt.subplots(figsize=a4_dims)
sns.lineplot(data = df_pred,markers=True, dashes=False)
fig.suptitle('VAR Model forecasting last time series for Engine 1 and Engine 2', fontsize=20)
plt.xlabel("Features", fontsize=20)
plt.ylabel("Predicted", fontsize=20)


# In[642]:


sns.set_theme(style="whitegrid")
a4_dims = (9,5)
fig, ax = plt.subplots(figsize=a4_dims)
sns.lineplot(data = df_actual,markers=True, dashes=False)
fig.suptitle('Last time series for Engine 1 and Engine 2', fontsize=20)
plt.xlabel("Features", fontsize=20)
plt.ylabel("Actual", fontsize=20)


# In[643]:


sns.set_theme(style="whitegrid")
a4_dims = (13,7)
fig, ax = plt.subplots(figsize=a4_dims)
sns.lineplot(data = df_diff,markers=True, dashes=False)
fig.suptitle('Deviation of VAR forecasting from actual (last) time series for Engine 1 and Engine 2', fontsize=20)
plt.xlabel("Features", fontsize=20)
plt.ylabel("Deviation", fontsize=20)


# # 2. For lookback = 5

# In[644]:


predictions = []
for i in range(len(y_new)):
    test_model = VAR(y_new[i][0:len(y_new[i])-5])
    test_model_fit = test_model.fit()
    test_pred = test_model_fit.forecast(test_model_fit.y, steps=5)
    predictions.append(test_pred)


# In[645]:


predictions=np.array(predictions)


# In[646]:


predictions.shape


# In[647]:


actual = []
for i in range(len(y_new)):
    actual.append(np.array(y_new[i].iloc[len(y_new[i])-5:]))


# In[648]:


actual = np.array(actual)


# In[649]:


actual.shape


# In[650]:


diff = []
for i in range(100):
    a = []
    for j in range(5):
        b = []
        for k in range(17):
            b.append(abs(predictions[i][j][k]-actual[i][j][k]))
        a.append(b)
    diff.append(a)
diff = np.array(diff)
diff.shape


# In[651]:


df1_actual=pd.DataFrame(actual[0])
df1_pred=pd.DataFrame(predictions[0])
df1_diff=pd.DataFrame(diff[0])
df1_actual,df1_pred,df1_diff = df1_actual.T,df1_pred.T,df1_diff.T


# In[652]:


columns=['1st Step','2nd Step','3rd Step','4th Step','5th Step']


# In[653]:


df1_actual.columns = columns
df1_pred.columns = columns
df1_diff.columns = columns


# In[654]:


df1_diff


# In[655]:


# display(df1_actual,df1_pred)


# In[656]:


df_diff.describe()


# In[657]:


sns.set_theme(style="whitegrid")

a4_dims = (9,5)
fig, ax = plt.subplots(figsize=a4_dims)
sns.lineplot(data = df1_pred,markers=True, dashes=False)
fig.suptitle('VAR Model forcasting last 5 time series for Engine 1 ', fontsize=20)
plt.xlabel("Features", fontsize=20)
plt.ylabel("Predicted", fontsize=20)
plt.legend(loc='lower right')


# In[658]:


sns.set_theme(style="whitegrid")

a4_dims = (9,5)
fig, ax = plt.subplots(figsize=a4_dims)
sns.lineplot(data = df1_actual,markers=True, dashes=False)
fig.suptitle('Last 5 time series for Engine 1 ', fontsize=20)
plt.xlabel("Features", fontsize=20)
plt.ylabel("Actual", fontsize=20)
plt.legend(loc='lower right')


# In[659]:


sns.set_theme(style="whitegrid")

a4_dims = (12,7)
fig, ax = plt.subplots(figsize=a4_dims)
sns.lineplot(data = df1_diff,markers=True, dashes=False)
fig.suptitle('Deviation of VAR Forecasting from actual last 5 time series for Engine 1 ', fontsize=20)
plt.xlabel("Features", fontsize=20)
plt.ylabel("Deviation", fontsize=20)
plt.legend(loc='upper right')


# In[ ]:





# In[ ]:





# # Lookback = 10

# In[660]:


predictions = []
for i in range(len(y_new)):
    test_model = VAR(y_new[i][0:len(y_new[i])-10])
    test_model_fit = test_model.fit()
    test_pred = test_model_fit.forecast(test_model_fit.y, steps=10)
    predictions.append(test_pred)

predictions=np.array(predictions)

print(predictions.shape)

actual = []
for i in range(len(y_new)):
    actual.append(np.array(y_new[i].iloc[len(y_new[i])-10:]))

actual = np.array(actual)

print(actual.shape)

# predictions = predictions.reshape((100,17,10))
# actual = actual.reshape((100,17,10))

diff = []
for i in range(100):
    a = []
    for j in range(10):
        b = []
        for k in range(17):
            b.append(abs(predictions[i][j][k]-actual[i][j][k]))
        a.append(b)
    diff.append(a)
diff = np.array(diff)
diff.shape

df1_actual=pd.DataFrame(actual[0])
df1_pred=pd.DataFrame(predictions[0])
df1_diff=pd.DataFrame(diff[0])
df1_actual,df1_pred,df1_diff = df1_actual.T,df1_pred.T,df1_diff.T

columns=['1st Step','2nd Step','3rd Step','4th Step','5th Step','6th Step','7th Step','8th Step','9th Step','10th Step']

df1_actual.columns = columns
df1_pred.columns = columns
df1_diff.columns = columns




# In[661]:


# display(df1_actual,df1_pred,df1_diff)


# In[662]:


df_diff.describe()


# In[663]:


sns.set_theme(style="whitegrid")
a4_dims = (9,5)
fig, ax = plt.subplots(figsize=a4_dims)
sns.lineplot(data = df1_pred,markers=True, dashes=False)
fig.suptitle('VAR Model forecasting last 10 time series for Engine 1 ', fontsize=20)
plt.xlabel("Features", fontsize=20)
plt.ylabel("Predicted", fontsize=20)
plt.legend(loc='lower right')


# In[664]:


sns.set_theme(style="whitegrid")
a4_dims = (9,5)
fig, ax = plt.subplots(figsize=a4_dims)
sns.lineplot(data = df1_actual,markers=True, dashes=False)
fig.suptitle('Last 10 time series for Engine 1 ', fontsize=20)
plt.xlabel("Features", fontsize=20)
plt.ylabel("Actual", fontsize=20)
plt.legend(loc='lower right')


# In[665]:


sns.set_theme(style="whitegrid")
a4_dims = (13,7)
fig, ax = plt.subplots(figsize=a4_dims)
sns.lineplot(data = df1_diff,markers=True, dashes=False)
fig.suptitle('Deviation of VAR Forecasting from actual last 10 time series for Engine 1 ', fontsize=20)
plt.xlabel("Features", fontsize=20)
plt.ylabel("Deviation", fontsize=20)
plt.legend(loc='upper right')


# In[ ]:





# In[ ]:





# # 4. Lookback=20

# In[666]:


predictions = []
for i in range(len(y_new)):
    test_model = VAR(y_new[i][0:len(y_new[i])-20])
    test_model_fit = test_model.fit()
    test_pred = test_model_fit.forecast(test_model_fit.y, steps=20)
    predictions.append(test_pred)

predictions=np.array(predictions)

print(predictions.shape)

actual = []
for i in range(len(y_new)):
    actual.append(np.array(y_new[i].iloc[len(y_new[i])-20:]))

actual = np.array(actual)

print(actual.shape)

# predictions = predictions.reshape((100,17,10))
# actual = actual.reshape((100,17,10))

diff = []
for i in range(100):
    a = []
    for j in range(20):
        b = []
        for k in range(17):
            b.append(abs(predictions[i][j][k]-actual[i][j][k]))
        a.append(b)
    diff.append(a)
diff = np.array(diff)
diff.shape

df1_actual=pd.DataFrame(actual[0])
df1_pred=pd.DataFrame(predictions[0])
df1_diff=pd.DataFrame(diff[0])
df1_actual,df1_pred,df1_diff = df1_actual.T,df1_pred.T,df1_diff.T

columns=[]
for i in range(20):
    columns.append('Step '+str(i+1))

df1_actual.columns = columns
df1_pred.columns = columns
df1_diff.columns = columns





# display(df1_actual,df1_pred,df1_diff)



df_diff.describe()

sns.set_theme(style="whitegrid")
a4_dims = (9,5)
fig, ax = plt.subplots(figsize=a4_dims)
sns.lineplot(data = df1_pred,markers=True, dashes=False)
fig.suptitle('VAR Model forecasting last 10 time series for Engine 1 ', fontsize=20)
plt.xlabel("Features", fontsize=20)
plt.ylabel("Predicted", fontsize=20)
plt.legend(loc='lower right')

sns.set_theme(style="whitegrid")
a4_dims = (9,5)
fig, ax = plt.subplots(figsize=a4_dims)
sns.lineplot(data = df1_actual,markers=True, dashes=False)
fig.suptitle('Last 20 time series for Engine 1 ', fontsize=20)
plt.xlabel("Features", fontsize=20)
plt.ylabel("Actual", fontsize=20)
plt.legend(loc='lower right')

sns.set_theme(style="whitegrid")
a4_dims = (13,7)
fig, ax = plt.subplots(figsize=a4_dims)
sns.lineplot(data = df1_diff,markers=True, dashes=False)
fig.suptitle('Deviation of VAR Forecasting from actual last 20 time series for Engine 1 ', fontsize=20)
plt.xlabel("Features", fontsize=20)
plt.ylabel("Deviation", fontsize=20)
plt.legend(loc='upper right')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




