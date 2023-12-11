#!/usr/bin/env python
# coding: utf-8

# # Run to Failure degradation simulation of NASA Turbo Jet Engine Fans

# # <a id='index'>Index</a>

# + <a href='#data_analysis'>1. Data Analysis</a>
#     + <a href='#info_about_data'>1.1 Info about data:</a>
# + <a href='#noise_removal'>2. Noise removal and Normalization</a>
# + <a href='#training_and_validation'>3. Training LSTM Model to predict RUL</a>
# + <a href='#testing_var'>4. Testing VAR</a>
# + <a href='#health_score'>5 Health Score Assignment</a>
# + <a href='#pred_analysis'>6. Analysing Prediction</a>

# In[263]:


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

from keras import optimizers
from keras.models import Sequential
from keras.layers import TimeDistributed, Flatten
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from sklearn.metrics import mean_squared_error

import warnings 
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')
cmap = cm.get_cmap('Spectral') # Colour map (there are many others)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score

from keras.models import load_model


# # <a id='data_analysis'>1. Data analysis</a>
# <a href='#index'>Go back to Index</a>

# In[264]:


train_file = "train_FD001.txt" 
test_file = "test_FD001.txt"
RUL_file = "RUL_FD001.txt"

df = pd.read_csv(train_file,sep=" ",header=None)
df.head()


# In[265]:


#columns = ['unit_number','time_in_cycles','setting_1','setting_2','TRA','T2','T24','T30','T50','P2','P15','P30','Nf',
#           'Nc','epr','Ps3 0','phi','NRf','NRc','BPR','farB','htBleed','Nf_dmd','PCNfR_dmd','W31','W32' ]
#delete NaN values
df.drop(columns=[26,27],inplace=True)
columns = ["Section-{}".format(i)  for i in range(26)]
df.columns = columns
df.head()


# #### Dataset statistics  for each parameter

# In[266]:


df.describe()


# ## <a id='info_about_data'>1.1 Info about data:</a>
# - Section-0 is MachineID
# - Section-1 is time in, Cycles
# - Section-2...4 is Opertional Settings
# - Section-5...25 is sensor's data 
# 
# 
# - Data Set: FD001
# - Train trjectories: 100
# - Test trajectories: 100
# - Conditions: ONE (Sea Level)
# - Fault Modes: ONE (HPC Degradation)

# In[267]:


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


# In[268]:


grp = RUL_data.groupby(MachineID_series)
max_cycles = np.array([max(grp.get_group(i)["Section-1"]) for i in MachineID_series.unique()])
print("Max Life >> ",max(max_cycles))
print("Mean Life >> ",np.mean(max_cycles))
print("Min Life >> ",min(max_cycles))


# In[269]:


for i in range(26):
    print(str(i))
    print(df['Section-'+str(i)])


# From the above vizulization its clear that 
# - Section-4 (Oprational Setting-3)
# - Section-5 (Sensor-1)
# - Section-9  (Sensor-5)
# - Section-10 (Sensor-6)
# - Section-14 (Sensor-10)
# - Section-20 (Sensor-16)
# - Section-22 (Sensor-18)
# - Section-23 (Sensor-19)
# 
# Does not play a vital role in variation of data and there std is also almost 0 so, these sensor data is useless for us hence, we can drop this coloumn data

# In[270]:


#delete columns with constant values that do not carry information about the state of the unit
#data = pd.concat([RUL_data,OS_data,Sensor_data], axis=1)
df.drop(columns=["Section-0",
                "Section-4", # Operatinal Setting
                "Section-5", # Sensor data
                "Section-9", # Sensor data
                "Section-10", # Sensor data
                "Section-14",# Sensor data
                "Section-20",# Sensor data
                "Section-22",# Sensor data
                "Section-23"] , inplace=True)


# In[271]:


df.head()


# In[ ]:





# In[272]:


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


# # <a id='testing_var'>4. Testing VAR</a>
# <a href='#index'>Go back to Index</a>

# # 1. Lookback = 1

# In[273]:


model = load_model('LSTM_with_lookback_1.h5')

lookback = 1
df_test = pd.read_csv(test_file, sep=" ",header=None)
df_test.drop(columns=[26,27],inplace=True)
df_test.columns = columns
df_test.head()

df_test_rul = pd.read_csv(RUL_file, names=['rul'])
df_test_rul.head()

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


gen = MinMaxScaler(feature_range=(0, 1))
df_test = gen.fit_transform(df_test)
df_test = pd.DataFrame(df_test)
#df_test = df_test.rolling(20).mean()
pt = PowerTransformer()
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

y_new = []
for i in range(len(test_data)):
    y_new.append(pd.DataFrame(test_data[i]))

y_new[1]

predictions = []
for i in range(len(y_new)):
    test_model = VAR(y_new[i][0:len(y_new[i])-lookback])
    test_model_fit = test_model.fit()
    test_pred = test_model_fit.forecast(test_model_fit.y, steps=1)
    predictions.append(test_pred)

predictions = np.array(predictions)

predictions.shape

lstm_pred = model.predict(predictions)

lstm_pred = np.array(lstm_pred)
for i in range(100):
    lstm_pred[i] = int(lstm_pred[i])

len(lstm_pred)

y_test = np.array(df_test_rul)

len(y_test)

fig = plt.figure(figsize=(18,10))
plt.plot(lstm_pred,color='red', label='prediction')
plt.plot(y_test,color='blue', label='y_test')

fig.suptitle('RUL Prediction using VAR Model with LSTM (Lookback=1)', fontsize=35)
plt.xlabel("Engine Number", fontsize=35)
plt.ylabel("Remaining Useful Life", fontsize=35)

plt.legend(loc='upper left')
plt.grid()
plt.show()

def scoring_function(actual,predicted):
    d = []
    for i in range(len(predicted)):
        d.append((predicted[i] - actual[i]))
    scores = []
    for i in range(len(d)):
        if d[i] >= 0:
            scores.append(math.exp(d[i]/10) - 1)
        else :
            scores.append(math.exp((-1*d[i])/13) - 1)
    return sum(scores)

print("mean_squared_error >> ", mean_squared_error(y_test,lstm_pred))
print("root_mean_squared_error >> ", math.sqrt(mean_squared_error(y_test,lstm_pred)))
print("mean_absolute_error >>",mean_absolute_error(y_test,lstm_pred))
print("scoring function >>",scoring_function(y_test,lstm_pred))


# In[274]:


df=pd.DataFrame(np.arange(1,101))
df['Actual']=y_test
df['Predicted']=lstm_pred
df=df.drop([0],axis=1)

sns.set_theme(style="whitegrid")
a4_dims = (18,10)
fig, ax = plt.subplots(figsize=a4_dims)
sns.lineplot(data = df,markers=True, dashes=False)

fig.suptitle('RUL Prediction using VAR Model with LSTM (Lookback=1)', fontsize=35)
plt.xlabel("Engine Number", fontsize=35)
plt.ylabel("Remaining Useful Life", fontsize=35)


# # 2. Lookback = 5

# In[285]:


model = load_model('LSTM_with_lookback_5.h5')
lookback = 5
lookback


# In[286]:


df_test = pd.read_csv(test_file, sep=" ",header=None)
df_test.drop(columns=[26,27],inplace=True)
df_test.columns = columns
df_test.head()


# In[287]:


df_test_rul = pd.read_csv(RUL_file, names=['rul'])
df_test_rul.head()


# In[288]:


RUL_name = ["Section-1"]
RUL_data = df_test[RUL_name]
MachineID_series = df_test["Section-0"]
grp = RUL_data.groupby(MachineID_series)
max_cycles = np.array([max(grp.get_group(i)["Section-1"]) for i in MachineID_series.unique()])


# In[289]:


print(max_cycles)


# In[290]:


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


# In[291]:


gen = MinMaxScaler(feature_range=(0, 1))
df_test = gen.fit_transform(df_test)
df_test = pd.DataFrame(df_test)
#df_test = df_test.rolling(20).mean()
pt = PowerTransformer()
df_test = pt.fit_transform(df_test)
df_test=np.nan_to_num(df_test)


# In[292]:


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


# In[293]:


print(len(test_data))


# In[294]:


y_new = []
for i in range(len(test_data)):
    y_new.append(pd.DataFrame(test_data[i]))

y_new[1]


# In[248]:


predictions = []
for i in range(100):
    test_model = VAR(y_new[i][0:len(y_new[i])-lookback])
    test_model_fit = test_model.fit()
    test_pred = test_model_fit.forecast(test_model_fit.y, steps=5)
    predictions.append(test_pred)


# In[249]:


predictions = np.array(predictions)


# In[296]:


predictions.shape


# In[297]:


lstm_pred = model.predict(predictions)


# In[ ]:





# In[298]:


(lstm_pred.shape)


# In[299]:


y_test = np.array(df_test_rul)


# In[300]:


len(y_test)


# In[301]:


fig = plt.figure(figsize=(18,10))
plt.plot(lstm_pred,color='red', label='prediction')
plt.plot(y_test,color='blue', label='y_test')

fig.suptitle('RUL Prediction using VAR Model with LSTM (lookback=5)', fontsize=35)
plt.xlabel("Engine Number", fontsize=35)
plt.ylabel("Remaining Useful Life", fontsize=35)

plt.legend(loc='upper left')
plt.grid()
plt.show()


# In[302]:


df=pd.DataFrame(np.arange(1,101))
df['Actual']=y_test
df['Predicted']=lstm_pred
df=df.drop([0],axis=1)

sns.set_theme(style="whitegrid")
a4_dims = (18,10)
fig, ax = plt.subplots(figsize=a4_dims)
sns.lineplot(data = df,markers=True, dashes=False)

fig.suptitle('RUL Prediction using VAR Model with LSTM (lookback=5)', fontsize=35)
plt.xlabel("Engine Number", fontsize=35)
plt.ylabel("Remaining Useful Life", fontsize=35)


# In[303]:


def scoring_function(actual,predicted):
    d = []
    for i in range(len(predicted)):
        d.append((predicted[i] - actual[i]))
    scores = []
    for i in range(len(d)):
        if d[i] >= 0:
            scores.append(math.exp(d[i]/10) - 1)
        else :
            scores.append(math.exp((-1*d[i])/13) - 1)
    return sum(scores)

print("mean_squared_error >> ", mean_squared_error(y_test,lstm_pred))
print("root_mean_squared_error >> ", math.sqrt(mean_squared_error(y_test,lstm_pred)))
print("mean_absolute_error >>",mean_absolute_error(y_test,lstm_pred))
print("scoring function >>",scoring_function(y_test,lstm_pred))


# # 3. Lookback = 10

# In[304]:


model = load_model('LSTM_with_lookback_10.h5')
lookback = 10
df_test = pd.read_csv(test_file, sep=" ",header=None)
df_test.drop(columns=[26,27],inplace=True)
df_test.columns = columns
df_test.head()

df_test_rul = pd.read_csv(RUL_file, names=['rul'])
df_test_rul.head()

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


gen = MinMaxScaler(feature_range=(0, 1))
df_test = gen.fit_transform(df_test)
df_test = pd.DataFrame(df_test)
#df_test = df_test.rolling(20).mean()
pt = PowerTransformer()
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

y_new = []
for i in range(len(test_data)):
    y_new.append(pd.DataFrame(test_data[i]))

y_new[1]

predictions = []
for i in range(100):
    test_model = VAR(y_new[i][0:len(y_new[i])-lookback])
    test_model_fit = test_model.fit()
    test_pred = test_model_fit.forecast(test_model_fit.y, steps=10)
    predictions.append(test_pred)

predictions = np.array(predictions)

predictions.shape

lstm_pred = model.predict(predictions)



(lstm_pred.shape)

y_test = np.array(df_test_rul)

len(y_test)

fig = plt.figure(figsize=(18,10))
plt.plot(lstm_pred,color='red', label='prediction')
plt.plot(y_test,color='blue', label='y_test')

fig.suptitle('RUL Prediction using VAR Model with LSTM (lookback = 10)', fontsize=35)
plt.xlabel("Engine Number", fontsize=35)
plt.ylabel("Remaining Useful Life", fontsize=35)

plt.legend(loc='upper left')
plt.grid()
plt.show()

def scoring_function(actual,predicted):
    d = []
    for i in range(len(predicted)):
        d.append((predicted[i] - actual[i]))
    scores = []
    for i in range(len(d)):
        if d[i] >= 0:
            scores.append(math.exp(d[i]/10) - 1)
        else :
            scores.append(math.exp((-1*d[i])/13) - 1)
    return sum(scores)

print("mean_squared_error >> ", mean_squared_error(y_test,lstm_pred))
print("root_mean_squared_error >> ", math.sqrt(mean_squared_error(y_test,lstm_pred)))
print("mean_absolute_error >>",mean_absolute_error(y_test,lstm_pred))
print("scoring function >>",scoring_function(y_test,lstm_pred))




# In[305]:


df=pd.DataFrame(np.arange(1,101))
df['Actual']=y_test
df['Predicted']=lstm_pred
df=df.drop([0],axis=1)

sns.set_theme(style="whitegrid")
a4_dims = (18,10)
fig, ax = plt.subplots(figsize=a4_dims)
sns.lineplot(data = df,markers=True, dashes=False)

fig.suptitle('RUL Prediction using VAR Model with LSTM (lookback = 10)', fontsize=35)
plt.xlabel("Engine Number", fontsize=35)
plt.ylabel("Remaining Useful Life", fontsize=35)


# # 4 Lookback = 20

# In[306]:


model = load_model('LSTM_with_lookback_20.h5')
lookback = 20
df_test = pd.read_csv(test_file, sep=" ",header=None)
df_test.drop(columns=[26,27],inplace=True)
df_test.columns = columns
df_test.head()

df_test_rul = pd.read_csv(RUL_file, names=['rul'])
df_test_rul.head()

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


gen = MinMaxScaler(feature_range=(0, 1))
df_test = gen.fit_transform(df_test)
df_test = pd.DataFrame(df_test)
#df_test = df_test.rolling(20).mean()
pt = PowerTransformer()
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

y_new = []
for i in range(len(test_data)):
    y_new.append(pd.DataFrame(test_data[i]))

y_new[1]

predictions = []
for i in range(100):
    test_model = VAR(y_new[i][0:len(y_new[i])-lookback])
    test_model_fit = test_model.fit()
    test_pred = test_model_fit.forecast(test_model_fit.y, steps=20)
    predictions.append(test_pred)

predictions = np.array(predictions)

predictions.shape

lstm_pred = model.predict(predictions)



(lstm_pred.shape)

y_test = np.array(df_test_rul)

len(y_test)

fig = plt.figure(figsize=(18,10))
plt.plot(lstm_pred,color='red', label='prediction')
plt.plot(y_test,color='blue', label='y_test')

fig.suptitle('RUL Prediction using VAR Model with LSTM', fontsize=35)
plt.xlabel("Engine Number", fontsize=35)
plt.ylabel("Remaining Useful Life", fontsize=35)

plt.legend(loc='upper left')
plt.grid()
plt.show()

def scoring_function(actual,predicted):
    d = []
    for i in range(len(predicted)):
        d.append((predicted[i] - actual[i]))
    scores = []
    for i in range(len(d)):
        if d[i] >= 0:
            scores.append(math.exp(d[i]/10) - 1)
        else :
            scores.append(math.exp((-1*d[i])/13) - 1)
    return sum(scores)

print("mean_squared_error >> ", mean_squared_error(y_test,lstm_pred))
print("root_mean_squared_error >> ", math.sqrt(mean_squared_error(y_test,lstm_pred)))
print("mean_absolute_error >>",mean_absolute_error(y_test,lstm_pred))
print("scoring function >>",scoring_function(y_test,lstm_pred))


# In[307]:


df=pd.DataFrame(np.arange(1,101))
df['Actual']=y_test
df['Predicted']=lstm_pred
df=df.drop([0],axis=1)

sns.set_theme(style="whitegrid")
a4_dims = (18,10)
fig, ax = plt.subplots(figsize=a4_dims)
sns.lineplot(data = df,markers=True, dashes=False)

fig.suptitle('RUL Prediction using VAR Model with LSTM (lookback = 20)', fontsize=35)
plt.xlabel("Engine Number", fontsize=35)
plt.ylabel("Remaining Useful Life", fontsize=35)


# In[ ]:





# In[ ]:




