#!/usr/bin/env python
# coding: utf-8

# # Run to Failure degradation simulation of NASA Turbo Jet Engine Fans

# # <a id='index'>Index</a>

# + <a href='#data_analysis'>1. Data Analysis</a>
#     + <a href='#info_about_data'>1.1 Info about data:</a>
# + <a href='#noise_removal'>2. Noise removal and Normalization</a>
# + <a href='#training_and_validation'>3. Training and Validation</a>
#     + <a href='#3.1'>3.1 Training Performance of different models</a> 
#     + <a href='#3.2'>3.2 Random Forest</a> 
#         + <a href='#3.2.1'>3.2.1 Random Forest Validation Performance</a>
#         + <a href='#3.2.2'>3.2.2 Random Forest Validation Prediction vs Actual</a>
#     + <a href='#3.3'>3.3 Linear Regression</a> 
#         + <a href='#3.3.1'>3.3.1 Linear Regression Validation Performance</a>
#         + <a href='#3.3.2'>3.3.2 Linear Regression Validation Prediction vs Actual</a>
#     + <a href='#3.4'>3.4 Logistic Regression</a> 
#         + <a href='#3.4.1'>3.4.1 Logistic Regression Validation Performance</a>
#         + <a href='#3.4.2'>3.4.2 Logistic Regression Validation Prediction vs Actual</a>
# + <a href='#testing'>4 Testing</a>
#     + <a href='#4.1'>4.1 Random Forest Testing</a>
#     + <a href='#4.2'>4.2 Linear Regression Testing</a>
#     + <a href='#4.3'>4.3 Logistic Regression Testing</a>
# 

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import math
import matplotlib.pyplot as plt
from matplotlib import cm

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

from keras.layers import Dense , LSTM
from keras.models import Sequential
from sklearn.metrics import mean_squared_error

import warnings 
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')
cmap = cm.get_cmap('Spectral') # Colour map (there are many others)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score

import pickle


# # <a id='data_analysis'>1. Data analysis</a>
# <a href='#index'>Go back to Index</a>

# In[3]:


train_file = "train_FD001.txt" 
test_file = "test_FD001.txt"
RUL_file = "RUL_FD001.txt"

df = pd.read_csv(train_file,sep=" ",header=None)
df.head()


# In[4]:


#columns = ['unit_number','time_in_cycles','setting_1','setting_2','TRA','T2','T24','T30','T50','P2','P15','P30','Nf',
#           'Nc','epr','Ps3 0','phi','NRf','NRc','BPR','farB','htBleed','Nf_dmd','PCNfR_dmd','W31','W32' ]
#delete NaN values
df.drop(columns=[26,27],inplace=True)
columns = ["Section-{}".format(i)  for i in range(26)]
df.columns = columns
df.head()


# #### Dataset statistics  for each parameter

# In[5]:


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

# In[5]:


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


# In[45]:


grp = RUL_data.groupby(MachineID_series)
max_cycles = np.array([max(grp.get_group(i)["Section-1"]) for i in MachineID_series.unique()])


# In[7]:


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


# From the above vizulization its clear that 
# - Section-4 (Oprational Setting-3)
# - Section-5 (Sensor-1)
# - Section-9  (Sensor-5)
# - Section-14 (Sensor-10)
# - Section-20 (Sensor-16)
# - Section-22 (Sensor-18)
# - Section-23 (Sensor-19)
# 
# Does not play a vital role in variation of data and there std is also almost 0 so, these sensor data is useless for us hence, we can drop this coloumn data

# In[8]:


df.head()


# # <a id='noise_removal'>2. Noise removal and Normalization</a>
# <a href='#index'>Go back to Index</a>

# In[9]:


print(type(df))
gen = MinMaxScaler(feature_range=(0, 1))
df = gen.fit_transform(df)
df = pd.DataFrame(df)
#df = df.rolling(20).mean()
pt = PowerTransformer()
df = pt.fit_transform(df)


# In[10]:


df=np.nan_to_num(df)


# In[11]:


df


# In[12]:


# grouping w.r.t MID (Machine ID)
# col_names = df.columns
# def grouping(datafile, mid_series):
#     data = [x for x in datafile.groupby(mid_series)]
#     return data 


# # <a id='training_and_validation'>3. Training and Validation</a>
# <a href='#index'>Go back to Index</a>

# In[13]:


def RUL_df():
    rul_lst = [j  for i in MachineID_series.unique() for j in np.array(grp.get_group(i)[::-1]["Section-1"])]
    rul_col = pd.DataFrame({"rul":rul_lst})
    return rul_col

RUL_df().head()


# ## <a id='3.1'>3.1 Training Performance of different models</a> 
# <a href='#index'>Go back to Index</a>

# In[14]:


X = np.array(df)

y = np.array(RUL_df()).reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20) 

forest_model = RandomForestRegressor()
forest_model.fit(X_train, y_train)
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)


# In[18]:


pred_f = forest_model.predict(X)
pred_lin = lin_model.predict(X)
pred_log = logistic_model.predict(X)

print("mean_squared_error >> ", mean_squared_error(y, pred_f))
print("mean_absolute_error >>",mean_absolute_error(y, pred_f))

print("\nmean_squared_error >> ", mean_squared_error(y, pred_lin))
print("mean_absolute_error >>",mean_absolute_error(y, pred_lin))

print("\nmean_squared_error >> ", mean_squared_error(y, pred_log))
print("mean_absolute_error >>",mean_absolute_error(y, pred_log))


# In[15]:


pickle.dump(logistic_model, open('logistic_regression.sav', 'wb'))

pickle.dump(lin_model, open('linear_regression.sav', 'wb'))

pickle.dump(forest_model, open('forest_regression.sav', 'wb'))


# In[17]:


print("Acc of Linear Regression >> ",lin_model.score(X_test, y_test))
print("Acc of Random Forest >> ",forest_model.score(X_test, y_test))
print("Acc of Logistic Regression >> ",logistic_model.score(X_test, y_test))


# ## <a id='3.2'>3.2 Random Forest</a> 
# <a href='#index'>Go back to Index</a>

# ### <a id='3.2.1'>3.2.1 Random Forest Validation Performance</a>

# In[17]:


forest_pred = forest_model.predict(X_test)
print("mean_squared_error >> ", mean_squared_error(y_test, forest_pred))
print("mean_absolute_error >>",mean_absolute_error(y_test, forest_pred))


# ### <a id='3.2.2'>3.2.2 Random Forest Validation Prediction vs Actual</a>

# In[18]:


plt.plot(y_test,c='k',label='Actual')
plt.plot(forest_pred,c='red',label='Predicted')
plt.legend()
plt.show()


# ## <a id='3.3'>3.3 Linear Regression</a>
# <a href='#index'>Go back to Index</a>

# ### <a id='3.3.1'>3.3.1 Linear Regression Validation Performance</a>

# In[19]:


lin_pred = lin_model.predict(X_test)
print("mean_squared_error >> ", mean_squared_error(y_test, lin_pred))
print("mean_absolute_error >>",mean_absolute_error(y_test, lin_pred))


# ### <a id='3.3.2'>3.3.2 Linear Regression Validation Prediction vs Actual</a>

# In[20]:


plt.plot(y_test,c='k',label='Actual')
plt.plot(lin_pred,c='red',label='Predicted')
plt.legend()
plt.show()


# ## <a id='3.4'>3.4 Logistic Regression </a>
# <a href='#index'>Go back to Index</a>

# ### <a id='3.4.1'>3.4.1 Logistic Regression Validation Performance</a>

# In[21]:


logistic_pred = logistic_model.predict(X_test)
print("mean_squared_error >> ", mean_squared_error(y_test, logistic_pred))
print("mean_absolute_error >>",mean_absolute_error(y_test, logistic_pred))


# ### <a id='3.4.2'>3.4.2 Logistic Regression Validation Prediction vs Actual</a>

# In[22]:


plt.plot(y_test,c='k',label='Actual')
plt.plot(logistic_pred,c='red',label='Predicted')
plt.legend()
plt.show()


# In[23]:


# from sklearn.linear_model import LinearRegression
# linear_reg = LinearRegression()
# linear_reg.fit(trainX,trainY)


# In[24]:


# print("acc of Linear Regressor >> ",linear_reg.score(testX, testY))


# # <a id='testing'>4 Testing</a>
# <a href='#index'>Go back to Index</a>

# In[6]:


forest_model = pickle.load(open('forest_regression.sav', 'rb'))
lin_model = pickle.load(open('linear_regression.sav', 'rb'))
logistic_model = pickle.load(open('logistic_regression.sav', 'rb'))


# In[7]:


df_test = pd.read_csv(test_file, sep=" ",header=None)
df_test.drop(columns=[26,27],inplace=True)
df_test.columns = columns
df_test.head()


# In[8]:


df_rul = pd.read_csv(RUL_file, names=['rul'])
df_rul.head()


# In[9]:


RUL_name = ["Section-1"]
RUL_data = df_test[RUL_name]
MachineID_series = df_test["Section-0"]
grp = RUL_data.groupby(MachineID_series)
max_cycles = np.array([max(grp.get_group(i)["Section-1"]) for i in MachineID_series.unique()])
max_cycles


# In[10]:


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


# In[11]:


gen = MinMaxScaler(feature_range=(0, 1))
df_test = gen.fit_transform(df_test)
df_test = pd.DataFrame(df_test)
#df_test = df_test.rolling(20).mean()
pt = PowerTransformer()
df_test = pt.fit_transform(df_test)
df_test=np.nan_to_num(df_test)


# In[12]:


df_test


# ## <a id='4.1'>4.1 Random Forest Testing</a>
# <a href='#index'>Go back to Index</a>

# In[13]:


forest_pred = forest_model.predict(df_test)


# In[14]:


forest_pred = np.array(forest_pred)


# In[15]:


forest_pred = forest_pred.flatten()


# In[16]:


forest_pred = forest_pred.reshape(forest_pred.shape[0],1)


# In[17]:


forest_pred.shape


# In[18]:


forest_pred


# In[19]:


final_forest_pred = []
count = 0
for i in range(100):
    temp = 0
    j = max_cycles[i] 
    while j>0:
        temp = temp + forest_pred[count]
        j=j-1
        count=count+1
    final_forest_pred.append(temp/max_cycles[i])


# In[20]:


final_forest_pred=np.array(final_forest_pred)
final_forest_pred = final_forest_pred.flatten()


# In[21]:


final_forest_pred[0]


# In[22]:


fig = plt.figure(figsize=(18,10))
plt.plot(final_forest_pred,c='red',label='preduction')
plt.plot(df_rul,c='blue',label='y_test')

fig.suptitle('RUL Prediction using Random Forest Regressin Model', fontsize=35)
plt.xlabel("Engine Number", fontsize=35)
plt.ylabel("Remaining Useful Life", fontsize=35)

plt.legend(loc='upper left')
plt.grid()
plt.show()


# In[43]:


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


print("mean_squared_error >> ", mean_squared_error(df_rul,final_forest_pred))
print("root mean_absolute_error >>",math.sqrt(mean_squared_error(df_rul,final_forest_pred)))
print("mean_absolute_error >>",mean_absolute_error(df_rul,final_forest_pred))
print("scoring function >>",scoring_function(np.array(df_rul),final_forest_pred))


# ## <a id='4.2'>4.2 Linear Regressor Testing</a>
# <a href='#index'>Go back to Index</a>

# In[24]:


lin_pred = lin_model.predict(df_test)
lin_pred = np.array(lin_pred)
lin_pred = lin_pred.flatten()
lin_pred = lin_pred.reshape(lin_pred.shape[0],1)
lin_pred.shape


# In[25]:


final_lin_pred = []
count = 0
for i in range(100):
    temp = 0
    j = max_cycles[i] 
    while j>0:
        temp = temp + lin_pred[count]
        j=j-1
        count=count+1
    final_lin_pred.append(temp/max_cycles[i])

final_lin_pred=np.array(final_lin_pred)
final_lin_pred = final_lin_pred.flatten()


# In[26]:


fig = plt.figure(figsize=(18,10))
plt.plot(final_lin_pred,c='red',label='prediction')
plt.plot(df_rul,c='blue',label='y_test')

fig.suptitle('RUL Prediction using Linear Regressin Model', fontsize=35)
plt.xlabel("Engine Number", fontsize=35)
plt.ylabel("Remaining Useful Life", fontsize=35)

plt.legend(loc='upper left')
plt.grid()
plt.show()


# In[42]:


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

print("mean_squared_error >> ", mean_squared_error(df_rul,final_lin_pred))
print("root mean_absolute_error >>",math.sqrt(mean_squared_error(df_rul,final_lin_pred)))
print("mean_absolute_error >>",mean_absolute_error(df_rul,final_lin_pred))
print("scoring function >>",scoring_function(np.array(df_rul),final_lin_pred))
# scoring_function(np.array(df_rul),final_lin_pred)


# ## <a id='4.3'>4.3 Logistic Regressor Testing</a>
# <a href='#index'>Go back to Index</a>

# In[28]:


logistic_pred = logistic_model.predict(df_test)
logistic_pred = np.array(logistic_pred)
logistic_pred = logistic_pred.flatten()
logistic_pred = logistic_pred.reshape(logistic_pred.shape[0],1)
logistic_pred.shape


# In[29]:


final_logistic_pred = []
count = 0
for i in range(100):
    temp = 0
    j = max_cycles[i] 
    while j>0:
        temp = temp + logistic_pred[count]
        j=j-1
        count=count+1
    final_logistic_pred.append(temp/max_cycles[i])

final_logistic_pred=np.array(final_logistic_pred)
final_logistic_pred = final_logistic_pred.flatten()


# In[30]:


fig = plt.figure(figsize=(18,10))
plt.plot(final_logistic_pred,c='red',label='prediction')
plt.plot(df_rul,c='blue',label='y_test')

fig.suptitle('RUL Prediction using Logistic Regressin Model', fontsize=35)
plt.xlabel("Engine Number", fontsize=35)
plt.ylabel("Remaining Useful Life", fontsize=35)

plt.legend(loc='upper left')
plt.grid()
plt.show()


# In[44]:


print("mean_squared_error >> ", mean_squared_error(df_rul,final_logistic_pred))
print("root mean_squared_error >> ", math.sqrt(mean_squared_error(df_rul,final_logistic_pred)))
print("mean_absolute_error >>",mean_absolute_error(df_rul,final_logistic_pred))
print("scoring function >>",scoring_function(np.array(df_rul),final_logistic_pred))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




