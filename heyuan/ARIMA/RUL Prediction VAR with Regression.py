#!/usr/bin/env python
# coding: utf-8

# # Run to Failure degradation simulation of NASA Turbo Jet Engine Fans

# In[26]:


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
import pickle


# In[27]:


train_file = "train_FD001.txt" 
test_file = "test_FD001.txt"
RUL_file = "RUL_FD001.txt"


# In[28]:


log_model = pickle.load(open('logistic_regression.sav', 'rb'))
lin_model = pickle.load(open('linear_regression.sav', 'rb'))
forest_model = pickle.load(open('forest_regression.sav', 'rb'))


# In[29]:


df_test = pd.read_csv(test_file, sep=" ",header=None)
df_test.drop(columns=[26,27],inplace=True)
columns = ["Section-{}".format(i)  for i in range(26)]
df_test.columns = columns
# df_test.head()


# In[30]:


df_test_rul = pd.read_csv(RUL_file, names=['rul'])
# df_test_rul.head()


# In[31]:


RUL_name = ["Section-1"]
RUL_data = df_test[RUL_name]
MachineID_series = df_test["Section-0"]
grp = RUL_data.groupby(MachineID_series)
max_cycles = np.array([max(grp.get_group(i)["Section-1"]) for i in MachineID_series.unique()])


# In[32]:


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


# In[33]:


gen = MinMaxScaler(feature_range=(0, 1))
df_test = gen.fit_transform(df_test)
df_test = pd.DataFrame(df_test)
#df_test = df_test.rolling(20).mean()
pt = PowerTransformer()
df_test = pt.fit_transform(df_test)
df_test=np.nan_to_num(df_test)


# In[34]:


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


# In[35]:


print(len(test_data))


# In[36]:


y_new = []
for i in range(len(test_data)):
    y_new.append(pd.DataFrame(test_data[i]))

y_new[1]


# In[51]:


predictions = []
for i in range(len(y_new)):
    test_model = VAR(y_new[i])
    test_model_fit = test_model.fit()
    test_pred = test_model_fit.forecast(test_model_fit.y, steps=1)
    predictions.append(test_pred)


# In[52]:


predictions[0].shape


# In[53]:


logistic_pred = []
for i in range(100):
    logistic_pred.append(log_model.predict(predictions[i]))


# In[54]:


logistic_pred = np.array(logistic_pred)


# In[55]:


len(logistic_pred)


# In[56]:


y_test = np.array(df_test_rul)


# In[57]:


len(y_test)


# In[58]:


fig = plt.figure(figsize=(18,10))
plt.plot(logistic_pred,color='red', label='prediction')
plt.plot(y_test,color='blue', label='y_test')

fig.suptitle('RUL Prediction using VAR Model with Logistic Regression', fontsize=35)
plt.xlabel("Engine Number", fontsize=35)
plt.ylabel("Remaining Useful Life", fontsize=35)

plt.legend(loc='upper left')
plt.grid()
plt.show()


# In[59]:


df=pd.DataFrame(np.arange(1,101))
df['Actual']=y_test
df['Predicted']=logistic_pred
df=df.drop([0],axis=1)

sns.set_theme(style="whitegrid")
a4_dims = (18,10)
fig, ax = plt.subplots(figsize=a4_dims)
sns.lineplot(data = df,markers=True, dashes=False)

fig.suptitle('RUL Prediction using VAR Model with Logistic Regression', fontsize=35)
plt.xlabel("Engine Number", fontsize=35)
plt.ylabel("Remaining Useful Life", fontsize=35)


# In[60]:


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

print("mean_squared_error >> ", mean_squared_error(y_test,logistic_pred))
print("root_mean_squared_error >> ", math.sqrt(mean_squared_error(y_test,logistic_pred)))
print("mean_absolute_error >>",mean_absolute_error(y_test,logistic_pred))
print("scoring function >>",scoring_function(y_test,logistic_pred))


# # 2. Linear Regression with VAR

# In[61]:


lin_pred = []
for i in range(100):
    lin_pred.append(lin_model.predict(predictions[i]))

lin_pred = np.array(lin_pred)

print(len(lin_pred))

y_test = np.array(df_test_rul)

print(len(y_test))
lin_pred = lin_pred.reshape((100,1))

fig = plt.figure(figsize=(18,10))
plt.plot(lin_pred,color='red', label='prediction')
plt.plot(y_test,color='blue', label='y_test')

fig.suptitle('RUL Prediction using VAR Model with Linear Regression', fontsize=35)
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


print(lin_pred.shape)

print("mean_squared_error >> ", mean_squared_error(y_test,lin_pred))
print("root_mean_squared_error >> ", math.sqrt(mean_squared_error(y_test,lin_pred)))
print("mean_absolute_error >>",mean_absolute_error(y_test,lin_pred))
print("scoring function >>",scoring_function(y_test,lin_pred))


# In[62]:


df=pd.DataFrame(np.arange(1,101))
df['Actual']=y_test
df['Predicted']=lin_pred
df=df.drop([0],axis=1)

sns.set_theme(style="whitegrid")
a4_dims = (18,10)
fig, ax = plt.subplots(figsize=a4_dims)
sns.lineplot(data = df,markers=True, dashes=False)

fig.suptitle('RUL Prediction using VAR Model with Linear Regression', fontsize=35)
plt.xlabel("Engine Number", fontsize=35)
plt.ylabel("Remaining Useful Life", fontsize=35)


# # 3. Random Forest with VAR

# In[63]:


forest_pred = []
for i in range(100):
    forest_pred.append(forest_model.predict(predictions[i]))

forest_pred = np.array(forest_pred)

print(len(forest_pred))

y_test = np.array(df_test_rul)

print(len(y_test))
forest_pred = forest_pred.reshape((100,1))

fig = plt.figure(figsize=(18,10))
plt.plot(forest_pred,color='red', label='prediction')
plt.plot(y_test,color='blue', label='y_test')

fig.suptitle('RUL Prediction using VAR Model with Random Forest Regression', fontsize=35)
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


print(forest_pred.shape)

print("mean_squared_error >> ", mean_squared_error(y_test,forest_pred))
print("root_mean_squared_error >> ", math.sqrt(mean_squared_error(y_test,forest_pred)))
print("mean_absolute_error >>",mean_absolute_error(y_test,forest_pred))
print("scoring function >>",scoring_function(y_test,forest_pred))


# In[64]:


df=pd.DataFrame(np.arange(1,101))
df['Actual']=y_test
df['Predicted']=forest_pred
df=df.drop([0],axis=1)

sns.set_theme(style="whitegrid")
a4_dims = (18,10)
fig, ax = plt.subplots(figsize=a4_dims)
sns.lineplot(data = df,markers=True, dashes=False)

fig.suptitle('RUL Prediction using VAR Model with Random Forest Regression', fontsize=35)
plt.xlabel("Engine Number", fontsize=35)
plt.ylabel("Remaining Useful Life", fontsize=35)


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





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




