#!/usr/bin/env python
# coding: utf-8

# In[227]:


import matplotlib.pyplot as plt


# In[228]:


import pandas as pd
import numpy as np
from math import log10


# In[229]:


models_label = ['Random Forest Regression','Random Forest Regression','Linear Regression','Linear Regression','Logistic Regression','Logistic Regression',
                'LSTM Lookback=1','LSTM Lookback=1','LSTM Lookback=5','LSTM Lookback=5',
                'LSTM Lookback=10','LSTM Lookback=10']
VAR = ['Without VAR','With VAR','Without VAR','With VAR','Without VAR','With VAR',
       'Without VAR','With VAR','Without VAR','With VAR','Without VAR','With VAR']
MSE = [3179.9, 788.57,
       2747.63, 1233.97,
       2454.39, 1262.74,
       1082.23, 773.07,
       1854.72, 2589.95,
       1694.33, 1688.7]
RMSE = [56.39,28.08,
        52.41,35.12,
        49.54,35.53,
        32.89,27.8,
        43.06,50.89,
        41.16,41.0]
MAE = [49.34,22.37,
       44.35,28.48,
       40.5,28.68,
       26.39,21.45,
       33.38,41.51,
       32.39,29.34]
scoring_function = [327970.27,1986.47,
                    282257.88,5721.58,
                    210561.72,5285.39,
                    3819.08,2131.60,
                    991089.91,43056.87,
                    173798.23,8879551.74]

log_scoring_function = []
for s in scoring_function:
    log_scoring_function.append(np.round(log10(s),3))


# In[230]:


dict = {'Model': models_label, 'With/Without VAR': VAR, 'MSE': MSE, 'RMSE':RMSE, 'MAE':MAE, 'Log10 of Scoring Function':log_scoring_function} 
# df = pd.DataFrame(models_label,VAR,MSE,RMSE,MAE,scoring_function)


# In[231]:


df = pd.DataFrame(dict)
df


# In[232]:


ax = sns.barplot(x="Model", y="MSE", hue="With/Without VAR", data=df)
plt.xticks(rotation=90)
plt.ylabel("Mean Sqaure Error", fontsize=12)


# In[233]:


ax = sns.barplot(x="Model", y="RMSE", hue="With/Without VAR", data=df)
plt.xticks(rotation=90)
plt.ylabel("Root Mean Square Error", fontsize=12)


# In[234]:


ax = sns.barplot(x="Model", y="MAE", hue="With/Without VAR", data=df)
plt.xticks(rotation=90)
plt.ylabel("Mean Aboslute Error", fontsize=12)


# In[235]:


ax = sns.barplot(x="Model", y="Log10 of Scoring Function", hue="With/Without VAR", data=df)
plt.xticks(rotation=90)

plt.ylabel("Log10 of Scoring Function Value", fontsize=12)



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[238]:


list(zip(models_label,VAR,log_scoring_function))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




