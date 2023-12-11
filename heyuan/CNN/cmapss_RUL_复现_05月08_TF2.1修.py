
import os 
import sys

from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import RMSprop,Adam,Adadelta,Nadam,Adamax,Adagrad
from tensorflow.keras.callbacks import History
from tensorflow.keras import callbacks,Input
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.utils import plot_model
from tensorflow import keras

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

# plt.style.use('seaborn-whitegrid')#绘图的主题
plt.rcParams['font.sans-serif'] = 'Times New Roman'
plt.rcParams.update({'font.size': 12}) # 改变所有字体大小，改变其他性质类似

import warnings
warnings.filterwarnings('ignore')


import tensorflow as tf
#TF2.0以上的设置方法
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

# config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
# tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


# In[77]:


# Setting seed for reproducibility
np.random.seed(1234)

# Input files don't have column names
dependent_var = ['RUL']#依赖的变量
index_columns_names =  ["UnitNumber","Cycle"]
operational_settings_columns_names = ["OpSet"+str(i) for i in range(1,4)]#添加操作条件1,2,3,
sensor_measure_columns_names =["Sensor"+str(i) for i in range(1,22)]#添加传感器编号1，…，21
#输入发动机编号/运行时间/设置/哪个传感器数据/
input_file_column_names = index_columns_names + operational_settings_columns_names + sensor_measure_columns_names


# ## 读取训练集

# In[3]:





# In[78]:


#pandas读取数据/True:以，为分割符/names:指定列名
df_train = pd.read_csv(r'F:\dataset\C-MAPSS\train_FD001.txt',delim_whitespace=True,names=input_file_column_names)

rul = pd.DataFrame(df_train.groupby('UnitNumber')['Cycle'].max()).reset_index()
rul.columns = ['UnitNumber', 'max']
#将每一Unitnumber中最大的Cycle找到，并在原来的df_train中添加新的colum，位置为Unitnumber的左边，也即在最右端
df_train = df_train.merge(rul, on=['UnitNumber'], how='left')
#计算出RUL对应的各个值
df_train['RUL'] = df_train['max'] - df_train['Cycle']
#然后在把最大的max去掉
df_train.drop('max', axis=1, inplace=True)


# In[7]:


from IPython.core.interactiveshell import InteractiveShell 
InteractiveShell.ast_node_interactivity = 'all' 
df_train = pd.read_csv(r'F:\dataset\C-MAPSS\train_FD001.txt',delim_whitespace=True,names=input_file_column_names)

rul1 = pd.DataFrame(df_train.groupby('UnitNumber'))
rul2 = pd.DataFrame(df_train.groupby('UnitNumber')['Cycle'].max())
rul3 = pd.DataFrame(df_train.groupby('UnitNumber')['Cycle'].max()).reset_index()
df_train.head()
rul1
rul2
rul3
print(rul1)
print(rul2)
print(rul3)


# In[5]:


#https://www.jb51.net/article/155584.htm
#https://www.jb51.net/article/151614.htm


# In[11]:


df_train.head()


# ## 读取测试集

# #### 方便可视化测试集，计算“截断的”RUL值：注意这里还不是Actual_RUL

# In[79]:


df_test = pd.read_csv(r'F:\dataset\C-MAPSS\test_FD001.txt',delim_whitespace=True,names=input_file_column_names)
"""为了方便测试集的每一个unit的可视化，需要给测试集算出真实的RUL列值"""
rul = pd.DataFrame(df_test.groupby('UnitNumber')['Cycle'].max()).reset_index()
rul.columns = ['UnitNumber', 'max']
#将每一Unitnumber中最大的Cycle找到，并在原来的df_train中添加新的colum，位置为Unitnumber的左边，也即在最右端
df_test = df_test.merge(rul, on=['UnitNumber'], how='left')
#计算出RUL对应的各个值
"""此时还不是真实剩余RUL"""
df_test['RUL'] = df_test['max'] - df_test['Cycle']
#然后在把最大的max去掉
df_test.drop('max', axis=1, inplace=True)

df_test.head()


# In[80]:


y_true = pd.read_csv('F:\dataset\C-MAPSS\RUL_FD001.txt',delim_whitespace=True,names=["RUL"])
"""注意：这里UnitNumber=1=测试集的UnitNumber"""
y_true["UnitNumber"] = y_true.index+1
y_true


# In[81]:


"""首先制作以UnitNumber为ID的真是RUL的DataFrame"""
actual_rul = pd.DataFrame(y_true.groupby('UnitNumber')['RUL'].max()).reset_index()
actual_rul.columns = ['UnitNumber', 'acrul']
df_test = df_test.merge(actual_rul, on=['UnitNumber'], how='left')
df_test.head()


# In[82]:


"""最终才是真实的测试集RUL"""
df_test['RUL'] = df_test['RUL']+df_test['acrul']
#然后在把最大的acrul去掉
df_test.drop('acrul', axis=1, inplace=True)

df_test


# In[83]:


"""采用分段线性退化假说：参照相关的文献，其分段的RUL认为>=130的RUL值，均标记为RUL=130
    注意：部分文献的分段RUL=125
    github上面一般也没有这样处理的代码，但是论文都是用的120-130
"""
def fun(x):
    if x >= 125:
        return 125
    else:
        return x


# In[85]:


df_test['RUL']=df_test['RUL'].apply(lambda x: fun(x))
df_test


# In[84]:


y_true['RUL']=y_true['RUL'].apply(lambda x: fun(x))
y_true


# ## 数据分析
df_train.isnull().sum()
# In[14]:


df_train.isnull().sum()

df_test.isnull().sum()
# ## 绘制小提琴图
temp_df = df_train[["UnitNumber","Cycle"]].groupby("UnitNumber").max()
sns.violinplot(temp_df.Cycle)
plt.title("Life of Engines")
# plt.xticks(fontsize=12, fontweight='bold')  # 默认字体大小为10
# In[12]:


temp_df = df_train[["UnitNumber","Cycle"]].groupby("UnitNumber").max()
sns.violinplot(temp_df.Cycle)
plt.title("Life of Engines")
temp_df
# plt.xticks(fontsize=12, fontweight='bold')  # 默认字体大小为10


# In[13]:


temp_df = df_train[["UnitNumber","Cycle"]].groupby("UnitNumber").max()
sns.violinplot(temp_df.Cycle)
plt.title("Life of Engines")
# plt.xticks(fontsize=12, fontweight='bold')  # 默认字体大小为10


# In[8]:


temp_df = df_train[["UnitNumber","Cycle"]].groupby("UnitNumber").max()
temp_df


# ## 拥有最大RUL的排序首位五个
df_train[["UnitNumber","Cycle"]].groupby("UnitNumber").max().sort_values(by = ["Cycle"], ascending= False).head(5)
# ## 拥有最大RUL的排序，末尾的五个
df_train[["UnitNumber","Cycle"]].groupby("UnitNumber").max().sort_values(by = ["Cycle"], ascending= False).tail(5)df_test[["UnitNumber","Cycle"]].groupby("UnitNumber").max().sort_values(by = ["Cycle"], ascending= False).tail(5)
# ## 三个工况的操作条件可视化，选择其中一个单元
fig,ax=plt.subplots(1,3,figsize=(30,8),sharex='all')
for i in range(0,3):
    df_u1=df_train.query('UnitNumber==8')
    ax[i].plot(df_u1.Cycle.values, df_u1['OpSet'+str(i+1)])
    ax[i].set_title('OpSet'+str(i+1))
    ax[i].set_xlabel("Cycle")
# In[7]:


fig,ax=plt.subplots(1,3,figsize=(30,8),sharex='all')
for i in range(0,3):
    df_u1=df_train.query('UnitNumber==8')
    ax[i].plot(df_u1.Cycle.values, df_u1['OpSet'+str(i+1)])
    ax[i].set_title('OpSet'+str(i+1))
    ax[i].set_xlabel("Cycle")


# In[ ]:





# In[ ]:





# ## 选择其中一个单元进行传感器数据的可视化
fig,ax=plt.subplots(7,3,figsize=(30,20),sharex=True)
df_u1=df_train.query('UnitNumber==50')
c=0
for i in range(0,7):
    for j in range(0,3):
        ax[i,j].plot(df_u1.Cycle.values, df_u1['Sensor'+str(c+1)])
        ax[i,j].set_title('Sensor'+str(c+1),fontsize=20)
        ax[i,j].axvline(0,c='r')
        c+=1
plt.suptitle('Sensor Traces: Unit 50',fontsize=25)
plt.show()
# In[9]:


fig,ax=plt.subplots(7,3,figsize=(30,20),sharex=True)
df_u1=df_train.query('UnitNumber==30')
c=0
for i in range(0,7):
    for j in range(0,3):
        ax[i,j].plot(df_u1.Cycle.values, df_u1['Sensor'+str(c+1)])
        ax[i,j].set_title('Sensor'+str(c+1),fontsize=20)
        ax[i,j].axvline(0,c='r')
        c+=1
plt.suptitle('Sensor Traces: Unit 50',fontsize=25)
plt.show()


# ### 去除没必要的传单其数据

# In[86]:


# necessary features for analysis

"""FD001&FD003"""
not_required_feats = ["Sensor1", "Sensor5", "Sensor6", "Sensor10", "Sensor16", "Sensor18", "Sensor19"]

# """FD002"""
# not_required_feats = []


feats = [feat for feat in sensor_measure_columns_names if feat not in not_required_feats]
feats


# In[12]:


feats + ["RUL"]


# ### 进行相关性分析
corr = df_train[feats + ["RUL"]].corr()

fig = plt.figure(figsize=(14,12))
ax = fig.add_subplot(111)
ax = sns.heatmap(corr, annot=True, cmap = "coolwarm", fmt=".2f")
# In[13]:


corr = df_train[feats + ["RUL"]].corr()

fig = plt.figure(figsize=(14,12))
ax = fig.add_subplot(111)
ax = sns.heatmap(corr, annot=True, cmap = "coolwarm", fmt=".2f")


# ### 可视化在失效之前，传感器的数值
pairPlot = sns.PairGrid(data = df_train[df_train.UnitNumber < 10], x_vars = "RUL", y_vars = feats, hue = "UnitNumber", size = 3, aspect = 2)
pairPlot = pairPlot.map(plt.scatter, alpha = 0.5)
pairPlot = pairPlot.set(xlim = (50, 0))
pairPlot = pairPlot.add_legend()
# In[14]:


pairPlot = sns.PairGrid(data = df_train[df_train.UnitNumber < 10], x_vars = "RUL", y_vars = feats, hue = "UnitNumber", size = 3, aspect = 2)
pairPlot = pairPlot.map(plt.scatter, alpha = 0.5)
pairPlot = pairPlot.set(xlim = (50, 0))
pairPlot = pairPlot.add_legend()


# ### 查看两个传感器之间的相关性
"""必要时，移除，以避免魔性的复杂化"""
plt.plot(df_train.Sensor9, df_train.Sensor14)
# In[15]:


plt.plot(df_train.Sensor9, df_train.Sensor14)


# ## 二.数据处理

# 为了确保特征的方差保持在相同的范围内，缩放特征是很重要的。如果一个特征的方差大于其他特征的方差的数量级，则该特定特征可能支配数据集中的其他特征，这是不可取的。

# In[87]:


sequence_length = 36
mask_value = 0
"""选择哪些特征进行预测"""
# feats.append('Cycle')
# feats=feats+operational_settings_columns_names


# In[88]:


#这里是否出现问题/ MinMaxScaler：归一到 [ 0，1 ] MaxAbsScaler：归一到 [ -1，1 ] 
from sklearn.preprocessing import MaxAbsScaler,MinMaxScaler
min_max_scaler = MinMaxScaler(feature_range=(0,1))

df_train[feats] = min_max_scaler.fit_transform(df_train[feats])
df_test[feats] = min_max_scaler.transform(df_test[feats])


# ### 自定义数据的三维转换函数

# In[89]:


# LSTM希望输入是三维numpy数组的形状，我需要相应地转换训练和测试数据。
def gen_train(id_df, seq_length, seq_cols):
 
    data_array = id_df[seq_cols].values
    #存储的array的shape,第一个维度必须是0，有且仅有这一个，代表这个维度是可拓展的。
    num_elements = data_array.shape[0]    # 样本数量
    lstm_array=[]
    
    for start, stop in zip(range(0, num_elements-seq_length+1), range(seq_length, num_elements+1)):
        lstm_array.append(data_array[start:stop, :])
    
    return np.array(lstm_array)
    

def gen_target(id_df, seq_length, label):
    data_array = id_df[label].values
    num_elements = data_array.shape[0]
    return data_array[seq_length-1:num_elements+1]


def gen_test(id_df, seq_length, seq_cols, mask_value):
    print(id_df.head())
    df_mask = pd.DataFrame(np.zeros((seq_length-1,id_df.shape[1])),columns=id_df.columns)
    print(df_mask.head())
    df_mask[:] = mask_value
    print(df_mask.head())
    id_df = df_mask.append(id_df,ignore_index=True)
    print(id_df.head())
    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    lstm_array=[]

    start = num_elements-seq_length
    stop = num_elements
    
    lstm_array.append(data_array[start:stop, :])
    
    return np.array(lstm_array)


# In[60]:


gen_target(df_train[df_train['UnitNumber']==1], sequence_length, "RUL")


# In[53]:


x = gen_train(df_train[df_train['UnitNumber']==1], sequence_length, feats)
x.shape
df_train[df_train['UnitNumber']==1].shape
print(type (x))
print(type (list(x)))
print(type (list(list(x))))
list(list(x))


# In[90]:


#generate train
# feats 为使用的传感器编号
# sequence_length 设置为36
x_train=np.concatenate(list(list(gen_train(df_train[df_train['UnitNumber']==unit], sequence_length, feats)) 
                            for unit in df_train['UnitNumber'].unique()))
print(x_train.shape)


# In[91]:


#generate target of train
y_train = np.concatenate(list(list(gen_target(df_train[df_train['UnitNumber']==unit], sequence_length, "RUL")) 
                              for unit in df_train['UnitNumber'].unique()))
y_train.shape


# In[67]:



df_test.head()
df_test['UnitNumber'].unique()


# In[92]:


#generate test
x_test=np.concatenate(list(list(gen_test(df_test[df_test['UnitNumber']==unit], sequence_length, feats, mask_value)) 
                           for unit in df_test['UnitNumber'].unique()))
print(x_test.shape)
x_test


# In[93]:


#true target of test 
y_test = y_true.RUL.values
y_test.shape


# In[94]:


# import keras.backend as K
import tensorflow.keras.backend as K
from sklearn.metrics import mean_squared_error

#自定义评价指标
def RMSE(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred -y_true)))

#自定义PHM2008评价函数
def Scoring_2008(Y_true, Y_pred):
    h = Y_pred - Y_true
    g = (-(h-K.abs(h))/2.0)  # 正数为0
    f = ((K.abs(h)+h)/2.0)   # 负数取0
    return K.sum(K.exp(g/13.0)-1)+K.sum(K.exp(f/10.0)-1)


# In[103]:


input_shape = (4, 10, 128)
x = tf.random.normal(input_shape)
y = tf.keras.layers.Conv1D(32, 3, activation='relu',input_shape=input_shape[1:])(x)
print(y.shape)
print(input_shape[1:])


# In[105]:


inputt=Input(shape=(x_train.shape[1],x_train.shape[2]))
x = tf.keras.layers.Conv1D(filters=10,kernel_size=10,activation='tanh',padding='same',
                           kernel_initializer='glorot_uniform')(inputt)
x = tf.keras.layers.Conv1D(filters=10,kernel_size=10,activation='tanh',padding='same',
                           kernel_initializer='glorot_uniform')(x)
x = tf.keras.layers.Conv1D(filters=10,kernel_size=10,activation='tanh',padding='same',
                           kernel_initializer='glorot_uniform')(x)
x = tf.keras.layers.Conv1D(filters=10,kernel_size=10,activation='tanh',padding='same',
                           kernel_initializer='glorot_uniform')(x)
x = tf.keras.layers.Conv1D(filters=10,kernel_size=3,activation='tanh',padding='same',
                           kernel_initializer='glorot_uniform')(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(100,activation='tanh')(x)
x = tf.keras.layers.Dense(1,activation='relu')(x)

DCNN_model=Model(inputs=inputt,outputs=x)
#查看网络结构
DCNN_model.summary()
#编译模型 RMSprop,Adam,Adadelta,Nadam,Adamax,Adagrad
"""分段学习率"""
def scheduler(epoch):
    if epoch > 200:
        return 0.0001
    else:
        return 0.001
"""打印学习率，方便查看"""
def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr#无衰减策略的学习率
    return lr

optimizer_diy =tf.keras.optimizers.Adam(learning_rate=0.001)
lr_metric = get_lr_metric(optimizer_diy)

DCNN_model.compile(optimizer=optimizer_diy,loss='mse',metrics=['mse','mae',RMSE,Scoring_2008,lr_metric]) 


# In[106]:


get_ipython().run_cell_magic('time', '', '#训练模型\n# """https://blog.csdn.net//article/details/106398881\n#     连续保存模型参数model_{epoch:03d}.hdf5；join(save_dir, \'model_Weights_{epoch:03d}.h5\')\n# """NewComerSyt:period=1,\n\nBatch_size=512\nsave_dir="../DCNN_复现/FD001_30time/"\nif not os.path.exists(save_dir):\n                os.makedirs(save_dir)\n"""\n注意：validation_data=(x_test,y_test)这样是不允许的，算是训练作弊\n要么是通过手动从x_train中单独划分(x_vaild,y_vaild),raise后采用validation_data=(x_vaild,y_vaild)\n要么通过keras或者tf.keras中的model.fit(validation_split=0.1-0.3)从训练集随机划分\n"""\nHistory = DCNN_model.fit(x_train, y_train, epochs=250, batch_size=Batch_size,validation_split=0.20,\n                        verbose=2,\n            callbacks =[keras.callbacks.EarlyStopping(monitor =\'val_loss\', min_delta=0,patience=125, verbose=0, mode=\'min\'),\n                        keras.callbacks.ModelCheckpoint(os.path.join(save_dir, \'model_Weights_best.h5\'),\n                                                        monitor=\'val_loss\',save_best_only=True,\n                                                        save_weights_only=True, mode=\'min\', verbose=0),\n                        keras.callbacks.LearningRateScheduler(scheduler)])')


# In[107]:


def plotTrainHistory(save_dir,model,title=' '):
    plt.grid(linestyle="--")
    plt.plot(model.history['loss'])
    plt.plot(model.history['val_loss'])
    plt.title(title)
    plt.ylabel('mse')
    plt.xlabel('epoch')
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(save_dir+'loss.png' ,bbox_inches='tight', dpi=300)  # 600
    plt.show()
    
print(History.history.keys())
plotTrainHistory(save_dir,History)


# In[30]:


DCNN_model.load_weights(os.path.join(save_dir, 'model_Weights_best.h5'))


# ### 方法一：模型 与 权重分开读取

# In[31]:


# ###模型保存2
# dir_P,json_P,weights_P = Model_to_save(model, Bpath="CNN_LSTM_Best_Weights/",
#                                       json_path="Functonal_151_j",
#                                       weights_path="model_Weights_last_only")

# ###模型读取
# # reloaded_model = Model_Load(dir_P,json_P,weights_P)

# reloaded_model = Model_Load(dirs =save_dir,
#                             json_path="Functonal_151_j",
#                             weights_path="model_Weights_best_only")


# In[98]:


scores = DCNN_model.evaluate(x_train, y_train, verbose=1, batch_size=Batch_size)
df = pd.DataFrame({'MSE': [scores[1]],
                   'MAE': [scores[2]],
                   'RMSE': [scores[3]],
                  'Scoring_2008':[scores[4]]})

df.transpose().to_csv(save_dir+"评估得分.txt",
                      mode='w',header=['train'],sep=' ',index=["MSE","MAE","RMSE","Scoring_2008"])


# In[99]:


scores = DCNN_model.evaluate(x_test,y_test, verbose=1, batch_size=Batch_size)
df = pd.DataFrame({'MSE': [scores[1]],
                   'MAE': [scores[2]],
                   'RMSE': [scores[3]],
                  'Scoring_2008':[scores[4]]})

df.transpose().to_csv(save_dir+"评估得分.txt",
                      mode='a',header=['test'],sep=' ',index=["MSE","MAE","RMSE","Scoring_2008"])


# ### 进行预测

# In[100]:


y_pred_test = DCNN_model.predict(x_test,verbose=2)
y_true_test = y_test.reshape(y_test.shape[0],1).astype(np.float32)


# In[101]:


"""保存预测的RUL值与实际的RUL值"""
y_pred_test=y_pred_test.reshape(y_pred_test.shape[0]*y_pred_test.shape[1]).astype(np.float32)
y_true_test=y_true_test.reshape(y_true_test.shape[0]*y_true_test.shape[1]).astype(np.float32)
y_All=pd.DataFrame({'Pre':y_pred_test,
                   'Actual':y_true_test})
y_All.to_csv(save_dir+"y_lastpoint_Pre.csv",
                      mode='w',header=["Pre","Actual"],sep=',',index=0)


# In[102]:


"""测试集所有unitNumber的最后一个值进行可视化"""
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_true_test,y_pred_test)
score_2008 = Scoring_2008(y_true_test,y_pred_test)

fig=plt.figure(figsize=(16,8))
plt.grid(linestyle="--")
plt.plot(y_pred_test, color='red', label='Prediction', marker='o',linestyle='-',linewidth=1.2)
plt.plot(y_true_test, color='blue', label='Ground Truth', marker='v',linestyle='-',linewidth=1.2)
plt.title('FD001'+ ', MSE: '+str('%.2f' % mse)+ ', Score2008: '+str('%.2f' % score_2008),fontsize=12)
plt.ylabel('RUL',fontsize=12)
plt.xlabel('Unit Number' ,fontsize=12)
plt.legend(loc='upper left',fontsize=12)
fig.tight_layout()
plt.savefig(save_dir+'FD001_Last_point_Pre.png' ,bbox_inches='tight', dpi=300)  # 600
plt.show()


# In[ ]:





# In[ ]:




