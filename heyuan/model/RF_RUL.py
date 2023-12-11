import math
import os
import copy
import numpy as np
import pandas as pd
import torch
from heyuan.utils.data_loader import CMPDataset

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader




class RF_RUL(torch.nn.Module):
    def __init__(self, seq_len, fea_dim):
        super(RF_RUL, self).__init__()

        # 创建 RF 模型
        self.rf_model = RandomForestRegressor(n_estimators=100,random_state=42)

        # 参数初始化 seq_length 序列长度 fea_dim特征维度
        self.seq_len, self.fea_dim = seq_len, fea_dim


    def forward(self, x_train, y_train, x_test):
        self.rf_model.fit(torch.reshape(x_train,[-1,self.seq_len*self.fea_dim]), y_train)
        # 将输入数据转换为2D数组
        n_samples = x_test.shape[0]
        x_test = x_test.reshape(n_samples, self.seq_len * self.fea_dim)

        # 使用随机森林模型进行预测
        predicted_target_2d = self.rf_model.predict(x_test)

        # 将预测结果转换tensor
        predicted_target = torch.from_numpy(predicted_target_2d)


        return predicted_target



class Dataset_Custom(Dataset):
    def __init__(self, root_path='/Users/apple/Downloads/Frame-pytorch--for-RUL-Prediction/time_dataset',
                 flag='train', size=[96, 48, 24],
                 features='M', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None, train_only=False):
        # size [seq_len, label_len, pred_len]
        # infoeq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.train_only = train_only

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        if self.features == 'S':
            cols.remove(self.target)
        cols.remove('date')
        # print(cols)
        num_train = int(len(df_raw) * (0.7 if not self.train_only else 1))
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        # border1和border2为三个划分部分的起始和终止索引
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            df_raw = df_raw[['date'] + cols]
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_raw = df_raw[['date'] + cols + [self.target]]
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            # print(self.scaler.mean_)
            # exit()
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        # elif self.timeenc == 1:
        #     # data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
        #     # data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        # index = 0
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        # x 0-96  y 48-192
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

if __name__ == '__main__':

    data = CMPDataset(r"F:\python demo\rul_collection\LunarLi-net\data", "FD001", 125, 30)
    data_loader = torch.utils.data.DataLoader(data, batch_size=16, shuffle=True)
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn import tree
    import matplotlib.pyplot as plt
    import io
    import pydotplus
    from PIL import Image

    # 模拟生成时间序列数据
    np.random.seed(0)
    n_samples = 200  # 输入数据长度
    n_predictions = 24  # 预测输出长度
    X = np.arange(n_samples).reshape(-1, 1)
    y = np.sin(0.1 * X) + np.random.normal(0, 0.1, size=(n_samples, 1))

    # 拆分输入数据和预测输出数据
    X_train = X[:-n_predictions]
    y_train = y[:-n_predictions].flatten()
    X_test = X[-n_predictions:]

    # 创建随机森林模型
    rf = RandomForestRegressor(n_estimators=5, random_state=0)

    # 训练模型
    rf.fit(X_train, y_train)

    # 进行预测
    y_pred = rf.predict(X_test)

    # 输出预测结果
    print("预测结果：", y_pred)

    # 绘制决策树图像
    for per_rf in rf.estimators_:
        dot_data = tree.export_graphviz(per_rf, out_file=None,
                                        feature_names=['x'],
                                        filled=True, rounded=True,
                                        special_characters=True)
        graph = pydotplus.graph_from_dot_data(dot_data)
        image_bytes = graph.create_png()
        image_stream = io.BytesIO(image_bytes)
        image = Image.open(image_stream)
        plt.figure()
        plt.imshow(image)
        plt.axis('off')
        plt.show()



