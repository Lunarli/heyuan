import os

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import Flatten
from torch.utils.data import Dataset, DataLoader
class LSTM1(nn.Module):
    """LSTM architecture"""

    def __init__(self, input_size, hidden_size, num_layers, seq_length=1):
        super(LSTM1, self).__init__()
        self.input_size = input_size  # input size 输入特征维度
        self.hidden_size = hidden_size  # hidden state  LSTM输出维度
        self.num_layers = num_layers  # number of layers LSTM层数
        self.seq_length = seq_length  # sequence length   序列长度

        self.lstm = nn.LSTM(input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                            dropout=0.1)
        self.fc_1 = nn.Linear(hidden_size, 7)  # fully connected 1
        self.fc_2 = nn.Linear(16, 8)  # fully connected 2
        self.fc = nn.Linear(8, 1)  # fully connected last layer

        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

    def forward(self, x):
        """

        :param x: input features
        :return: prediction results
        """
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))  # hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))  # internal state
        ###gpu配置
        h_0 = h_0
        c_0 = c_0
        # h_0 = h_0
        # c_0 = c_0
        output, (hn, cn) = self.lstm(x, (h_0, c_0))  # lstm with input, hidden, and internal state
        out = self.fc_1(output)
        out = out[:,-24:,:]
        return out

# class ETTDataset(Dataset):
#     def __init__(self, root_path='F:\python demo\informer\data\ETT',
#                  flag='train', size=[96, 48, 24],
#                  features='M', data_path='ETTh1.csv',
#                  target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None, train_only=False):
#         # size [seq_len, label_len, pred_len]
#         # infoeq_len, label_len, pred_len]
#         # info
#         if size == None:
#             self.seq_len = 24 * 4 * 4
#             self.label_len = 24 * 4
#             self.pred_len = 24 * 4
#         else:
#             self.seq_len = size[0]
#             self.label_len = size[1]
#             self.pred_len = size[2]
#         # init
#         assert flag in ['train', 'test', 'val']
#         type_map = {'train': 0, 'val': 1, 'test': 2}
#         self.set_type = type_map[flag]
#
#         self.features = features
#         self.target = target
#         self.scale = scale
#         self.timeenc = timeenc
#         self.freq = freq
#         self.train_only = train_only
#
#         self.root_path = root_path
#         self.data_path = data_path
#         self.__read_data__()
#
#     def __read_data__(self):
#         self.scaler = StandardScaler()
#         df_raw = pd.read_csv(os.path.join(self.root_path,
#                                           self.data_path))
#
#         '''
#         df_raw.columns: ['date', ...(other features), target feature]
#         '''
#         cols = list(df_raw.columns)
#         if self.features == 'S':
#             cols.remove(self.target)
#         cols.remove('date')
#         # print(cols)
#         num_train = int(len(df_raw) * (0.7 if not self.train_only else 1))
#         num_test = int(len(df_raw) * 0.2)
#         num_vali = len(df_raw) - num_train - num_test
#         border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
#         border2s = [num_train, num_train + num_vali, len(df_raw)]
#         # border1和border2为三个划分部分的起始和终止索引
#         border1 = border1s[self.set_type]
#         border2 = border2s[self.set_type]
#
#         if self.features == 'M' or self.features == 'MS':
#             df_raw = df_raw[['date'] + cols]
#             cols_data = df_raw.columns[1:]
#             df_data = df_raw[cols_data]
#         elif self.features == 'S':
#             df_raw = df_raw[['date'] + cols + [self.target]]
#             df_data = df_raw[[self.target]]
#
#         if self.scale:
#             train_data = df_data[border1s[0]:border2s[0]]
#             self.scaler.fit(train_data.values)
#             # print(self.scaler.mean_)
#             # exit()
#             data = self.scaler.transform(df_data.values)
#         else:
#             data = df_data.values
#
#         df_stamp = df_raw[['date']][border1:border2]
#         df_stamp['date'] = pd.to_datetime(df_stamp.date)
#         if self.timeenc == 0:
#             df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
#             df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
#             df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
#             df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
#             data_stamp = df_stamp.drop(['date'], 1).values
#         # elif self.timeenc == 1:
#         #     # data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
#         #     # data_stamp = data_stamp.transpose(1, 0)
#
#         self.data_x = data[border1:border2]
#         self.data_y = data[border1:border2]
#         self.data_stamp = data_stamp
#
#     def __getitem__(self, index):
#         # index = 0
#         s_begin = index
#         s_end = s_begin + self.seq_len
#         r_begin = s_end - self.label_len
#         r_end = r_begin + self.label_len + self.pred_len
#         # x 0-96  y 48-192
#         seq_x = self.data_x[s_begin:s_end]
#         seq_y = self.data_y[r_begin:r_end]
#         seq_x_mark = self.data_stamp[s_begin:s_end]
#         seq_y_mark = self.data_stamp[r_begin:r_end]
#
#         return seq_x, seq_y
#
#     def __len__(self):
#         return len(self.data_x) - self.seq_len - self.pred_len + 1
#
#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)


class ETTDataset(Dataset):
    def __init__(self, root_path='F:\python demo\informer\data\ETT\ETTh1.csv',seq_len = 96 ,pred_len = 24,mode = 'train'):
        self.seq_len = seq_len  # 输入序列长度
        self.pred_len = pred_len  # 输出序列长度
        self.root_path = root_path  # 文件目录
        self.scale = StandardScaler() # 标准化
        self.mode = mode
        self.datax = []
        self.datay = []

        self.__read_data__()

    def __read_data__(self):

        df_raw = pd.read_csv(self.root_path)

        '''
        df_raw.columns: ['date', ...(other features)]
        '''
        num_train = int(len(df_raw))  # 训练
        # num_test = int(len(df_raw) * 0.2)  # 测试
        # num_vali = len(df_raw) - num_train - num_test # 验证

        if self.mode == 'train':
            train_data = df_raw.iloc[:num_train, 1]    # 原始数据集有多列 只取一列进行拟合预测
            normalized_data = self.scale.fit_transform(train_data.values.reshape(-1, 1))

            for i in range(num_train - self.seq_len - self.pred_len):
                self.datax.append(torch.Tensor(normalized_data[i:self.seq_len + i]))
                self.datay.append(torch.Tensor(normalized_data[self.seq_len + i:self.seq_len + self.pred_len + i]))
        elif self.mode == 'test':
            train_data = df_raw.iloc[:num_train, 1]
            normalized_data = self.scale.fit_transform(train_data.values.reshape(-1, 1))

            # 只取最后一个时刻的数据进行预测
            self.datax.append(torch.Tensor(normalized_data[num_train-self.seq_len-self.pred_len:num_train-self.pred_len]))
            self.datay.append(torch.Tensor(normalized_data[num_train-self.pred_len:]))





    def __getitem__(self, index):
        # index = 0


        return self.datax[index], self.datay[index]

    def __len__(self):
        return len(self.datax)

    def inverse_transform(self, data):
        return self.scale.inverse_transform(data)



if __name__ == '__main__':

    test_set = ETTDataset()
    test_set = ETTDataset(mode='test')
    shuffle_flag = True
    drop_last = False
    batch_size = 1
    data_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=1,
        drop_last=drop_last)

    model = LSTM1(7,50,1)  # 特征维度  嵌入特征维度  LSTM层数
    model_optim = optim.Adam(model.parameters(), lr=3e-5)
    for epoch in range(30):
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
        # for i, (batch_x, batch_y) in enumerate(train_loader):
        #     iter_count += 1
        #     model_optim.zero_grad()
            model_optim.zero_grad()
            batch_x = batch_x.float()
            output = model(batch_x)
            batch_y = batch_y[:,-24:,:].float()
            loss = criterion(output, batch_y)
            loss.backward()
            model_optim.step()
            if(i%10==0):
                print(loss)
            # print(batch_x)
        print(loss)