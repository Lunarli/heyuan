
import copy

import torch.nn

from torch.autograd import Variable

import torch.nn.functional as F

import torch
import torch.nn as nn
import math
import numpy as np

class CNN_RUL(nn.Module):
    def __init__(self,sequence_length = 30,FD_feature_columns = 14):
        super(CNN_RUL, self).__init__()
        self.conv1 = nn.Sequential(
            # 第一层卷积层，输入通道数为1，输出通道数为10，卷积核大小为(10, 1)，步长为1，填充大小为(5, 0)
            nn.Conv2d(1, 10, kernel_size=(10,1), stride=1, padding=(5,0)),
            # 批量归一化层，对卷积层的输出进行归一化
            nn.BatchNorm2d(10),
            # Tanh激活函数
            nn.Tanh()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 10, kernel_size=(10,1), stride=1, padding=(5, 0)),
            nn.BatchNorm2d(10),# 批量归一化层
            nn.Tanh()  # Tanh激活函数
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(10, 10, kernel_size=(10,1), stride=1, padding=(5, 0)),
            nn.BatchNorm2d(10),# 批量归一化层
            nn.Tanh()  # Tanh激活函数
        )

        # 继续定义conv3、conv4、conv5，具有相同的结构，区别在于输入输出通道数不同

        self.conv4 = nn.Sequential(
            nn.Conv2d(10, 10, kernel_size=(10,1), stride=1, padding=(5, 0)),
            nn.BatchNorm2d(10),# 批量归一化层
            nn.Tanh()  # Tanh激活函数
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(10, 1, kernel_size=(3,1), stride=1, padding=(1, 0)),
            nn.BatchNorm2d(1),# 批量归一化层
            nn.Tanh()  # Tanh激活函数
        )
        # Dropout层，以指定的概率对输入进行随机置零
        self.dropout = nn.Dropout(p=0.5)
        # 全连接层，输入特征数为540，输出特征数为100
        self.Dense1 = nn.Sequential(
            nn.Linear(in_features=540, out_features=100),
            nn.Tanh()
        )

        self.Dense2 = nn.Sequential(
            nn.Linear(100,1),# 全连接层，输入特征数为100，输出特征数为1
            # nn.ReLU()  # ReLU激活函数
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight,1)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight,1)



    def forward(self, x):

        #   batch C H W
        x = x.reshape(-1, 1, x.size(2), x.size(1))
        # 将输入x进行reshape，使其符合卷积层的输入格式
        # 第一层卷积层的前向传播
        conv1 = self.conv1(x)
        # 层的前向传播
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv_raw = self.conv5(conv4)
        a,b,c,d = conv_raw.shape
        # 继续传播conv3、conv4、conv5

        conv5 = conv_raw.reshape(conv_raw.size(0),-1)
        # 将conv5进行reshape，将其转换为一维向量
        self.dropout(conv5)

        dense1 = self.Dense1(conv5)
        # 全连接层Dense1的前向传播
        dense2 = self.Dense2(dense1)

        # 返回dense2作为输出

        return dense2



class LSTM_RUL(nn.Module):
    """LSTM architecture"""

    def __init__(self, input_size=14, hidden_size=50, num_layers=2, seq_length=30):
        super(LSTM_RUL, self).__init__()
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state
        self.num_layers = num_layers  # number of layers
        self.seq_length = seq_length  # sequence length

        # Define LSTM layer with dropout
        self.lstm = nn.LSTM(input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0.1)

        # Define fully connected layers
        self.fc_1 = nn.Linear(hidden_size, 16)  # fully connected 1
        self.fc_2 = nn.Linear(16, 8)  # fully connected 2
        self.fc = nn.Linear(8, 1)  # fully connected last layer

        self.dropout = nn.Dropout(0.1)  # dropout layer
        self.relu = nn.ReLU()  # activation function

    def forward(self, x):
        """Defines forward pass"""
        # Initializing hidden state and internal state
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))  # hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))  # internal state

        # Forward pass through LSTM layer
        output, (hn, cn) = self.lstm(x, (h_0, c_0))

        # Select last hidden states of LSTM
        hn_o = torch.Tensor(hn.detach()[-1, :, :])
        hn_1 = torch.Tensor(hn.detach()[1, :, :])
        hn_o = hn_o.view(-1, self.hidden_size)
        hn_1 = hn_1.view(-1, self.hidden_size)

        # Forward pass through fully connected layers
        out = self.relu(self.fc_1(self.relu(hn_o + hn_1)))
        out = self.relu(self.fc_2(out))
        out = self.dropout(out)
        out = self.fc(out)

        return out

class Transformer_RUL(nn.Module):
    def __init__(self, d_model, num_layers,dff, heads,seq_len,FD_feature, dropout):
        super().__init__()

        self.encoder = Encoder(d_model, num_layers, heads,dff,seq_len,FD_feature, dropout)  # Encoder
        self.out = nn.Linear(seq_len*d_model, 1)  # Linear

    def forward(self, x):   # batch seq_len fea_dim

        x = x.transpose(2,1).unsqueeze(-1)
        encoder_output = self.encoder(x)
        output = self.out(encoder_output.reshape(encoder_output.size(0),-1))
        output = output.unsqueeze(-1)
        return output

class Encoder(nn.Module):
    def __init__(self, d_model, num_layers, heads,dff, seq_len,FD_feature,dropout): #d_model = 128  # dimension in encoder, heads = 4  #number of heads in multi-head attention, N = 2  #encoder layers, m = 14  #number of features
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model

        self.embed = nn.Linear(seq_len*FD_feature, seq_len*d_model)
        self.pe = PositionalEncoder(maximum_position_encoding = 100, d_model = d_model)

        self.enc_layers = [EncoderLayer(d_model, heads, dff, dropout)
                           for _ in range(num_layers)]


        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        seq_len = x.size(2)
        x = x.reshape(x.size(0),-1)
        x = self.embed(x)
        x *= math.sqrt(self.d_model)
        x = x.reshape(-1,seq_len,self.d_model)
        x += self.pe[:, :seq_len, :]
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, None)
        return x  # (batch_size, input_seq_len, d_model)

def PositionalEncoder(maximum_position_encoding, d_model):
    def get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates
    angle_rads = get_angles(np.arange(maximum_position_encoding)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # 将 sin 应用于数组中的偶数索引（indices）；2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # 将 cos 应用于数组中的奇数索引；2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]
    return   torch.tensor(pos_encoding,dtype=torch.float32)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads,dff, dropout=0.5):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout)
        self.ff = FeedForward(d_model,dff,dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):

        attn_output = self.dropout_1(self.attn(x, x, x, mask))
        out1 = self.norm_1(x + attn_output)

        ffn_output = self.dropout_2(self.ff(out1))

        out2 = self.norm_1(out1 + ffn_output)


        return out2

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.5):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.dropout = dropout

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)    # (batch_size, num_heads, seq_len_q, depth)
        q = q.transpose(1, 2)    # (batch_size, num_heads, seq_len_k, depth)
        v = v.transpose(1, 2)    # (batch_size, num_heads, seq_len_v, depth)

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)

        output = self.out(concat)

        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=512, dropout=0.5):
        super().__init__()
        # set d_ff as a default to 512
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x



def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1)
    # scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)

    output = torch.matmul(scores, v)
    return output












class CNN_Trend(nn.Module):

    def __init__(self,sequence_length = 96,FD_feature_columns = 1):
        super(CNN_Trend, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 5, kernel_size=10, stride=1, padding=0),
            nn.BatchNorm1d(5),
            nn.Tanh()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(5, 5, kernel_size=10, stride=1, padding=0),
            nn.BatchNorm1d(5),
            nn.Tanh()
        )

        self.dropout = nn.Dropout(p=0.5)

        self.Dense1 = nn.Sequential(
            nn.Linear(in_features=390, out_features=24),
            nn.Tanh()
        )



        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight,1)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight,1)

    def forward(self, x):   # batch seq_len channel

        #  batch Channel(feature_dim) seq_len
        x = x.transpose(1,2)

        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        self.dropout(conv2)
        dense1 = self.Dense1(conv2.reshape(conv2.size(0),-1))

        return dense1.unsqueeze(2)



class LSTM_Trend(nn.Module):
    """
    LSTM architecture
    fea_dim : feature_dim
    hidden_size : lstm hidden size
    num_layers : lstm layers
    """

    def __init__(self, fea_dim = 1, hidden_size = 32, num_layers = 1,pred_sequence = 24):
        super(LSTM_Trend, self).__init__()
        self.fea_dim = fea_dim  # input size
        self.hidden_size = hidden_size  # hidden state
        self.num_layers = num_layers  # number of layers
        self.pred_sequence = pred_sequence

        self.lstm = nn.LSTM(fea_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                            dropout=0.1)
        self.fc_1 = nn.Linear(hidden_size, pred_sequence * fea_dim)  # fully connected 1


    def forward(self, x):
        """

        :param x: input features
        :return: prediction results
        """
        # h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))  # hidden state
        # c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))  # internal state

        output, (hn, cn) = self.lstm(x)  # lstm with input, hidden, and internal state
        out = self.fc_1(hn.squeeze())
        out = out.reshape((-1,self.pred_sequence,self.fea_dim))
        return out














if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib
    from matplotlib.pylab import mpl

    x = np.arange(0, 201)
    y = np.where(x <= 75, 125, 125 - (x - 75))

    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xlabel('Cycles')
    ax.set_ylabel('Remaining useful life')
    ax.tick_params(axis='both', direction='in')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylim(0, 140)

    # 调整绘图尺寸，使其占满画板
    fig.set_size_inches(8, 4)
    plt.savefig('time_series_plot.svg', format='svg',bbox_inches='tight')

    plt.show()




    model = CNN_Trend()
    model_LSTM = LSTM_Trend()


    data = torch.randn(16, 96, 1)
    model(data)
    model_LSTM(data)

    import onnx
    import onnx.utils
    import onnx.version_converter

    torch.onnx.export(
        model_LSTM,
        data,
        'lstm_trend.onnx',
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
        export_params=True,
        opset_version=14,
    )

    torch.onnx.export(
        model,
        data,
        'cnn_trend.onnx',
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
        export_params=True,
        opset_version=14,
    )






    model = CNN_RUL()
    model_LSTM = LSTM_RUL()
    # model_trans = Transformer_RUL(d_model=128, num_layers=2, dff=256, heads=4, seq_len=30, FD_feature=14,
    #                                             dropout=0.5)

    data = torch.randn(16, 30, 14)

    import onnx
    import onnx.utils
    import onnx.version_converter

    output = model(data)
    output_LSRTM = model_LSTM(data)
    # output_trans = model_trans(data)

    torch.onnx.export(
        model,
        data,
        'cnn_rul.onnx',
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
        export_params=True,
        opset_version=14,
    )
    torch.onnx.export(
        model_LSTM,
        data,
        'lstm_rul.onnx',
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
        export_params=True,
        opset_version=14,
    )
    # torch.onnx.export(
    #     model_trans,
    #     data,
    #     'transformer_rul.onnx',
    #     operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
    #     export_params=True,
    #     opset_version=14,
    # )