import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# https://github.com/jiaxiang-cheng/PyTorch-Transformer-for-RUL-Prediction
# Remaining useful life estimation via transformer encoder enhanced by a gated convolutional unit
class Transformer_RUL(nn.Module):
    def __init__(self, d_model, num_layers,dff, heads,seq_len,FD_feature, dropout):
        super().__init__()

        self.encoder = Encoder(d_model, num_layers, heads,dff,seq_len,FD_feature, dropout)  # Encoder
        self.out = nn.Linear(seq_len*d_model, 1)  # Linear

    def forward(self, x):   # batch seq_len fea_dim

        x = x.transpose(2,1).unsqueeze(-1)
        encoder_output = self.encoder(x)
        output = self.out(encoder_output.reshape(encoder_output.size(0),-1))

        return output

class Encoder(nn.Module):
    def __init__(self, d_model, num_layers, heads,dff, seq_len,FD_feature,dropout): #d_model = 128  # dimension in encoder, heads = 4  #number of heads in multi-head attention, N = 2  #encoder layers, m = 14  #number of features
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model

        self.embed = nn.Linear(seq_len*FD_feature, seq_len*d_model)
        self.pe = PositionalEncoder(maximum_position_encoding = 100, d_model = d_model)

        self.enc_layers = nn.ModuleList([EncoderLayer(d_model, heads, dff, dropout)
                           for _ in range(num_layers)])


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


if __name__ == "__main__":
    model_RUL = Transformer_RUL(d_model=128, num_layers=2, dff=256, heads=4, seq_len=30, FD_feature=14, dropout=0.5)

    data = torch.randn(16, 30, 14)

    model_RUL(data)

    import onnx
    import onnx.utils
    import onnx.version_converter

    torch.onnx.export(
        model_RUL,
        data,
        'trans_RUL.onnx',
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
        export_params=True,
        opset_version=14,
    )

    print()
