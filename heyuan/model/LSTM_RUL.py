import torch
import torch.nn as nn
from torch.autograd import Variable


class LSTM_RUL(nn.Module):
    """LSTM architecture"""

    def __init__(self, input_size = 14, hidden_size = 50, num_layers = 2, seq_length=30):
        super(LSTM_RUL, self).__init__()
        self.input_size = input_size  # input size 即尺度
        self.hidden_size = hidden_size  # hidden state
        self.num_layers = num_layers  # number of layers
        self.seq_length = seq_length  # sequence length

        self.lstm = nn.LSTM(input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                            dropout=0.1)
        self.fc_1 = nn.Linear(hidden_size, 16)  # fully connected 1
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
        ###gpu配置
        hn = hn.cpu()
        hn_o = torch.Tensor(hn.detach().numpy()[-1, :, :])
        hn_1 = torch.Tensor(hn.detach().numpy()[1, :, :])
        # hn_o = torch.Tensor(hn.detach().numpy()[-1, :, :])
        # hn_1 = torch.Tensor(hn.detach().numpy()[1, :, :])
        hn_o = hn_o.view(-1, self.hidden_size)
        hn_1 = hn_1.view(-1, self.hidden_size)

        out = self.relu(self.fc_1(self.relu(hn_o + hn_1)))
        out = self.relu(self.fc_2(out))
        out = self.dropout(out)
        out = self.fc(out)
        return out
