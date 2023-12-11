import torch
import torch.nn as nn
from torch.autograd import Variable


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


if __name__ == '__main__':

    data = torch.randn(16,96,1)

    model = LSTM_Trend()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    output = model(data)

    print(output)
