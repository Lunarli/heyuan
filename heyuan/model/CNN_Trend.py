import  torch.nn as nn
import  torch

class CNN_Trend(nn.Module):

    def __init__(self,sequence_length = 96,FD_feature_columns = 1):
        super(CNN_Trend, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 5, kernel_size=10, stride=1, padding='same'),
            nn.BatchNorm1d(5),
            nn.Tanh()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(5, 5, kernel_size=10, stride=1, padding='same'),
            nn.BatchNorm1d(5),
            nn.Tanh()
        )

        self.dropout = nn.Dropout(p=0.5)

        self.Dense1 = nn.Sequential(
            nn.Linear(in_features=96 * 5, out_features=24),
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