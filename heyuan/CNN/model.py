import torch
from torch import nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.cnn1 = nn.Conv1d()
        self.attenion = Attention3dBlock()
        self.linear = nn.Sequential(
            nn.Linear(in_features=1500, out_features=50),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=50, out_features=10),
            nn.ReLU(inplace=True)
        )
        self.handcrafted = nn.Sequential(
            nn.Linear(in_features=34, out_features=10),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2)
        )

        self.output = nn.Sequential(
            nn.Linear(in_features=20, out_features=1)
        )

    def forward(self, inputs, handcrafted_feature):
        y = self.handcrafted(handcrafted_feature)
        x, (hn, cn) = self.lstm(inputs) # [batch seq_len out_dim]
        x = self.attenion(x)  # [batch seq_len out_dim]
        # flatten
        x = x.reshape(-1, 1500)
        x = self.linear(x)  # [batch 10]
        out = torch.cat((x, y), dim=1)  # for torch.__version__ > 1.5.1 use torch.concat
        out = self.output(out)
        return out



