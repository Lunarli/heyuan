from torch import nn
import torch

from model import LSTM_RUL


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

        return dense2,conv5

if __name__ == "__main__":
    model = CNN_RUL()
    model = LSTM_RUL()
    model = Transformer()
    data = torch.randn(16, 30, 14)

    import onnx
    import onnx.utils
    import onnx.version_converter

    output = model(data)

    torch.onnx.export(
        model,
        data,
        'model.onnx',
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
        export_params=True,
        opset_version=14,
    )
