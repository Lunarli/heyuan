import  torch.nn as nn
import torch


class CNN_RUL(nn.Module):

    def __init__(self,sequence_length = 30,FD_feature_columns = 14):
        super(CNN_RUL, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=(10,1), stride=1, padding=(5,0)),
            nn.BatchNorm2d(10),
            nn.Tanh()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 10, kernel_size=(10,1), stride=1, padding=(5, 0)),
            nn.BatchNorm2d(10),
            nn.Tanh()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(10, 10, kernel_size=(10,1), stride=1, padding=(5, 0)),
            nn.BatchNorm2d(10),
            nn.Tanh()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(10, 10, kernel_size=(10,1), stride=1, padding=(5, 0)),
            nn.BatchNorm2d(10),
            nn.Tanh()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(10, 1, kernel_size=(3,1), stride=1, padding=(1, 0)),
            nn.BatchNorm2d(1),
            nn.Tanh()
        )

        self.dropout = nn.Dropout(p=0.5)
        self.Dense1 = nn.Sequential(
            nn.Linear(in_features=540, out_features=100),
            nn.Tanh()
        )

        self.Dense2 = nn.Sequential(
            nn.Linear(100,1),
            nn.ReLU()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight,1)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight,1)



    def forward(self, x):

        #   batch C H W
        x = x.reshape(-1, 1, x.size(2), x.size(1))

        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)


        conv5 = conv5.reshape(conv5.size(0),-1)

        self.dropout(conv5)

        dense1 = self.Dense1(conv5)
        dense2 = self.Dense2(dense1)


        return dense2

if __name__ == "__main__":



    model  = CNN_RUL()
    data  = torch.randn(512,30,14)

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




    print(list(model.parameters()))
    print()