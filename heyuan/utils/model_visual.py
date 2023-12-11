import matplotlib.pyplot as plt
from torchviz import make_dot
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from model import CNN_RUL
models = CNN_RUL(30,14)
output = models(torch.randn(512,30,14))

# 1、网络结构显示字符串
# 2、网络结构显示图片
# 3、机器学习网络结构展示



class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = MyModel()

print(models.Dense1[0].weight.device)
from torchinfo import summary
summary(models, input_size=(512, 30, 14),col_names=["input_size", "output_size", "num_params", "mult_adds"],device='cpu')
print(models.Dense1[0].weight.device)

import hiddenlayer as h
print(models.Dense1[0].weight.device)
vis_graph = h.build_graph(models, torch.zeros((512, 30, 14)))  # 获取绘制图像的对象
vis_graph.theme = h.graph.THEMES["blue"].copy()  # 指定主题颜色
h.view_graph(vis_graph)
vis_graph.save("model.png")
plt.show()
print(models.Dense1[0].weight.device)




from torchsummary import summary

def modelSummary(model):
    summary(model, (30,14))

modelSummary(models)
print('model')