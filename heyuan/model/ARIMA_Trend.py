import numpy as np
import torch
import torch.nn as nn
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

class ARIMA_Trend(nn.Module):
    def __init__(self):
        super(ARIMA_Trend, self).__init__()

        # 创建 ARIMA 模型
        self.arima_model = None





    def forward(self, x_train):

        self.arima_model = ARIMA(np.array(x_train), order=(2, 1, 2)).fit()
        # 预测未来的24个值
        forecast = self.arima_model.forecast(steps=24)

        return forecast


if __name__ == '__main__':

    model = ARIMA(np.random.randn(96),order=(2, 1, 2))
    data = np.random.randn(96)
    data_y = np.random.randn(24)
    model = ARIMA_Trend()

    predicted = model(data)
    print(model.arima_model.aic)
    print(model.arima_model.summary())
    print(predicted)

