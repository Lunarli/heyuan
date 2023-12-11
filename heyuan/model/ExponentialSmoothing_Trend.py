import numpy as np
import torch
import torch.nn as nn
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np

class ExponentialSmoothing_Trend(nn.Module):
    def __init__(self):
        super(ExponentialSmoothing_Trend, self).__init__()

        # 创建 ARIMA 模型
        self.ES_model = None





    def forward(self, x_train,trend = 'add',seasonal = None,seasonal_periods = None):

        self.ES_model = ExponentialSmoothing(np.array(x_train),  trend= trend,seasonal = seasonal,seasonal_periods = seasonal_periods).fit()
        # 预测未来的24个值
        forecast = self.ES_model.forecast(steps=24)




        return forecast


if __name__ == '__main__':
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    file_path = r'F:\python demo\informer\data\ETT\ETTh1.csv'
    df_raw = pd.read_csv(file_path)
    scale = StandardScaler()  # 标准化
    train_data = df_raw.iloc[:, -1]  # 原始数据集有多列 取油温列进行拟合预测
    normalized_data = scale.fit_transform(train_data.values.reshape(-1, 1))

    # 选择最后120个点 前96的个用于训练 24个点用于预测
    train_x = normalized_data[-120:-24]
    train_y = normalized_data[-24:]

    ExponentialSmoothing_model = ExponentialSmoothing_Trend()
    ExponentialSmoothing_model(train_x, trend='add', seasonal='add',
                               seasonal_periods=None)