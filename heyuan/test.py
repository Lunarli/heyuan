import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# 定义数组数据
data = np.array([0.25421766, 0.24858746, 0.24437478, 0.24297722, 0.24579232, 0.24858746,
                 0.24437478, 0.24157966, 0.23736698, 0.23455188, 0.19382274, 0.22191387,
                 0.17836964, 0.2148861, 0.20927586, 0.21628366, 0.15449118, 0.15449118,
                 0.16431409, 0.17415697, 0.20083056, 0.16854673, 0.17415697, 0.19803542,
                 0.19943298, 0.20083056, 0.19803542, 0.19803542, 0.18819255, 0.18961007,
                 0.18961007, 0.19661789, 0.19943298, 0.20646076, 0.2148861, 0.21067343,
                 0.20927586, 0.22049634, 0.23173678, 0.23594945, 0.23736698, 0.23455188,
                 0.24437478, 0.23736698, 0.24437478, 0.23736698, 0.24718988, 0.24157966,
                 0.23455188, 0.22892168, 0.22752411, 0.22470901, 0.21909877, 0.21348853,
                 0.20927586, 0.21207099, 0.22049634, 0.21909877, 0.2303392, 0.2303392,
                 0.250005, 0.24016211, 0.25140256, 0.26122546, 0.26685568, 0.26543814,
                 0.26685568, 0.26825324, 0.27106833, 0.26543814, 0.26404058, 0.262643,
                 0.25843034, 0.24858746, 0.24858746, 0.25140256, 0.25561522, 0.26122546,
                 0.27246592, 0.27246592, 0.25140256, 0.25421766, 0.2598279, 0.27246592,
                 0.30055703, 0.3103999, 0.30476971, 0.29494679, 0.28652145, 0.29352925,
                 0.30618722, 0.31319505, 0.32864814, 0.3103999, 0.30476971, 0.29213169,
                 0.28089125, 0.28089125, 0.27246592, 0.2696508, 0.27528101, 0.27388343,
                 0.27667858, 0.27246592, 0.2696508, 0.26685568, 0.26685568, 0.27388343,
                 0.2696508, 0.262643, 0.2598279, 0.26543814, 0.26543814, 0.2696508,
                 0.28510391, 0.29915947, 0.30195459, 0.28652145, 0.27667858, 0.27246592])

# 将数据分为训练集和测试集
train_data = data[:96]
test_data = data[96:]

# 设置搜索范围
p_values = range(0, 10)
d_values = range(0, 2)
q_values = range(0, 10)
best_aic = float("inf")
best_order = None

# 遍历参数组合，选择AIC最小的模型
for p in p_values:
    for d in d_values:
        for q in q_values:
            try:
                model = sm.tsa.ARIMA(train_data, order=(p, d, q))
                model_fit = model.fit(disp=False)
                aic = model_fit.aic
                if aic < best_aic:
                    best_aic = aic
                    best_order = (p, d, q)

                # 创建画板和子图
                fig, axs = plt.subplots(1, 2, figsize=(12, 5))

                # 左边的子图：预测结果与真实值
                axs[0].plot(model_fit.forecast(24)[0], label='predict')
                axs[0].plot(test_data, label='real')
                axs[0].set_title('预测结果与真实值')
                axs[0].legend()

                # 右边的子图：拟合结果与真实值
                axs[1].plot(model_fit.fittedvalues, label='fitted_value')
                axs[1].plot(train_data.ravel(), label='real_value')
                axs[1].set_title('拟合结果与真实值')
                axs[1].legend(loc=1)

                # 显示图形
                plt.tight_layout()
                plt.show()

                print(p,d,q)


            except:
                continue

# 拟合最优模型并预测后续24个点
model = sm.tsa.ARIMA(train_data, order=best_order)
model_fit = model.fit(disp=False)
predictions = model_fit.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1)
predict = model_fit.forecast(24)
# 计算预测误差
errors = test_data - predictions

plt.plot(predictions,label = 'predict')
plt.plot(test_data,label = 'real')
plt.legend()
plt.show()

# 绘制预测值与真实值之间的误差图
plt.plot(errors)
plt.xlabel('Time')
plt.ylabel('Prediction Error')
plt.title('ARIMA Model - Prediction Errors')
plt.show()
print()