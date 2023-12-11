import numpy as np
import pandas as pd
import seaborn as sns
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import QuantileTransformer , PowerTransformer
import warnings

from matplotlib.pylab import  mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']   #显示中文
mpl.rcParams['axes.unicode_minus']=False       #显示负号
warnings.filterwarnings('ignore')

# get file names
train_file = "train_FD001.txt"
test_file = "test_FD001.txt"
RUL_file = "RUL_FD001.txt"

class ARIMA():
    def __init__(self):
        # load file and set columns name
        df = pd.read_csv(train_file, sep=" ", header=None)
        df.drop(columns=[26, 27], inplace=True)
        columns = ["Section-{}".format(i) for i in range(26)]
        df.columns = columns

        # get columns data and plot
        raw_data = df.groupby('Section-0').get_group(1)['Section-6']

        minmaxscaler = MinMaxScaler()
        raw_data_norm = minmaxscaler.fit_transform(np.array(raw_data).reshape(-1, 1))

        fig, ax = plt.subplots()
        plt.plot(raw_data, label='real_value')
        plt.title('原始数据')
        plt.legend(loc=1)
        # plt.savefig(fname="原始数据.svg",format="svg")
        fig, ax = plt.subplots()
        plt.plot(raw_data.diff(1).dropna(), label='diff1_value')
        plt.title('一阶差分数据')
        plt.legend(loc=1)
        # plt.savefig(fname="一阶原始数据.svg",format="svg")
        plt.show()

        # stationnary check
        from statsmodels.tsa.stattools import adfuller
        print(adfuller(raw_data))
        print(adfuller(raw_data.diff(1).dropna()))
        # (-5.2350403606036302,     adt检验结果 t统计量的值  若小于99%置信区间下的临界值证明平稳
        #  7.4536580061930903e-06,             t统计量的p值
        #  0,                       计算过程使用的延迟阶数
        #  60,                      用于ADF回归和计算的观测值的个数
        #  {'1%': -3.5443688564814813, '5%': -2.9110731481481484, '10%': -2.5931902777777776},  在99%，95%，90%置信区间下的临界的ADF检验的值
        #  1935.4779504450603)



        #  plot acf and pacf   绘制平稳时间序列的自相关图和偏自相关图
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
        plot_acf(raw_data.diff(1).dropna())
        plot_pacf(raw_data.diff(1).dropna())

        # get p q according to the aic bic
        from statsmodels.tsa.arima.model import ARIMA
        import itertools
        # 设定参数搜索范围
        pmax = 5
        qmax = 5
        d = 1

        model_eval = pd.DataFrame(columns=['model', 'aic', 'bic'])
        for p, q in itertools.product(range(0, pmax), range(0, qmax)):
            model = ARIMA(raw_data, order=(p, d, q)).fit()
            param = 'ARIMA({0},{1},{2})'.format(p, d, q)
            AIC = model.aic
            BIC = model.bic
            print([param, AIC, BIC])
            model_eval = model_eval.append(pd.DataFrame([[param, AIC, BIC]]))
            model_eval = pd.concat([model_eval,pd.DataFrame([[param, AIC, BIC]])])
        # 排序，选取BIC最小值的参数组合
        model_eval.sort_values(by='bic')
        print(model_eval)

        # model evaluation
        model = ARIMA(raw_data.diff(1).dropna(), order=(0, 1, 1)).fit()
        model.plot_diagnostics(figsize=(16, 12))
        plt.show()

        # model prediction
        ahead = 10  # 向前预测多少个值

        pred_value = model.predict(1, len(raw_data.diff(1).dropna()) + ahead)  # 预测的第一个数据是0

        print(len(pred_value))
        print(len(raw_data.diff(1).dropna()))

        # 差分还原
        def inv_diff(diff_df, first_value, add_first=True):
            diff_df.reset_index(drop=True, inplace=True)
            diff_df.index = diff_df.index + 1
            diff_df = pd.DataFrame(diff_df)
            diff_df = diff_df.cumsum()
            df = diff_df + first_value
            if add_first:
                df.loc[0] = first_value
                df.sort_index(inplace=True)
            return df

        # pred_value = inv_diff(pred_value,raw_data.values[0])
        diff = raw_data.diff(1).dropna() - pred_value.values[:191]
        pred_value = np.array(inv_diff(raw_data.diff(1).dropna(), raw_data.values[0])[1:]).reshape(-1) + np.array(diff)

        plt.figure(figsize=(25, 8))
        plt.plot(diff, label='residual')
        plt.title('预测误差')
        # plt.plot(diff,label='diff')
        plt.legend()
        # plt.savefig(fname="预测误差.svg",format="svg")
        plt.show()

        # 画出拟合图形
        plt.figure(figsize=(25, 8))
        plt.rcParams['font.family'] = ['SimHei']
        plt.plot(raw_data[:191], label='real_value')
        plt.plot(pred_value, label='pred_value')
        plt.title('ARIMA趋势预测')
        # plt.plot(diff,label='diff')
        plt.legend(loc=1)
        # plt.savefig(fname="ARIMA趋势预测.svg",format="svg")
        plt.show()

a= ARIMA()
