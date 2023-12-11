import numpy as np
import pandas as pd
from scipy.optimize import minimize
from statsmodels.tsa.arima_model import ARIMA

class ARIMA:
    def __init__(self, p, d, q):
        self.p = p
        self.d = d
        self.q = q

    def fit(self, X):
        self.X = X

        # 预处理
        X = np.array(X)
        X_diff = np.diff(X, self.d)

        # 构建初始参数
        theta = np.ones(self.p + self.q + 1)

        # 定义优化目标函数
        def objective(theta):
            ar_params = np.append(1, -theta[:self.p])
            ma_params = np.append(1, theta[self.p+1:])
            return np.sum(np.square(self.residuals(X_diff, ar_params, ma_params)))

        # 优化参数
        result = minimize(objective, theta, method='BFGS')
        self.params = result.x

    def predict(self, X_test):
        X_test = np.array(X_test)
        n = len(X_test)
        y_hat = np.zeros(n)

        # 构建ARMA模型
        ar_params = np.append(1, -self.params[:self.p])
        ma_params = np.append(1, self.params[self.p+1:])
        model = ARIMA(self.X, order=(self.p, self.d, self.q), freq=None, dates=None,
                      exog=None, validate_specification=True)

        # 预测
        for i in range(n):
            model_fit = model.fit(trend='nc', start=len(self.X), end=len(self.X)+i)
            y_hat[i] = model_fit.forecast()[0]

        return y_hat

    def residuals(self, X_diff, ar_params, ma_params):
        n = len(X_diff)
        eps = np.zeros(n)

        for i in range(self.p, n):
            ar_term = np.dot(ar_params[::-1], X_diff[i-self.p:i])
            ma_term = np.dot(ma_params[::-1], eps[i-self.q:i])
            eps[i] = X_diff[i] - ar_term - ma_term

        return eps[self.q:]
