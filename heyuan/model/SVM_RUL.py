from torch import  nn
import numpy as np
from sklearn.svm import SVR
from  sklearn import  metrics
import sys
import math

from heyuan.utils.data_loader import CMPDataset
from heyuan.utils.model_saved import *
import matplotlib.pyplot as plt



class SVM_RUL(nn.Module):
    def __init__(self, seq_len, fea_dim,verbose=False):
        super(SVM_RUL, self).__init__()

        # 创建 SVM 模型
        self.svm_model = SVR(kernel='linear',verbose=verbose)

        # 参数初始化 seq_length 序列长度 fea_dim特征维度
        self.seq_len, self.fea_dim = seq_len, fea_dim


    def forward(self, x_train, y_train, x_test = None):

        if x_test != None:
            self.svm_model.fit(np.reshape(x_train,[-1,self.seq_len*self.fea_dim]), y_train)
            # 将输入数据转换为2D数组
            n_samples = x_test.shape[0]
            x_test = x_test.reshape(n_samples, self.seq_len * self.fea_dim)

            # 使用随机森林模型进行预测
            predicted_target_2d = self.svm_model.predict(x_test)

            # 将预测结果转换tensor
            predicted_target = predicted_target_2d

            return predicted_target
        else:
            self.svm_model.fit(x_train, y_train)

    # 返回预测结果
    def forecast(self, x):
        return self.svm_model.predict(x)



def score_func(Y_test, Y_pred):
    s = 0
    for i in range(len(Y_pred)):
        if Y_pred[i] > Y_test[i]:
            # s = s + math.exp(args.max_rul*(Y_pred[i] - Y_test[i]) / 10) - 1
            s = s + math.exp(125 * (Y_pred[i] - Y_test[i]) / 10) - 1
        else:
            # s = s + math.exp(args.max_rul*(Y_test[i] - Y_pred[i]) / 13) - 1
            s = s + math.exp(125 * (Y_test[i] - Y_pred[i]) / 13) - 1

    return s


def rmse(Y_test, Y_pred):


    return np.sqrt(np.mean(125*125*(Y_pred - Y_test.ravel()) ** 2))

if __name__ == '__main__':

    # 获取待预测数据  此处为C-MAPSS据集 FD001
    fd_number = '1'
    data = CMPDataset('../data', 'FD00' + str(fd_number), 125, 30)



    # 初始化支持向量机模型
    SVM_model = SVM_RUL(seq_len = 30, fea_dim = 14,verbose = True)

    # 加载网格搜索类
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import make_scorer

    # 定义参数网格
    param_grid = {
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        'degree': [2, 3, 4],
        'epsilon': [0.1, 0.2, 0.3],
        'coef0': [-1, 0, 1]
    }

    # 定义多个评价指标
    scoring = {
        'score': make_scorer(score_func),
        'rmse': make_scorer(rmse),
    }

    # 创建网格搜索对象
    grid_search = GridSearchCV(estimator=SVM_model.svm_model, param_grid=param_grid, scoring=scoring, verbose=2, cv=5,refit='rmse')

    # 拟合数据
    grid_search.fit(data.train_normalized_machine.values,data.train_y_machine.values.ravel())


    # 打印拟合参数
    for i in range(len(grid_search.cv_results_['params'])):
        params = grid_search.cv_results_['params'][i]
        mean_score = grid_search.cv_results_['mean_test_rmse'][i]
        std_score = grid_search.cv_results_['mean_test_score'][i]
        print(f"参数: {params}, RMSE损失: {mean_score:.4f}  Score得分： {std_score:.4f}")

    print()

    # 保存拟合好的模型
    model_saved('SVR', SVM_model.svm_model)
    # 加载拟合好的模型
    # SVM_model.svm_model = model_loaded('../saved_models/SVR.pkl','SVR')

    # 获取最优模型
    SVM_model.svm_model = grid_search.best_estimator_

    # 加载测试集数据
    start_index = 0

    # 存放全部数据
    test_y = []
    test_x = []
    for i in range(len(data.test_rul)):  # for every engine
        end_index = start_index + data.test_rul.loc[i, 'max']
        test_y.append(data.test_y_machine.iloc[start_index:end_index].values)
        test_x.append(data.train_normalized_machine.iloc[start_index:end_index].values)
        start_index = end_index

    # 存放最后一个样本点 用于计算测试损失
    last_test_x = [arr[-1] for arr in test_x]
    last_test_y = [arr[-1] for arr in test_y]



    # # 计算测试集损失
    predict = SVM_model.forecast(np.array(last_test_x))



    RMSE = np.sqrt(np.mean(125 * 125 * (predict.ravel() - np.array(last_test_y).ravel()) ** 2))
    Score = score_func(data.test_y.ravel(), predict)

    plt.plot(SVM_model.forecast(np.array(data.train_normalized_machine[:192])))
    plt.plot(data.train_y_machine[:192])
    plt.show()


    # 绘制测试集寿命预测结果图
    predict_array = SVM_model.forecast(test_x[3])
    real_value = test_y[3].ravel()

    plt.plot(predict_array)
    plt.plot(real_value)
    plt.show()

    # 模型可视化
