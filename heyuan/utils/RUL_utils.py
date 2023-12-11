
def train_epoch_RUL(train_loader ,model ,criterion ,optimizer ,):

    model.train()
    epoch_loss = 0
    for i, (input, target ) in enumerate(train_loader):
        output = model(input)
        loss = criterion(output ,target)
        epoch_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return  epoch_loss


# 绘制支持向量机三维可视化结果图
def drwa3d():
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.svm import SVR

    # 生成训练样本
    X = np.random.rand(100, 2)  # 特征矩阵，维度为2
    y = np.sin(X[:, 0] * 2 * np.pi)  # 目标值

    # 训练SVR模型
    svr = SVR(kernel='rbf')  # 使用RBF核函数
    svr.fit(X, y)

    # 生成测试数据
    x1 = np.linspace(0, 1, 100)
    x2 = np.linspace(0, 1, 100)
    X_test = np.array(np.meshgrid(x1, x2)).T.reshape(-1, 2)

    # 预测输出
    y_pred = svr.predict(X)

    # 绘制可视化图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], y, c='b', label='Training data')
    ax.scatter(X[:, 0], X[:, 1], y_pred, c='r', label='Predicted output')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Output')
    ax.legend()
    plt.show()

