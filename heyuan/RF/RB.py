from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import math
import sklearn
from sklearn import metrics
import pickle
import sys
from sklearn.model_selection import train_test_split

# 未传参时设置参数默认值      para1~4 n_estimators/max_depth/max_features/min_samples_leaf
para1 = sys.argv[1] if len(sys.argv) == 10 else 300
para2 = sys.argv[2] if len(sys.argv) == 10 else 25
para3 = sys.argv[3] if len(sys.argv) == 10 else 'auto'
para4 = sys.argv[4] if len(sys.argv) == 10 else 3
mode = sys.argv[5] if len(sys.argv) == 10 else "train"  # 模式选择  train/save/predict
csvFile = "D:\\mk\\real\\" + sys.argv[6] if len(sys.argv) == 10 else r"C:\Users\86132\Desktop\凯斯西储.csv"  # 训练或测试文件
ID = sys.argv[7] if len(sys.argv) == 10 else '1'  # 模型选择  共有3组 ID=1/2/3对应model/model2/model3
start = float(sys.argv[8]) if len(sys.argv) == 10 else 0  # 数据分割起始点
stop = float(sys.argv[9]) if len(sys.argv) == 10 else 1  # 数据分割结束点

if ID == '1':
    trainModel = r'C:\Users\86132\Desktop\svmModel\RBCache.pickle'
    saveModel = r'C:\Users\86132\Desktop\svmModel\RB.pickle'
elif ID == '2':
    trainModel = r'C:\Users\86132\Desktop\svmModel\RBCache2.pickle'
    saveModel = r'C:\Users\86132\Desktop\svmModel\RB2.pickle'
elif ID == '3':
    trainModel = r'C:\Users\86132\Desktop\svmModel\RBCache3.pickle'
    saveModel = r'C:\Users\86132\Desktop\svmModel\RB3.pickle'

frep = 300
predictor = RandomForestClassifier(n_estimators=int(para1), max_depth=int(para2), max_features=para3,
                                   min_samples_leaf=int(para4))
target = []
data = []


# 当 mode = "train"时
def trainRB():
    df = pd.read_csv(csvFile, skiprows=0, usecols=[1, 2, 3, 4])
    df = np.array(df)
    sampleNumber = math.floor(df.shape[0] / frep)

    for i in range(sampleNumber):
        for j in range(4):
            target.append(j)
    for i in range(sampleNumber):
        data.append(df[i * frep:(i + 1) * frep, 0])
        data.append(df[i * frep:(i + 1) * frep, 1])
        data.append(df[i * frep:(i + 1) * frep, 2])
        data.append(df[i * frep:(i + 1) * frep, 3])

    datastart = math.floor(len(data) * start)
    datastop = math.floor(len(data) * stop)

    xTrain, xTest, yTrain, yTest = train_test_split(data[datastart:datastop], target[datastart:datastop], test_size=0.2,random_state=1)
    predictor.fit(xTrain, yTrain)
    # 预测结果
    # 准确率估计
    # accurancy = np.sum(np.equal(result, yTrain)) / len(yTrain)
    print(predictor.score(xTrain, yTrain))  # 训练集准确率

    result = predictor.predict(xTrain)
    testResult = predictor.predict(xTest)
    trueCount = 0
    faultCount = 0
    truePreCount = 0
    faultPreCount = 0
    for i in range(len(testResult)):
        if yTest[i] == 0:
            trueCount += 1
            if testResult[i] == 0:
                truePreCount += 1
        else:
            faultCount += 1
            if testResult[i] == yTest[i]:
                faultPreCount += 1
    # onehot编码
    y_true = pd.get_dummies(yTrain).values
    y_pred = pd.get_dummies(result).values
    print(metrics.log_loss(y_true, y_pred))  # 交叉熵损失得分
    print(metrics.mean_squared_error(yTrain, result))  # 均方误差
    print(truePreCount / trueCount)  # 正确类别准确率
    print(faultPreCount / faultCount)  # 故障类别准确率


    with open(trainModel, 'wb') as fw:
        pickle.dump(predictor, fw)


# 当 mode = "save" 时
def saveRB():
    with open(trainModel, 'rb') as fw:
        svmClassfi = pickle.load(fw)
        with open(saveModel, 'wb') as fw1:
            pickle.dump(svmClassfi, fw1)


# 当 mode = "predict" 时
def predict():
    with open(saveModel, 'rb') as fr:
        df = pd.read_csv(csvFile, skiprows=0, usecols=[1])
        df = np.array(df)
        sampleNumber = math.floor(df.shape[0] / frep)
        for i in range(sampleNumber):
            target.append(0)
        for i in range(sampleNumber):
            data.append(df[i * frep:(i + 1) * frep, 0])
        RBClassfi = pickle.load(fr)  # 加载保存的模型
        result = RBClassfi.predict(data)  # 根据数据进行预测
        resultProba = RBClassfi.predict_proba(data)  # 预测概率
        faultCount = np.bincount(np.argmax(resultProba, axis=1), minlength=4)
        resultProba = np.mean(resultProba, axis=0)


        # 打印各个类别的预测数量及其百分比
        for i in range(len(resultProba)):
            print(round(resultProba[i], 4))
        for i in range(len(faultCount)):
            print(faultCount[i])



if mode == "train":
    trainRB()
elif mode == "save":
    saveRB()
elif mode == "predict":
    predict()

# predict()
