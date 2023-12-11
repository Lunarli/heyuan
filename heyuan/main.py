import time
from threading import Thread
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False       #显示负号
import torch.utils.data
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMessageBox, QApplication
from PyQt5.QtWidgets import QFileDialog ,QMainWindow, QVBoxLayout, QWidget, QApplication
from utils.data_loader import CMPDataset
from utils.data_loader_Trend import ETTDataset
from model import  *
import re
import torch.nn as nn
from torch import optim as optim
from PyQt5 import QtCore
from torch.utils.data import Dataset





class CollapsibleBox(QtWidgets.QWidget):
    def __init__(self, title="", parent=None):
        super(CollapsibleBox, self).__init__(parent)
        self.title = title   # 菜单栏标题 此处为二级标题 寿命预测/趋势预测
        self.toggle_button = QtWidgets.QToolButton(
            text=title, checkable=True, checked=False
        )
        self.toggle_button.setStyleSheet("QToolButton { border: none; }")   # 设置无边框
        self.toggle_button.setToolButtonStyle(   # 设置按钮的样式，文字显示在图标旁边
            QtCore.Qt.ToolButtonTextBesideIcon
        )
        self.toggle_button.setArrowType(QtCore.Qt.RightArrow) # 设置按钮的箭头类型为向右箭头
        self.toggle_button.pressed.connect(self.on_pressed)  # 连接按钮的pressed信号到on_pressed槽函数

        self.toggle_animation = QtCore.QParallelAnimationGroup(self)

        self.content_area = QtWidgets.QScrollArea(  # 创建一个滚动区域，用于容纳折叠框的内容。初始时最大和最小高度都为0，即折叠状态
            maximumHeight=0, minimumHeight=0
        )
        self.content_area.setSizePolicy(    # 设置滚动区域的大小策略，允许在垂直方向上扩展，但高度保持不变
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
        )
        self.content_area.setFrameShape(QtWidgets.QFrame.NoFrame)  # 设置滚动区域的边框形状为无边框

        lay = QtWidgets.QVBoxLayout(self)
        lay.setSpacing(0)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.toggle_button) # 添加按钮
        lay.addWidget(self.content_area)  # 添加目录项

        self.toggle_animation.addAnimation(
            QtCore.QPropertyAnimation(self, b"minimumHeight") # 设置要操作的对象和属性 折叠区域的最小高度
        )
        self.toggle_animation.addAnimation(
            QtCore.QPropertyAnimation(self, b"maximumHeight")  # 设置要操作的对象和属性 折叠区域的最大高度
        )
        self.toggle_animation.addAnimation(
            QtCore.QPropertyAnimation(self.content_area, b"maximumHeight")  # 设置弹出栏菜单的最大高度
        )

    # 点击按钮弹出子菜单槽函数  改变箭头方向和按钮选中状态
    # 设置动画方向
    @QtCore.pyqtSlot()
    def on_pressed(self):
        checked = self.toggle_button.isChecked()
        self.toggle_button.setArrowType(
            QtCore.Qt.DownArrow if not checked else QtCore.Qt.RightArrow
        )
        self.toggle_animation.setDirection(
            QtCore.QAbstractAnimation.Forward
            if not checked
            else QtCore.QAbstractAnimation.Backward
        )
        self.toggle_animation.start()

    def setContentLayout(self, layout):

        self.content_area.setLayout(layout)

        # 点击算法按钮显示当前任务和算法信息
        def buttonClicked():
            sender = self.sender()

            trans_text = (self.title, sender.text())
            self.ui.update_textEdit_4('当前任务为：' + self.title + '   |   ' + '当前算法为：' + sender.text(), trans_text)

        button_names = ["cnn", "lstm", "transformer", "svm", "forest", "graph", "ARIMA", "指数移动平均"]

        # 设置算法样式
        for idx, name in enumerate(button_names):
            button = QtWidgets.QPushButton(name)

            button.setStyleSheet("background-color: {}; color: white;".format(
                QtGui.QColor(*[random.randint(0, 255) for _ in range(3)]).name()))
            setattr(self, f"button_{idx}", button)  # 为每个对象设置一个属性
            button.clicked.connect(lambda: buttonClicked())
            layout.addWidget(button)




        collapsed_height = self.sizeHint().height() - self.content_area.maximumHeight()
        content_height = layout.sizeHint().height()  # 得到推荐高度
        for i in range(self.toggle_animation.animationCount()-1):
            animation = self.toggle_animation.animationAt(i)
            animation.setDuration(500)
            animation.setStartValue(collapsed_height)
            animation.setEndValue(collapsed_height + content_height)

        content_animation = self.toggle_animation.animationAt(self.toggle_animation.animationCount() - 1)
        content_animation.setDuration(500)
        content_animation.setStartValue(0)
        content_animation.setEndValue(content_height)


class CollapsibleBox1(QtWidgets.QWidget):
    def __init__(self, nested_box, nested_box1, title="", parent=None):
        super(CollapsibleBox1, self).__init__(parent)

        self.nested_box = nested_box  # 折叠栏1  寿命预测
        self.nested_box1 = nested_box1  # 折叠栏2 趋势预测

        self.title = title # 设置菜单栏标题  故障趋势预测 内嵌两个子折叠栏

        self.toggle_button = QtWidgets.QToolButton(
            text=title, checkable=True, checked=False
        )
        self.toggle_button.setStyleSheet("QToolButton { border: none; }")
        self.toggle_button.setToolButtonStyle(
            QtCore.Qt.ToolButtonTextBesideIcon
        )
        self.toggle_button.setArrowType(QtCore.Qt.RightArrow)
        self.toggle_button.pressed.connect(self.on_pressed)

        self.toggle_animation = QtCore.QParallelAnimationGroup(self)

        self.content_area = QtWidgets.QScrollArea(
            maximumHeight=0, minimumHeight=0
        )
        self.content_area.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
        )
        self.content_area.setFrameShape(QtWidgets.QFrame.NoFrame)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setSpacing(0)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.toggle_button)
        lay.addWidget(self.content_area)

        self.toggle_animation.addAnimation(
            QtCore.QPropertyAnimation(self, b"minimumHeight")
        )
        self.toggle_animation.addAnimation(
            QtCore.QPropertyAnimation(self, b"maximumHeight")
        )
        self.toggle_animation.addAnimation(
            QtCore.QPropertyAnimation(self.content_area, b"maximumHeight")
        )

    @QtCore.pyqtSlot()
    def on_pressed(self):
        checked = self.toggle_button.isChecked()
        self.toggle_button.setArrowType(
            QtCore.Qt.DownArrow if not checked else QtCore.Qt.RightArrow
        )
        self.toggle_animation.setDirection(
            QtCore.QAbstractAnimation.Forward
            if not checked
            else QtCore.QAbstractAnimation.Backward
        )
        self.toggle_animation.start()

    def setContentLayout(self, layout):

        self.content_area.setLayout(layout)

        # Create nested collapsible box
        nested_box = self.nested_box
        nested_box.setContentLayout(QVBoxLayout())
        layout.addWidget(nested_box)

        nested_box1 = self.nested_box1
        nested_box1.setContentLayout(QVBoxLayout())
        layout.addWidget(nested_box1)

        layout.addItem(
            QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding))






        collapsed_height = self.sizeHint().height() - self.content_area.maximumHeight()
        content_height = layout.sizeHint().height()
        for i in range(self.toggle_animation.animationCount()):
            animation = self.toggle_animation.animationAt(i)
            animation.setDuration(500)
            animation.setStartValue(collapsed_height)
            animation.setEndValue(collapsed_height + content_height)

        content_animation = self.toggle_animation.animationAt(self.toggle_animation.animationCount() - 1)
        content_animation.setDuration(500)
        content_animation.setStartValue(0)
        content_animation.setEndValue(content_height)


# 调用matplotlib绘图类
class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)



class Ui_Form(QtWidgets.QWidget):

    def __init__(self):
        super(Ui_Form, self).__init__()

        self.running = True  # 设置运行状态全局变量


        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self)     # 右侧垂直组件
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")      # 左侧垂直组件
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")  #  中心水平组件
        # 设置text
        #


        self.textEdit = QtWidgets.QTextEdit(self)
        self.textEdit.setObjectName("textEdit")

        self.horizontalLayout.addWidget(self.textEdit)

        # 设置button
        self.pushButton = QtWidgets.QPushButton(self)
        self.pushButton.setObjectName("pushButton")
        self.pushButton_3 = QtWidgets.QPushButton(self)
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(self)
        self.pushButton_4.setObjectName("pushButton_4")
        self.horizontalLayout.addWidget(self.pushButton)
        self.horizontalLayout.addWidget(self.pushButton_3)
        self.horizontalLayout.addWidget(self.pushButton_4)
        self.verticalLayout.addLayout(self.horizontalLayout)

        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.textEdit_2 = QtWidgets.QTextEdit(self)
        self.textEdit_2.setObjectName("textEdit_2")
        self.horizontalLayout_2.addWidget(self.textEdit_2)
        self.pushButton_2 = QtWidgets.QPushButton(self)
        self.pushButton_2.setObjectName("pushButton_2")
        self.horizontalLayout_2.addWidget(self.pushButton_2)
        self.pushButton_5 = QtWidgets.QPushButton(self)
        self.pushButton_5.setObjectName("pushButton_5")
        self.horizontalLayout_2.addWidget(self.pushButton_5)
        self.pushButton_6 = QtWidgets.QPushButton(self)
        self.pushButton_6.setObjectName("pushButton_6")
        self.horizontalLayout_2.addWidget(self.pushButton_6)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.verticalLayout_2.addLayout(self.verticalLayout)
        self.textEdit_4 = QtWidgets.QTextEdit(self)
        self.textEdit_4.setObjectName("textEdit_4")
        self.verticalLayout_2.addWidget(self.textEdit_4)
        self.scrollArea = QtWidgets.QScrollArea(self)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")

        self.sc = MplCanvas(self, width=5, height=4, dpi=100)  # 绘图图层
        # Create toolbar, passing canvas as first parament, parent (self, the MainWindow) as second.
        self.toolbar = NavigationToolbar(self.sc,self)   # 绘图窗口菜单栏


        self.plotwidget = QtWidgets.QWidget()
        self.plotwidget.setObjectName("plotwidget")
        plotwidget_layout= QVBoxLayout()
        plotwidget_layout.addWidget(self.toolbar)
        plotwidget_layout.addWidget(self.sc)  # 绘图窗口
        self.plotwidget.setLayout(plotwidget_layout)

        self.scrollArea.setWidget(self.plotwidget)
        self.verticalLayout_2.addWidget(self.scrollArea)
        _translate = QtCore.QCoreApplication.translate
        # Form.setWindowTitle(_translate("Form", "Form"))
        self.pushButton.setText(_translate("Form", "训练文件选择"))
        self.pushButton_3.setText(_translate("Form", "开始训练"))
        self.pushButton_4.setText(_translate("Form", "保存模型"))
        self.pushButton_4.setEnabled(False)  # 保存模型按钮初始不可用
        self.pushButton_2.setText(_translate("Form", "测试文件选择"))
        self.pushButton_5.setText(_translate("Form", "开始测试"))
        self.pushButton_6.setText(_translate("Form", "加载模型"))


        # 训练文件选择按钮被点击
        self.pushButton.clicked.connect(lambda: self.TrainFileSelect(self.textEdit))

        # 测试文件选择按钮被点击
        self.pushButton_2.clicked.connect(lambda: self.TestFileSelect(self.textEdit_2))


        # 开始训练按钮被点击
        self.pushButton_3.clicked.connect(lambda: self.TrainFunc(self.textEdit_4))

        # 开始测试按钮被点击
        self.pushButton_5.clicked.connect(lambda: self.TestFunc(self.textEdit_4))

        # 保存按钮被点击
        self.pushButton_4.clicked.connect(lambda: self.SaveModel(self.textEdit_4))

        # 加载按钮被点击
        self.pushButton_6.clicked.connect(lambda: self.LoadModel(self.textEdit_4))

        self.model = None   # 训练模型初始化为None
        self.model_test = None   # 测试模型初始化为None
        self.mode = None    # 模式初始化为None
        self.algorithm_name = None    # 算法名初始化为None
        self.trainFileName = None     # 训练文件名初始化为None
        self.TestFileName = None      # 测试文件名初始化为None

    # 训练文件选择函数 传入text组件打印文件信息
    def TrainFileSelect(self, a):
        folder_path = QFileDialog.getOpenFileName(self, '请选择训练文件')
        if folder_path[0] is not "":   # 判断是否未选择文件
            self.trainFileName = folder_path[0]
        a.setText(self.trainFileName)

        # 训练文件选择函数 传入text组件打印文件信息

    def TestFileSelect(self, a):
        folder_path = QFileDialog.getOpenFileName(self, '请选择测试文件')
        if folder_path[0] is not "":
            self.TestFileName = folder_path[0]
        a.setText(self.TestFileName)



    def update_textEdit_4(self, button_name, text_array):
        if (text_array[0] == '寿命预测'):
            self.mode = 'RUL'
        else:
            self.mode = 'Prediction'
        self.algorithm_name = text_array[1]
        # box类按钮传参,并修改self.mode
        self.textEdit_4.append(button_name)

    def TrainFunc(self, a):
        def train():
            # 加载训练数据
            self.running = True  # 异步停止判断位
            while self.running:

                mode = self.mode    # 'RUL' or 'Prediction'   判断模式 是寿命预测还是趋势预测
                algorithm_name = self.algorithm_name    #  具体算法名  cnn/lstm
                algorithm_type = " "                    # 算法类型判断  None/Statistical_learning/machine_learning
                try:
                    self.pushButton_3.setText('停止训练')  # 进入训练时将开始训练按钮文字设置为停止训练 用于停止训练
                    if mode == 'RUL' and self.trainFileName is not None:  # 加载寿命预测数据 需要训练文件不为空

                        a.append("训练文件加载中...")
                        data = CMPDataset(os.path.dirname(self.trainFileName),
                                          'FD00' + re.findall(r'[1-4]', os.path.basename(self.trainFileName))[0], 125, 30, )
                        data_loader = torch.utils.data.DataLoader(data, batch_size=16, shuffle=True)

                        a.append("文件加载完毕!")
                        self.trainData = None

                        # 选择相应算法模型
                        # 初始化相应算法模型

                        criterion = None
                        optimizer = None
                        if algorithm_name == 'cnn':
                            self.model = CNN_RUL()
                            criterion = nn.MSELoss()
                            optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
                        elif algorithm_name == 'lstm':
                            self.model = LSTM_RUL()
                            criterion = nn.MSELoss()
                            optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
                        elif algorithm_name == 'transformer':
                            self.model = Transformer_RUL(d_model=128, num_layers=2, dff=256, heads=4, seq_len=30, FD_feature=14,
                                                        dropout=0.5)
                            criterion = nn.MSELoss()
                            optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
                        elif algorithm_name == 'svm':
                            algorithm_type = 'machine_learning'
                            self.model = SVM_RUL(seq_len=30,fea_dim=14)
                            criterion = nn.MSELoss()
                        elif algorithm_name == 'forest':
                            algorithm_type = 'machine_learning'
                            self.model = RF_RUL(seq_len=30, fea_dim=14)
                            criterion = nn.MSELoss()
                        else:
                            a.append("算法暂不支持")
                            break
                        # 加载模型开始训练
                        a.append("开始训练...")

                        if algorithm_type == "machine_learning":  # 机器学习算法训练分支
                            for epoch in range(1):
                                epoch_loss = 0
                                for i, (input, target) in enumerate(data_loader):
                                    input = input[:, :, :].float()
                                    target = target[:, 0].float()
                                    index = np.round(input.shape[0] * 0.5)
                                    index = int(index)
                                    x_train = input[:index, :, :]
                                    x_test = input[index:, :, :]
                                    y_train = target[:index]
                                    y_test = target[index:]
                                    # 输入数据为1*96*1，标签为1*1
                                    y_pred = self.model(x_train, y_train, x_test)
                                    if i % 50 == 0:
                                        a.append("batch loss: {}".format(criterion(y_pred, y_test)))
                                    epoch_loss += criterion(y_pred, y_test)
                                a.append("fit loss: {}".format(epoch_loss))
                        elif algorithm_type == "":
                            pass
                        else:
                            for epoch in range(10):    # 深度学习训练分支
                                # 训练单个epoch
                                if self.running == False:
                                    break
                                self.model.train()
                                epoch_loss = 0
                                for i, (input, target) in enumerate(data_loader):
                                    if self.running:
                                        if i % 200 == 0: print("batch {}/{}".format(i, len(data_loader)))
                                        output = self.model(input)
                                        loss = criterion(output, target)
                                        epoch_loss += loss
                                        optimizer.zero_grad()
                                        loss.backward()
                                        optimizer.step()
                                    else:
                                        a.append("停止训练!")
                                        break
                                if self.running:
                                    a.append("epoch {}  loss: {}".format(epoch, epoch_loss))
                        if self.running:
                            a.append("训练完毕!")




                    elif mode == 'Prediction' and self.trainFileName is not None:  # 加载趋势预测数据
                        a.append("训练文件加载中...")
                        data = ETTDataset()  # 传入csv路径

                        data_loader = torch.utils.data.DataLoader(data, batch_size=16, shuffle=True)

                        a.append("文件加载完毕!")
                        self.trainData = None

                        # 选择相应算法模型
                        # 初始化相应算法模型

                        criterion = None
                        optimizer = None


                        # 选择相应算法模型
                        if algorithm_name == 'cnn':
                            self.model = CNN_Trend()
                            criterion = nn.MSELoss()
                            optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
                        elif algorithm_name == 'lstm':
                            self.model = LSTM_Trend()
                            criterion = nn.MSELoss()
                            optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
                        elif algorithm_name == 'transformer':
                            self.model = Transformer_Trend(d_model=128, num_layers=2, dff=256, heads=4, seq_len=96,
                                                            FD_feature=1, dropout=0.5)
                            criterion = nn.MSELoss()
                            optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
                        elif algorithm_name == 'svm':
                            a.append("算法暂不支持")
                            break
                        elif algorithm_name == 'forest':
                            algorithm_type = 'machine_learning'
                            self.model = RF_Trend(seq_len=96,fea_dim=1)
                            criterion = nn.MSELoss()
                        elif algorithm_name == 'ARIMA':
                            algorithm_type = 'Statistical_learning'
                            self.model = ARIMA_Trend()
                        elif algorithm_name == '指数移动平均':
                            algorithm_type = 'Statistical_learning'
                            self.model = ExponentialSmoothing_Trend()
                        else:
                            a.append("算法暂不支持")
                            break
                        # 加载模型开始训练
                        a.append("开始训练...")
                        self.pushButton_3.setText("停止训练") # 点击训练按钮后显示文字切换到停止训练

                        if algorithm_type == "machine_learning":
                            for epoch in range(5):
                                if self.running == False:
                                    break
                                epoch_loss = 0
                                for i, (input, target) in enumerate(data_loader):
                                    if self.running == False:
                                        a.append("停止训练!")
                                        break
                                    input = input[:, :, :].float()
                                    target = target[:, -24:, :].float()
                                    index = np.round(input.shape[0] * 0.5)
                                    index = int(index)
                                    x_train = input[:index, :, 0]
                                    x_test = input[index:, :, 0]
                                    y_train = target[:index, :, 0]
                                    y_test = target[index:, :, 0]
                                    y_pred = self.model(x_train, y_train, x_test)
                                    if i % 50 == 0:
                                        a.append("batch loss: {}".format(criterion(y_pred.squeeze(), y_test)))
                                    epoch_loss = criterion(y_pred.squeeze(), y_test)
                                a.append("fit loss: {}".format(epoch_loss))
                        elif algorithm_type == "Statistical_learning":    # 统计学习算法趋势预测分支
                            predict_output = self.model(data[random.randint(0, len(data)-1)][0])
                            a.append("拟合完毕!")
                            print(predict_output)
                        else:                                              # 深度学习趋势预测分支
                            for epoch in range(10):
                                if self.running == False:
                                    break
                                # 训练单个epoch
                                self.model.train()
                                epoch_loss = 0
                                for i, (input, target) in enumerate(data_loader):
                                    if self.running == False:
                                        a.append("停止训练!")
                                        break
                                    if i % 200 == 0: # 控制台日志信息打印
                                        print("batch {}/{}".format(i, len(data_loader)))
                                    output = self.model(input)
                                    loss = criterion(output, target)
                                    epoch_loss += loss
                                    optimizer.zero_grad()
                                    loss.backward()
                                    optimizer.step()
                                a.append("epoch {}  loss: {}".format(epoch, epoch_loss))

                        a.append("训练完毕!")
                        self.pushButton_3.setText("开始训练")  # 训练完毕后按钮文字恢复到开始训练

                    else:
                        a.append("算法或训练文件未选择")
                    self.pushButton_3.setText('开始训练')
                    self.pushButton_4.setEnabled(True)  # 训练完毕后可保存模型
                    self.running = False
                except Exception as e:
                    a.append("解析出错 请检查文件 {}".format(e))
                    self.running = False  # 出现错误将running置为false



        if self.pushButton_3.text() == '开始训练':
            # train_Thread.run()   #  调试用
            train_Thread = Thread(target=train)
            train_Thread.start() #  正常运行时开启多线程
            self.running = True
        else:
            self.pushButton_3.setText('开始训练')
            self.running = False

    def TestFunc(self, a):
        def test():
            # 加载测试数据

            # 判断模式 是寿命预测还是趋势预测
            mode = self.mode  # 'RUL' or 'Prediction'
            algorithm_name = self.algorithm_name
            try:
                if mode == 'RUL' and self.TestFileName is not None:  # 加载寿命预测数据
                    a.append("测试文件加载中...")
                    data = CMPDataset(os.path.dirname(self.TestFileName),
                                      'FD00' + re.findall(r'[1-4]', os.path.basename(self.TestFileName))[0], 125, 30,
                                      mode='test')
                    data_loader = torch.utils.data.DataLoader(data, batch_size=16, shuffle=True)

                    a.append("文件加载完毕!")

                    # 选择相应算法模型
                    # 初始化相应算法模型
                    self.model_test = None

                    if algorithm_name == 'cnn':
                        self.model_test = CNN_RUL()
                    elif algorithm_name == 'lstm':
                        self.model_test = LSTM_RUL()
                    elif algorithm_name == 'transformer':
                        self.model_test = Transformer_RUL(d_model=128, num_layers=2, dff=256, heads=4,
                                                    seq_len=30, FD_feature=14,
                                                    dropout=0.5)
                    else:
                        a.append("算法暂不支持")
                    if self.model_test:
                        # 加载模型开始训练
                        a.append("开始测试...")
                        print(mode,algorithm_name)
                        self.model_test.eval()
                        result_RUL = []
                        for i, (input, target) in enumerate(data_loader):
                            if i % 200 == 0: print("batch {}/{}".format(i, len(data_loader)))
                            print(i)
                            output = self.model_test(input)
                            result_RUL.extend(np.array(output.detach()).flatten())
                        # a.append("epoch {}  loss: {}".format(, output))
                        self.sc.axes.cla()  # Clear the canvas.
                        self.sc.axes.plot([i for i in range(len(result_RUL))],result_RUL)
                        self.sc.draw()

                        a.append("测试完毕!")
                        print("plot finish")



                elif mode == 'Prediction' and self.TestFileName is not None:  # 加载趋势预测数据
                    a.append("测试文件加载中...")
                    data = ETTDataset(mode='test')  # 传入csv路径

                    data_loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=True)

                    a.append("文件加载完毕!")
                    self.trainData = None

                    # 选择相应算法模型
                    # 初始化相应算法模型
                    model_Trend = None


                    print("Load prediction data")
                    # 选择相应算法模型
                    if algorithm_name == 'cnn':
                        model_Trend = CNN_Trend()

                    elif algorithm_name == 'lstm':
                        model_Trend = LSTM_Trend()

                    elif algorithm_name == 'transformer':
                        model_Trend = Transformer_Trend(d_model=128, num_layers=2, dff=256, heads=4, seq_len=96,
                                                        FD_feature=1, dropout=0.5)

                    else:
                        a.append("算法暂不支持")

                    if model_Trend:
                        # 加载模型开始训练
                        a.append("开始测试...")

                        model_Trend.eval()
                        result_Trend = []
                        for i, (input, target) in enumerate(data_loader):
                            if i % 200 == 0:  # 控制台日志信息打印
                                print("batch {}/{}".format(i, len(data_loader)))
                            output = model_Trend(input)
                            input_inverse,output_inverse = data.inverse_transform(input.detach().cpu()),data.inverse_transform(output.detach().cpu())
                            result_Trend = result_Trend + list(input_inverse.flatten()) + list(output_inverse.flatten())
                        self.sc.axes.cla()
                        self.sc.axes.plot(result_Trend[:96],linestyle='dashed',label='输入序列')
                        self.sc.axes.plot(np.arange(96,120),result_Trend[96:],label='预测序列')
                        self.sc.axes.legend()
                        self.sc.draw()

                        a.append("测试完毕!")

                else:
                    a.append("算法或测试文件未选择")
            except Exception as e:
                a.append("解析出错 请检查文件 {}".format(e))


        test_Thread = Thread(target=test)
        test_Thread.run()

    # 保存训练模型函数
    def SaveModel(self, a):

        try:
            savedModelName = self.model.__class__.__name__ + '.pth'
            fileSavePath, _ = QFileDialog.getSaveFileName(self, "请选择保存路径",savedModelName)
            if fileSavePath is not "":

                torch.save(self.model.state_dict(), fileSavePath)
                a.append('模型保存成功 ' +  fileSavePath)
        except Exception as e:
            # 打印错误信息
            a.append(f"Error: {e}")
            a.append("模型保存失败！")
            print(f"Error: {e}")

    # 加载训练模型函数
    def LoadModel(self, a):

        try:
            folder_path = QFileDialog.getOpenFileName(self, '请选择训练文件')
            ModelFileName = folder_path[0]
            print(ModelFileName)
            a.append(ModelFileName)
            self.model = torch.load(ModelFileName)
            self.model_test = torch.load(ModelFileName)
            print("load model")
            a.append("模型加载成功")
        except Exception as e:
            # 打印错误信息
            a.append(f"Error: {e}")
            a.append("模型加载失败！")
            print(f"Error: {e}")


if __name__ == "__main__":
    import sys
    import random

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("中国核动力研究设计院")    # 设置软件名
    app.setWindowIcon(QIcon(r"F:\idea_demo\核院\HDL\src\main\resources\static\images\logo.jpg"))  # 设置应用图标

    w = QtWidgets.QMainWindow()

    dock = QtWidgets.QDockWidget("故障趋势预测算法工具箱") # 创建折叠组件
    w.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock)
    scroll = QtWidgets.QScrollArea()
    dock.setWidget(scroll)

    content = QtWidgets.QWidget()
    scroll.setWidget(content)
    scroll.setWidgetResizable(True) # 自动改变窗口大小 通过代码创建的默认值为false

    hlay = QtWidgets.QHBoxLayout(content)  # 设置内容为水平布局  然后添加两个垂直布局   创建水平布局对象
    vlay = QtWidgets.QVBoxLayout()  # 左侧内容
    vlay2 = QtWidgets.QVBoxLayout()  # 右侧内容
    hlay.addLayout(vlay)
    hlay.addLayout(vlay2)

    nested_box = CollapsibleBox(title="寿命预测")
    nested_box1 = CollapsibleBox(title="趋势预测")
    box = CollapsibleBox1(nested_box, nested_box1, "故障趋势预测")  # 一级目录  内嵌两个二级目录
    vlay.addWidget(box)
    lay = QtWidgets.QVBoxLayout()
    box.setContentLayout(lay)

    box1 = CollapsibleBox("健康评估")
    vlay.addWidget(box1)
    lay = QtWidgets.QVBoxLayout()
    box1.setContentLayout(lay)

    box2 = CollapsibleBox("运维决策")
    vlay.addWidget(box2)
    lay = QtWidgets.QVBoxLayout()
    box2.setContentLayout(lay)

    vlay.addStretch()  # 拉伸  设置空白处所占大小 使得开始的三个菜单栏在最开始处
    # box2 = CollapsibleBox2("故障趋势预测 Header-{}".format(i))

    # vlay2.addWidget(box2)

    ui = Ui_Form()
    nested_box.ui = ui  # 动态添加ui成员变量
    nested_box1.ui = ui
    box.ui = ui
    box1.ui = ui
    box2.ui = ui

    vlay2.addWidget(ui)
    vlay2.addStretch()
    w.resize(800, 800)
    w.show()
    sys.exit(app.exec_())














