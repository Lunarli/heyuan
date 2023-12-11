import time
from threading import Thread
import os

import numpy as np
import torch.utils.data
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QMessageBox, QApplication
from PyQt5.QtWidgets import QFileDialog
from utils_backup.data_loader import CMPDataset
from utils_backup.data_loader_Trend import ETTDataset
from model import  *
import re
import torch.nn as nn
from torch import optim as optim
from PyQt5 import QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


class CollapsibleBox(QtWidgets.QWidget):
    def __init__(self, title="", parent=None):
        super(CollapsibleBox, self).__init__(parent)

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
        lay = self.content_area.layout()
        del lay
        self.content_area.setLayout(layout)

        text_edit = QtWidgets.QTextEdit()
        text_edit.setPlainText("This is a QTextEdit widget.")
        layout.addWidget(text_edit)

        collapsed_height = (
            self.sizeHint().height() - self.content_area.maximumHeight()
        )
        content_height = layout.sizeHint().height()+ text_edit.sizeHint().height()
        for i in range(self.toggle_animation.animationCount()):
            animation = self.toggle_animation.animationAt(i)
            animation.setDuration(500)
            animation.setStartValue(collapsed_height)
            animation.setEndValue(collapsed_height + content_height)

        content_animation = self.toggle_animation.animationAt(
            self.toggle_animation.animationCount() - 1
        )
        content_animation.setDuration(500)
        content_animation.setStartValue(0)
        content_animation.setEndValue(content_height)


class CollapsibleBox2(QtWidgets.QWidget):
    def __init__(self, title="", parent=None):
        super(CollapsibleBox, self).__init__(parent)

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
        lay = self.content_area.layout()
        del lay
        self.content_area.setLayout(layout)

        text_edit = QtWidgets.QTextEdit()
        text_edit.setPlainText("This is a QTextEdit widget.")
        layout.addWidget(text_edit)

        collapsed_height = (
            self.sizeHint().height() - self.content_area.maximumHeight()
        )
        content_height = layout.sizeHint().height()+ text_edit.sizeHint().height()
        for i in range(self.toggle_animation.animationCount()):
            animation = self.toggle_animation.animationAt(i)
            animation.setDuration(500)
            animation.setStartValue(collapsed_height)
            animation.setEndValue(collapsed_height + content_height)

        content_animation = self.toggle_animation.animationAt(
            self.toggle_animation.animationCount() - 1
        )
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

    def __init__(self, title="", parent=None):
        super(Ui_Form, self).__init__(parent)

        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")

        # 设置text
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
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 581, 72))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.textEdit_3 = QtWidgets.QTextEdit(self.scrollAreaWidgetContents)
        self.textEdit_3.setGeometry(QtCore.QRect(20, 10, 104, 87))
        self.textEdit_3.setObjectName("textEdit_3")
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.verticalLayout_2.addWidget(self.scrollArea)
        _translate = QtCore.QCoreApplication.translate
        # Form.setWindowTitle(_translate("Form", "Form"))
        self.pushButton.setText(_translate("Form", "训练文件选择"))
        self.pushButton_3.setText(_translate("Form", "开始训练"))
        self.pushButton_4.setText(_translate("Form", "保存模型"))
        self.pushButton_2.setText(_translate("Form", "测试文件选择"))
        self.pushButton_5.setText(_translate("Form", "开始测试"))
        self.pushButton_6.setText(_translate("Form", "加载模型"))
        self.textEdit_3.setHtml(_translate("Form",
                                           "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                                           "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
                                           "p, li { white-space: pre-wrap; }\n"
                                           "</style></head><body style=\" font-family:\'.AppleSystemUIFont\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
                                           "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'SimSun\'; font-size:9pt;\">阿发我avast</span></p>\n"
                                           "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'SimSun\'; font-size:9pt;\">VAVA是</span></p>\n"
                                           "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'SimSun\'; font-size:9pt;\">阿萨</span></p>\n"
                                           "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'SimSun\'; font-size:9pt;\">阿萨斯</span></p>\n"
                                           "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'SimSun\'; font-size:9pt;\">阿萨斯    </span></p>\n"
                                           "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'SimSun\'; font-size:9pt;\">萨巴斯</span></p>\n"
                                           "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'SimSun\'; font-size:9pt;\">阿发我avast</span></p>\n"
                                           "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'SimSun\'; font-size:9pt;\">VAVA是</span></p>\n"
                                           "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'SimSun\'; font-size:9pt;\">阿萨</span></p>\n"
                                           "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'SimSun\'; font-size:9pt;\">阿萨斯</span></p>\n"
                                           "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'SimSun\'; font-size:9pt;\">阿萨斯    </span></p>\n"
                                           "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'SimSun\'; font-size:9pt;\">萨巴斯</span></p>\n"
                                           "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'SimSun\'; font-size:9pt;\">阿女食杂</span></p>\n"
                                           "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'SimSun\'; font-size:9pt;\">阿发我avast</span></p>\n"
                                           "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'SimSun\'; font-size:9pt;\">VAVA是</span></p>\n"
                                           "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'SimSun\'; font-size:9pt;\">阿萨</span></p>\n"
                                           "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'SimSun\'; font-size:9pt;\">阿萨斯</span></p>\n"
                                           "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'SimSun\'; font-size:9pt;\">阿萨斯    </span></p>\n"
                                           "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'SimSun\'; font-size:9pt;\">萨巴斯</span></p>\n"
                                           "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'SimSun\'; font-size:9pt;\">阿女食杂阿发我avast</span></p>\n"
                                           "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'SimSun\'; font-size:9pt;\">VAVA是</span></p>\n"
                                           "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'SimSun\'; font-size:9pt;\">阿萨</span></p>\n"
                                           "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'SimSun\'; font-size:9pt;\">阿萨斯</span></p>\n"
                                           "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'SimSun\'; font-size:9pt;\">阿萨斯    </span></p>\n"
                                           "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'SimSun\'; font-size:9pt;\">萨巴斯</span></p>\n"
                                           "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'SimSun\'; font-size:9pt;\">阿女食杂</span></p>\n"
                                           "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'SimSun\'; font-size:9pt;\">阿女食杂</span></p>\n"
                                           "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'SimSun\'; font-size:9pt;\">撒</span></p></body></html>"))

        # 训练文件选择按钮被点击
        self.pushButton.clicked.connect(lambda:self.TrainFileSelect(self.textEdit))

        # 测试文件选择按钮被点击
        self.pushButton_2.clicked.connect(lambda:self.TestFileSelect(self.textEdit_2))
        # self.pushButton_3.clicked.connect(lambda:self.TrainClick('train'))

        # 开始训练按钮被点击
        self.pushButton_3.clicked.connect(lambda: self.TrainFunc(self.textEdit_4))

        # 开始训练按钮被点击
        self.pushButton_5.clicked.connect(lambda: self.TestFunc(self.textEdit_4))


        # 保存按钮被点击
        self.pushButton_4.clicked.connect(lambda: self.SaveModel(self.textEdit_4))

    # 训练文件选择函数 传入text组件打印文件信息
    def TrainFileSelect(self,a):
        folder_path = QFileDialog.getOpenFileName(self, '请选择训练文件')
        self.trainFileName = folder_path[0]
        a.setText(self.trainFileName)

        # 训练文件选择函数 传入text组件打印文件信息

    def TestFileSelect(self, a):
        folder_path = QFileDialog.getOpenFileName(self, '请选择测试文件')
        self.TestFileName = folder_path[0]
        a.setText(self.TestFileName)


    def SetText(self,a):
        self.textEdit_4.setText(a)

    def TrainFunc(self,a):
        def train():
            # 加载训练数据

            # 判断模式 是寿命预测还是趋势预测
            mode = 'RUL'   # 'RUL' or 'Prediction'
            algorithm_name = 'Transformer'
            if mode == 'RUL':   # 加载寿命预测数据
                data = CMPDataset(os.path.dirname(self.trainFileName), 'FD00' + re.findall(r'[1-4]', os.path.basename(self.trainFileName))[0], 125, 30, )
                data_loader = torch.utils.data.DataLoader(data,batch_size=16,shuffle=True)

                a.append("文件加载完毕!")
                self.trainData = None

                # 选择相应算法模型
                # 初始化相应算法模型
                model_RUL = None
                criterion = None
                optimizer = None
                if algorithm_name == 'CNN':
                    model_RUL = CNN()
                    criterion = nn.MSELoss()
                    optimizer = optim.Adam(model_RUL.parameters(),lr=1e-3)
                elif algorithm_name == 'LSTM':
                    model_RUL = LSTM()
                    criterion = nn.MSELoss()
                    optimizer = optim.Adam(model_RUL.parameters(), lr=1e-3)
                elif algorithm_name == 'Transformer':
                    model_RUL = Transformer_RUL(d_model=128, num_layers=2, dff=256, heads=4,seq_len=30,FD_feature=14, dropout=0.5)
                    criterion = nn.MSELoss()
                    optimizer = optim.Adam(model_RUL.parameters(), lr=1e-3)
                elif algorithm_name == 'LSTM':
                    pass
                elif algorithm_name == 'LSTM':
                    pass
                # 加载模型开始训练
                a.append("开始训练...")
                for epoch in range(10):
                    # 训练单个epoch
                    model_RUL.train()
                    epoch_loss = 0
                    for i, (input, target) in enumerate(data_loader):
                        if i % 200 == 0:print("batch {}/{}".format(i,len(data_loader)))
                        output = model_RUL(input)
                        loss = criterion(output, target)
                        epoch_loss += loss
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    a.append("epoch {}  loss: {}".format(epoch, epoch_loss))
                a.append("训练完毕!")



            elif mode == 'Prediction':  # 加载趋势预测数据

                data = ETTDataset()  # 传入csv路径

                data_loader = torch.utils.data.DataLoader(data, batch_size=16, shuffle=True)

                a.append("文件加载完毕!")
                self.trainData = None

                # 选择相应算法模型
                # 初始化相应算法模型
                model_RUL = None
                criterion = None
                optimizer = None


                print("Load prediction data")
                # 选择相应算法模型
                if algorithm_name == 'CNN':
                    model_Trend = CNN_Trend()
                    criterion = nn.MSELoss()
                    optimizer = optim.Adam(model_Trend.parameters(), lr=1e-3)
                elif algorithm_name == 'LSTM':
                    model_Trend = LSTM_Trend()
                    criterion = nn.MSELoss()
                    optimizer = optim.Adam(model_Trend.parameters(), lr=1e-3)
                elif algorithm_name == 'Transformer':
                    model_Trend = Transformer_Trend(d_model=128, num_layers=2, dff=256, heads=4,seq_len=96,FD_feature=1, dropout=0.5)
                    criterion = nn.MSELoss()
                    optimizer = optim.Adam(model_Trend.parameters(), lr=1e-3)
                elif algorithm_name == 'LSTM':
                    pass
                elif algorithm_name == 'LSTM':
                    pass

                # 加载模型开始训练
                a.append("开始训练...")
                for epoch in range(10):
                    # 训练单个epoch
                    model_Trend.train()
                    epoch_loss = 0
                    for i, (input, target) in enumerate(data_loader):
                        if i % 200 == 0: print("batch {}/{}".format(i, len(data_loader)))
                        output = model_Trend(input)
                        loss = criterion(output, target)
                        epoch_loss += loss
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    a.append("epoch {}  loss: {}".format(epoch, epoch_loss))

                a.append("训练完毕!")







        a.setText("训练文件加载中...")


        train_Thread = Thread(target=train)
        train_Thread.start()


    def TestFunc(self,a):
        def test():
            # 加载测试数据

            # 判断模式 是寿命预测还是趋势预测
            mode = 'RUL'  # 'RUL' or 'Prediction'
            algorithm_name = 'CNN'
            if mode == 'RUL':  # 加载寿命预测数据
                data = CMPDataset(os.path.dirname(self.TestFileName),
                                  'FD00' + re.findall(r'[1-4]', os.path.basename(self.TestFileName))[0], 125, 30, mode='test')
                data_loader = torch.utils.data.DataLoader(data, batch_size=16, shuffle=True)

                a.append("文件加载完毕!")


                # 选择相应算法模型
                # 初始化相应算法模型
                model_RUL = None
                criterion = None
                optimizer = None
                if algorithm_name == 'CNN':
                    model_RUL = CNN()
                elif algorithm_name == 'LSTM':
                    model_RUL = LSTM()
                elif algorithm_name == 'Transformer':
                    model_RUL = Transformer_RUL(d_model=128, num_layers=2, dff=256, heads=4, seq_len=30, FD_feature=14,
                                                dropout=0.5)
                elif algorithm_name == 'LSTM':
                    pass
                elif algorithm_name == 'LSTM':
                    pass
                # 加载模型开始训练
                a.append("开始测试...")
                model_RUL.eval()
                result_RUL = []
                for i, (input, target) in enumerate(data_loader):
                    if i % 200 == 0: print("batch {}/{}".format(i, len(data_loader)))
                    output = model_RUL(input)
                    result_RUL.extend(np.array(output.detach()).flatten())
                # a.append("epoch {}  loss: {}".format(, output))

                print("hbsasa")



            elif mode == 'Prediction':  # 加载趋势预测数据

                data = ETTDataset()  # 传入csv路径

                data_loader = torch.utils.data.DataLoader(data, batch_size=16, shuffle=True)

                a.append("文件加载完毕!")
                self.trainData = None

                # 选择相应算法模型
                # 初始化相应算法模型
                model_RUL = None
                criterion = None
                optimizer = None

                print("Load prediction data")
                # 选择相应算法模型
                if algorithm_name == 'CNN':
                    model_Trend = CNN_Trend()
                    criterion = nn.MSELoss()
                    optimizer = optim.Adam(model_Trend.parameters(), lr=1e-3)
                elif algorithm_name == 'LSTM':
                    model_Trend = LSTM_Trend()
                    criterion = nn.MSELoss()
                    optimizer = optim.Adam(model_Trend.parameters(), lr=1e-3)
                elif algorithm_name == 'Transformer':
                    model_Trend = Transformer_Trend(d_model=128, num_layers=2, dff=256, heads=4, seq_len=96,
                                                    FD_feature=1, dropout=0.5)
                    criterion = nn.MSELoss()
                    optimizer = optim.Adam(model_Trend.parameters(), lr=1e-3)
                elif algorithm_name == 'LSTM':
                    pass
                elif algorithm_name == 'LSTM':
                    pass

                # 加载模型开始训练
                a.append("开始训练...")
                for epoch in range(10):
                    # 训练单个epoch
                    model_Trend.train()
                    epoch_loss = 0
                    for i, (input, target) in enumerate(data_loader):
                        if i % 200 == 0: print("batch {}/{}".format(i, len(data_loader)))
                        output = model_Trend(input)
                        loss = criterion(output, target)
                        epoch_loss += loss
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    a.append("epoch {}  loss: {}".format(epoch, epoch_loss))

                a.append("训练完毕!")


        a.setText("测试文件加载中...")


        test_Thread = Thread(target=test)
        test_Thread.run()




    # 保存训练模型函数
    def SaveModel(self,a):

        try:
            print("saved model")
            a.append("模型保存成功")
        except:
            # 打印错误信息
            a.append("模型保存失败！")



if __name__ == "__main__":
    import sys
    import random

    app = QtWidgets.QApplication(sys.argv)

    w = QtWidgets.QMainWindow()
    w.setCentralWidget(QtWidgets.QWidget())
    dock = QtWidgets.QDockWidget("Collapsible Demo")
    w.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock)
    scroll = QtWidgets.QScrollArea()
    dock.setWidget(scroll)
    content = QtWidgets.QWidget()
    scroll.setWidget(content)
    scroll.setWidgetResizable(True)

    hlay = QtWidgets.QHBoxLayout(content)
    vlay = QtWidgets.QVBoxLayout()
    vlay2 = QtWidgets.QVBoxLayout()
    hlay.addLayout(vlay)
    hlay.addLayout(vlay2)
    for i in range(9):
        box = CollapsibleBox("故障趋势预测 Header-{}".format(i))
        vlay.addWidget(box)
        lay = QtWidgets.QVBoxLayout()
        name_ = ['cnn','lstm','transformer','svm','forest','graph','ARIMA','xxxx']
        for j in range(8):
            label = QtWidgets.QPushButton("{}".format(name_[j]))
            color = QtGui.QColor(*[random.randint(0, 255) for _ in range(3)])
            label.setStyleSheet(
                "background-color: {}; color : white;".format(color.name())
            )
            # label.setAlignment(QtCore.Qt.AlignCenter)
            lay.addWidget(label)



        box.setContentLayout(lay)


    vlay.addStretch()   # 添加一个伸缩量组件
    # box2 = CollapsibleBox2("故障趋势预测 Header-{}".format(i))

    # vlay2.addWidget(box2)


    ui = Ui_Form()


    vlay2.addWidget(ui)
    vlay2.addStretch()
    w.resize(800, 800)
    w.show()
    sys.exit(app.exec_())










