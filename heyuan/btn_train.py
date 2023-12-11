# -*- coding: utf-8 -*-
import sys
from threading import Thread
from time import sleep

from PyQt5.QtWidgets import QMainWindow, QTextEdit, QAction, QFileDialog, QApplication, QPushButton, QWidget, \
    QGridLayout
from PyQt5.QtGui import QIcon





class MainUi(QMainWindow):

    def __init__(self):
        super().__init__()

        self.initUI()



    def initUI(self):
        self.setFixedSize(960, 700)
        self.main_widget = QWidget()  # 创建窗口主部件
        self.main_layout = QGridLayout()  # 创建主部件的网格布局
        self.main_widget.setLayout(self.main_layout)  # 设置窗口主部件布局为网格布局

        self.left_widget = QWidget()  # 创建左侧部件
        self.left_widget.setObjectName('left_widget')
        self.left_layout = QGridLayout()  # 创建左侧部件的网格布局
        self.left_widget.setLayout(self.left_layout)  # 设置左侧部件布局为网格布局

        self.right_widget = QWidget()  # 创建右侧部件
        self.right_widget.setObjectName('right_widget')
        self.right_layout = QGridLayout()  # 创建右侧部件的网格布局
        self.right_widget.setLayout(self.right_layout)  # 设置右侧部件布局为网格布局

        self.main_layout.addWidget(self.left_widget, 0, 0, 12, 2)
        self.main_layout.addWidget(self.right_widget, 0, 2, 12, 10)
        self.setCentralWidget(self.main_widget)  # 设置窗口主部件

        self.button1 = QPushButton(QIcon(''), '训练')  # 创建按钮
        self.button2 = QPushButton(QIcon(''), '保存文件')  # 创建按钮
        self.button3 = QPushButton(QIcon(''), '加载文件')  # 创建按钮
        self.left_layout.addWidget(self.button1, 2, 0, 1, 3)
        self.left_layout.addWidget(self.button2, 3, 0, 1, 4)
        self.left_layout.addWidget(self.button3, 4, 0, 1, 5)
        self.textEdit = QTextEdit()  # 创建文本框用于显示
        self.right_layout.addWidget(self.textEdit, 0, 0, 4, 8)
        # self.button1.clicked.connect(self.showDialog1)





    # 定义打开文件夹目录的函数
    def showDialog1(self):
        QFileDialog.getOpenFileName()
        fname = QFileDialog.getOpenFileName(self, 'Open file', '.')
        self.textEdit.setText(fname[0])

        if fname[0]:
            f = open(fname[0], 'r',encoding='utf-8')
            with f:
                data = f.read()

from PyQt5 import QtCore
from PyQt5.QtCore import QThread






from sklearn import svm
from sklearn import datasets
import pickle
global model


def train(textedit):

    global model
    model = svm.SVC()
    textedit.setText("开始训练")
    textedit.append("训练完毕")

def save(textedit):

    # 模型的保存
    global model
    with open('./saved_models/clf.pickle', 'wb') as f:
        pickle.dump(model, f)  # 将训练好的模型clf存储在变量f中，且保存到本地
    textedit.append("保存文件成功")

def load(textedit):
    # 模型的重新加载和使用
    from sklearn import svm
    from sklearn import datasets
    import pickle
    global model
    try:
        with open('./saved_models/clf.pickle', 'rb') as f:
            model = pickle.load(f)  # 将模型存储在变量clf_load中
            textedit.append("加载文件成功")
    except:
        textedit.append("文件加载失败")

def main():

    app = QApplication(sys.argv)
    gui = MainUi()
    gui.button1.clicked.connect(lambda :train(gui.textEdit))
    gui.button2.clicked.connect(lambda :save(gui.textEdit))
    gui.button3.clicked.connect(lambda :load(gui.textEdit))


    gui.show()
    app.exec_()



if __name__ == '__main__':
    main()
