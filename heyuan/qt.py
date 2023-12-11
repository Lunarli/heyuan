import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QPushButton, QVBoxLayout, QHBoxLayout
import requests

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.initUI()

    def initUI(self):
        self.setGeometry(300, 300, 400, 300)
        self.setWindowTitle('HTTP Request Demo')

        centralWidget = QtWidgets.QWidget()
        self.setCentralWidget(centralWidget)

        vbox = QVBoxLayout()

        self.textEdit = QTextEdit()
        vbox.addWidget(self.textEdit)

        hbox1 = QHBoxLayout()
        self.getText = QTextEdit()
        self.postText = QTextEdit()
        hbox1.addWidget(self.getText)
        hbox1.addWidget(self.postText)
        vbox.addLayout(hbox1)

        hbox2 = QHBoxLayout()
        self.getBtn = QPushButton('GET')
        self.postBtn = QPushButton('POST')
        hbox2.addWidget(self.getBtn)
        hbox2.addWidget(self.postBtn)
        vbox.addLayout(hbox2)

        centralWidget.setLayout(vbox)





        # self.getText = QTextEdit(self)
        # self.getText.move(10, 10)
        # self.getText.resize(180, 240)
        #
        # self.postText = QTextEdit(self)
        # self.postText.move(210, 10)
        # self.postText.resize(180, 240)
        #
        # self.getBtn = QPushButton('GET', self)
        # self.getBtn.move(10, 260)
        self.getBtn.clicked.connect(self.getBtnClicked)
        #
        # self.postBtn = QPushButton('POST', self)
        # self.postBtn.move(210, 260)
        self.postBtn.clicked.connect(self.postBtnClicked)

    def getBtnClicked(self):
        response = requests.get('http://localhost:8000')
        self.getText.append(response.text)

    def postBtnClicked(self):
        data = self.textEdit.toPlainText()
        response = requests.post('http://localhost:8000',data=data)
        self.postText.append(response.text)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())