from PyQt5.QtWidgets import QApplication, QWidget, QTabWidget, QRadioButton, QPushButton, QVBoxLayout
import sys

class MyTabWidget(QTabWidget):
  def __init__(self, parent=None) -> None:
    super().__init__(parent)

  def remove_tab_handler(self):
    '''
    槽函数, 移除索引为0的选项卡
    '''
    super().removeTab(0)


if __name__ == '__main__':
  app = QApplication(sys.argv)

  top_widget = QWidget()

  # 创建3个容器
  tab1_widget = QWidget()
  tab2_widget = QWidget()
  tab3_widget = QWidget()

  # 创建一些子部件
  remove_tab_btn = QPushButton('removeTab')
  page1_radio1 = QRadioButton('A')
  page1_radio2 = QRadioButton('B')
  page2_radio1 = QRadioButton('C')
  page2_radio2 = QRadioButton('D')
  page3_radio1 = QRadioButton('E')
  page3_radio2 = QRadioButton('F')

  # 为3个标签页分别创建布局
  page1_vbox_layout = QVBoxLayout()
  page1_vbox_layout.addWidget(page1_radio1)
  page1_vbox_layout.addWidget(page1_radio2)
  tab1_widget.setLayout(page1_vbox_layout)

  page2_vbox_layout = QVBoxLayout()
  page2_vbox_layout.addWidget(page2_radio1)
  page2_vbox_layout.addWidget(page2_radio2)
  tab2_widget.setLayout(page2_vbox_layout)

  page3_vbox_layout = QVBoxLayout()
  page3_vbox_layout.addWidget(page3_radio1)
  page3_vbox_layout.addWidget(page3_radio2)
  tab3_widget.setLayout(page3_vbox_layout)

  # 创建QTabWidget部件
  my_tabwidget = MyTabWidget()

  # 把容器添加到对应的选项卡之下
  my_tabwidget.addTab(tab1_widget, 'tabA')
  my_tabwidget.addTab(tab2_widget, 'tabB')
  my_tabwidget.addTab(tab3_widget, 'tabC')

  # 顶层窗口的布局
  top_vbox_layout = QVBoxLayout()
  top_vbox_layout.addWidget(remove_tab_btn)
  top_vbox_layout.addWidget(my_tabwidget)
  top_widget.setLayout(top_vbox_layout)

  # 关联信号和槽
  remove_tab_btn.clicked.connect(my_tabwidget.remove_tab_handler)
  # 使用QTabWidget就可以省略类似于下面的选项卡与容器的信号和槽的关联步骤
  # my_tabwidget.currentChanged.connect(my_stacked_layout.setCurrentIndex)

  top_widget.show()
  sys.exit(app.exec_())

