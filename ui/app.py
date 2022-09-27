import sys 
from PyQt5 import uic
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

UI_FirstWindow, QtBaseClass = uic.loadUiType("form.ui")

class MainApp(QMainWindow, UI_FirstWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

def main():
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()  