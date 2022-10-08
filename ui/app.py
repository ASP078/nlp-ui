import sys 
from PyQt5 import uic
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import pyperclip
import logging

UI_FirstWindow, QtBaseClass = uic.loadUiType("form.ui")

class MainApp(QMainWindow, UI_FirstWindow):
    def __init__(self, logger):
        super().__init__()

        self._logger = logger

        self.setupUi(self)
        self.clrBtn.clicked.connect(self.clearInput)
        self.cpyBtn.clicked.connect(self.copyOutput)
        self.translateBtn.clicked.connect(self.translateInput)


    @pyqtSlot()
    def clearInput(self):
        self._logger.info("Clear Button Clicked")

        self.engTxt.setPlainText("")
        self.hinTxt.setPlainText("")

        self._logger.warning("Text Cleared from the Boxes")

    @pyqtSlot()
    def copyOutput(self):
        self._logger.info("Copy Button Clicked")

        pyperclip.copy(self.hinTxt.toPlainText())
        
        self._logger.warning("Hindi Text Copied to Clipboard")

    @pyqtSlot()
    def translateInput(self):
        self._logger.info("Translate Button Clicked")

        # print(self.engTxt.toPlainText())

        self._logger.warning("Text Translated to Hindi")

        self.hinTxt.setPlainText(self.engTxt.toPlainText())

        self._logger.warning("Hindi Text set to the Text Box")


def main():
    import os
    path = "logs"
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)

    logging.basicConfig(filename="logs/debug.log",
                            format='%(asctime)s %(created)f  %(funcName)s  %(levelname)s %(process)d  %(processName)s  %(thread)d   %(threadName)s  %(message)s',
                            filemode='a')

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    logger.info("Application Initialized")

    app = QApplication(sys.argv)
    window = MainApp(logger=logger)

    logger.warning("Application UI created")

    logger.warning("Application Running")

    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()  