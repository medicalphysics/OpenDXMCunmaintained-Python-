# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 11:42:37 2015

@author: erlean
"""
import sys
import os
from PyQt4 import QtGui, QtCore
from gui_widgets.viewer import View
import logging

logger = logging.getLogger('OpenDXMC')
logger.setLevel(10)
LOG_FORMAT = ("[%(asctime)s %(name)s %(levelname)s]  -  %(message)s  -  in method %(funcName)s line:"
    "%(lineno)d filename: %(filename)s")



class LogHandler(QtCore.QObject, logging.Handler):
    message = QtCore.pyqtSignal(str)
    def __init__(self, parent=None):
        QtCore.QObject.__init__(self, parent)
        logging.Handler.__init__(self)

        self.setFormatter(logging.Formatter(LOG_FORMAT, '%H:%M'))
#        self.log_formater = logging.Formatter()

    def emit(self, log_record):
        self.message.emit(self.format(log_record) + os.linesep)

class LogWidget(QtGui.QTextEdit):
    closed = QtCore.pyqtSignal(bool)
    def __init__(self, parent=None):
        super().__init__(parent)

    def closeEvent(self, event):
        self.closed.emit(False)
        super().closeEvent(event)

class StatusBarButton(QtGui.QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setFlat(True)
        self.setCheckable(True)

class MainWindow(QtGui.QMainWindow):
    def __init__(self):
        super().__init__()

        central_widget = QtGui.QWidget()
        central_widget.setContentsMargins(0, 0, 0, 0)

        central_layout = QtGui.QHBoxLayout()
        central_splitter = QtGui.QSplitter(central_widget)
        central_layout.addWidget(central_splitter)

        w1 = QtGui.QLabel()
        w1.setText('Test 1 label')
        w2 = View()
        central_splitter.addWidget(w1)
        central_splitter.addWidget(w2)
        self.timer = QtCore.QTimer(self)

        self.timer.timeout.connect(w2.set_random)
        self.timer.start(1000)

        status_bar = QtGui.QStatusBar()
        statusbar_log_button = StatusBarButton('Log', None)
        status_bar.addPermanentWidget(statusbar_log_button)

        central_widget.setLayout(central_layout)
        self.setCentralWidget(central_widget)
        self.setStatusBar(status_bar)

        #logging
        self.log_widget = LogWidget()
        self.log_handler = LogHandler(self)
        self.log_handler.message.connect(self.log_widget.insertPlainText)
        self.log_widget.closed.connect(statusbar_log_button.setChecked)
        statusbar_log_button.toggled.connect(self.log_widget.setVisible)
        logger.addHandler(self.log_handler)


def main(args):
    app = QtGui.QApplication(args)
    app.setOrganizationName("SSHF")
#    app.setOrganizationDomain("https://code.google.com/p/ctqa-cp/")
    app.setApplicationName("OpenDXMC")
    win = MainWindow()
    win.show()

    return app.exec_()

if __name__ == "__main__":
    # exit code 1 triggers a restart
    # Also testing for memory error
    try:
        while main(sys.argv) == 1:
            continue
    except MemoryError as e:
        msg = QtGui.QMessageBox()
        msg.setText("Ouch, OpenDXMC ran out of memory.")
        msg.setIcon(msg.Critical)
        msg.exec_()
    sys.exit(0)
