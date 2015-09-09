# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 11:42:37 2015

@author: erlean
"""
import sys
import os
from PyQt4 import QtGui, QtCore
from opendxmc.app.view import View, ViewController
from opendxmc.app.model import DatabaseInterface, ListView, ListModel
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

        # statusbar
        status_bar = QtGui.QStatusBar()
        statusbar_log_button = StatusBarButton('Log', None)
        status_bar.addPermanentWidget(statusbar_log_button)
        self.setStatusBar(status_bar)

        # logging
        self.log_widget = LogWidget()
        self.log_handler = LogHandler(self)
        self.log_handler.message.connect(self.log_widget.insertPlainText)
        self.log_widget.closed.connect(statusbar_log_button.setChecked)
        statusbar_log_button.toggled.connect(self.log_widget.setVisible)
        logger.addHandler(self.log_handler)

        # central widget setup
        central_widget = QtGui.QWidget()
        central_widget.setContentsMargins(0, 0, 0, 0)
        self.setContentsMargins(0, 0, 0, 0)
        central_layout = QtGui.QHBoxLayout()
        central_splitter = QtGui.QSplitter(central_widget)
        central_layout.addWidget(central_splitter)
        central_layout.setContentsMargins(0, 0, 0, 0)

        # Databse interface
        self.interface = DatabaseInterface(QtCore.QUrl.fromLocalFile('E:/test.h5'))

        # Models
        self.simulation_list_model = ListModel(self.interface, self,
                                               simulations=True)
        simulation_list_view = ListView()
        simulation_list_view.setModel(self.simulation_list_model)

        self.material_list_model = ListModel(self.interface, self,
                                             materials=True)
        material_list_view = ListView()
        material_list_view.setModel(self.material_list_model)

        # Widgets

        list_view_collection_widget = QtGui.QWidget()
        list_view_collection_widget.setContentsMargins(0, 0, 0, 0)
        list_view_collection_widget.setLayout(QtGui.QVBoxLayout())
        list_view_collection_widget.layout().setContentsMargins(0, 0, 0, 0)
        list_view_collection_widget.layout().addWidget(simulation_list_view, 3)
        list_view_collection_widget.layout().addWidget(material_list_view, 1)
        central_splitter.addWidget(list_view_collection_widget)

        view = View()
        self.viewcontroller = ViewController(self.interface, view)
        central_splitter.addWidget(view)

        central_widget.setLayout(central_layout)
        self.setCentralWidget(central_widget)

        # threading
        self.database_thread = QtCore.QThread(self)
        self.interface.moveToThread(self.database_thread)
        self.database_thread.start()

    def __init_database(self):
        pass


def main(args):

    app = QtGui.QApplication(args)
    app.setOrganizationName("SSHF")
#    app.setOrganizationDomain("https://code.google.com/p/ctqa-cp/")
    app.setApplicationName("OpenDXMC")
    win = MainWindow()
    win.show()

    return app.exec_()



def start():
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


if __name__ == "__main__":
    pass
