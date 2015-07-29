# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 11:42:37 2015

@author: erlean
"""
import sys
import os
from PyQt4 import QtGui, QtCore


def main(args):
    app = QtGui.QApplication(args)
    app.setOrganizationName("SSHF")
#    app.setOrganizationDomain("https://code.google.com/p/ctqa-cp/")
    app.setApplicationName("CTQA_cp")
    win = Ctqa()
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
        msg.setText("Ouch, CTQA-cp ran out of memory.")
        msg.setIcon(msg.Critical)
        msg.exec_()
    sys.exit(0)