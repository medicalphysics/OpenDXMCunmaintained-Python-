# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 22:27:17 2016

@author: ander
"""
from PyQt4 import QtGui, QtCore
import sys
def main(args):

    app = QtGui.QApplication(args)
    splash_pix = QtGui.QPixmap('splash_loading.png')
    splash = QtGui.QSplashScreen(splash_pix, QtCore.Qt.WindowStaysOnTopHint)
    splash.setMask(splash_pix.mask())
    splash.show()
    app.processEvents()
    app.setOrganizationName("SSHF")
    
    app.setApplicationName("OpenDXMC")
    import time
    time.sleep(5)
    from opendxmc.app.gui import MainWindow
    win = MainWindow()
    win.show()
    splash.finish(win)

    return app.exec_()



def start():
    # exit code 1 triggers a restart
    # Also testing for memory error
    try:
        while main(sys.argv) == 1:
            continue
    except MemoryError:
        msg = QtGui.QMessageBox()
        msg.setText("Ouch, OpenDXMC ran out of memory.")
        msg.setIcon(msg.Critical)
        msg.exec_()
    sys.exit(0)


if __name__ == "__main__":
    pass
