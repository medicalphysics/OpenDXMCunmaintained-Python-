# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 10:10:58 2015

@author: erlean
"""

from PyQt4 import QtGui, QtCore
from opendxmc.database import Database
from opendxmc.study import import_ct_series
import numpy as np
import logging
logger = logging.getLogger('OpenDXMC')


class DatabaseInterface(QtCore.QObject):
    """ Async database interface, async provided with signal/slots resulting in
    two connections per task wished to be done, ie, signal/slot hell
    """
    recive_simulation_list = QtCore.pyqtSignal(list)
    recive_material_list = QtCore.pyqtSignal(list)


    def __init__(self, database_qurl, parent=None):
        super().__init__(parent)
        self.__db = None

        self.set_database(database_qurl)


    @QtCore.pyqtSlot(QtCore.QUrl)
    def set_database(self, database_qurl):
        if self.__db:
            self.__db.close()
        fileinfo = QtCore.QFileInfo(database_qurl.toLocalFile())
        if fileinfo.isDir():
            folder = QtCore.QDir(fileinfo.absoluteFilePath())
            fname = 'database.h5'
            logger.info('Database path is a directory, setting database filename to: {0}'.format(fname))
        else:
            folder = fileinfo.dir()
            fname = fileinfo.fileName()

        msg = ''
        while not folder.exists():
            if not folder.mkpath(folder.absolutePath):
                msg = 'Could not create folder: {0}, setting databasefolder to: {1}'.format(folder.absolutePath(), QtCore.QDir.current().absolutePath())
                logger.error(msg)
                folder = QtCore.QDir.current()

        path = QtCore.QFileInfo(folder.absolutePath() + folder.separator() + fname)
#        if not folder.isWritable():
#            if len(msg) > 0:
#                msg += ' '
#            msg += 'Could not write or create database path: {0}'.format(path)
#            QtGui.QMessageBox.critical(title='Error in database file path', text=msg)
#            QtGui.qApp.quit()

        logger.debug('Attemting to use database in {0}'.format(path.absoluteFilePath()))

        self.__db = Database(path.absoluteFilePath())

    @QtCore.pyqtSlot()
    def get_simulation_list(self):
        sims = self.__db.simulation_list()
        self.recive_simulation_list.emit(sims)

    @QtCore.pyqtSlot()
    def get_material_list(self):
        mats = self.__db.material_list()
        self.recive_material_list.emit(mats)

    @QtCore.pyqtSlot(list)
    def import_dicom(self, qurl_list):
        paths = [url.toLocalFile() for url in qurl_list]
        for sim in import_ct_series(paths):
            self.__db.add_simulation(sim)
            self.get_simulation_list()


class ListModel(QtCore.QAbstractListModel):

    request_data_list = QtCore.pyqtSignal()
    request_import_dicom = QtCore.pyqtSignal(list)

    def __init__(self, interface, parent=None, simulations=False,
                 materials=False):
        super().__init__(parent)
        self.__data = []

        # connecting interface
        # outbound signals
        if simulations:
            self.request_data_list.connect(interface.get_simulation_list)
        elif materials:
            self.request_data_list.connect(interface.get_material_list)
        self.request_import_dicom.connect(interface.import_dicom)

        # inbound signals
        if simulations:
            interface.recive_simulation_list.connect(self.recive_data_list)
        elif materials:
            interface.recive_material_list.connect(self.recive_data_list)

        # setting up
        self.request_data_list.emit()

    @QtCore.pyqtSlot(list)
    def recive_data_list(self, sims):
        self.layoutAboutToBeChanged.emit()
        #muste update persistent index
        self.__data = sims
        self.layoutChanged.emit()

    def rowCount(self, index):
        if not index.isValid():
            return len(self.__data)
        return 0

    def data(self, index, role):
        row = index.row()
        if role == QtCore.Qt.DisplayRole:
            return self.__data[row]
        elif role == QtCore.Qt.DecorationRole:
            pass
        elif role == QtCore.Qt.ToolTipRole:
            pass
        elif role == QtCore.Qt.SizeHintRole:
            pass
        elif role == QtCore.Qt.BackgroundRole:
            pass
        elif role == QtCore.Qt.ForegroundRole:
            pass
        return None

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        return str(section)

    def flags(self, index):
        if index.isValid():
            return QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsDropEnabled
        return QtCore.Qt.ItemIsDropEnabled

    def mimeTypes(self):
        return ['text/uri-list']

    def dropMimeData(self, mimedata, action, row, column, index):
        if mimedata.hasUrls():
            self.request_import_dicom.emit(mimedata.urls())
            logger.debug(' '.join([u.toLocalFile() for u in mimedata.urls()]))
            return True
        return False

    def supportedDropActions(self):
        return QtCore.Qt.CopyAction | QtCore.Qt.MoveAction


class ListView(QtGui.QListView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(self.DropOnly)

#class Model(QtCore.QAbstractItemModel):
#    request_simulation_list = QtCore.pyqtSignal()
#    request_import_dicom = QtCore.pyqtSignal(list)
#    def __init__(self, database_path, parent=None):
#        super().__init__(parent)
#
#        interface = Database_interface(database_path, None)
#        # connecting interface
#        # outbound signals
#        self.request_simulation_list.connect(interface.get_simulation_list)
#        self.request_import_dicom.connect(interface.import_dicom)
#        # inbound signals
#        interface.recive_simulation_list.connect(self.recive_simulation_list)
#
#
#        self.__data = []
#        self.request_simulation_list.emit()
#
#    @QtCore.pyqtSlot(list)
#    def recive_simulation_list(self, sims):
#        self.layoutAboutToBeChanged.emit()
#        #muste update persistent index
#        self.__data = sims
#        self.layoutChanged.emit()
#
#
#    def flags(self, index):
#        return QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled
#
#    def data(self, index, role):
#        row = index.row()
#        if role == QtCore.Qt.DisplayRole:
#            return self.__data[row]
#        elif role == QtCore.Qt.DecorationRole:
#            pass
#        elif role == QtCore.Qt.ToolTipRole:
#            pass
#        elif role == QtCore.Qt.SizeHintRole:
#            pass
#        elif role == QtCore.Qt.BackgroundRole:
#            pass
#        elif role == QtCore.Qt.ForegroundRole:
#            pass
#        return None
#
#    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
#        return str(section)
#    def rowCount(self):
#        return len(self.__data)
#
#
#    def index(self, row, column, parent_index):
#        return self.createIndex(row, column, self.__data[row])
#
#    def parent(self, index):
#        return QtCore.QModelIndex()
#    def mimeTypes(self):
#        return ['text/uri-list']
#    def supportedDropActions(self):
#        return QtCore.Qt.CopyAction | QtCore.Qt.MoveAction
#
#    def dropMimeData(self, mimedata, action, row, column, index):
#        if mimedata.hasUrls():
#            self.request_import_dicom.emit(mimedata.urls())
#            return True
#        return False


