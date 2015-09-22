# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 10:10:58 2015

@author: erlean
"""
import numpy as np
import copy
from PyQt4 import QtGui, QtCore
from opendxmc.database import Database
from opendxmc.study import import_ct_series, Simulation, SIMULATION_DESCRIPTION
from opendxmc.materials import Material
from opendxmc.runner import ct_runner
import logging
logger = logging.getLogger('OpenDXMC')


class DatabaseInterface(QtCore.QObject):
    """ Async database interface, async provided with signal/slots resulting in
    two connections per task wished to be done, ie, signal/slot hell
    """
    recive_simulation_list = QtCore.pyqtSignal(list)
    recive_material_list = QtCore.pyqtSignal(list)


    request_simulation_run = QtCore.pyqtSignal(Simulation, list)
    request_simulation_view = QtCore.pyqtSignal(Simulation)
    request_material_view = QtCore.pyqtSignal(Material)
    database_busy = QtCore.pyqtSignal(bool)


    def __init__(self, database_qurl, parent=None):
        super().__init__(parent)
        self.__db = None

        self.set_database(database_qurl)

    @QtCore.pyqtSlot(QtCore.QUrl)
    def set_database(self, database_qurl):
        self.database_busy.emit(True)
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
            if not folder.mkpath(folder.absolutePath()):
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
        self.database_busy.emit(False)

    @QtCore.pyqtSlot()
    def get_simulation_list(self):
        self.database_busy.emit(True)
        sims = self.__db.simulation_list()
        self.recive_simulation_list.emit(sims)
        self.database_busy.emit(False)

    @QtCore.pyqtSlot()
    def get_material_list(self):
        self.database_busy.emit(True)
        mats = self.__db.material_list()
        self.recive_material_list.emit(mats)
        self.database_busy.emit(False)

    @QtCore.pyqtSlot(list)
    def import_dicom(self, qurl_list):
        paths = [url.toLocalFile() for url in qurl_list]
        for sim in import_ct_series(paths):
            self.database_busy.emit(True)
            try:
                self.__db.add_simulation(sim, overwrite=False)
            except ValueError:
               name = self.__db.get_unique_simulation_name()
               logger.info('Simulation {0} already exist in database, renaming to {1}'.format(sim.name, name))
               sim.name = name
               self.__db.add_simulation(sim, overwrite=False)

            self.get_simulation_list()
            self.database_busy.emit(False)

    @QtCore.pyqtSlot(str)
    def select_simulation(self, name):
        self.database_busy.emit(True)
        try:
            sim = self.__db.get_simulation(name)
        except ValueError:
            pass
        else:
            logger.debug('Emmitting signal for request to view Simulation {}'.format(sim.name))
            self.request_simulation_view.emit(sim)
        self.database_busy.emit(False)

    @QtCore.pyqtSlot(str)
    def select_material(self, name):
        self.database_busy.emit(True)
        try:
            mat = self.__db.get_material(name)
        except ValueError:
            pass
        else:
            self.request_material_view.emit(mat)
        self.database_busy.emit(False)

    @QtCore.pyqtSlot(list)
    def copy_simulation(self, names):
        for name in names:
            if isinstance(name, bytes):
                name = str(name, encoding='utf-8')
            self.database_busy.emit(True)
            self.__db.copy_simulation(name)
            self.database_busy.emit(False)
        self.get_simulation_list()

    @QtCore.pyqtSlot(dict, dict, bool)
    def update_simulation_properties(self, prop_dict, arr_dict, purge_volatiles):
        logger.debug('Request database to update simulation properties.')
        self.database_busy.emit(True)
        self.__db.update_simulation(prop_dict, arr_dict, purge_volatiles)
        self.database_busy.emit(False)

    @QtCore.pyqtSlot()
    def get_run_simulation(self):
        self.database_busy.emit(True)
        try:
            sim = self.__db.get_MCready_simulation()
        except ValueError:
            logger.debug('Request to run simulations failed.')
            self.database_busy.emit(False)
            logger.warning('Failed')
            return
        else:
            pass

        try:
            materials = self.__db.get_materials(organic_only=True)
        except ValueError:
            self.database_busy.emit(False)
            logger.warning('failed')
            return

        logger.debug('Emmitting signal for request to run simulation {}'.format(sim.name))
        self.request_simulation_run.emit(sim, materials)
        self.database_busy.emit(False)

#    @QtCore.pyqtSlot(Simulation)
#    def set_run_simulation(self, sim):
#        self.database_busy.emit(True)
#        self.__db.add_simulation(sim)
#        self.database_busy.emit(False)
#        self.get_run_simulation()

class RunManager(QtCore.QObject):
    mc_calculation_finished = QtCore.pyqtSignal()

    request_update_simulation = QtCore.pyqtSignal(dict, dict, bool)
    def __init__(self, interface, parent=None):
        super().__init__(parent)
        interface.request_simulation_run.connect(self.run_simulation)
        self.mc_calculation_finished.connect(interface.get_run_simulation)
        self.request_update_simulation.connect(interface.update_simulation_properties)
        self.timer_interval = 1000 * 60 * 10
        self.timer = QtCore.QBasicTimer()

    @QtCore.pyqtSlot(Simulation, list)
    def run_simulation(self, sim, mat_list):
        self.timer.stop()
        self.timer.start(1000 * 60 * 10, self)
        ct_runner(sim, mat_list, energy_imparted_to_dose_conversion=True, callback=self.update_simulation_iteration)

        self.request_update_simulation.emit(sim.description, sim.volatiles, False)
        self.mc_calculation_finished.emit()
        self.timer.stop()

    def update_simulation_iteration(self, name, energy_imparted, exposure_number):
        if not self.timer.isActive():
            desc = {'name': name,
                    'start_at_exposure_no': exposure_number}
            arrs = {'energy_imparted': energy_imparted}
            self.request_update_simulation.emit(desc, arrs, False)
            self.timer.start(self.timer_interval)





class ListModel(QtCore.QAbstractListModel):

    request_data_list = QtCore.pyqtSignal()
    request_import_dicom = QtCore.pyqtSignal(list)
    request_viewing = QtCore.pyqtSignal(str)
    request_copy_elements = QtCore.pyqtSignal(list)
    def __init__(self, interface, parent=None, simulations=False,
                 materials=False):
        super().__init__(parent)
        self.__data = []

        # connecting interface
        # outbound signals
        if simulations:
            self.request_viewing.connect(interface.select_simulation)
            self.request_data_list.connect(interface.get_simulation_list)
            self.request_copy_elements.connect(interface.copy_simulation)
        elif materials:
            self.request_viewing.connect(interface.select_material)
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

    @QtCore.pyqtSlot(str)
    def element_activated(self, name):
        self.request_viewing.emit(name)


    def rowCount(self, index):
        if not index.isValid():
            return len(self.__data)
        return 0

    def data(self, index, role):
        if not index.isValid():
            return None
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
            return QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsDropEnabled | QtCore.Qt.ItemIsDragEnabled
        return QtCore.Qt.ItemIsDropEnabled

    def mimeData(self, index_list):
        mimedata = QtCore.QMimeData()
        names = [self.__data[index.row()] for index in index_list if index.isValid()]
        if len(names) > 0:
            mimedata.setData('text/plain', ','.join(names))
        return mimedata

    def mimeTypes(self):
#        return ['text/uri-list']
        return ['text/plain', 'text/uri-list']

    def dropMimeData(self, mimedata, action, row, column, index):
        if mimedata.hasUrls():
            urls = [u for u in mimedata.urls() if u.isLocalFile()]
            self.request_import_dicom.emit(urls)
            logger.debug(' '.join([u.toLocalFile() for u in urls]))
            return True
        elif mimedata.hasText():
            names = mimedata.data('text/plain').split(',')
            self.request_copy_elements.emit([str(n, encoding='ascii') for n in names])
        return False

    def supportedDropActions(self):
        return QtCore.Qt.CopyAction | QtCore.Qt.MoveAction


class ListView(QtGui.QListView):
    name_activated = QtCore.pyqtSignal(str)

    def __init__(self, parent=None, simulation=True):
        super().__init__(parent)
        if simulation:
            self.setAcceptDrops(True)
            self.viewport().setAcceptDrops(True)
            self.setDropIndicatorShown(True)
            self.setDragDropMode(self.DragDrop)
            self.setDefaultDropAction(QtCore.Qt.CopyAction)
        else:
            self.setAcceptDrops(False)
            self.viewport().setAcceptDrops(False)
            self.setDropIndicatorShown(True)
            self.setDragDropMode(self.NoDragDrop)
            self.setDefaultDropAction(QtCore.Qt.CopyAction)

        self.activated.connect(self.activation_name)

    def setModel(self, model):
        try:
            self.name_activated.disconnect()
        except TypeError:
            pass
        super().setModel(model)
        self.name_activated.connect(model.element_activated)


    @QtCore.pyqtSlot(QtCore.QModelIndex)
    def activation_name(self, index):
        if index.isValid():
            self.name_activated.emit(index.data())


