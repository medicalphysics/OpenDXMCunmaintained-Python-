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
import time

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

    simulation_updated = QtCore.pyqtSignal(dict, dict)

    def __init__(self, database_qurl, importer, parent=None):
        super().__init__(parent)
        self.__db = None

        importer.request_add_sim_to_database.connect(self.import_simulation)
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

    @QtCore.pyqtSlot(Simulation)
    def import_simulation(self, sim):
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

    @QtCore.pyqtSlot(dict, dict, bool, bool)
    def update_simulation_properties(self, prop_dict, arr_dict, purge_volatiles, cancel_if_running=True):
        logger.debug('Request database to update simulation properties.')
        self.database_busy.emit(True)
        self.__db.update_simulation(prop_dict, arr_dict, purge_volatiles, cancel_if_running)
        if prop_dict.get('MC_ready', False) and not prop_dict.get('MC_running', False):
            self.get_run_simulation()
        self.database_busy.emit(False)
        self.simulation_updated.emit(prop_dict, arr_dict)

    @QtCore.pyqtSlot()
    def get_run_simulation(self):
        self.database_busy.emit(True)
        try:
            sim = self.__db.get_MCready_simulation()
        except ValueError:
            logger.debug('Request to get ready simulations failed for a mysterious reason.')
            self.database_busy.emit(False)
            return
        else:
            pass

        try:
            materials = self.__db.get_materials(organic_only=False)
        except ValueError:
            self.database_busy.emit(False)
            logger.warning('Request to materials for a ready simulations failed for a mysterious reason.')
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
class Importer(QtCore.QObject):
    request_add_sim_to_database = QtCore.pyqtSignal(Simulation)
    running = QtCore.pyqtSignal(bool)
    def __init__(self, parent=None):
        super().__init__(parent)

    @QtCore.pyqtSlot(list)
    def import_urls(self, qurl_list):
        self.running.emit(True)
        paths = [url.toLocalFile() for url in qurl_list]
        for sim in import_ct_series(paths):
            self.request_add_sim_to_database.emit(sim)
        self.running.emit(False)

class Runner(QtCore.QThread):
    mc_calculation_finished = QtCore.pyqtSignal()
    request_update_simulation = QtCore.pyqtSignal(dict, dict, bool, bool)
    request_view_update = QtCore.pyqtSignal(dict, dict)

    start_timer = QtCore.pyqtSignal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self.request_save = False
        self.timer = QtCore.QTimer()
        self.timer.setInterval(600000)
#        self.time_interval = 600000 # saving data every 10 minutes
        self.started.connect(self.timer.start)
        self.timer.setSingleShot(False)
        self.terminated.connect(self.timer.stop)
        self.finished.connect(self.timer.stop)

        self.timer.timeout.connect(self.set_request_save)
        self.mutex = QtCore.QMutex()
        self.simulation = None
        self.material_list = None

    @QtCore.pyqtSlot()
    def set_request_save(self):
        self.mutex.lock()
        self.request_save = True
        self.mutex.unlock()

    def update_simulation_iteration(self, name, energy_imparted, exposure_number):
        desc = {'name': name,
                'start_at_exposure_no': exposure_number}
        arrs = {'energy_imparted': energy_imparted}
        if self.request_save:
            self.request_update_simulation.emit(desc, arrs, False, False)
            self.mutex.lock()
            self.request_save = False
            self.mutex.unlock()
        else:
            self.request_view_update.emit(desc, arrs)

    def run(self):
        if self.simulation is None:
            return
        if self.material_list is None:
            return

        self.request_update_simulation.emit({'name': self.simulation.name,
                                             'MC_running': True},
                                            {},
                                            False, False)
        ct_runner(self.simulation, self.material_list,
                  energy_imparted_to_dose_conversion=True,
                  callback=self.update_simulation_iteration)
        self.simulation.MC_running = False
        self.simulation.MC_ready = False
        self.simulation.MC_finished = True
        self.request_update_simulation.emit(self.simulation.description,
                                            self.simulation.volatiles,
                                            False, False)

        self.mc_calculation_finished.emit()



class RunManager(QtCore.QObject):
    mc_calculation_running = QtCore.pyqtSignal(bool)
    def __init__(self, interface, view_controller, parent=None):
        super().__init__(parent)
        self.runner = Runner()
        interface.request_simulation_run.connect(self.run_simulation)
        self.runner.mc_calculation_finished.connect(interface.get_run_simulation)
        self.runner.request_update_simulation.connect(interface.update_simulation_properties)
        self.runner.request_view_update.connect(view_controller.updateSimulation)
        self.runner.finished.connect(self.run_finished)
        self.runner.started.connect(self.run_started)

    @QtCore.pyqtSlot()
    def run_started(self):
        self.mc_calculation_running.emit(True)

    @QtCore.pyqtSlot()
    def run_finished(self):
        self.mc_calculation_running.emit(False)

    @QtCore.pyqtSlot(Simulation, list)
    def run_simulation(self, sim, mat_list):
        logger.debug('Attemp to start MC thread')
        if not self.runner.isRunning():
            self.runner.simulation = sim
            self.runner.material_list = mat_list
            self.runner.start()
#            self.runner.run2()
            logger.debug('MC thread started')





class ListModel(QtCore.QAbstractListModel):

    request_data_list = QtCore.pyqtSignal()
    request_import_dicom = QtCore.pyqtSignal(list)
    request_viewing = QtCore.pyqtSignal(str)
    request_copy_elements = QtCore.pyqtSignal(list)
    def __init__(self, interface, importer=None, parent=None, simulations=False,
                 materials=False):
        super().__init__(parent)
        self.__data = []

        # connecting interface
        # outbound signals
        if simulations:
            self.request_viewing.connect(interface.select_simulation)
            self.request_data_list.connect(interface.get_simulation_list)
            self.request_copy_elements.connect(interface.copy_simulation)
            if importer:
                self.request_import_dicom.connect(importer.import_urls)
        elif materials:
            self.request_viewing.connect(interface.select_material)
            self.request_data_list.connect(interface.get_material_list)


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


