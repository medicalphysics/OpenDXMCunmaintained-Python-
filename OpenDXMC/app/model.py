# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 10:10:58 2015

@author: erlean
"""
import numpy as np
import copy
from PyQt4 import QtGui, QtCore
from opendxmc.database import Database
from opendxmc.data.import_phantoms import read_phantoms
from opendxmc.study import import_ct_series, Simulation, SIMULATION_DESCRIPTION
from opendxmc.materials import Material
from opendxmc.runner import ct_runner
import logging
logger = logging.getLogger('OpenDXMC')
import time


from opendxmc.runner.ct_study_runner import ct_runner_validate_simulation


class ImportScalingValidator(QtGui.QValidator):
    def __init__(self, parent):
        super().__init__(parent)

    def fixup(self, instr):
        instr = ''.join([b for b in instr.replace(',', ' ') if b in "1234567890. "])

        fstr = ""
        d_in_word = False
        for s in instr:
            if s == '.':
                if not d_in_word:
                    fstr += s
                else:
                    fstr += ' '
                d_in_word = True
            elif s == ' ':
                d_in_word = False
                fstr += ' '
            else:
                fstr += s


        return fstr
#        nums = [float(a) for a in instr.split()]
#        return ' '.join([str(n) for n in nums])

    def validate(self, rawstr, pos):
        instr = ''.join([b for b in rawstr.replace(',', ' ') if b in "1234567890. "])
        pos -= (len(instr) - len(rawstr))

        if len(instr) < 5:
            return self.Intermediate, instr, pos
        numbers = []
        state = self.Acceptable
        last = instr[-1]
        for word in instr.split():
            try:
                float(word)
            except:
                state = self.Intermediate
            numbers.append(word)
        if len(numbers) > 3:
            rstr = ' '.join(numbers[:3])
        else:
            rstr = ' '.join(numbers)
        if last == ' ':
            rstr += last
        return state, rstr, pos


class ImportScalingEdit(QtGui.QLineEdit):
    request_set_import_scaling = QtCore.pyqtSignal(tuple)
    def __init__(self, importer, parent=None):
        super().__init__(parent)
        self.base_color = self.palette().color(self.palette().Base)

        self.request_set_import_scaling.connect(importer.set_import_scaling)

        self.editingFinished.connect(self.set_import_scaling)
        self.textEdited.connect(self.text_was_edited)
        self.setValidator(ImportScalingValidator(self))
        self.setText("2.0 2.0 1.0")
        self.set_import_scaling()

    @QtCore.pyqtSlot(str)
    def text_was_edited(self, txt):
        palette = self.palette()
        palette.setColor(palette.Base, QtCore.Qt.red)
        self.setPalette(palette)

    @QtCore.pyqtSlot()
    def set_import_scaling(self):
        txt = self.text()
        d = tuple(float(s) for s in txt.split())
        self.request_set_import_scaling.emit(d)
        self.setText(' '.join([str(n) for n in d]))
        palette = self.palette()
        palette.setColor(palette.Base, self.base_color)
        self.setPalette(palette)


class DatabaseInterface(QtCore.QObject):
    """ Async database interface, async provided with signal/slots resulting in
    two connections per task wished to be done, ie, signal/slot hell
    """
    send_simulation_list = QtCore.pyqtSignal(list)
    send_material_list = QtCore.pyqtSignal(list)

    send_array_slice = QtCore.pyqtSignal(str, np.ndarray, str, int, int)  # simulation dict, array_slice, array_name, index, orientation
    send_array = QtCore.pyqtSignal(str, np.ndarray, str)  # simulation dict, array_slice, array_name, index, orientation
    send_sim_propeties = QtCore.pyqtSignal(dict)

    send_material_obj = QtCore.pyqtSignal(Material)
    database_busy = QtCore.pyqtSignal(bool)

    send_mc_ready = QtCore.pyqtSignal(dict, list)  # sim properties and list of Material objects



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

        logger.debug('Attemting to use database in {0}'.format(path.absoluteFilePath()))

        self.__db = Database(path.absoluteFilePath())
        self.database_busy.emit(False)
        self.emit_material_list()
        self.emit_simulation_list()


    @QtCore.pyqtSlot()
    def emit_simulation_list(self):
        self.database_busy.emit(True)
        sims = self.__db.simulation_list()
        self.send_simulation_list.emit(sims)
        self.database_busy.emit(False)

    @QtCore.pyqtSlot()
    def emit_material_list(self):
        self.database_busy.emit(True)
        mats = self.__db.material_list()
        self.send_material_list.emit(mats)
        self.database_busy.emit(False)

#    @QtCore.pyqtSlot(Simulation)
#    def import_simulation(self, sim):
#        self.database_busy.emit(True)
#        try:
#            self.__db.add_simulation(sim, overwrite=False)
#        except ValueError:
#            if sim.is_phantom:
#                logger.info('Phantom {0} already exist in database'.format(sim.name))
#            else:
#                name = self.__db.get_unique_simulation_name(sim.name)
#                logger.info('Simulation {0} already exist in database, renaming to {1}'.format(sim.name, name))
#                sim.name = name
#                self.__db.add_simulation(sim, overwrite=False)
#
#
#        self.get_simulation_list()
#        self.database_busy.emit(False)

    @QtCore.pyqtSlot(str, np.ndarray, str, bool)
    def store_array(self, name, array, array_name, volatile=False):
        self.database_busy.emit(True)
        try:
            self.__db.get_simulation_array(simulation_name, array_name)
        except ValueError:
            pass
        else:
            self.send_array.emit(simulation_name, arr, array_name)
        self.database_busy.emit(False)

    @QtCore.pyqtSlot(str, str)
    def request_array(self, simulation_name, array_name):
        self.database_busy.emit(True)
        try:
            arr = self.__db.get_simulation_array(simulation_name, array_name)
        except ValueError:
            pass
        else:
            self.send_array.emit(simulation_name, arr, array_name)
        self.database_busy.emit(False)

    @QtCore.pyqtSlot(str, str, int, int)
    def request_array_slice(self, simulation_name, array_name, index, orientation):
        self.database_busy.emit(True)
        try:
            arr = self.__db.get_simulation_array_slice(simulation_name, array_name, index, orientation)
        except:
            print(simulation_name, array_name, index, orientation)
#            raise e
            pass
        else:
            self.send_array_slice.emit(simulation_name, arr, array_name, index, orientation)
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
    def update_simulation_properties(self, propeties_dict, array_dict, volatiles_dict, purge_volatiles=True, cancel_if_running=True):
        logger.debug('Request database to update simulation properties.')
        self.database_busy.emit(True)
        self.__db.update_simulation(propeties_dict, array_dict, volatiles_dict, purge_volatiles, cancel_if_running)
        self.database_busy.emit(False)

    @QtCore.pyqtSlot()
    def request_run_simulation(self):
        self.database_busy.emit(True)
        try:
            props = self.__db.get_MCready_simulation()
        except ValueError:
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

        logger.debug('Emmitting signal request to run simulation {}'.format(props['name']))
        self.send_mc_ready.emit(props, materials)
        self.database_busy.emit(False)


class Importer(QtCore.QObject):
    request_add_sim_to_database = QtCore.pyqtSignal(Simulation)
    running = QtCore.pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.__import_scaling = (1, 1, 1)

    @QtCore.pyqtSlot(tuple)
    def set_import_scaling(self, im_scaling):
        self.__import_scaling = im_scaling


    @QtCore.pyqtSlot(list)
    def import_urls(self, qurl_list):
        self.running.emit(True)
        paths = [url.toLocalFile() for url in qurl_list]
        for sim in import_ct_series(paths, import_scaling=self.__import_scaling):
            self.request_add_sim_to_database.emit(sim)
        self.running.emit(False)

    @QtCore.pyqtSlot()
    def import_phantoms(self):
        self.running.emit(True)
        for sim in read_phantoms():
            self.request_add_sim_to_database.emit(sim)
        self.running.emit(False)

class Runner(QtCore.QThread):
    mc_calculation_finished = QtCore.pyqtSignal()
    request_update_simulation = QtCore.pyqtSignal(dict, dict, bool, bool)
    request_view_update = QtCore.pyqtSignal(dict, dict)

    start_timer = QtCore.pyqtSignal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self.request_save = True
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

    def update_simulation_iteration(self, name, array_dict, exposure_number):
        desc = {'name': name,
                'start_at_exposure_no': exposure_number}
        if self.request_save:
            self.request_update_simulation.emit(desc, array_dict, False, False)
            self.mutex.lock()
            self.request_save = False
            self.mutex.unlock()
        else:
            self.request_view_update.emit(desc, array_dict)

    def run(self):
        if self.simulation is None:
            return
        if self.material_list is None:
            return

#        ct_runner_validate_simulation(self.simulation, self.material_list)

        self.request_update_simulation.emit({'name': self.simulation.name,
                                             'MC_running': True},
                                            {},
                                            False, False)
        try:
            ct_runner(self.simulation, self.material_list,
                      energy_imparted_to_dose_conversion=True,
                      callback=self.update_simulation_iteration)
        except MemoryError:
            logger.error('MEMORY ERROR: Could not run simulation {0}, memory to low. Try to increase dose matrix scaling or use 64 bit version of OpenDXMC'.format(self.simulation.name))
            self.simulation.MC_finished = False
            self.simulation.MC_running = False
            self.simulation.MC_ready = False
            self.request_update_simulation.emit(self.simulation.description,
                                                {},
                                                True, False)
        except ValueError or AssertionError as e:
            print(e)
            raise e
            logger.error('UNKNOWN ERROR: Could not run simulation {0}'.format(self.simulation.name))
            self.simulation.MC_finished = False
            self.simulation.MC_running = False
            self.simulation.MC_ready = False
            self.request_update_simulation.emit(self.simulation.description,
                                                {},
                                                True, False)

        else:
            self.simulation.MC_finished = True
            self.simulation.MC_running = False
            self.simulation.MC_ready = False
            self.request_update_simulation.emit(self.simulation.description,
                                                self.simulation.volatiles,
                                                False, False)
        self.mc_calculation_finished.emit()
        self.request_save = True


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


