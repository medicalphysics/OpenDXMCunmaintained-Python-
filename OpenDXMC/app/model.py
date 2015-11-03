# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 10:10:58 2015

@author: erlean
"""
import numpy as np
from PyQt4 import QtGui, QtCore
from opendxmc.database import Database, PROPETIES_DICT_TEMPLATE, Validator
from opendxmc.database.import_phantoms import read_phantoms
from opendxmc.database import import_ct_series
from opendxmc.materials import Material
from opendxmc.runner import ct_runner
import logging
logger = logging.getLogger('OpenDXMC')


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
    database_busy = QtCore.pyqtSignal(bool)

    send_simulation_list = QtCore.pyqtSignal(list)
    send_material_list = QtCore.pyqtSignal(list)

    send_view_array = QtCore.pyqtSignal(str, np.ndarray, str)  # simulation dict, array_slice, array_name, index, orientation
    send_view_array_slice = QtCore.pyqtSignal(str, np.ndarray, str, int, int)  # simulation dict, array_slice, array_name, index, orientation
    send_view_sim_propeties = QtCore.pyqtSignal(dict)

    send_MC_ready_simulation = QtCore.pyqtSignal(dict, dict, list)



#    send_import_array = QtCore.pyqtSignal(str, np.ndarray, str)  # simulation dict, array_slice, array_name, index, orientation
#    send_view_sim_propeties = QtCore.pyqtSignal(dict)
#    send_view_sim_propeties = QtCore.pyqtSignal(dict)

#    send_material_obj = QtCore.pyqtSignal(Material)


#    send_mc_ready = QtCore.pyqtSignal(dict, list)  # sim properties and list of Material objects



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



    @QtCore.pyqtSlot(dict, dict, bool)
    def add_simulation(self, properties, array_dict, overwrite):
        self.database_busy.emit(True)
        self.__db.add_simulation(properties, array_dict, overwrite)
        self.emit_simulation_list()
        self.database_busy.emit(False)

    @QtCore.pyqtSlot(str, str)
    def request_view_array(self, simulation_name, array_name):
        self.database_busy.emit(True)
        try:
            arr = self.__db.get_simulation_array(simulation_name, array_name)
        except ValueError:
            pass
        else:
            self.send_view_array.emit(simulation_name, arr, array_name)
        self.database_busy.emit(False)

    @QtCore.pyqtSlot(str, str, int, int)
    def request_view_array_slice(self, simulation_name, array_name, index, orientation):
        self.database_busy.emit(True)
        try:
            arr = self.__db.get_simulation_array_slice(simulation_name, array_name, index, orientation)
        except:
            pass
        else:
            self.send_view_array_slice.emit(simulation_name, arr, array_name, index, orientation)
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


    @QtCore.pyqtSlot(str)
    def request_simulation_properties(self, name):
        logger.debug('Request simulation metadata for {} from database.'.format(name))
        self.database_busy.emit(True)
        try:
            data = self.__db.get_simulation_metadata(name)
        except ValueError:
            logger.debug('Could not read metadata for simulation')
        else:
            self.send_view_sim_propeties.emit(data)
        self.database_busy.emit(False)

    @QtCore.pyqtSlot(dict, bool, bool)
    def set_simulation_properties(self, propeties_dict, purge_volatiles=True, cancel_if_running=True):
        logger.debug('Request database to update simulation properties.')
        self.database_busy.emit(True)
        self.__db.set_simulation_metadata(propeties_dict, purge_volatiles, cancel_if_running)
        try:
            update_data = self.__db.get_simulation_metadata(propeties_dict.get('name', ''))
        except ValueError:
            logger.debug('Could not read metadata for simulation')
        else:
            self.send_view_sim_propeties.emit(update_data)
        self.database_busy.emit(False)

    @QtCore.pyqtSlot(str, dict)
    def write_simulation_arrays(self, name, array_dict):
        logger.debug('Request database to write arrays.')
        self.database_busy.emit(True)
        for arr_name, array in array_dict.items():
            self.__db.set_simulation_array(name, array, arr_name)
        self.database_busy.emit(False)

    @QtCore.pyqtSlot()
    def request_MC_ready_simulation(self):
        self.database_busy.emit(True)
        try:
            props, arrays = self.__db.get_MCready_simulation()
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
        self.send_MC_ready_simulation.emit(props,arrays, materials)
        self.database_busy.emit(False)


class Importer(QtCore.QObject):
    request_add_sim_to_database = QtCore.pyqtSignal(dict, dict, bool)
    running = QtCore.pyqtSignal(bool)

    def __init__(self, database_interface, parent=None):
        super().__init__(parent)
        self.__import_scaling = (1, 1, 1)
        self.request_add_sim_to_database.connect(database_interface.add_simulation)


    @QtCore.pyqtSlot(tuple)
    def set_import_scaling(self, im_scaling):
        self.__import_scaling = im_scaling


    @QtCore.pyqtSlot(list)
    def import_urls(self, qurl_list):
        self.running.emit(True)
        paths = [url.toLocalFile() for url in qurl_list]
        for props, arrays in import_ct_series(paths, import_scaling=self.__import_scaling):
            self.request_add_sim_to_database.emit(props, arrays, True)
        self.running.emit(False)

    @QtCore.pyqtSlot()
    def import_phantoms(self):
        self.running.emit(True)
        for props, arrays in read_phantoms():
            self.request_add_sim_to_database.emit(props, arrays, False)
        self.running.emit(False)

class Runner(QtCore.QThread):
    mc_calculation_finished = QtCore.pyqtSignal()
    request_write_simulation_arrays = QtCore.pyqtSignal(str, dict)
    request_set_simulation_properties = QtCore.pyqtSignal(dict, bool, bool)
    request_runner_view_update = QtCore.pyqtSignal(str, dict, dict)

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
        self.simulation_properties = None
        self.simulation_arrays = None
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
            self.request_set_simulation_properties.emit(desc, False, False)
            self.request_write_simulation_arrays.emit(name, array_dict)
            self.mutex.lock()
            self.request_save = False
            self.mutex.unlock()
        else:
            self.request_runner_view_update.emit(name, desc, array_dict)

    def run(self):
        if self.simulation is None:
            return
        if self.material_list is None:
            return

        self.request_set_simulation_properties.emit({'name': self.simulation.name,
                                                     'MC_running': True},
                                                     False, False)

        try:
            props_dict, arr_dict = ct_runner(self.material_list, self.simulation_properties,
                                             energy_imparted_to_dose_conversion=True,
                                             callback=self.update_simulation_iteration,
                                             **self.simulation_arrays)
        except MemoryError:
            logger.error('MEMORY ERROR: Could not run simulation {0}, memory to low. Try to increase dose matrix scaling or use 64 bit version of OpenDXMC'.format(self.simulation.name))
            self.simulation_properties['MC_finished'] = False
            self.simulation_properties['MC_running'] = False
            self.simulation_properties['MC_ready'] = False
            self.request_set_simulation_properties.emit(self.simulation_properties, True, False)

        except ValueError or AssertionError as e:
            print(e)
            raise e
            logger.error('UNKNOWN ERROR: Could not run simulation {0}'.format(self.simulation.name))
            self.simulation_properties['MC_finished'] = False
            self.simulation_properties['MC_running'] = False
            self.simulation_properties['MC_ready'] = False
            self.request_set_simulation_properties.emit(self.simulation_properties, True, False)

        else:
            self.request_set_simulation_properties.emit(props_dict, False, False)

        self.simulation_properties = None
        self.simulation_arrays = None
        self.material_list = None
        self.mc_calculation_finished.emit()
        self.request_save = True


class RunManager(QtCore.QObject):
    mc_calculation_running = QtCore.pyqtSignal(bool)
    def __init__(self, interface, parent=None):
        super().__init__(parent)
        self.runner = Runner()
        interface.send_MC_ready_simulation.connect(self.run_simulation)
        self.runner.mc_calculation_finished.connect(interface.request_MC_ready_simulation)
        self.runner.request_set_simulation_properties.connect(interface.set_simulation_properties)

        self.runner.finished.connect(self.run_finished)
        self.runner.started.connect(self.run_started)

    @QtCore.pyqtSlot()
    def run_started(self):
        self.mc_calculation_running.emit(True)

    @QtCore.pyqtSlot()
    def run_finished(self):
        self.mc_calculation_running.emit(False)

    @QtCore.pyqtSlot(dict, dict, list)
    def run_simulation(self, props, arrays, mat_list):
        logger.debug('Attemp to start MC thread')
        if not self.runner.isRunning():
            self.runner.simulation_properties = props
            self.runner.simulation_arrays = arrays
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
            self.request_data_list.connect(interface.emit_simulation_list)
            self.request_copy_elements.connect(interface.copy_simulation)
            if importer:
                self.request_import_dicom.connect(importer.import_urls)
        elif materials:
            self.request_data_list.connect(interface.emit_material_list)


        # inbound signals
        if simulations:
            interface.send_simulation_list.connect(self.recive_data_list)
        elif materials:
            interface.send_material_list.connect(self.recive_data_list)

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

class PropertiesEditModelItem(QtGui.QStandardItem):
    def __init__(self, key, value):
        super().__init__()
        self.key = key
        self.value = None

        init_val, dtype, volatile, editable, description, order = PROPETIES_DICT_TEMPLATE[key]
        self.dtype = dtype
        self.setEditable(editable)

        if dtype.type is np.bool_:
            self.setCheckable(True)
        self.update_data(value)




    def update_data(self, value):
        self.value = value
        if len(self.dtype.shape) > 0:
            val_t = " ".join([str(val) for val in value.astype(self.dtype.base.type)])
            self.setData(val_t, QtCore.Qt.DisplayRole)
        else:
            if self.dtype.type is np.bool_:
                if value:
                    c_val = QtCore.Qt.Checked
                else:
                    c_val = QtCore.Qt.Unchecked
                self.setData(c_val, QtCore.Qt.CheckStateRole)
            else:
                self.setData(value, QtCore.Qt.DisplayRole)


class PropertiesEditModel(QtGui.QStandardItemModel):
    request_properties_from_database = QtCore.pyqtSignal(str)
    request_update_properties_to_database = QtCore.pyqtSignal(dict)
    request_write_properties_to_database = QtCore.pyqtSignal(dict, bool, bool)
    has_unsaved_changes = QtCore.pyqtSignal(bool)
    def __init__(self, database_interface, simulation_list_model, parent=None):
        super().__init__(parent)
        self.current_simulation = ""
        database_interface.send_view_sim_propeties.connect(self.set_simulation_properties)
        self.request_properties_from_database.connect(database_interface.request_simulation_properties)
        self.request_write_properties_to_database.connect(database_interface.set_simulation_properties)

        simulation_list_model.request_viewing.connect(self.set_simulation)

        self.validator = Validator()
        self.unsaved_items = {}

        propeties_dict, array_dict = self.validator.get_data()
        row = 0
        for key, value in propeties_dict.items():
            self.setItem(row, 0, QtGui.QStandardItem(PROPETIES_DICT_TEMPLATE[key][4]))
            self.setItem(row, 1, PropertiesEditModelItem(key, value))
            row += 1

    @QtCore.pyqtSlot(str)
    def set_simulation(self, name):
        self.current_simulation = name
        self.request_properties_from_database.emit(self.current_simulation)

    @QtCore.pyqtSlot(dict)
    def set_simulation_properties(self, data_dict):
        if data_dict['name'] != self.current_simulation:
            return
        self.validator.set_data(props=data_dict, reset=True)
        self.unsaved_items = {}
        for row in range(self.rowCount()):
            item = self.item(row, 1)
            item.update_data(self.validator._props[item.key])
        self.test_unsaved_changes()

    def test_unsaved_changes(self):
        keys_for_deletion = []
        for key, value in self.unsaved_items.items():
            if isinstance(value, np.ndarray):
                if np.sum(np.nonzero(value - getattr(self.validator, key))) == 0:
                    keys_for_deletion.append(key)
            else:
                if (value - getattr(self.validator, key)) == 0:
                    keys_for_deletion.append(key)
        for key in keys_for_deletion:
            del self.unsaved_items[key]

        self.has_unsaved_changes.emit(len(self.unsaved_items) > 0)


    def setData(self, index, value, role):
        if not index.isValid():
            return False
        if index.column() == 0:
            return super().setData(index, value, role)

        if role in [QtCore.Qt.DisplayRole, QtCore.Qt.EditRole, QtCore.Qt.CheckStateRole]:
            item = self.itemFromIndex(index)
            if not self.validator._pt[item.key][3]:

                return False
            if role == QtCore.Qt.CheckStateRole:
                value = value == QtCore.Qt.Checked

            try:
                setattr(self.validator, item.key, value)
            except AssertionError:
                return False
            else:
                if item.key not in self.unsaved_items:

                    self.unsaved_items[item.key] = item.value
                item.update_data(getattr(self.validator, item.key))
                self.test_unsaved_changes()
            return True
        return super().setData(index, value, role)

    @QtCore.pyqtSlot()
    def apply_changes(self):
        self.request_write_properties_to_database.emit(self.validator._props, True, True)

    @QtCore.pyqtSlot()
    def reset_changes(self):
        self.validator.set_data(self.unsaved_items, reset=False)
        self.set_simulation_properties(self.validator.get_data()[0])



class PropertiesEditWidget(QtGui.QWidget):
    def __init__(self, database_interface, simulation_list_model, parent=None):
        super().__init__(parent)
        layout = QtGui.QVBoxLayout()
        table = QtGui.QTableView()
        model = PropertiesEditModel(database_interface, simulation_list_model)
        table.setModel(model)
        layout.addWidget(table)
        sub_layout = QtGui.QHBoxLayout()
        layout.addLayout(sub_layout)
        apply_button = QtGui.QPushButton()
        reset_button = QtGui.QPushButton()
        run_button = QtGui.QPushButton()
        apply_button.setText('Apply')
        reset_button.setText('Reset')
        run_button.setText('Run')
        sub_layout.addWidget(reset_button)
        sub_layout.addWidget(apply_button)
        sub_layout.addWidget(run_button)
        layout.setContentsMargins(0, 0, 0, 0)
        sub_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)


        reset_button.clicked.connect(model.reset_changes)
        apply_button.clicked.connect(model.apply_changes)
        model.has_unsaved_changes.connect(reset_button.setEnabled)
        model.has_unsaved_changes.connect(apply_button.setEnabled)






#class PropertiesModel(QtCore.QAbstractTableModel):
#    request_update_simulation = QtCore.pyqtSignal(dict, dict, bool, bool)
#    unsaved_data_changed = QtCore.pyqtSignal(bool)
#    properties_is_set = QtCore.pyqtSignal(bool)
#
#    def __init__(self, interface, parent=None):
#        super().__init__(parent)
#        self.__data = copy.copy(SIMULATION_DESCRIPTION)
#        self.unsaved_data = {}
#        self.__indices = list(self.__data.keys())
#        self.__indices.sort()
#        interface.request_simulation_view.connect(self.set_data)
#        interface.simulation_updated.connect(self.update_data)
#        self.request_update_simulation.connect(interface.update_simulation_properties)
#        self.__simulation = Simulation('None')
#
#
#    def properties_data(self):
#        return self.__data, self.__indices
#
#    @QtCore.pyqtSlot()
#    def reset_properties(self):
#        self.unsaved_data = {}
#        self.dataChanged.emit(self.createIndex(0,0), self.createIndex(len(self.__indices)-1 , 1))
#        self.test_for_unsaved_changes()
#
#    @QtCore.pyqtSlot()
#    def apply_properties(self):
#        self.__init_data = self.__data
#        self.unsaved_data['name'] = self.__data['name'][0]
#        self.unsaved_data['MC_ready'] = True
#        self.unsaved_data['MC_finished'] = False
#        self.unsaved_data['MC_running'] = False
#        self.test_for_unsaved_changes()
#        self.request_update_simulation.emit(self.unsaved_data, {}, True, True)
#        self.properties_is_set.emit(True)
##        self.request_simulation_update.emit({key: value[0] for key, value in self.__data.items()})
#        self.unsaved_data = {}
#        self.test_for_unsaved_changes()
#
##    @QtCore.pyqtSlot()
##    def run_simulation(self):
##        self.__data['MC_running'][0] = True
##        self.__data['MC_ready'][0] = True
###        self.request_simulation_update.emit({key: value[0] for key, value in self.__data.items()})
##        self.unsaved_data_changed.emit(False)
###        self.request_simulation_start.emit()
#
#    def test_for_unsaved_changes(self):
#        for key, value in self.__simulation.description.items():
#            if self.__data[key][3]:
#                if isinstance(self.__data[key][0], np.ndarray):
#                    if (value - self.__data[key][0]).sum() != 0.0:
#                        self.unsaved_data[key] = value
#                elif self.__data[key][0] != value:
#                    self.unsaved_data[key] = value
#        self.unsaved_data_changed.emit(len(self.unsaved_data) > 0)
#        self.layoutAboutToBeChanged.emit()
#        self.layoutChanged.emit()
#
#    @QtCore.pyqtSlot(Simulation)
#    def set_data(self, sim):
#        sim_description = sim.description
#        self.update_data(sim_description, {})
#
#    @QtCore.pyqtSlot(dict, dict)
#    def update_data(self, sim_description, array_dict):
#        self.unsaved_data = {}
#        self.layoutAboutToBeChanged.emit()
#        self.__simulation = Simulation('None', sim_description)
#        for key, value in sim_description.items():
#            self.__data[key][0] = value
#
#        self.dataChanged.emit(self.createIndex(0,0), self.createIndex(len(self.__indices)-1 , 1))
#        self.layoutChanged.emit()
#        self.test_for_unsaved_changes()
#        self.properties_is_set.emit(self.__data['MC_running'][0])
#
#    def rowCount(self, index):
#        if not index.isValid():
#            return len(self.__data)
#        return 0
#
#    def columnCount(self, index):
#        if not index.isValid():
#            return 2
#        return 0
#
#    def data(self, index, role):
#        if not index.isValid():
#            return None
#        row = index.row()
#        column = index.column()
#
#        var = self.__indices[row]
#        if column == 0:
#            value = self.__data[var][4]
#        else:
#            value = self.unsaved_data.get(var, self.__data[var][0])
#
#        if role == QtCore.Qt.DisplayRole:
#            if (column == 1) and isinstance(value, np.ndarray):
#                return ' '.join([str(round(p, 3)) for p in value])
#            elif (column == 1) and isinstance(value, bool):
#                return ''
#            return value
#        elif role == QtCore.Qt.DecorationRole:
#            pass
#        elif role == QtCore.Qt.ToolTipRole:
#            pass
#        elif role == QtCore.Qt.BackgroundRole:
#            if not self.__data[var][3] and index.column() == 1:
#                return QtGui.qApp.palette().brush(QtGui.qApp.palette().Window)
#        elif role == QtCore.Qt.ForegroundRole:
#            pass
#        elif role == QtCore.Qt.CheckStateRole:
#            if (column == 1) and isinstance(value, bool):
#                if value:
#                    return QtCore.Qt.Checked
#                else:
#                    return QtCore.Qt.Unchecked
#        return None
#
#    def setData(self, index, value, role):
#        if not index.isValid():
#            return False
#        if index.column() != 1:
#            return False
#        if role == QtCore.Qt.DisplayRole or role == QtCore.Qt.EditRole:
##            var = self.__indices[index.row()]
##            self.unsaved_data[var] = value
##            self.dataChanged.emit(index, index)
##            return True
##        elif role == QtCore.Qt.EditRole:
#            var = self.__indices[index.row()]
#            try:
#                setattr(self.__simulation, var, value)
#            except Exception as e:
#                logger.error(str(e))
#                return False
#            else:
#                if value != self.__data[var][0]:
#                    self.unsaved_data[var] = value
#                else:
#                    try:
#                        del self.unsaved_data[var]
#                    except KeyError:
#                        pass
#
#            self.dataChanged.emit(index, index)
#            self.test_for_unsaved_changes()
#            return True
#        elif role == QtCore.Qt.CheckStateRole:
#            var = self.__indices[index.row()]
#            if self.__data[var][0] != bool(value == QtCore.Qt.Checked):
#                self.unsaved_data[var] = bool(value == QtCore.Qt.Checked)
#            else:
#                if var in self.unsaved_data:
#                    del self.unsaved_data[var]
#            self.test_for_unsaved_changes()
#            self.dataChanged.emit(index, index)
#            return True
#
#    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
#        return str(section)
#
#    def flags(self, index):
#        if index.isValid():
#            if self.__data[self.__indices[index.row()]][3] and index.column() == 1:
#                if self.unsaved_data.get('MC_running', False):
#                    return QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled
#                if isinstance(self.__data[self.__indices[index.row()]][0], bool):
#                    return  QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsUserCheckable# | QtCore.Qt.ItemIsEditable
#                return QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsEditable
#            return QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled
#        return QtCore.Qt.NoItemFlags
#
#class ArrayEdit(QtGui.QLineEdit):
#    def __init__(self, parent=None):
#        super().__init__(parent)
#
#    def set_data(self, value):
#        self.setText(' '.join([str(r) for r in value]))
#
#
#class LineEdit(QtGui.QLineEdit):
#    def __init__(self, parent=None):
#        super().__init__(parent)
#
#    def set_data(self, value):
#        self.setText(str(value))
#
#class IntSpinBox(QtGui.QSpinBox):
#    def __init__(self, parent=None):
#        super().__init__(parent)
#        self.setRange(-1e9, 1e9)
#
#    def set_data(self, value):
#        self.setValue(int(value))
#
#
#class DoubleSpinBox(QtGui.QDoubleSpinBox):
#    def __init__(self, parent=None):
#        super().__init__(parent)
#        self.setRange(-1e9, 1e9)
#
#    def set_data(self, value):
#        self.setValue(float(value))
#
#class CheckBox(QtGui.QCheckBox):
#    def __init__(self, parent=None):
#        super().__init__(parent)
#
#    def set_data(self, value):
#        self.setChecked(bool(value))
#
#
#class PropertiesDelegate(QtGui.QItemDelegate):
#    def __init__(self, parent=None):
#        super().__init__(parent)
#
#    def createEditor(self, parent, option, index):
#        data , ind= index.model().properties_data()
#        var = ind[index.row()]
#        if data[var][1] is np.bool:
##            return CheckBox(parent)
#            return None
#        elif data[var][1] is np.double:
#            return DoubleSpinBox(parent)
#        elif data[var][1] is np.int:
#            return IntSpinBox(parent)
#        elif isinstance(data[var][0], np.ndarray):
#            return ArrayEdit(parent)
#        return None
#
#    def setEditorData(self, editor, index):
#        data, ind= index.model().properties_data()
#        var = ind[index.row()]
#        editor.set_data(data[var][0])
##        if isinstance(editor, QtGui.QCheckBox):
##            editor.setChecked(data[var][0])
##        elif isinstance(editor, QtGui.QSpinBox) or isinstance(editor, QtGui.QDoubleSpinBox):
##            editor.setValue(data[var][0])
##        elif isinstance(editor, QtGui.QTextEdit):
##            editor.setText(data[var][0])
###        self.setProperty('bool', bool)
##        factory = QtGui.QItemEditorFactory()
##        print(factory.valuePropertyName(QtCore.QVariant.Bool))
##
###        factory.registerEditor(QtCore.QVariant.Bool, QtGui.QCheckBox())
##        self.setItemEditorFactory(factory)
###        self.itemEditorFactory().setDefaultFactory(QtGui.QItemEditorFactory())
#
#class PropertiesView(QtGui.QTableView):
#    def __init__(self, properties_model, parent=None):
#        super().__init__(parent)
#        self.setModel(properties_model)
#        self.setItemDelegateForColumn(1, PropertiesDelegate())
#
#        self.setWordWrap(False)
##        self.setTextElideMode(QtCore.Qt.ElideMiddle)
##        self.verticalHeader().setResizeMode(0, QtGui.QHeaderView.ResizeToContents)
#        self.horizontalHeader().setResizeMode(0, QtGui.QHeaderView.ResizeToContents)
##        self.horizontalHeader().setMinimumSectionSize(-1)
#        self.horizontalHeader().setResizeMode(1, QtGui.QHeaderView.Stretch)
#        self.verticalHeader().setResizeMode(QtGui.QHeaderView.Stretch)
#
#    def resizeEvent(self, ev):
##        self.resizeColumnsToContents()
##        self.resizeRowsToContents()
#        super().resizeEvent(ev)
#
#class PropertiesWidget(QtGui.QWidget):
#    def __init__(self, properties_model, parent=None):
#        super().__init__(parent)
#        self.setLayout(QtGui.QVBoxLayout())
#        self.layout().setContentsMargins(0, 0, 0, 0)
#        view = PropertiesView(properties_model)
#        self.layout().addWidget(view)
#
#        apply_button = QtGui.QPushButton()
#        apply_button.setText('Reset')
#        apply_button.clicked.connect(properties_model.reset_properties)
#        apply_button.setEnabled(False)
#        properties_model.unsaved_data_changed.connect(apply_button.setEnabled)
#
#        run_button = QtGui.QPushButton()
#        run_button.setText('Apply and Run')
#        run_button.clicked.connect(properties_model.apply_properties)
#        properties_model.properties_is_set.connect(run_button.setDisabled)
#
#
##        run_button = QtGui.QPushButton()
##        run_button.setText('Run')
##        run_button.clicked.connect(properties_model.request_simulation_start)
#
#        button_layout = QtGui.QHBoxLayout()
#        button_layout.setContentsMargins(0, 0, 0, 0)
#        button_layout.addWidget(apply_button)
#        button_layout.addWidget(run_button)
#
#        self.layout().addLayout(button_layout)
