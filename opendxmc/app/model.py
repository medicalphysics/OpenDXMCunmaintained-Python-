# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 10:10:58 2015

@author: erlean
"""
import numpy as np
from PyQt4 import QtGui, QtCore
from scipy.ndimage.interpolation import affine_transform
from opendxmc.database import Database, PROPETIES_DICT_TEMPLATE, Validator, PROPETIES_DICT_TEMPLATE_GROUPING
from opendxmc.database.import_phantoms import read_phantoms
from opendxmc.database import import_ct_series
from opendxmc.runner import ct_runner
from opendxmc.utils import find_all_files
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
        help_txt = "Select scaling factor per dimension when importing DICOM images"
        self.setWhatsThis(help_txt)
        self.setToolTip(help_txt)
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

class ArrayBuffer(object):
    def __init__(self):
        self.array_name = ""
        self.name = ""
        self.buffer_size = 2  # number of slices to buffer
        self.indices = np.arange(2, dtype=np.int)
        self.slices = np.empty((2,2, 2))
        self.view_orientation = 2
        self.index_threshold_edge = (0, 1)

    def set_buffer(self, slices, name, array_name, indices, orientation):
        self.indices = indices
        self.slices = slices
        self.name = name
        self.array_name = array_name
        self.view_orientation = orientation

        self.buffer_size = slices.shape[orientation]
        self.index_threshold_edge = (indices[0], indices[-1])
        logger.debug('Updating buffer, indices: {}'.format(indices))


    def is_slice_available(self, name, array_name, index, orientation):
        if orientation != self.view_orientation:
            return False, 0
        if name != self.name:
            return False, 0
        if array_name != self.array_name:
            return False, 0
        if index in self.indices:
            if index == self.index_threshold_edge[0]:
                return True, -1
            if index == self.index_threshold_edge[1]:
                return True, 1
            return True, 0
        return False, 0

    def get_slice(self, name, array_name, index, orientation):
        if orientation != self.view_orientation:
            raise ValueError('Not same orientation')
        if name != self.name:
            raise ValueError('Not same name')
        if array_name != self.array_name:
            raise ValueError('Not same array_name')
        if index not in self.indices:
            raise ValueError('Array not in buffer')
        sub_ind = np.nonzero(self.indices == index)
        if self.view_orientation == 0:
            return np.squeeze(self.slices[sub_ind, :, :])
        if self.view_orientation == 1:
            return np.squeeze(self.slices[:, sub_ind, :])
        return np.squeeze(self.slices[:, :, sub_ind])


class DatabaseInterface(QtCore.QObject):
    """ Async database interface, async provided with signal/slots resulting in
    two connections per task wished to be done, ie, signal/slot hell
    """
    database_busy = QtCore.pyqtSignal(bool)
    send_simulation_list = QtCore.pyqtSignal(list)
    send_material_list = QtCore.pyqtSignal(list)
    send_material_for_viewing = QtCore.pyqtSignal(object)
    send_view_array = QtCore.pyqtSignal(str, np.ndarray, str)  # simulation dict, array_slice, array_name, index, orientation
    send_view_array_bytescaled = QtCore.pyqtSignal(str, np.ndarray, str)  # simulation dict, array_slice, array_name, index, orientation
    send_view_array_slice = QtCore.pyqtSignal(str, np.ndarray, str, int, int)  # simulation dict, array_slice, array_name, index, orientation
    send_view_sim_propeties = QtCore.pyqtSignal(dict)
    send_MC_ready_simulation = QtCore.pyqtSignal(dict, dict, list)


    send_proper_database_path = QtCore.pyqtSignal(QtCore.QUrl)

    def __init__(self, database_qurl, parent=None):
        super().__init__(parent)
        self.__db = None
        self.set_database(database_qurl)
        self.array_buffer = ArrayBuffer()
        self.array_buffer_size = 60

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
        for_revert = self.__db
        try:
            self.__db = Database(path.absoluteFilePath())
        except OSError:    
            self.database_busy.emit(False)
            self.__db = for_revert            
            logger.debug('Failed to use database in {0}'.format(path.absoluteFilePath()))
        self.database_busy.emit(False)
        self.emit_material_list()
        self.emit_simulation_list()
        self.send_proper_database_path.emit(QtCore.QUrl.fromLocalFile(path.absoluteFilePath()))


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


    @QtCore.pyqtSlot(str)
    def emit_material_for_viewing(self, mat_name):
        self.database_busy.emit(True)
        try:
            mat = self.__db.get_material(mat_name)
        except ValueError:
            pass
        else:
            self.send_material_for_viewing.emit(mat)
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

    @QtCore.pyqtSlot(str, str, float, float, bool)
    def request_view_array_bytescaled(self, simulation_name, array_name, amin, amax, vals_is_modifier):
        self.database_busy.emit(True)
        try:
            arr = self.__db.get_simulation_array_bytescaled(simulation_name, array_name, amin, amax, vals_is_modifier)
        except ValueError:
            pass
        else:
            self.send_view_array_bytescaled.emit(simulation_name, arr, array_name)
        self.database_busy.emit(False)

    @QtCore.pyqtSlot(str, str, int, int)
    def request_view_array_slice(self, simulation_name, array_name, index, orientation):
        self.database_busy.emit(True)
        buffer_has_index, buffer_direction = self.array_buffer.is_slice_available(simulation_name, array_name, index, orientation)
        if buffer_has_index:
            logger.debug('Buffer has index {0} in {1}'.format(index, array_name))
            arr = self.array_buffer.get_slice(simulation_name, array_name, index, orientation)
            self.send_view_array_slice.emit(simulation_name, arr, array_name, index, orientation)

            if buffer_direction != 0:
                if buffer_direction > 0:
                    indices = np.arange(self.array_buffer_size, dtype=np.int) + index
                else:
                    indices = np.arange(self.array_buffer_size, dtype=np.int) + index - self.array_buffer_size + 1
                try:
                    arr = self.__db.get_simulation_array_slice(simulation_name, array_name, indices, orientation)
                except:
                    pass
                else:
                    logger.debug('Buffer need update, updating indices {0} in {1}'.format(indices, array_name))
                    self.array_buffer.set_buffer(arr, simulation_name, array_name, indices, orientation)

        else:
            try:
                arr = self.__db.get_simulation_array_slice(simulation_name, array_name, index, orientation)
            except:
                pass
            else:
                self.send_view_array_slice.emit(simulation_name, arr, array_name, index, orientation)

            indices = np.arange(self.array_buffer_size, dtype=np.int) + index - self.array_buffer_size // 2

            try:
                arr = self.__db.get_simulation_array_slice(simulation_name, array_name, indices, orientation)
            except:
                pass
            else:
                logger.debug('Buffer need update, updating indices {0} in {1}'.format(indices, array_name))
                self.array_buffer.set_buffer(arr, simulation_name, array_name, indices, orientation)

        self.database_busy.emit(False)

    @QtCore.pyqtSlot(list)
    def copy_simulation(self, names):
        for name in names:
            if isinstance(name, bytes):
                name = str(name, encoding='utf-8')
            self.database_busy.emit(True)
            self.__db.copy_simulation(name)
            self.database_busy.emit(False)
            self.emit_simulation_list()

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


    @QtCore.pyqtSlot(str)
    def remove_simulation(self, name):
        logger.debug('Attempting to remove simulation {}'.format(name))
        self.__db.remove_simulation(name)
        self.emit_simulation_list()



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

    @QtCore.pyqtSlot(list)
    def import_phantoms(self, qurl_list):
        self.running.emit(True)
        paths = [url.toLocalFile() for url in qurl_list]
        for props, arrays in read_phantoms(find_all_files(paths)):
            self.request_add_sim_to_database.emit(props, arrays, False)
        self.running.emit(False)

class Runner(QtCore.QThread):
    request_write_simulation_arrays = QtCore.pyqtSignal(str, dict)
    request_set_simulation_properties = QtCore.pyqtSignal(dict, bool, bool)
    request_runner_view_update = QtCore.pyqtSignal(np.ndarray, float, float, str, bool)

    start_timer = QtCore.pyqtSignal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self.request_save = False
        self.timer = QtCore.QTimer()
        self.timer.setInterval(600000)

        self.started.connect(self.timer.start)
        self.timer.setSingleShot(False)
        self.terminated.connect(self.timer.stop)
        self.finished.connect(self.timer.stop)

        self.timer.timeout.connect(self.set_request_save)
        self.mutex = QtCore.QMutex()
        self.simulation_properties = None
        self.simulation_arrays = None
        self.material_list = None

        self.kill_me = False
        self.is_running=False

    @QtCore.pyqtSlot()
    def set_request_save(self):
        self.mutex.lock()
        self.request_save = True
        self.mutex.unlock()
    @QtCore.pyqtSlot()
    def cancel_run(self):
        self.mutex.lock()
        self.kill_me = True
        self.mutex.unlock()

    def update_simulation_iteration(self, name, array_dict=None, exposure_number=None, progressbar_data=None, save=True):
        if self.kill_me:
            self.request_runner_view_update.emit(np.ones((3, 3)), 1, 1, '', False)
            self.mutex.lock()
            self.kill_me = False
            self.mutex.unlock()
            self.is_running=False
            self.terminated.emit()
            self.terminate()


#        if all([self.request_save, save, array_dict is not None,
#                exposure_number is not None]):
#            desc = {'name': name,
#                'start_at_exposure_no': exposure_number,
#                }
#            self.request_set_simulation_properties.emit(desc, False, False)
#            self.request_write_simulation_arrays.emit(name, array_dict)
#            self.mutex.lock()
#            self.request_save = False
#            self.mutex.unlock()
        if progressbar_data is not None:
            self.request_runner_view_update.emit(*progressbar_data)

    def run(self):
        self.is_running=True
        self.mutex.lock()
        self.kill_me = False





        simulation_properties = self.simulation_properties
        simulation_arrays = self.simulation_arrays
        material_list = self.material_list




        self.mutex.unlock()

        if simulation_properties is None or simulation_arrays is None:
            return
        if material_list is None:
            return

        self.request_set_simulation_properties.emit({'name': simulation_properties['name'],
                                                     'MC_running': True},
                                                     False, False)
        try:
            props_dict, arr_dict = ct_runner(material_list, simulation_properties,
                                             energy_imparted_to_dose_conversion=True,
                                             callback=self.update_simulation_iteration,
                                             **simulation_arrays)
        except MemoryError:
            logger.error('MEMORY ERROR: Could not run simulation {0}, memory to low. Try to increase dose matrix scaling or use 64 bit version of OpenDXMC'.format(simulation_properties['name']))
            simulation_properties['MC_finished'] = False
            simulation_properties['MC_running'] = False
            simulation_properties['MC_ready'] = False
            self.request_set_simulation_properties.emit(simulation_properties, True, False)
        except ValueError or AssertionError as e:
            raise e
            logger.error('UNKNOWN ERROR: Could not run simulation {0}'.format(simulation_properties['name']))
            simulation_properties['MC_finished'] = False
            simulation_properties['MC_running'] = False
            simulation_properties['MC_ready'] = False
            self.request_set_simulation_properties.emit(self.simulation_properties, True, False)
        else:
            self.request_write_simulation_arrays.emit(props_dict['name'], arr_dict)
            # generating doe array, watching memory
            energy_imparted = arr_dict.get('energy_imparted', None)
            density = arr_dict.get('density', None)
            c_factor = props_dict.get('conversion_factor_ctdiair')
            if c_factor == 0:
                c_factor = props_dict.get('conversion_factor_ctdiw')

            if energy_imparted is not None and density is not None:
                if c_factor > 0:
                    del arr_dict
                    try:
                        dose = (energy_imparted * c_factor) / (density * np.prod(props_dict['spacing'] * props_dict['scaling']))
                    except MemoryError:
                        logger.error('Memory error in generating dose array')
                    else:
                        self.request_write_simulation_arrays.emit(props_dict['name'], {'dose': dose})
            self.request_set_simulation_properties.emit(props_dict, False, False)
        self.simulation_properties = None
        self.simulation_arrays = None
        self.material_list = None
        self.request_save = False
        self.is_running=False


class RunManager(QtCore.QObject):
    mc_calculation_running = QtCore.pyqtSignal(bool)
    kill_runner = QtCore.pyqtSignal()
    def __init__(self, interface, progressbar, parent=None):
        super().__init__(parent)
        self.runner = Runner()
        self.kill_runner.connect(self.runner.cancel_run)
        interface.send_MC_ready_simulation.connect(self.run_simulation)
        self.runner.finished.connect(interface.request_MC_ready_simulation)
        self.runner.terminated.connect(interface.request_MC_ready_simulation)
        self.runner.request_set_simulation_properties.connect(interface.set_simulation_properties)
        self.runner.request_write_simulation_arrays.connect(interface.write_simulation_arrays)
        self.runner.finished.connect(self.run_finished)
        self.runner.terminated.connect(self.run_finished)
        self.runner.started.connect(self.run_started)
        self.runner.request_runner_view_update.connect(progressbar.set_data)
        self.current_simulation = ""


    @QtCore.pyqtSlot()
    def run_started(self):
        self.mc_calculation_running.emit(True)

    @QtCore.pyqtSlot(str)
    def cancel_run(self, name):
        if name == self.current_simulation:
            self.kill_runner.emit()
            self.current_simulation = ''

    @QtCore.pyqtSlot()
    def run_finished(self):
        self.current_simulation = ""
        self.mc_calculation_running.emit(False)

    @QtCore.pyqtSlot(dict, dict, list)
    def run_simulation(self, props, arrays, mat_list):
        logger.debug('Attemp to start MC thread')

        if not self.runner.is_running:
            self.runner.simulation_properties = props
            self.runner.simulation_arrays = arrays
            self.runner.material_list = mat_list
            self.current_simulation = props['name']
            self.runner.start()
#            self.runner.run()
            logger.debug('MC thread started')


class ListModel(QtCore.QAbstractListModel):

    request_data_list = QtCore.pyqtSignal()
    request_import_dicom = QtCore.pyqtSignal(list)
    request_import_phantom = QtCore.pyqtSignal(list)
    request_viewing = QtCore.pyqtSignal(str)
    request_copy_elements = QtCore.pyqtSignal(list)
    request_removal = QtCore.pyqtSignal(str)
    def __init__(self, interface, importer=None, phantom_importer=None, parent=None, simulations=False,
                 materials=False):
        super().__init__(parent)
        self.__data = []

        # connecting interface
        # outbound signals
        if simulations:
            self.request_data_list.connect(interface.emit_simulation_list)
            self.request_copy_elements.connect(interface.copy_simulation)
            self.request_removal.connect(interface.remove_simulation)
            if importer:
                self.request_import_dicom.connect(importer.import_urls)
            if phantom_importer:
                self.request_import_phantom.connect(phantom_importer.import_phantoms)
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
        self.__data = sims
        self.layoutChanged.emit()

    @QtCore.pyqtSlot(str)
    def element_activated(self, name):
        self.request_viewing.emit(name)

    @QtCore.pyqtSlot(str)
    def request_removal_emit(self, name):
        self.layoutAboutToBeChanged.emit()
        self.request_removal.emit(name)


    def rowCount(self, index):
        if not index.isValid():
            return len(self.__data)
        return 0

    def data(self, index, role):
        if not index.isValid():
            return None
        row = index.row()
        if row >= len(self.__data):
            return None
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
            self.request_import_phantom.emit(urls)
            logger.debug(' '.join([u.toLocalFile() for u in urls]))
            return True
        elif mimedata.hasText():
            names = mimedata.data('text/plain').split(',')
            self.request_copy_elements.emit([str(n, encoding='ascii') for n in names])
        return False

    def supportedDropActions(self):
        return QtCore.Qt.CopyAction | QtCore.Qt.MoveAction

    def __len__(self):
        return len(self.__data)

class ListView(QtGui.QListView):
    name_activated = QtCore.pyqtSignal(str)
    request_removal = QtCore.pyqtSignal(str)
    def __init__(self, parent=None, simulation=True):
        super().__init__(parent)
        self.tooltip = ""
        if simulation:
            self.setAcceptDrops(True)
            self.viewport().setAcceptDrops(True)
            self.setDropIndicatorShown(True)
            self.setDragDropMode(self.DragDrop)
            self.setDefaultDropAction(QtCore.Qt.CopyAction)
            self.tooltip = 'Drag DiCOM images or digital\nphantoms here to import'
            self.setToolTip(self.tooltip)

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
        self.request_removal.connect(model.request_removal_emit)


    @QtCore.pyqtSlot(QtCore.QModelIndex)
    def activation_name(self, index):
        if index.isValid():
            self.name_activated.emit(index.data())

    def keyPressEvent(self, event):
        super().keyPressEvent(event)
        if event.key() == QtCore.Qt.Key_Delete:
            indices = self.selectedIndexes()
            for ind in indices:
                if ind.isValid():
                    self.request_removal.emit(ind.data())

    def paintEvent(self, event):
        super().paintEvent(event)
        if len(self.model()) == 0:
            painter = QtGui.QPainter(self.viewport())
            f = painter.font()
            f.setItalic(True)
            painter.setFont(f)
            painter.drawText(self.viewport().rect(),QtCore.Qt.AlignCenter | QtCore.Qt.TextWordWrap, self.tooltip)


class PropertiesEditModelItem(QtGui.QStandardItem):
    def __init__(self, key, value):
        super().__init__()
        self.key = key
        self.value = None

        init_val, dtype, volatile, editable, description, order, grouping = PROPETIES_DICT_TEMPLATE[key]
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
    request_write_properties_to_database = QtCore.pyqtSignal(dict, bool, bool)
    has_unsaved_changes = QtCore.pyqtSignal(bool)
    current_simulation_is_running = QtCore.pyqtSignal(bool)
    request_run_simulation = QtCore.pyqtSignal()
    request_cancel_simulation = QtCore.pyqtSignal(str)

    def __init__(self, database_interface, simulation_list_model, parent=None):
        super().__init__(parent)
        self.current_simulation = ""
        database_interface.send_view_sim_propeties.connect(self.set_simulation_properties)
        self.request_properties_from_database.connect(database_interface.request_simulation_properties)
        self.request_write_properties_to_database.connect(database_interface.set_simulation_properties)

        self.request_run_simulation.connect(database_interface.request_MC_ready_simulation)

        simulation_list_model.request_viewing.connect(self.set_simulation)

        self.validator = Validator()
        self.unsaved_items = {}

        propeties_dict, array_dict = self.validator.get_data()
        row = 0
        parent_items = {}

        values = list(PROPETIES_DICT_TEMPLATE_GROUPING.items())
        values.sort(key=lambda x:x[0])

        for key, value in values:
                parent_item = QtGui.QStandardItem(PROPETIES_DICT_TEMPLATE_GROUPING[key])
                parent_item.font().setBold(True)
                parent_item.setEditable(False)
                self.setItem(row, 0, parent_item)
                self.setItem(row, 1, QtGui.QStandardItem(' '))
                row += 1
                parent_items[key] = parent_item


        for key, value in propeties_dict.items():
            parent_ind = PROPETIES_DICT_TEMPLATE[key][6]
            if parent_ind in parent_items.keys():
                parent_item = parent_items[parent_ind]
            else:
                parent_item = QtGui.QStandardItem(PROPETIES_DICT_TEMPLATE_GROUPING.get(parent_ind, 'Unknown'))
                parent_item.font().setBold(True)
                parent_item.setEditable(False)
                self.setItem(row, 0, parent_item)
                self.setItem(row, 1, QtGui.QStandardItem(' '))
                row += 1
                parent_items[parent_ind] = parent_item

            item = QtGui.QStandardItem(PROPETIES_DICT_TEMPLATE[key][4])
            item.setEditable(False)
            parent_item.appendRow([item, PropertiesEditModelItem(key, value)])

        for parent_item in parent_items.values():
            parent_item.sortChildren(0)
#            parent_item.setChild(0, 0, item)
#            parent_item.setChild(0, 1, PropertiesEditModelItem(key, value))
#            self.setItem(row, 0, item)
#            self.setItem(row, 1, PropertiesEditModelItem(key, value))
#            row += 1

    def headerData(self, ind, orient, role):
        if (ind < 2) and (orient == QtCore.Qt.Horizontal):
            return ['Description', 'Value'][ind]
        return super().headerData(ind, orient, role)


    def data_item_iterator(self):
        for row in range(self.rowCount()):
            parent_item = self.item(row, 0)
            for child_row in range(parent_item.rowCount()):
                yield parent_item.child(child_row, 1)

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
        for item in self.data_item_iterator():
            item.update_data(self.validator._props[item.key])
#        for row in range(self.rowCount()):
#            item = self.item(row, 1)
#            item.update_data(self.validator._props[item.key])
        self.test_unsaved_changes()
        self.current_simulation_is_running.emit(self.validator.MC_running)

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

#                for item in [self.item(i, 1) for i in range(self.rowCount())]:
                for item in self.data_item_iterator():
                    item.update_data(getattr(self.validator, item.key))
#
                self.test_unsaved_changes()
            return True
        return super().setData(index, value, role)

    @QtCore.pyqtSlot()
    def run_simulation(self):
        self.validator.MC_ready = True
        self.request_write_properties_to_database.emit(self.validator._props, True, True)
        self.request_run_simulation.emit()

    @QtCore.pyqtSlot()
    def apply_changes(self):
        self.request_write_properties_to_database.emit(self.validator._props, True, True)

    @QtCore.pyqtSlot()
    def reset_changes(self):
        self.validator.set_data(self.unsaved_items, reset=False)
        self.set_simulation_properties(self.validator.get_data()[0])

    @QtCore.pyqtSlot()
    def cancel_MC_run(self):
        if self.current_simulation == "":
            return

        self.validator.MC_ready = False
        self.validator.MC_running = False
        self.request_write_properties_to_database.emit(self.validator.get_data()[0], False, False)
        self.request_cancel_simulation.emit(self.current_simulation)



class PropertiesEditWidget(QtGui.QWidget):
    def __init__(self, database_interface, simulation_list_model, run_manager, parent=None):
        super().__init__(parent)
        layout = QtGui.QVBoxLayout()
        table = QtGui.QTreeView()
#        table.horizontalHeader().setResizeMode(QtGui.QHeaderView.Stretch)
        table.header().setResizeMode(QtGui.QHeaderView.ResizeToContents)
        table.setAnimated(True)
        model = PropertiesEditModel(database_interface, simulation_list_model)
        self.model = model
        model.request_cancel_simulation.connect(run_manager.cancel_run)
        table.setModel(model)

        table.showColumn(0)
        table.showColumn(1)
        layout.addWidget(table)
        sub_layout = QtGui.QHBoxLayout()
        layout.addLayout(sub_layout)
        apply_button = QtGui.QPushButton()
        reset_button = QtGui.QPushButton()
        run_button = QtGui.QPushButton()
        cancel_button = QtGui.QPushButton()
        apply_button.setText('Apply')
        reset_button.setText('Reset')
        run_button.setText('Run')
        cancel_button.setText('Cancel')
        sub_layout.addWidget(reset_button)
        sub_layout.addWidget(apply_button)
        sub_layout.addWidget(run_button)
        sub_layout.addWidget(cancel_button)
        layout.setContentsMargins(0, 0, 0, 0)
        sub_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)


        reset_button.clicked.connect(model.reset_changes)
        cancel_button.clicked.connect(model.cancel_MC_run)
        apply_button.clicked.connect(model.apply_changes)
        run_button.clicked.connect(model.run_simulation)
        model.has_unsaved_changes.connect(reset_button.setEnabled)
        model.has_unsaved_changes.connect(apply_button.setEnabled)
        model.current_simulation_is_running.connect(cancel_button.setEnabled)
        model.current_simulation_is_running.connect(run_button.setDisabled)


class OrganDoseModel(QtCore.QAbstractTableModel):
    request_array = QtCore.pyqtSignal(str, str)
    request_array_slice = QtCore.pyqtSignal(str, str, int, int)
    hide_view = QtCore.pyqtSignal(bool)
    def __init__(self, database_interface, simulation_list_model, parent=None):
        super().__init__(parent)
        self.current_simulation = ""

        self.dose_z_lenght = []

        self.organ_array = None
        self.organ_material_map = None
        self.organ_map = None

        self._data = {}
        self._data_keys = []

        database_interface.send_view_array.connect(self.set_requested_array)
        database_interface.send_view_array_slice.connect(self.reload_slice)
        database_interface.send_view_sim_propeties.connect(self.set_simulation_properties)
        self.request_array.connect(database_interface.request_view_array)
        self.request_array_slice.connect(database_interface.request_view_array_slice)
        simulation_list_model.request_viewing.connect(self.set_simulation)


    def headerData(self, section, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            if section == 0:
                return 'Organ'
            elif section == 1:
                return 'Material'
            elif section == 2:
                return 'Dose [mGy/100mAs]'
        return None

    def data(self, index, role):
        if role == QtCore.Qt.DisplayRole:
            r = index.row()
            c = index.column()
            if r >= len(self._data) or c > 2:
               return None
            if c == 2:
                if self._data[self._data_keys[r]][c+1] == 0:
                    return None
                return str(round(self._data[self._data_keys[r]][c] / self._data[self._data_keys[r]][c+1], 2))
            return self._data[self._data_keys[r]][c]
        return None
    def sort(self, column, order):
        self.layoutAboutToBeChanged.emit()
        reverse = True if order == QtCore.Qt.DescendingOrder else False
        if column in [0, 1]:
            self._data_keys.sort(key=lambda x: self._data[x][column], reverse=reverse)
        elif column == 2:
            self._data_keys.sort(key=lambda x: self._data[x][column]/self._data[x][column+1], reverse=reverse)
        self.layoutChanged.emit()

    def rowCount(self, index):
        return len(self._data)
    def columnCount(self, index):
        return 3

    @QtCore.pyqtSlot(str)
    def set_simulation(self, name):
        self.layoutAboutToBeChanged.emit()
        self.organ_array = None
        self.organ_material_map = None
        self.organ_map = None
        self._data = {}
        self._data_keys = []
        self.dose_z_lenght = []
        self.current_simulation = name
        self.layoutChanged.emit()

    @QtCore.pyqtSlot(dict)
    def set_simulation_properties(self, props_dict):
        self.hide_view.emit(True)
        if props_dict.get('name', "") != self.current_simulation:
            return
        self.scale = props_dict.get('scaling', np.ones(3))
        self.organ_array = None
        self.organ_material_map = None
        self.organ_map = None
        self.request_array.emit(self.current_simulation, 'organ_map')
        self.request_array.emit(self.current_simulation, 'organ_material_map')
        self.request_array.emit(self.current_simulation, 'organ')

    @QtCore.pyqtSlot(str, np.ndarray, str)
    def set_requested_array(self, name, array, array_name):
        if name != self.current_simulation:
            return
        if array_name == 'organ':
            self.organ_array = affine_transform(array,
                                 self.scale,
                                 output_shape=np.floor(np.array(array.shape)/self.scale).astype(np.int),
                                 cval=0, output=np.uint8, prefilter=True,
                                 order=0).astype(np.uint8)
        elif array_name == 'organ_map':
            self.organ_map = {array['organ'][i]: str(array['organ_name'][i], encoding='utf-8') for i in range(len(array))}
        elif array_name == 'organ_material_map':
            self.organ_material_map = {array['organ'][i]: str(array['material_name'][i], encoding='utf-8') for i in range(len(array))}

        if self.organ_array is not None and self.organ_map is not None and self.organ_material_map is not None:
            self.layoutAboutToBeChanged.emit()
            self._data = {}
            self._data_keys = []
            for key, value in self.organ_map.items():
                self._data[key] = [value, self.organ_material_map.get(key, 'Unknown'), 0, 0]
                self._data_keys.append(key)
            self.dose_z_lenght = list(range(self.organ_array.shape[0]))
#            self.dataChanged.emit(self.index(0, 0), self.index(len(self._data), 2))
            self.request_array_slice.emit(self.current_simulation, 'dose', 0, 0)
            self.layoutChanged.emit()
            self.hide_view.emit(False)


    @QtCore.pyqtSlot(str, np.ndarray, str, int, int)
    def reload_slice(self, name, arr, array_name, index, orientation):
        if orientation != 0 or array_name != 'dose' or name != self.current_simulation:
            return
        if self.organ_array is None:
            return
        if index in self.dose_z_lenght:
            for organ in np.unique(self.organ_array[index, :, :]):
                if organ not in self._data:
                    continue
#                self.layoutAboutToBeChanged.emit()
                ind_x, ind_y = np.nonzero(self.organ_array[index, :, :] == organ)
                self._data[organ][2] += arr[ind_x, ind_y].sum()
                self._data[organ][3] += len(ind_x)
#                import pdb
#                pdb.set_trace()
                model_index = self.index(self._data_keys.index(organ), 2)
#                self.layoutChanged.emit()
                self.dataChanged.emit(model_index, model_index)

            self.dose_z_lenght.remove(index)
        if len(self.dose_z_lenght) > 0:
            self.request_array_slice.emit(self.current_simulation, 'dose', self.dose_z_lenght[0], 0)


class OrganDoseView(QtGui.QTableView):
    def __init__(self, model, parent=None):
        super().__init__(parent)
        self.setModel(model)
        self.model().hide_view.connect(self.setHidden)
        self.horizontalHeader().setResizeMode(QtGui.QHeaderView.Stretch)
#        self.horizontalHeader().setVisible(True)
        self.setSortingEnabled(True)
        self.setDragEnabled(True)
        self.setAlternatingRowColors(True)
        self.setHidden(True)


    def copy_to_clipboard(self):
        indices = self.selectedIndexes()
        n_columns = self.model().columnCount(0)
        indices.sort(key = lambda x: x.row()*n_columns + x.column())

        c_row=0
        c_column=0

        html = "<table><tr>"
        txt = ""
        for index in indices:
            if index.isValid():
                if index.row() > c_row:
                    html += "</tr>"+"<tr>"
                    txt+="\n"
                if index.column() > c_column:
                    txt += "; "
                txt += str(index.data())
                html += "<td>" + str(index.data()) +"</td>"
                c_row = index.row()
                c_column = index.column()
        html += "</tr></table>"
        if len(html) > 0:
            mime=QtCore.QMimeData()
            mime.setHtml(html)
            mime.setText(txt)
            QtGui.qApp.clipboard().setMimeData(mime)


    def keyPressEvent(self, event):
        if event.matches(QtGui.QKeySequence.Copy):
            self.copy_to_clipboard()
            event.accept()
            return
        super().keyPressEvent(event)

