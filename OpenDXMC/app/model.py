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
        self.timer.start(1000 * 60 * 10)
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


class PropertiesModel(QtCore.QAbstractTableModel):
    error_setting_value = QtCore.pyqtSignal(str)
    request_simulation_update = QtCore.pyqtSignal(dict)
    request_simulation_start = QtCore.pyqtSignal()
    unsaved_data_changed = QtCore.pyqtSignal(bool)
    
    def __init__(self, interface, parent=None):
        super().__init__(parent)
        self.__data = copy.copy(SIMULATION_DESCRIPTION)
        self.__init_data = copy.copy(self.__data)
        self.__indices = list(self.__data.keys())
        self.__indices.sort()
        interface.request_simulation_view.connect(self.update_data)
        self.request_simulation_update.connect(interface.update_simulation_properties)
        self.request_simulation_start.connect(interface.get_run_simulation)
        self.__simulation = Simulation('None')
        

    def properties_data(self):
        return self.__data, self.__indices

    @QtCore.pyqtSlot()
    def apply_properties(self):
        self.__data['MC_ready'][0] = True
        self.__init_data = self.__data
        self.request_simulation_update.emit({key: value[0] for key, value in self.__data.items()})
        self.unsaved_data_changed.emit(False)

    @QtCore.pyqtSlot()
    def run_simulation(self):
        self.__data['MC_running'][0] = True
        self.__data['MC_ready'][0] = True
        self.request_simulation_update.emit({key: value[0] for key, value in self.__data.items()})
        self.unsaved_data_changed.emit(False)        
        self.request_simulation_start.emit()        
        
    def test_for_unsaved_changes(self):
        for key, value in self.__data.items():
            if self.__init_data[key] != value:
                self.unsaved_data_changed.emit(True)
                return
        self.unsaved_data_changed.emit(False)

    @QtCore.pyqtSlot(Simulation)
    def update_data(self, sim):
        print('Viewed')
        self.layoutAboutToBeChanged.emit()
        for key, value in sim.description.items():
            self.__data[key][0] = value
        self.dataChanged.emit(self.createIndex(0,0), self.createIndex(len(self.__indices)-1 , 1))
        self.layoutChanged.emit()
        self.unsaved_data_changed.emit(False)

    def rowCount(self, index):
        if not index.isValid():
            return len(self.__data)
        return 0

    def columnCount(self, index):
        if not index.isValid():
            return 2
        return 0

    def data(self, index, role):
        if not index.isValid():
            return None
        row = index.row()
        column = index.column()
        if column == 0:
            pos = 4
        elif column == 1:
            pos = 0
        else:
            return None

        var = self.__indices[row]

        if role == QtCore.Qt.DisplayRole:
            if (column == 1) and isinstance(self.__data[var][0], np.ndarray):
                return ', '.join([str(round(p, 3)) for p in self.__data[var][pos]])
            elif (column == 1) and isinstance(self.__data[var][0], bool):
                return ''
            return self.__data[var][pos]
        elif role == QtCore.Qt.DecorationRole:
            pass
        elif role == QtCore.Qt.ToolTipRole:
            pass
        elif role == QtCore.Qt.BackgroundRole:
            if not self.__data[var][3] and index.column() == 1:
                return QtGui.qApp.palette().brush(QtGui.qApp.palette().Window)
        elif role == QtCore.Qt.ForegroundRole:
            pass
        elif role == QtCore.Qt.CheckStateRole:
            if (column == 1) and isinstance(self.__data[var][0], bool):
                if self.__data[var][0]:
                    return QtCore.Qt.Checked
                else:
                    return QtCore.Qt.Unchecked
        return None

    def setData(self, index, value, role):
        if not index.isValid():
            return False
        if index.column() != 1:
            return False
        if role == QtCore.Qt.DisplayRole:
            self.__data[self.__indices[index.row()]][0] = value

            self.dataChanged.emit(index, index)
            return True
        elif role == QtCore.Qt.EditRole:
            var = self.__indices[index.row()]
            try:
                setattr(self.__simulation, var, value)
            except Exception as e:
                self.error_setting_value.emit(str(e))
                return False
            else:
                self.__data[var][0] = value
            self.dataChanged.emit(index, index)
            self.test_for_unsaved_changes()
            return True
        elif role == QtCore.Qt.CheckStateRole:
            self.__data[self.__indices[index.row()]][0] = bool(value == QtCore.Qt.Checked)
            self.test_for_unsaved_changes()
            self.dataChanged.emit(index, index)
            return True

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        return str(section)

    def flags(self, index):
        if index.isValid():
            if self.__data[self.__indices[index.row()]][3] and index.column() == 1:
                if isinstance(self.__data[self.__indices[index.row()]][0], bool):
                    return  QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsUserCheckable# | QtCore.Qt.ItemIsEditable
                return QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsEditable
            return QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled
        return QtCore.Qt.NoItemFlags


class LineEdit(QtGui.QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
     
    def set_data(self, value):
        self.setText(str(value))

class IntSpinBox(QtGui.QSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRange(-1e9, 1e9)

    def set_data(self, value):
        self.setValue(int(value))


class DoubleSpinBox(QtGui.QDoubleSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRange(-1e9, 1e9)
        
    def set_data(self, value):
        self.setValue(float(value))

class CheckBox(QtGui.QCheckBox):
    def __init__(self, parent=None):
        super().__init__(parent)

    def set_data(self, value):
        self.setChecked(bool(value))


class PropertiesDelegate(QtGui.QItemDelegate):
    def __init__(self, parent=None):
        super().__init__(parent)

    def createEditor(self, parent, option, index):
        data , ind= index.model().properties_data()
        var = ind[index.row()]
        if data[var][1] is np.bool:
#            return CheckBox(parent)
            return None
        elif data[var][1] is np.double:
            return DoubleSpinBox(parent)
        elif data[var][1] is np.int:
            return IntSpinBox(parent)
        return None
    
    def setEditorData(self, editor, index):
        data, ind= index.model().properties_data()
        var = ind[index.row()]
        editor.set_data(data[var][0])
#        if isinstance(editor, QtGui.QCheckBox):
#            editor.setChecked(data[var][0])
#        elif isinstance(editor, QtGui.QSpinBox) or isinstance(editor, QtGui.QDoubleSpinBox):
#            editor.setValue(data[var][0])
#        elif isinstance(editor, QtGui.QTextEdit):
#            editor.setText(data[var][0])
##        self.setProperty('bool', bool)
#        factory = QtGui.QItemEditorFactory()
#        print(factory.valuePropertyName(QtCore.QVariant.Bool))
#
##        factory.registerEditor(QtCore.QVariant.Bool, QtGui.QCheckBox())
#        self.setItemEditorFactory(factory)
##        self.itemEditorFactory().setDefaultFactory(QtGui.QItemEditorFactory())

class PropertiesView(QtGui.QTableView):
    def __init__(self, interface, parent=None):
        super().__init__(parent)
        self.setModel(PropertiesModel(interface))
        self.setItemDelegateForColumn(1, PropertiesDelegate())
        
        self.setWordWrap(False)
#        self.setTextElideMode(QtCore.Qt.ElideMiddle)
#        self.verticalHeader().setResizeMode(0, QtGui.QHeaderView.ResizeToContents)
        self.horizontalHeader().setResizeMode(0, QtGui.QHeaderView.ResizeToContents)
#        self.horizontalHeader().setMinimumSectionSize(-1)
        self.horizontalHeader().setResizeMode(1, QtGui.QHeaderView.Stretch)
        self.verticalHeader().setResizeMode(QtGui.QHeaderView.Stretch)

    def resizeEvent(self, ev):
#        self.resizeColumnsToContents()
#        self.resizeRowsToContents()
        super().resizeEvent(ev)
        
class PropertiesWidget(QtGui.QWidget):
    def __init__(self, interface, parent=None):
        super().__init__(parent)
        self.setLayout(QtGui.QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        view = PropertiesView(interface)
        model = view.model()
        self.layout().addWidget(view)

        apply_button = QtGui.QPushButton()
        apply_button.setText('Apply')
        apply_button.clicked.connect(model.apply_properties)
        apply_button.setEnabled(False)
        model.unsaved_data_changed.connect(apply_button.setDisabled)
                
        
        run_button = QtGui.QPushButton()    
        run_button.setText('Run')
        run_button.clicked.connect(model.request_simulation_start)        
        
        button_layout = QtGui.QHBoxLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.addWidget(apply_button)
        button_layout.addWidget(run_button)
        
        self.layout().addLayout(button_layout)
    