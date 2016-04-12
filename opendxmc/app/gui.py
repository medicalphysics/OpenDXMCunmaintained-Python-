# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 11:42:37 2015

@author: erlean
"""
import sys
import os
from PyQt4 import QtGui, QtCore
from opendxmc.app.view import ViewController, RunnerView
from opendxmc.app.model import DatabaseInterface, ListView, ListModel, RunManager, Importer, ImportScalingEdit, PropertiesEditWidget, OrganDoseModel, OrganDoseView
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
        self.setReadOnly(True)

    def closeEvent(self, event):
        self.closed.emit(False)
        super().closeEvent(event)
    
    @QtCore.pyqtSlot(str)
    def insertPlainText(self, txt):
        self.moveCursor(QtGui.QTextCursor.End)
        super().insertPlainText(txt)
        self.ensureCursorVisible()


class SelectDatabaseWidget(QtGui.QWidget):
    set_database_path = QtCore.pyqtSignal(QtCore.QUrl)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setLayout(QtGui.QHBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.setContentsMargins(0, 0, 0, 0)
  
        help_msg = "Location of database file"        
        self.setToolTip(help_msg)        
        
        settings = QtCore.QSettings('OpenDXMC', 'gui')
        if settings.contains('database/path'):
            self.path = QtCore.QUrl.fromLocalFile(settings.value('database/path'))
        else:
            self.path = QtCore.QUrl.fromLocalFile(os.path.join(os.path.dirname(sys.argv[0]), 'database.h5'))
            settings.setValue('database/path', self.path.toLocalFile())
        
        self.applybutton = QtGui.QToolButton()
        self.applybutton.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        self.applybutton.setText('Apply')
        self.applybutton.setContentsMargins(0, 0, 0, 0)
        
        self.txtedit = QtGui.QLineEdit()
        self.txtedit.setToolTip(help_msg)
        self.txtedit.setContentsMargins(0, 0, 0, 0)
        self.txtedit.setText(self.path.toLocalFile())
        completer = QtGui.QCompleter(self)
        self.model = QtGui.QFileSystemModel(completer)
        self.model.setRootPath(os.path.dirname(self.path.toLocalFile()))
        completer.setModel(self.model)
        self.txtedit.setCompleter(completer)
        
        self.layout().addWidget(self.txtedit, 10)
        self.layout().addWidget(self.applybutton, 1)
    
        self.applybutton.clicked.connect(self.apply)    
        self.applybutton.setToolTip(help_msg)
#        self.setMaximumHeight(max([self.txtedit.minimumHeight(), self.applybutton.minimumHeight()]))
        self.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
    
    @QtCore.pyqtSlot()    
    def apply(self):
        self.path = QtCore.QUrl.fromLocalFile(self.txtedit.text())
        self.set_database_path.emit(self.path)
   
    @QtCore.pyqtSlot(QtCore.QUrl)    
    def validate_apply(self, url):
        self.path = url
        self.txtedit.setText(self.path.toLocalFile())
        settings = QtCore.QSettings('OpenDXMC', 'gui')
        settings.setValue('database/path', self.path.toLocalFile())
        self.model.setRootPath(self.path.toLocalFile())
    
    @QtCore.pyqtSlot(bool)    
    def locked(self, val):
        self.setDisabled(val)
        self.applybutton.setDisabled(val)
    
   
class StatusBarButton(QtGui.QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setFlat(True)
        self.setCheckable(True)


class BusyWidget(QtGui.QWidget):
    def __init__(self, parent=None, tooltip=''):
        super().__init__(parent)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.progress)
        self.timer.timeout.connect(self.update)
        self.progress = 0
        self.pen = QtGui.QPen(QtGui.QBrush(QtCore.Qt.white), 50.,
                              cap=QtCore.Qt.RoundCap)
        self.setLayout(QtGui.QHBoxLayout())
        label = QtGui.QLabel('!')
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.setToolTip(tooltip)
        self.setToolTip(tooltip)
        self.layout().addWidget(label)
        self.layout().setContentsMargins(0, 0, 0, 0)

        self.setMinimumSize(QtGui.qApp.fontMetrics().size(QtCore.Qt.TextSingleLine, 'OpenDXMC'))

        self.setVisible(False)

    @QtCore.pyqtSlot()
    def progress(self):
        self.progress = (self.progress + 64) % 5760

    @QtCore.pyqtSlot(bool)
    def busy(self, start):
        self.setMinimumSize(QtGui.qApp.fontMetrics().size(QtCore.Qt.TextSingleLine, '!!!')*2)
        if start and not self.isVisible():
            self.timer.start(50)
            self.show()
        elif not start:
            self.hide()
            self.timer.stop()

    def paintEvent(self, ev):
        if self.width() > self.height():
            d = self.height()
        else:
            d = self.width()

        rect = QtCore.QRectF(0, 0, d*.70, d*.70)
        rect.moveCenter(QtCore.QPointF(self.rect().center()))

        self.pen.setWidthF(d * .15)
        p = QtGui.QPainter(self)
        p.setRenderHint(p.Antialiasing, True)

        self.pen.setColor(QtGui.QColor.fromHsv((self.progress // 16) % 180 + 0,
                                               255, 255))
        p.setPen(self.pen)
        p.drawArc(rect, self.progress, 960)
        self.pen.setColor(QtGui.QColor.fromHsv((self.progress // 16) % 180 + 180,
                                               255, 255))
        p.setPen(self.pen)
        p.drawArc(rect, self.progress + 2880, 960)


class MainWindow(QtGui.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowIcon(QtGui.QIcon(os.path.join(os.path.dirname(__file__), 'icon.png')))
        self.setWindowIconText('OpenDXMC')
        self.setWindowTitle('OpenDXMC')        
        
        database_busywidget = BusyWidget(tooltip='Writing or Reading to Database')
        simulation_busywidget = BusyWidget(tooltip='Monte Carlo simulation in progress')
        importer_busywidget = BusyWidget(tooltip='Importing DICOM files')
        importer_phantoms_busywidget = BusyWidget(tooltip='Importing digital phantoms')

        mc_progressbar = RunnerView()

        # statusbar
        status_bar = QtGui.QStatusBar()
        statusbar_log_button = StatusBarButton('Log', None)
        status_bar.addPermanentWidget(importer_busywidget)
        status_bar.addPermanentWidget(importer_phantoms_busywidget)
        status_bar.addPermanentWidget(simulation_busywidget)
        status_bar.addPermanentWidget(database_busywidget)
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
        database_selector_widget = SelectDatabaseWidget()
        self.interface = DatabaseInterface(database_selector_widget.path)
        self.interface.database_busy.connect(database_busywidget.busy)
        self.interface.database_busy.connect(database_selector_widget.locked)
        database_selector_widget.set_database_path.connect(self.interface.set_database)
        self.interface.send_proper_database_path.connect(database_selector_widget.validate_apply)



        # importer
        self.importer = Importer(self.interface)
        self.importer.running.connect(importer_busywidget.busy)
        self.importer.running.connect(database_selector_widget.locked)
        self.importer_phantom = Importer(self.interface)
        self.importer_phantom.running.connect(importer_phantoms_busywidget.busy)
        self.importer_phantom.running.connect(database_selector_widget.locked)

        ## import scaling setter
        import_scaling_widget = ImportScalingEdit(self.importer, self)

        self.viewcontroller = ViewController(self.interface)


        ## MC runner
        self.mcrunner = RunManager(self.interface, mc_progressbar)
        self.mcrunner.mc_calculation_running.connect(simulation_busywidget.busy)
        self.mcrunner.mc_calculation_running.connect(database_selector_widget.locked)


        # Models
        self.simulation_list_model = ListModel(self.interface, self.importer, self.importer_phantom, self,
                                               simulations=True)
        simulation_list_view = ListView()
        simulation_list_view.setModel(self.simulation_list_model)
        self.simulation_list_model.request_viewing.connect(self.viewcontroller.set_simulation)



        self.material_list_model = ListModel(self.interface, self,
                                             materials=True)
        self.material_list_model.request_viewing.connect(self.interface.emit_material_for_viewing)
        material_list_view = ListView()
        material_list_view.setModel(self.material_list_model)

        # Widgets

        list_view_collection_widget = QtGui.QWidget()
        list_view_collection_widget.setContentsMargins(0, 0, 0, 0)
        list_view_collection_widget.setLayout(QtGui.QVBoxLayout())
        list_view_collection_widget.layout().setContentsMargins(0, 0, 0, 0)
        list_view_collection_widget.layout().addWidget(import_scaling_widget, 1)
        list_view_collection_widget.layout().addWidget(database_selector_widget, 1)
        list_view_collection_widget.layout().addWidget(simulation_list_view, 300)
        list_view_collection_widget.layout().addWidget(material_list_view, 100)
        central_splitter.addWidget(list_view_collection_widget)

        properties_collection_widget = QtGui.QWidget()
        properties_collection_widget.setContentsMargins(0, 0, 0, 0)
        properties_collection_widget.setLayout(QtGui.QVBoxLayout())
        properties_collection_widget.layout().setContentsMargins(0, 0, 0, 0)
        properties_collection_widget.layout().addWidget(import_scaling_widget, 1)
        
        simulation_editor = PropertiesEditWidget(self.interface, self.simulation_list_model, self.mcrunner)
        self.viewcontroller.set_simulation_editor(simulation_editor.model)
        properties_collection_widget.layout().addWidget(simulation_editor, 3)
        properties_collection_widget.layout().addWidget(mc_progressbar, 1)
        central_splitter.addWidget(properties_collection_widget)

        for wwid in self.viewcontroller.view_widget():
            central_splitter.addWidget(wwid)

        self.organdosemodel = OrganDoseModel(self.interface, self.simulation_list_model)
        organdoseview = OrganDoseView(self.organdosemodel)
        central_splitter.addWidget(organdoseview)

        central_splitter.setSizes([100, 100, 100, 10, 0])

        central_widget.setLayout(central_layout)
        self.setCentralWidget(central_widget)

        # threading
        self.database_thread = QtCore.QThread(self)
        self.interface.moveToThread(self.database_thread)

#        self.mc_thread = QtCore.QThread(self)
#        self.mcrunner.moveToThread(self.mc_thread)

        self.dose_thread = QtCore.QThread(self)
        self.organdosemodel.moveToThread(self.dose_thread)

        self.import_thread = QtCore.QThread(self)
        self.importer.moveToThread(self.import_thread)
        self.import_phantom_thread = QtCore.QThread(self)
        self.importer_phantom.moveToThread(self.import_phantom_thread)
#        self.importer.moveToThread(self.database_thread)

        self.import_thread.start()
        self.import_phantom_thread.start()
#        self.mc_thread.start()
        self.dose_thread.start()
        self.database_thread.start()

        self.mcrunner.runner.finished.emit()


