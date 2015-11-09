# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 10:38:15 2015

@author: erlean
"""

import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import interp1d, RegularGridInterpolator
from PyQt4 import QtGui, QtCore
from .dicom_lut import get_lut
from opendxmc.app.ffmpeg_writer import FFMPEG_VideoWriter
import logging
logger = logging.getLogger('OpenDXMC')


class SceneSelectGroup(QtGui.QActionGroup):
    scene_selected = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setExclusive(True)
        self.triggered.connect(self.relay_clicked)

    def addAction(self, name, pretty_name=None):
        if pretty_name is None:
            pretty_name = name
        action = super().addAction(pretty_name)
        action.scene_name = name

    @QtCore.pyqtSlot(QtGui.QAction)
    def relay_clicked(self, action):
        self.scene_selected.emit(action.scene_name)

    @QtCore.pyqtSlot(str)
    def sceneSelected(self, name):
        for action in self.actions():
            if action.scene_name == name:
                action.setChecked()
                return

class ViewController(QtCore.QObject):
    request_metadata = QtCore.pyqtSignal(str)
    request_array_slice = QtCore.pyqtSignal(str, str, int, int)

    def __init__(self, database_interface, parent=None):
        super().__init__(parent)

        self.current_simulation = " "
        self.current_index = 0
        self.current_view_orientation = 2
        self.current_scene = 'planning'

        self.graphicsview = View()
        self.graphicsview.request_array.connect(database_interface.request_view_array)
        database_interface.send_view_array.connect(self.graphicsview.cine_film_creation)

        self.scenes = {'planning': PlanningScene(),
                       'running':  RunningScene(),
                       'material': MaterialScene(),
                       'energy imparted': DoseScene(),
                       'dose': DoseScene(front_array='dose')}
        for name, scene in self.scenes.items():
            # connecting scenes to request array slot
            scene.update_index.connect(self.update_index)
            scene.request_array.connect(database_interface.request_view_array)
            database_interface.send_view_array.connect(scene.set_requested_array)





        self.request_array_slice.connect(database_interface.request_view_array_slice)
        database_interface.send_view_array_slice.connect(self.reload_slice)

        self.request_metadata.connect(database_interface.request_simulation_properties)
        database_interface.send_view_sim_propeties.connect(self.set_simulation_properties)

    def set_mc_runner(self, runner):
        if runner is not None:
            if 'running' in self.scenes:
                runner.request_runner_view_update.connect(self.scenes['running'].set_running_data)

    @QtCore.pyqtSlot(str)
    def set_simulation(self, name):
        self.current_simulation = name
#        self.request_metadata.emit(self.current_simulation) # Not needed, propetiesmodel will request metadata

    @QtCore.pyqtSlot(dict)
    def set_simulation_properties(self, data_dict):
        if data_dict.get('name', '') != self.current_simulation:
            return
        for scene in self.scenes.values():
            scene.set_metadata(data_dict)
        self.update_index(self.current_index)
        self.graphicsview.fitInView(self.graphicsview.sceneRect(), QtCore.Qt.KeepAspectRatio)

    @QtCore.pyqtSlot(int)
    def update_index(self, index):
        self.current_index = index
        for arr_name in self.scenes[self.current_scene].array_names:
            self.request_array_slice.emit(self.current_simulation, arr_name, self.current_index, self.current_view_orientation)

    @QtCore.pyqtSlot(str, str, np.ndarray, int, int)
    def reload_slice(self, name, arr_name, arr, index, orientation):
        if name == self.current_simulation and index == self.current_index:
            self.scenes[self.current_scene].reload_slice(name, arr_name, arr, index, orientation)


    @QtCore.pyqtSlot()
    def selectViewOrientation(self):
        self.current_view_orientation += 1
        self.current_view_orientation %= 3
        for scene in self.scenes.values():
            scene.set_view_orientation(self.current_view_orientation, self.current_index)
        self.update_index(self.current_index)
        self.graphicsview.fitInView(self.graphicsview.sceneRect(), QtCore.Qt.KeepAspectRatio)

    @QtCore.pyqtSlot(str)
    def selectScene(self, scene_name):
        if scene_name in self.scenes:
            self.current_scene = scene_name
            self.graphicsview.setScene(self.scenes[self.current_scene])
            self.update_index(self.current_index)

    def view_widget(self):
        wid = QtGui.QWidget()
        main_layout = QtGui.QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        view_layout = QtGui.QVBoxLayout()
        view_layout.setContentsMargins(0, 0, 0, 0)

        main_layout.addLayout(view_layout)


        menu_widget = QtGui.QMenuBar(wid)
        menu_widget.setContentsMargins(0, 0, 0, 0)

        orientation_action = QtGui.QAction('Orientation', menu_widget)
        orientation_action.triggered.connect(self.selectViewOrientation)
        menu_widget.addAction(orientation_action)

        sceneSelect = SceneSelectGroup(wid)
        for scene_name in self.scenes.keys():
            sceneSelect.addAction(scene_name)
        sceneSelect.scene_selected.connect(self.selectScene)
#        self.scene_selected.connect(sceneSelect.sceneSelected)
        for action in sceneSelect.actions():
            menu_widget.addAction(action)


        cine_action = QtGui.QAction('Cine', menu_widget)
        cine_action.triggered.connect(self.graphicsview.request_cine_film_creation)
        menu_widget.addAction(cine_action)



        view_layout.addWidget(menu_widget)

        view_layout.addWidget(self.graphicsview)
        wid.setLayout(main_layout)
        return wid



class Scene(QtGui.QGraphicsScene):
    update_index = QtCore.pyqtSignal(int)
    request_array = QtCore.pyqtSignal(str, str)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.name = ''
        self.array_names = []
        self.view_orientation = 2
        self.index = 0
        self.shape = np.ones(3, np.int)
        self.spacing = np.ones(3, np.double)
        self.scaling = np.ones(3, np.double)


    def set_metadata(self, sim, index=0):
        self.name = sim.get('name', '')
        self.spacing = sim.get('spacing', np.ones(3, np.double))
        self.shape = sim.get('shape', np.ones(3, np.int))
        self.scaling = sim.get('scaling', np.ones(3, np.double))
        self.index = index % self.shape[self.view_orientation]

    @QtCore.pyqtSlot(str, np.ndarray, str)
    def set_requested_array(self, name, array, array_name):
        pass



    @QtCore.pyqtSlot(str, np.ndarray, str, int, int)
    def reload_slice(self, simulation_name, arr, array_name, index, orientation):
        pass

    def set_view_orientation(self, orientation, index):
        self.view_orientation = orientation
        self.index = index % self.shape[self.view_orientation]

    def wheelEvent(self, ev):
        if ev.delta() > 0:
            self.index += 1
        elif ev.delta() < 0:
            self.index -= 1
        self.index %= self.shape[self.view_orientation]
        self.update_index.emit(self.index)
        ev.accept()




#class ViewController(QtCore.QObject):
#    def __init__(self, database_interface, properties_model, parent=None):
#        super().__init__(parent)
#
#
#
#        database_interface.simulation_updated.connect(self.updateSimulation)
#        self.scenes = {'planning': PlanningScene(self),
#                       'energy_imparted': DoseScene(self),
#                       'running': RunningScene(self),
#                       'material': MaterialScene(self),
##                       'dose': DoseScene(self),
#                       }
#
#        self.scenes['planning'].request_reload_slice.connect(database_interface.get_array_slice)
#        self.scenes['material'].request_reload_slice.connect(database_interface.get_array_slice)
#        self.scenes['energy_imparted'].request_reload_slice.connect(database_interface.get_array_slice)
#
#        database_interface.request_array_slice_view.connect(self.scenes['planning'].reload_slice)
#        database_interface.request_array_slice_view.connect(self.scenes['material'].reload_slice)
#        database_interface.request_array_slice_view.connect(self.scenes['energy_imparted'].reload_slice)
#
#        self.current_simulation = None
#        self.current_scene = 'planning'
#        self.current_view_orientation = 2
#
#        self.graphicsview = View()
#
#
#        self.properties_widget = PropertiesWidget(properties_model)
#        self.organ_dose_widget = OrganDoseWidget()
#        properties_model.request_update_simulation.connect(self.updateSimulation)
#        self.simulation_properties_data.connect(properties_model.update_data)
#        self.selectScene('planning')
#
#    def reload_slice(self, )
#
#    def view_widget(self):
#        wid = QtGui.QWidget()
#        main_layout = QtGui.QHBoxLayout()
#        main_layout.setContentsMargins(0, 0, 0, 0)
#        view_layout = QtGui.QVBoxLayout()
#        view_layout.setContentsMargins(0, 0, 0, 0)
#
#        main_layout.addWidget(self.properties_widget)
#
#        main_layout.addLayout(view_layout)
#
#        main_layout.addWidget(self.organ_dose_widget)
#
#        menu_widget = QtGui.QMenuBar(wid)
#        menu_widget.setContentsMargins(0, 0, 0, 0)
#        orientation_action = QtGui.QAction('Orientation', menu_widget)
#        orientation_action.triggered.connect(self.selectViewOrientation)
#
#        menu_widget.addAction(orientation_action)
#
#        sceneSelect = SceneSelectGroup(wid)
#        for scene_name in self.scenes.keys():
#            sceneSelect.addAction(scene_name)
#        sceneSelect.scene_selected.connect(self.selectScene)
#        self.scene_selected.connect(sceneSelect.sceneSelected)
#        for action in sceneSelect.actions():
#            menu_widget.addAction(action)
#
#        view_layout.addWidget(menu_widget)
#
#        view_layout.addWidget(self.graphicsview)
#        wid.setLayout(main_layout)
#
##        sub_layout = QtGui.QVBoxLayout()
#        return wid
#
#    @QtCore.pyqtSlot()
#    def selectViewOrientation(self):
#        self.current_view_orientation += 1
#        self.current_view_orientation %= 3
#        self.scenes[self.current_scene].setViewOrientation(self.current_view_orientation)
#        self.graphicsview.fitInView(self.scenes[self.current_scene].sceneRect(), QtCore.Qt.KeepAspectRatio)
#
#    @QtCore.pyqtSlot(str)
#    def selectScene(self, scene_name):
#        if scene_name in self.scenes:
#            self.current_scene = scene_name
#            self.graphicsview.setScene(self.scenes[self.current_scene])
#            self.update_scene_data(scene_name)
#
#    def update_scene_data(self, name):
#        if self.current_simulation is None:
#            return
#        if not name in self.scenes:
#            return
#
#        if name == 'planning':
#            self.scenes[name].update_data(self.current_simulation)
##            if self.current_simulation.ctarray is not None:
##                self.scenes[name].setCtArray(self.current_simulation.ctarray,
##                                             self.current_simulation.spacing,
##                                             self.current_simulation.exposure_modulation)
##
##            elif self.current_simulation.organ is not None:
##                self.scenes[name].setBitArray(self.current_simulation.organ,
##                                             self.current_simulation.spacing,
##                                             self.current_simulation.exposure_modulation)
#
#        elif name == 'running':
#            if self.current_simulation.energy_imparted is not None:
#                self.scenes[name].setArray(self.current_simulation.energy_imparted,
#                                           self.current_simulation.spacing,
#                                           self.current_simulation.scaling)
#            else:
#                self.scenes[name].setNoData()
#        elif name == 'energy_imparted':
#            self.scenes[name].update_data(self.current_simulation)
##            if self.current_simulation.energy_imparted is not None:
##
###                self.scenes[name].setCtDoseArrays(self.current_simulation.ctarray,
###                                                  self.current_simulation.energy_imparted,
###                                                  self.current_simulation.spacing,
###                                                  self.current_simulation.scaling)
##            else:
##                self.scenes[name].setNoData()
#        elif name == 'material':
#            self.scenes[name].update_data(self.current_simulation)
##            if self.current_simulation.material is not None and self.current_simulation.material_map is not None:
##                self.scenes[name].setMaterialArray(self.current_simulation.material,
##                                                     self.current_simulation.material_map,
##                                                     self.current_simulation.spacing,
##                                                     self.current_simulation.scaling)
##            else:
##                self.scenes[name].setNoData()
#
#        elif name == 'dose':
#            try:
#                dose = self.current_simulation.dose
#            except ValueError:
#                self.scenes[name].setNoData()
#            else:
#                if self.current_simulation.ctarray is not None:
#                    background = self.current_simulation.ctarray
#                elif self.current_simulation.organ is not None:
#                    background = self.current_simulation.organ
#                else:
#                    background=None
#                self.scenes[name].setCtDoseArrays(background,
#                                                  dose,
#                                                  self.current_simulation.spacing,
#                                                  self.current_simulation.scaling)
#                self.organ_dose_widget.set_data(dose,
#                                                self.current_simulation.organ,
#                                                self.current_simulation.organ_map)
#
#
#        self.scenes[self.current_scene].setViewOrientation(self.current_view_orientation)
#        self.graphicsview.fitInView(self.scenes[self.current_scene].sceneRect(),
#                                    QtCore.Qt.KeepAspectRatio)
#        self.properties_widget.setVisible(name == 'planning')
#        self.organ_dose_widget.setVisible(name == 'dose')
#
#
#    @QtCore.pyqtSlot(dict)
#    def applySimulation(self, sim):
#        self.current_simulation = sim
#        logger.debug('Got signal request to view Simulation {}'.format(sim.name))
#
#
#        if sim.MC_running: ##############################################!!!!!!!!
#            scene = 'running'
#            self.selectScene(scene)
#        else:
#            self.update_scene_data(self.current_scene)
#
#
#
#    @QtCore.pyqtSlot(dict, dict)
#    def updateSimulation(self, description, volatiles):
#        if self.current_simulation is None:
#            return
#        if description.get('name', None) == self.current_simulation.name:
#            for key, value in itertools.chain(description.items(), volatiles.items()):
#                setattr(self.current_simulation, key, value)
#            if self.current_simulation.MC_running and (self.current_simulation.energy_imparted is not None):
#                self.selectScene('running')
#


#def intColor(index, hues=9, values=1, maxValue=255, minValue=150, maxHue=360,
#             minHue=0, sat=255, alpha=255, firstBlack=True):
#    """
#    Creates a QColor from a single index. Useful for stepping through a
#    predefined list of colors.
#
#    The argument *index* determines which color from the set will be returned.
#    All other arguments determine what the set of predefined colors will be
#
#    Colors are chosen by cycling across hues while varying the value
#    (brightness).
#    By default, this selects from a list of 9 hues."""
#    if firstBlack and index == 0:
#        return QtGui.QColor(QtCore.Qt.black)
#    hues = int(hues)
#    values = int(values)
#    ind = int(index) % (hues * values)
#    indh = ind % hues
#    indv = ind / hues
#    if values > 1:
#        v = minValue + indv * ((maxValue-minValue) / (values-1))
#    else:
#        v = maxValue
#    h = minHue + (indh * (maxHue-minHue)) / hues
#
#    c = QtGui.QColor()
#    c.setHsv(h, sat, v)
#    c.setAlpha(alpha)
#    return c

def blendArrayToQImage(f_array, b_array, front_level, back_level,
                       front_lut, back_lut):
    """Convert the 2D numpy array `gray` into a 8-bit QImage with a gray
    colormap.  The first dimension represents the vertical image axis.
    ATTENTION: This QImage carries an attribute `ndimage` with a
    reference to the underlying numpy array that holds the data. On
    Windows, the conversion into a QPixmap does not copy the data, so
    that you have to take care that the QImage does not get garbage
    collected (otherwise PyQt will throw away the wrapper, effectively
    freeing the underlying memory - boom!)."""

    front_array = np.clip(f_array, front_level[0]-front_level[1],
                          front_level[0]+front_level[1])

    front_array -= (front_level[0]-front_level[1])
    front_array *= 255./(front_level[1]*2.)

    back_array = np.clip(b_array, back_level[0]-back_level[1],
                         back_level[0]+back_level[1])
    back_array -= (back_level[0]-back_level[1])
    back_array *= 255./(back_level[1]*2.)

    front_array = np.require(front_array, np.uint8, 'C')
    back_array = np.require(back_array, np.uint8, 'C')

    front_h, front_w = front_array.shape
    back_h, back_w = back_array.shape

    front_qim = QtGui.QImage(front_array.data, front_w, front_h, front_w,
                             QtGui.QImage.Format_Indexed8)
    back_qim = QtGui.QImage(back_array.data, back_w, back_h, back_w,
                            QtGui.QImage.Format_Indexed8)
    front_qim.setColorTable(front_lut)
    back_qim.setColorTable(back_lut)

    back_qim = back_qim.convertToFormat(QtGui.QImage.Format_ARGB32_Premultiplied, back_lut)#, flags=QtCore.Qt.DiffuseAlphaDither)

    p = QtGui.QPainter(back_qim)
    p.setCompositionMode(QtGui.QPainter.CompositionMode_DestinationOut)
    p.drawImage(QtCore.QRectF(back_qim.rect()), front_qim)
    p.setCompositionMode(QtGui.QPainter.CompositionMode_SourceOver)

    p.drawImage(QtCore.QRectF(back_qim.rect()), front_qim)
#    p.setPen(QtCore.Qt.white)
#    p.drawText(QtCore.QPointF(0, back_w), "({0}, {1})".format(front_level[0], front_level[1]))

    return back_qim


def arrayToQImage(array_un, level, lut):
    """Convert the 2D numpy array `gray` into a 8-bit QImage with a gray
    colormap.  The first dimension represents the vertical image axis.
    ATTENTION: This QImage carries an attribute `ndimage` with a
    reference to the underlying numpy array that holds the data. On
    Windows, the conversion into a QPixmap does not copy the data, so
    that you have to take care that the QImage does not get garbage
    collected (otherwise PyQt will throw away the wrapper, effectively
    freeing the underlying memory - boom!)."""

    WC, WW = level[0], level[1]
    array = np.clip(array_un, WC-WW, WC+WW)

    array -= (WC - WW)
    array *= 255./(WW*2)


#    array = (np.clip(array, WC - 0.5 - (WW-1) / 2, WC - 0.5 + (WW - 1) / 2) -
#             (WC - 0.5 - (WW - 1) / 2)) * 255 / ((WC - 0.5 + (WW - 1) / 2) -
#                                                 (WC - 0.5 - (WW - 1) / 2))
    array = np.require(array, np.uint8, ['C', 'A'])
    h, w = array.shape

    result = QtGui.QImage(array.data, w, h, w, QtGui.QImage.Format_Indexed8)
#    result.ndarray = array
    result.setColorTable(lut)
#    result = result.convertToFormat(QtGui.QImage.Format_ARGB32, lut)
    result.ndarray = array
    return result


class NoDataItem(QtGui.QGraphicsItem):
    def __init__(self, parent=None):
        super().__init__(parent)
#        self.fontMetrics = QtGui.qApp.fontMetrics()
        self.msg = "Sorry, no data here yet. Run a simulation to compute."
#        self.rect = QtCore.QRectF(self.fontMetrics.boundingRect(self.msg))

    def boundingRect(self):
#        return  QtCore.QRectF(self.fontMetrics.boundingRect(self.msg))
        return QtCore.QRectF(0, 0, 200, 200)

    def paint(self, painter, style, widget=None):
        painter.setPen(QtGui.QPen(QtCore.Qt.white))

#        h = self.fontMetrics.boundingRect('A').height()
#        painter.drawText(0, h ,self.msg)

        painter.drawText(QtCore.QRectF(0, 0, 200, 200), QtCore.Qt.AlignCenter, self.msg)
#
#
#        self.fontMetrics = QtGui.qApp.fontMetrics()
#        self.box_size = self.fontMetrics.boundingRect('A').height()
#        self.rect = QtCore.QRectF(0, 0, self.box_size, self.box_size)
#        self.map = []

#
#    def set_map(self, mapping, colors):
#        self.map = []
#
#        for ind in range(len(mapping)):
#            self.map.append((colors[ind], str(mapping['value'][ind], encoding='utf-8')))
#
#        max_str_index = 0
#        max_len_str = 0
#        for ind, item in enumerate(self.map):
#            if len(item[1]) > max_len_str:
#                max_len_str = len(item[1])
#                max_str_index = ind
#
#        sub_rect = self.fontMetrics.boundingRect(self.map[max_str_index][1])
#        self.rect = QtCore.QRectF(0, 0,
#                                  self.box_size * 1.25 + sub_rect.width(),
#                                  sub_rect.height() * len(self.map) * 2)
#
#    def boundingRect(self):
#        return self.rect
#
#    def paint(self, painter, style, widget=None):
#        painter.setPen(QtGui.QPen(QtCore.Qt.white))
#        painter.setRenderHint(painter.Antialiasing, True)
#        h = self.fontMetrics.boundingRect('A').height()
#        for ind, value in enumerate(self.map):
#            key, item = value
#            painter.fillRect(QtCore.QRectF(0, ind*2*h, self.box_size, self.box_size), QtGui.QColor(key))
#            painter.drawText(self.box_size * 1.25, ind*2*h + self.box_size, item)
#            painter.drawRect(QtCore.QRectF(0, ind*2*h, self.box_size, self.box_size))



class BlendImageItem(QtGui.QGraphicsItem):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.back_image = np.zeros((8, 8))
        self.back_level = (0, 500)
        self.front_image = np.zeros((8, 8))
        self.front_level = (1000000, 10000)

        self.back_alpha = 255
        self.front_alpha = 255
        self.back_lut = get_lut('gray', self.back_alpha)
        self.front_lut = get_lut('pet', self.front_alpha)

        self.qimage = None
        self.setAcceptedMouseButtons(QtCore.Qt.RightButton | QtCore.Qt.MiddleButton)
#        self.setAcceptHoverEvents(True)

    def qImage(self):
        if self.qimage is None:
            self.render()
        return self.qimage

    def boundingRect(self):
        return QtCore.QRectF(self.qImage().rect())

    def setImage(self, front_image=None, back_image=None):
        if front_image is not None:
            self.front_image = np.copy(front_image)
        if back_image is not None:
            self.back_image = np.copy(back_image)
        self.prepareGeometryChange()
        self.qimage = None
        self.update(self.boundingRect())

    def setLevels(self, front=None, back=None):
        update = False
        if front is not None:
            self.front_level = front
            update = True
        if back is not None:
            self.back_level = back
            update = True
        if update:
            self.qimage = None
            self.update(self.boundingRect())

    def setLut(self, front_lut=None, back_lut=None, front_alpha=None,
               back_alpha=None):
        update = False
        if front_lut is not None:
            if front_alpha is not None:
                alpha = front_alpha
            else:
                alpha = self.front_alpha
            self.front_lut = get_lut(front_lut, alpha)
            update = True
        if back_lut is not None:
            if back_alpha is not None:
                alpha = back_alpha
            else:
                alpha = self.back_alpha
            self.back_lut = get_lut(back_lut, alpha)
            update = True
        if update:
            self.qimage = None
            self.update(self.boundingRect())

    def render(self):
        self.qimage = blendArrayToQImage(self.front_image, self.back_image,
                                         self.front_level, self.back_level,
                                         self.front_lut, self.back_lut)

    def shape(self):
        path = QtGui.QPainterPath()
        path.addEllipse(self.boundingRect())
        return path

    def paint(self, painter, style, widget=None):
        if self.qimage is None:
            self.render()
        painter.drawImage(QtCore.QPointF(0, 0), self.qimage)

    def mousePressEvent(self, event):
        event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == QtCore.Qt.RightButton:
            event.accept()
#            print(event.pos()- event.lastPos())
            dp = event.pos()- event.lastPos()
            x, y = self.front_level
            x += dp.x()*.01*abs(x)
            y += dp.y()*.01*abs(y)
            if x < 0:
                x=0
            if y < 0:
                y=0
            self.setLevels(front=(x, y))
        elif event.buttons() == QtCore.Qt.MiddleButton:
            event.accept()
#            print(event.pos()- event.lastPos())
            dp = event.pos()- event.lastPos()
            x, y = self.back_level
            x += dp.x()
            y += dp.y()
            if x < 0:
                x=0
            if y < 0:
                y=0
            self.setLevels(back=(x, y))

class BitImageItem(QtGui.QGraphicsItem):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFlag(QtGui.QGraphicsItem.ItemSendsGeometryChanges)
        self.image = np.zeros((8, 8), dtype=np.uint8)
        self.prepareGeometryChange()
        self.qimage = None
        self.lut = get_lut('pet')

    def qImage(self):
        if self.qimage is None:
            self.render()
        return self.qimage

    def boundingRect(self):
        return QtCore.QRectF(self.qImage().rect())

    def set_lut(self, lut):
        self.lut = lut

    def setImage(self, image):
        self.image = np.require(image, np.uint8, ['C', 'A'])
        self.prepareGeometryChange()
        self.qimage = None
        self.update(self.boundingRect())

    def render(self):
        h, w = self.image.shape

        self.qimage = QtGui.QImage(self.image.data, w, h, w, QtGui.QImage.Format_Indexed8)
#    result.ndself.image = self.image
        self.qimage.setColorTable(self.lut)
#    result = result.convertToFormat(QtGui.QImage.Format_ARGB32, lut)
        self.qimage.ndarray = self.image

    def shape(self):
        path = QtGui.QPainterPath()
        path.addEllipse(self.boundingRect())
        return path

    def paint(self, painter, style, widget=None):
        painter.drawImage(QtCore.QPointF(0, 0), self.qImage())

class ImageItem(QtGui.QGraphicsItem):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFlag(QtGui.QGraphicsItem.ItemSendsGeometryChanges)
        self.image = np.zeros((8, 8))
        self.level = (0, 700)
        self.qimage = None
        self.lut = get_lut('gray')
        self.setLevels((0., 1.))

    def qImage(self):
        if self.qimage is None:
            self.render()
        return self.qimage

    def boundingRect(self):
        return QtCore.QRectF(self.qImage().rect())

    def setImage(self, image):
        self.image = image
        self.prepareGeometryChange()
        self.qimage = None
        self.update(self.boundingRect())

    def setLevels(self, level=None):
        if level is None:
            p = self.image.max() - self.image.min()
            level = (p/2., p / 2. * .75)
        self.level = level
        self.qimage = None
        self.update(self.boundingRect())

    def setLut(self, lut):
        self.lut = lut
        self.qimage = None
        self.update(self.boundingRect())

    def render(self):
        self.qimage = arrayToQImage(self.image, self.level,
                                    self.lut)

    def paint(self, painter, style, widget=None):
        painter.drawImage(QtCore.QPointF(self.pos()), self.qImage())


class AecItem(QtGui.QGraphicsItem):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.aec = np.zeros((5, 2))
        self.view_orientation = 2
        self.__shape = (1, 1, 1)
        self.index = 0
        self.__path = None
        self.__path_pos = None

    def set_aec(self, aec, view_orientation, shape):
        self.__shape = shape
        self.view_orientation = view_orientation

        self.aec = aec
        self.aec[:, 0] -= self.aec[:, 0].min()

        self.aec[:, 0] /= self.aec[:, 0].max()

        if self.aec[:, 1].min() == self.aec[:, 1].max():
            self.aec[:, 1] = .5
        else:
            self.aec[:, 1] -= self.aec[:, 1].min()
            self.aec[:, 1] /= self.aec[:, 1].max()

        self.__path = None
        self.__path_pos = None
        self.prepareGeometryChange()

        self.update(self.boundingRect())

    def boundingRect(self):
        shape = tuple(self.__shape[i] for i in [0, 1, 2] if i != self.view_orientation)
        return QtCore.QRectF(0, 0, shape[1], shape[0]*.1)

    def setIndex(self, index):
        self.index = index % self.__shape[self.view_orientation]
        self.__path_pos = None
#        self.prepareGeometryChange()
        self.update(self.boundingRect())

    def setViewOrientation(self, view_orientation):
        self.view_orientation = view_orientation
        self.__path = None
        self.__path_pos = None
        self.prepareGeometryChange()
        self.update(self.boundingRect())

    def aec_path(self):

        if self.__path is None:
            shape = tuple(self.__shape[i] for i in [0, 1, 2] if i != self.view_orientation)
            self.__path = QtGui.QPainterPath()

            x = self.aec[:, 0] * shape[1]
            y = (1. - self.aec[:, 1]) * shape[0] * .1
            self.__path.moveTo(x[0], y[0])
            for i in range(1, self.aec.shape[0]):
                self.__path.lineTo(x[i], y[i])

            self.__path.moveTo(0, 0)
            self.__path.lineTo(0, shape[0]*.1)
            self.__path.lineTo(shape[1], shape[0]*.1)
            self.__path.lineTo(shape[1], 0)
            self.__path.lineTo(0, 0)

        if self.__path_pos is None:
            self.__path_pos = QtGui.QPainterPath()
            if self.view_orientation == 2:
                shape = tuple(self.__shape[i] for i in [0, 1, 2] if i != self.view_orientation)
                self.__path_pos = QtGui.QPainterPath()
                x = self.aec[:, 0] * shape[1]
                y = (1. - self.aec[:, 1]) * shape[0] * .1
                x_c = self.index / self.__shape[2]
                y_c = 1. - np.interp(x_c, self.aec[:, 0], self.aec[:, 1])
                self.__path_pos.addEllipse(QtCore.QPointF(x_c * shape[1], y_c * shape[0] *.1), shape[0]*.005, shape[0]*.01)

        p = QtGui.QPainterPath()
        p.addPath(self.__path)
        p.addPath(self.__path_pos)
        return p

    def paint(self, painter, style, widget=None):
        painter.setPen(QtGui.QPen(QtCore.Qt.white))
        painter.setRenderHint(painter.Antialiasing, True)
        painter.drawPath(self.aec_path())


class PlanningScene(Scene):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_item = ImageItem()
        self.image_item_bit = BitImageItem()
        self.addItem(self.image_item)
        self.addItem(self.image_item_bit)
        self.array_names = ['ctarray', 'organ']

        self.aec_item = AecItem()
        self.addItem(self.aec_item)
        self.image_item.setLevels((0, 500))
        self.is_bit_array = False
        self.lut = get_lut('pet')

    def set_metadata(self, sim, index=0):
        super().set_metadata(sim, index)
        self.is_bit_array = sim.get('is_phantom', False)
        self.image_item.setVisible(not self.is_bit_array)
        self.image_item_bit.setVisible(self.is_bit_array)

        if self.is_bit_array:
            self.array_names = ['organ']
            self.request_array.emit(self.name, 'organ_map')
        else:
            self.array_names = ['ctarray']
        self.request_array.emit(self.name, 'exposure_modulation')

        self.updateSceneTransform()



    @QtCore.pyqtSlot(str, np.ndarray, str)
    def set_requested_array(self, name, array, array_name):
        if name != self.name:
            return
        if array_name == 'organ_map':
            organ_max_value = array['organ'].max()
            lut =  [self.lut[i*255 // organ_max_value] for i in range(organ_max_value + 1)]
            self.image_item_bit.set_lut(lut)
        elif array_name == 'exposure_modulation':
            self.aec_item.set_aec(array, self.view_orientation, self.shape)

    def set_view_orientation(self, view_orientation, index):
        self.view_orientation = view_orientation
        self.aec_item.setViewOrientation(view_orientation)
        self.index = index % self.shape[self.view_orientation]
        self.updateSceneTransform()

    def updateSceneTransform(self):
        sx, sy = [self.spacing[i] for i in range(3) if i != self.view_orientation]
        transform = QtGui.QTransform.fromScale(sy / sx, 1.)
        if self.is_bit_array:
            self.image_item_bit.setTransform(transform)
        else:
            self.image_item.setTransform(transform)

        self.aec_item.setTransform(transform)
        self.aec_item.prepareGeometryChange()
        shape = tuple(sh for ind, sh in enumerate(self.shape) if ind != self.view_orientation)
        rect = QtCore.QRectF(0, 0, shape[1], shape[0])
        if self.is_bit_array:
            self.aec_item.setPos(self.image_item_bit.mapRectToScene(rect).bottomLeft())
            self.setSceneRect(self.image_item_bit.mapRectToScene(rect).united(self.aec_item.sceneBoundingRect()))
        else:
            self.aec_item.setPos(self.image_item.mapRectToScene(rect).bottomLeft())
            self.setSceneRect(self.image_item.mapRectToScene(rect).united(self.aec_item.sceneBoundingRect()))


    @QtCore.pyqtSlot(str, np.ndarray, str, int, int)
    def reload_slice(self, simulation_name, arr, array_name, index, orientation):
        if simulation_name != self.name:
            return

        if array_name not in self.array_names:
            return
        self.index = index
        if self.is_bit_array:
            self.image_item_bit.setImage(arr)
        else:
            self.image_item.setImage(arr)
        self.aec_item.setIndex(self.index)

#    def wheelEvent(self, ev):
#        if ev.delta() > 0:
#            self.index += 1
#        elif ev.delta() < 0:
#            self.index -= 1
#        self.index %= self.shape[self.view_orientation]
#        self.request_reload_slice.emit(self.name, self.array_name, self.index, self.view_orientation)
#        ev.accept()

#class PlanningScene(QtGui.QGraphicsScene):
#    def __init__(self, parent=None):
#        super().__init__(parent)
#        self.image_item = ImageItem()
#        self.image_item_bit = BitImageItem()
#        self.addItem(self.image_item)
#        self.addItem(self.image_item_bit)
#
#        self.aec_item = AecItem()
#        self.addItem(self.aec_item)
#        self.array = np.random.uniform(size=(8, 8, 8))
#        self.shape = np.array((8, 8, 8))
#        self.spacing = np.array((1., 1., 1.))
#        self.index = 0
#        self.view_orientation = 2
#        self.image_item.setLevels((0, 500))
#        self.is_bit_array = False
#        self.bit_lut = get_lut('pet')
#
#    def setCtArray(self, ct, spacing, aec):
#        self.is_bit_array = False
#        self.image_item.setVisible(True)
#        self.image_item_bit.setVisible(False)
#        self.array = ct
#        self.shape = ct.shape
#        self.spacing = spacing
#        self.index = self.index % self.shape[self.view_orientation]
#        if aec is None:
#            aec = np.ones((2,2))
#            aec[0, 0] = 0
#        self.aec_item.set_aec(aec, self.view_orientation, ct.shape)
#        self.reloadImages()
#        self.updateSceneTransform()
#
#    def setBitArray(self, ct, spacing, aec):
#        self.is_bit_array = True
#        self.image_item.setVisible(False)
#        self.image_item_bit.setVisible(True)
#
#        self.array = ct
#        self.shape = ct.shape
#        self.spacing = spacing
#        self.index = self.index % self.shape[self.view_orientation]
#        if aec is None:
#            aec = np.ones((2,2))
#            aec[0, 0] = 0
#        self.aec_item.set_aec(aec, self.view_orientation, ct.shape)
#        number_of_elements = self.array.max()
#        qlut = [QtGui.QColor(self.bit_lut[i * 255 // number_of_elements]) for i in range(number_of_elements)]
#        self.image_item_bit.set_lut(qlut)
#        self.reloadImages()
#        self.updateSceneTransform()
#
#    @QtCore.pyqtSlot(int)
#    def setViewOrientation(self, view_orientation):
#        self.view_orientation = view_orientation
#        self.aec_item.setViewOrientation(view_orientation)
#        self.reloadImages()
#        self.updateSceneTransform()
#
#    def updateSceneTransform(self):
#        sx, sy = [self.spacing[i] for i in range(3) if i != self.view_orientation]
#        transform = QtGui.QTransform.fromScale(sy / sx, 1.)
#        if self.is_bit_array:
#            self.image_item_bit.setTransform(transform)
#        else:
#            self.image_item.setTransform(transform)
#
#        self.aec_item.setTransform(transform)
#        self.aec_item.prepareGeometryChange()
#        if self.is_bit_array:
#            self.aec_item.setPos(self.image_item_bit.mapToScene(self.image_item.boundingRect().bottomLeft()))
#            self.setSceneRect(self.image_item_bit.sceneBoundingRect().united(self.aec_item.sceneBoundingRect()))
#        else:
#            self.aec_item.setPos(self.image_item.mapToScene(self.image_item.boundingRect().bottomLeft()))
#            self.setSceneRect(self.image_item.sceneBoundingRect().united(self.aec_item.sceneBoundingRect()))
#
##        self.setSceneRect(self.itemsBoundingRect())
#
#    def getSlice(self, array, index):
#        if self.view_orientation == 2:
#            return np.copy(np.squeeze(array[: ,: ,index % self.shape[self.view_orientation]]))
#        elif self.view_orientation == 1:
#            return np.copy(np.squeeze(array[:, index % self.shape[self.view_orientation], :]))
#        elif self.view_orientation == 0:
#            return np.copy(np.squeeze(array[index % self.shape[self.view_orientation], :, :]))
#        raise ValueError('view must select one of 0,1,2 dimensions')
#
#    def reloadImages(self):
#        if self.is_bit_array:
#            self.image_item_bit.setImage(self.getSlice(self.array, self.index))
#        else:
#            self.image_item.setImage(self.getSlice(self.array, self.index))
#        self.aec_item.setIndex(self.index)
#
#    def wheelEvent(self, ev):
#        if ev.delta() > 0:
#            self.index += 1
#        elif ev.delta() < 0:
#            self.index -= 1
#        self.index %= self.shape[self.view_orientation]
#        self.reloadImages()
#        ev.accept()


class RunningScene(Scene):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_item = ImageItem()
        self.addItem(self.image_item)
        self.image_item.setLut(get_lut('hot_iron'))
        self.nodata_item = NoDataItem()
        self.addItem(self.nodata_item)
        self.nodata_item.setVisible(True)

        self.progress_item = self.addText('0 %')
        self.progress_item.setDefaultTextColor(QtCore.Qt.white)

        self.n_exposures = 1
        self.array = None
        self.view_orientation = 1


    def set_metadata(self, props, index=0):
        if props['name'] != self.name:
            self.setNoData()
        super().set_metadata(props, index)
        collimation = props['detector_rows']*props['detector_width']
        if props['is_spiral']:
            self.n_exposures = props['exposures'] * (1 + abs(props['start'] - props['stop']) / collimation / props['pitch'])
        else:
            self.n_exposures = props['exposures'] * np.ceil(abs(props['start'] - props['stop']) / props['step'])
            if self.n_exposures < 1:
                self.n_exposures = 1
        self.updateSceneTransform()


    def setNoData(self):
        self.array = None
        self.image_item.setVisible(False)
        self.progress_item.setVisible(False)
        self.nodata_item.setVisible(True)

    def defaultLevels(self, array):

        p = array.max() - array.min()
        return (p/2., p / 2. )

    @QtCore.pyqtSlot(dict, dict)
    def set_running_data(self, props, arr_dict):
        self.array = arr_dict.get('energy_imparted', None)

        self.nodata_item.setVisible(False)
        self.image_item.setVisible(True)
        self.progress_item.setVisible(True)
        msg = ""
        if 'start_at_exposure_no' in props:
            msg += "{} %".format(round(props['start_at_exposure_no'] / self.n_exposures *100, 1))
        if 'eta' in props:
            msg += " ETA: {}".format(props['eta'])
        self.progress_item.setPlainText(msg)
        if self.array is not None:
            self.image_item.setLevels(self.defaultLevels(self.array))
        self.reloadImages()
        self.updateSceneTransform()

    def set_view_orientation(self, orientation, index):
        self.view_orientation = orientation
        self.reloadImages()
        self.updateSceneTransform()

    def updateSceneTransform(self):
        sx, sy = [self.spacing[i]*self.scaling[i] for i in range(3) if i != self.view_orientation]
        transform = QtGui.QTransform.fromScale(sy / sx, 1.)
        self.image_item.setTransform(transform)
        if self.nodata_item.isVisible():
            self.setSceneRect(self.nodata_item.sceneBoundingRect())
        else:
            self.setSceneRect(self.image_item.sceneBoundingRect())

    def reloadImages(self):
        if self.array is not None:
            self.image_item.setImage(self.array.max(axis=self.view_orientation))

    def wheelEvent(self, ev):
        pass


class MaterialMapItem(QtGui.QGraphicsItem):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.fontMetrics = QtGui.qApp.fontMetrics()
        self.box_size = self.fontMetrics.boundingRect('A').height()
        self.rect = QtCore.QRectF(0, 0, self.box_size, self.box_size)
        self.map = []


    def set_map(self, mapping, colors):
        self.map = []

        for ind in range(len(mapping)):
            self.map.append((colors[ind], str(mapping['material_name'][ind], encoding='utf-8')))

        max_str_index = 0
        max_len_str = 0
        for ind, item in enumerate(self.map):
            if len(item[1]) > max_len_str:
                max_len_str = len(item[1])
                max_str_index = ind

        sub_rect = self.fontMetrics.boundingRect(self.map[max_str_index][1])
        self.rect = QtCore.QRectF(0, 0,
                                  self.box_size * 1.25 + sub_rect.width(),
                                  sub_rect.height() * len(self.map) * 2)

    def boundingRect(self):
        return self.rect

    def paint(self, painter, style, widget=None):
        painter.setPen(QtGui.QPen(QtCore.Qt.white))
        painter.setRenderHint(painter.Antialiasing, True)
        h = self.fontMetrics.boundingRect('A').height()
        for ind, value in enumerate(self.map):
            key, item = value
            painter.fillRect(QtCore.QRectF(0, ind*2*h, self.box_size, self.box_size), QtGui.QColor(key))
            painter.drawText(self.box_size * 1.25, ind*2*h + self.box_size, item)
            painter.drawRect(QtCore.QRectF(0, ind*2*h, self.box_size, self.box_size))


class MaterialScene(Scene):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.lut = get_lut('pet')
        self.image_item = BitImageItem()
        self.addItem(self.image_item)
        self.map_item = MaterialMapItem()
        self.addItem(self.map_item)
        self.nodata_item = NoDataItem()
        self.addItem(self.nodata_item)

        self.array_names = ['material']


    def setNoData(self):
        self.nodata_item.setVisible(True)
        self.map_item.setVisible(False)
        self.image_item.setVisible(False)
        self.setSceneRect(self.nodata_item.sceneBoundingRect())

    def set_metadata(self, sim, index=0):
        super().set_metadata(sim, index)
        self.setNoData()
        self.index = self.index % self.shape[self.view_orientation]
        self.request_array.emit(self.name, 'material_map')
        self.updateSceneTransform()

    @QtCore.pyqtSlot(str, np.ndarray, str)
    def set_requested_array(self, name, array, array_name):
        if name != self.name:
            return
        if array_name == 'material_map':
            organ_max_value = array['material'].max()
            lut =  [self.lut[i*255 // organ_max_value] for i in range(organ_max_value+1)]
            self.image_item.set_lut(lut)
            self.map_item.set_map(array, lut)

            self.nodata_item.setVisible(False)
            self.map_item.setVisible(True)
            self.image_item.setVisible(True)
            self.updateSceneTransform()

    def set_view_orientation(self, orientation, index):
        self.view_orientation = orientation
        self.index = index % self.shape[self.view_orientation]
        self.updateSceneTransform()


    def updateSceneTransform(self):
        sx, sy = [self.spacing[i]*self.scaling[i] for i in range(3) if i != self.view_orientation]
        transform = QtGui.QTransform.fromScale(sy / sx, 1.)
        self.image_item.setTransform(transform)

        self.map_item.prepareGeometryChange()

#        shape = tuple(sh for ind, sh in enumerate(self.shape) if ind != self.view_orientation)
        shape = tuple(int(self.shape[i] / self.scaling[i]) for i in range(3) if i != self.view_orientation)
        rect = QtCore.QRectF(0, 0, shape[1], shape[0])
        self.map_item.setScale(self.image_item.mapRectToScene(rect).height() / self.map_item.boundingRect().height())
        self.map_item.setPos(self.image_item.mapRectToScene(rect).topRight())
        self.setSceneRect(self.image_item.mapRectToScene(rect).united(self.map_item.sceneBoundingRect()))

    @QtCore.pyqtSlot(str, np.ndarray, str, int, int)
    def reload_slice(self, simulation_name, arr, array_name, index, orientation):
        if simulation_name != self.name:
            return
        if array_name != 'material':
            return
        self.index = index
        self.image_item.setImage(arr)

class DoseScene(Scene):
    def __init__(self, parent=None, front_array='energy_imparted'):
        super().__init__(parent)

        self.image_item = BlendImageItem()
        self.addItem(self.image_item)
        self.nodata_item = NoDataItem()
        self.addItem(self.nodata_item)
        self.array_names = ['ctarray', 'organ']
        self.nodata_item.setVisible(True)
        self.image_item.setVisible(False)

        self.front_array_name = front_array

        self.front_array = None

        alpha = 1. -np.exp(-np.linspace(0, 6, 256))
        alpha *= 255./alpha.max()

        self.image_item.setLut(front_lut='jet', front_alpha=alpha.astype(np.int))


    def setNoData(self):
        self.front_array = None
        self.nodata_item.setVisible(True)
        self.image_item.setVisible(False)

    def set_metadata(self, sim, index=0):
        self.front_array = None
        super().set_metadata(sim, index)
        self.request_array.emit(self.name, self.front_array_name)
        if sim['is_phantom']:
            self.request_array.emit(self.name, 'organ_map')
        else:
            self.image_item.setLevels(back=(0, 500))
        self.nodata_item.setVisible(False)
        self.image_item.setVisible(True)

        self.updateSceneTransform()

        index2 = self.index % self.shape[self.view_orientation]
        index2 /= self.scaling[self.view_orientation]

    def updateSceneTransform(self):
        sx, sy = [self.spacing[i] for i in range(3) if i != self.view_orientation]
        transform = QtGui.QTransform.fromScale(sy / sx, 1.)
        self.image_item.setTransform(transform)
        if self.nodata_item.isVisible():
            self.setSceneRect(self.nodata_item.sceneBoundingRect())
        else:
            shape = tuple(sh for ind, sh in enumerate(self.shape) if ind != self.view_orientation)
            rect = QtCore.QRectF(0, 0, shape[1], shape[0])
            self.setSceneRect(self.image_item.mapRectToScene(rect))



    @QtCore.pyqtSlot(str, np.ndarray, str)
    def set_requested_array(self, name, array, array_name):
        if name != self.name:
            return
        if array_name == 'organ_map':
            max_level = array['organ'].max()
            self.image_item.setLevels(back=(max_level/2, max_level/2))

        elif array_name == self.front_array_name:
            self.front_array = gaussian_filter(array, 1.)
            max_level = array.max()/ 4
            min_level = max_level / 4
            self.image_item.setLevels(front=(min_level/2. + max_level/2.,min_level/2. + max_level/2.))


    @QtCore.pyqtSlot(str, np.ndarray, str, int, int)
    def reload_slice(self, name, arr, array_name, index, orientation):
        if name != self.name:
            return

        if array_name in self.array_names:
            if self.front_array is not None:
                index_front = (index % self.shape[self.view_orientation]) // self.scaling[self.view_orientation]
                index_front %= self.front_array.shape[self.view_orientation]
                if self.view_orientation == 0:
                    im = np.squeeze(self.front_array[index_front, :, :])[:, :]
                elif self.view_orientation == 1:
                    im = np.squeeze(self.front_array[:, index_front, :])[:, :]
                elif self.view_orientation == 2:
                    im = np.squeeze(self.front_array[:, :, index_front])[:, :]
                self.image_item.setImage(front_image=im, back_image=arr)
        else:
            self.image_item.setImage(back_image=arr)


    @QtCore.pyqtSlot(int, int)
    def set_view_orientation(self, view_orientation, index):
        self.view_orientation = view_orientation % 3
        self.index = index % self.shape[self.view_orientation]
        self.updateSceneTransform()


class View(QtGui.QGraphicsView):
    request_array = QtCore.pyqtSignal(str, str)


    def __init__(self, parent=None):
        super().__init__(parent)
        self.setContentsMargins(0, 0, 0, 0)
        self.setBackgroundBrush(QtGui.QBrush(QtCore.Qt.black))

        self.setRenderHints(QtGui.QPainter.Antialiasing |
#                            QtGui.QPainter.SmoothPixmapTransform |
                            QtGui.QPainter.TextAntialiasing)

        self.mouse_down_pos = QtCore.QPoint(0, 0)
        self.cine_film_data = {}


    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self.fitInView(self.sceneRect(), QtCore.Qt.KeepAspectRatio)

    def setScene(self, scene):
        super().setScene(scene)
        self.fitInView(self.sceneRect(), QtCore.Qt.KeepAspectRatio)

    def mouseMoveEvent(self, e):
        if e.buttons() == QtCore.Qt.LeftButton:
            dist = self.mouse_down_pos - e.globalPos()
            if dist.manhattanLength() > QtGui.QApplication.startDragDistance():
                e.accept()
                drag = QtGui.QDrag(self)
                # lager mimedata
                qim = self.toQImage()
                md = QtCore.QMimeData()
                md.setImageData(qim)
                drag.setMimeData(md)
                pix = QtGui.QPixmap.fromImage(qim.scaledToWidth(64))
                drag.setPixmap(pix)
                # initialiserer drops
                drag.exec_(QtCore.Qt.CopyAction)
        else:
            return super().mouseMoveEvent(e)
    def mousePressEvent(self, e):
        self.mouse_down_pos = e.globalPos()
#       e.setAccepted(False)
        return super().mousePressEvent(e)

    @QtCore.pyqtSlot()
    def request_cine_film_creation(self):
        if not isinstance(self.scene(), Scene):
            return
        self.cine_film_data = {}
        self.cine_film_data['view_orientation'] = self.scene().view_orientation
        self.cine_film_data['array_names'] = self.scene().array_names
        self.cine_film_data['name'] = self.scene().name

        for arr_name in self.cine_film_data['array_names']:
            self.request_array.emit(self.cine_film_data['name'], arr_name)


    @QtCore.pyqtSlot(str, np.ndarray, str)
    def cine_film_creation(self, name, array, array_name):
        if len(self.cine_film_data) == 0:
            return
        if name != self.cine_film_data.get('name', ''):
            return
        if array_name not in self.cine_film_data['array_names']:
            return

        rect = self.toQImage().rect()
        height, width = rect.height(), rect.width()

        filename = QtGui.QFileDialog.getSaveFileName(self,
                                                     "Save cineloop",
                                                     "/cine.mp4",
                                                     "Movie (*.mp4)")
        writer = FFMPEG_VideoWriter(filename,
                                    (width, height),
                                    24)

        for index in range(array.shape[self.cine_film_data['view_orientation']]):
            if self.cine_film_data['view_orientation'] == 1:
                arr_slice = np.squeeze(array[:, index, :])
            elif self.cine_film_data['view_orientation'] == 0:
                arr_slice = np.squeeze(array[index, :, :])
            else:
                arr_slice = np.squeeze(array[:,:,index])

            self.scene().reload_slice(self.cine_film_data['name'],
                                      arr_slice,
                                      array_name,
                                      index,
                                      self.cine_film_data['view_orientation']
                                      )
            QtGui.qApp.processEvents(flags=QtCore.QEventLoop.ExcludeUserInputEvents)
            qim = self.toQImage().convertToFormat(QtGui.QImage.Format_RGB888)
            ptr = qim.bits()
            ptr.setsize(qim.byteCount())
            arr = np.array(ptr).reshape(height, width, 3)
            writer.write_frame(arr)
        writer.close()


        self.cine_film_data = {}

#    @QtCore.pyqtSlot(str, np.ndarray, str)
#    def cine_film_creation(self, name, array, array_name):
#        if len(self.cine_film_data) == 0:
#            return
#        if name != self.cine_film_data.get('name', ''):
#            return
#        if array_name not in self.cine_film_data['array_names']:
#            return
#
#        rect = self.toQImage().rect()
#        height, width = rect.height(), rect.width()
#
#
#        FFMPEG_BIN = "E://ffmpeg//bin//ffmpeg"
#        command = [ FFMPEG_BIN,
#                   '-y', # (optional) overwrite output file if it exists
#                   '-f', 'rawvideo',
#                   '-vcodec','rawvideo',
#                   '-s', "{0}x{1}".format(height, width),###'420x360', # size of one frame
##                   '-pix_fmt', 'rgb24',
#                   '-r', '24', # frames per second
#                   '-i', '-', # The imput comes from a pipe
#                   '-an', # Tells FFMPEG not to expect any audio
#                   '-vcodec', 'mpeg4',
#                   'my_output_videofile.mp4',
#                   "-loglevel", "debug"]
##        command = ['-y', '-vcodec', 'png', '-i', '-', '-vcodec', 'mpeg4', '-qscale', '5', '-r', '24', 'c:/test/video.avi',"-loglevel", "debug"]
##        process = QtCore.QProcess(QtGui.qApp)
##        process.setProcessChannelMode(process.ForwardedChannels)
##        process.start(FFMPEG_BIN, command)
#
#        pipe = subprocess.Popen(command,
#                                stdin=subprocess.PIPE,
#                                stderr=subprocess.PIPE)
#
#        for index in range(array.shape[self.cine_film_data['view_orientation']]):
#            if index > 10:
#                break
#            if self.cine_film_data['view_orientation'] == 1:
#                arr_slice = np.squeeze(array[:, index, :])
#            elif self.cine_film_data['view_orientation'] == 0:
#                arr_slice = np.squeeze(array[index, :, :])
#            else:
#                arr_slice = np.squeeze(array[:,:,index])
#
#            self.scene().reload_slice(self.cine_film_data['name'],
#                                      arr_slice,
#                                      array_name,
#                                      index,
#                                      self.cine_film_data['view_orientation']
#                                      )
#            QtGui.qApp.processEvents(flags=QtCore.QEventLoop.ExcludeUserInputEvents)
#            QtGui.qApp.processEvents()
#            qim = self.toQImage().convertToFormat(QtGui.QImage.Format_RGB32)
#            ptr = qim.bits()
#            ptr.setsize(qim.byteCount())
#            arr = np.array(ptr).reshape(height, width, 4)
#
##            qpx = QtGui.QPixmap.fromImage(self.toQImage())
##            qpx.save(process, 'png')
##            qpx.save("C:\\test\\test\\{0}.jpeg".format(index))
#
##            pipe.proc.stdin.write(qim.bits())
#            pipe.stdin.write(arr.tostring())
#
##        process.closeWriteChannel()
##        process.terminate()
#
#        self.cine_film_data = {}

    def toQImage(self):
        rect = self.scene().sceneRect()
        h = rect.height()
        w = rect.width()
        b = max([h, w])
        if b >= 1024:
            scale = 1
        else:
            scale = 1024/b
        qim = QtGui.QImage(int(w * scale), int(h * scale),
                           QtGui.QImage.Format_ARGB32_Premultiplied)
        qim.fill(0)
        painter = QtGui.QPainter()
        painter.begin(qim)
        painter.setRenderHints(painter.Antialiasing | painter.TextAntialiasing)
        self.scene().render(painter, source=rect)
        painter.end()
        return qim

#
#class OrganListModelPopulator(QtCore.QThread):
#    new_dose_item = QtCore.pyqtSignal(str, float)
#    def __init__(self, parent=None):
#        super().__init__(parent)
#        self.organ = None
#        self.dose = None
#        self.organ_map = None
#
#    def run(self):
#        if self.organ is None or self.dose is None or self.organ_map is None:
#            return
#        shape_scale = np.array(self.organ.shape, dtype=np.double) / np.array(self.dose.shape, dtype=np.double)
#        interp = RegularGridInterpolator(tuple(np.arange(self.dose.shape[i]) * shape_scale[i] for i in range(3)),
#                                         self.dose,
#                                         method='nearest',
#                                         bounds_error=False,
#                                         fill_value=0)
#        for key, value in zip(self.organ_map['key'], self.organ_map['value']):
#            points = np.array(np.nonzero(self.organ == key), dtype=np.int).T
#            dose = np.mean(interp(points))
#            if np.isnan(dose):
#                dose = 0.
#            self.new_dose_item.emit(str(value, encoding='utf-8'), round(dose, 2))
#
#
#class OrganListModel(QtGui.QStandardItemModel):
#    def __init__(self, parent=None):
#        super().__init__(parent)
#        self.populator = OrganListModelPopulator()
#        self.populator.new_dose_item.connect(self.add_dose_item)
#
#
#    @QtCore.pyqtSlot(str, float)
#    def add_dose_item(self, organ, dose):
#        item1 = QtGui.QStandardItem(organ)
#        item2 = QtGui.QStandardItem(str(dose))
#        self.layoutAboutToBeChanged.emit()
#
#        self.appendRow([item1, item2])
#        self.sort(0)
#        item2.setData(dose, QtCore.Qt.DisplayRole)
#        self.layoutChanged.emit()
#
#    def set_data(self, dose, organ, organ_map):
#        self.clear()
#        self.setHorizontalHeaderLabels(['Organ', 'Dose [mGy/100mAs]'])
#        if self.populator.isRunning():
#            self.populator.wait()
#        self.populator.dose = dose
#        self.populator.organ = organ
#        self.populator.organ_map = organ_map
#        self.populator.start()
#
#
#
#class OrganDoseWidget(QtGui.QTableView):
#    def __init__(self, parent=None):
#        super().__init__(parent)
#        self.setModel(OrganListModel(self))
#        self.model().layoutChanged.connect(self.resizeColumnToContents)
#        self.name = ""
#        self.setWordWrap(False)
##        self.setTextElideMode(QtCore.Qt.ElideMiddle)
##        self.verticalHeader().setResizeMode(0, QtGui.QHeaderView.ResizeToContents)
#        self.horizontalHeader().setStretchLastSection(True)
##        self.horizontalHeader().setMinimumSectionSize(-1)
##        self.horizontalHeader().setResizeMode(1, QtGui.QHeaderView.Stretch)
##        self.verticalHeader().setResizeMode(QtGui.QHeaderView.Stretch)
#        self.setSortingEnabled(True)
#
#    @QtCore.pyqtSlot()
#    def resizeColumnToContents(self, col=0):
#        super().resizeColumnToContents(col)
#
#    def set_data(self, dose, organ, organ_map):
#        self.model().set_data(dose, organ, organ_map)
