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
import copy
import itertools
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
    simulation_properties_data = QtCore.pyqtSignal(dict)
    scene_selected = QtCore.pyqtSignal(str)
    def __init__(self, database_interface, properties_model, parent=None):
        super().__init__(parent)
        database_interface.request_simulation_view.connect(self.applySimulation)
        database_interface.simulation_updated.connect(self.updateSimulation)
        self.scenes = {'planning': PlanningScene(self),
                       'energy_imparted': DoseScene(self),
                       'running': RunningScene(self),
                       'material': MaterialScene(self),
#                       'dose': DoseScene(self),
                       }

        self.scenes['planning'].request_reload_slice.connect(database_interface.get_array_slice)
        self.scenes['material'].request_reload_slice.connect(database_interface.get_array_slice)
        self.scenes['energy_imparted'].request_reload_slice.connect(database_interface.get_array_slice)

        database_interface.request_array_slice_view.connect(self.scenes['planning'].reload_slice)
        database_interface.request_array_slice_view.connect(self.scenes['material'].reload_slice)
        database_interface.request_array_slice_view.connect(self.scenes['energy_imparted'].reload_slice)

        self.current_simulation = None
        self.current_scene = 'planning'
        self.current_view_orientation = 2

        self.graphicsview = View()


        self.properties_widget = PropertiesWidget(properties_model)
        self.organ_dose_widget = OrganDoseWidget()
        properties_model.request_update_simulation.connect(self.updateSimulation)
        self.simulation_properties_data.connect(properties_model.update_data)
        self.selectScene('planning')


    def view_widget(self):
        wid = QtGui.QWidget()
        main_layout = QtGui.QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        view_layout = QtGui.QVBoxLayout()
        view_layout.setContentsMargins(0, 0, 0, 0)

        main_layout.addWidget(self.properties_widget)

        main_layout.addLayout(view_layout)

        main_layout.addWidget(self.organ_dose_widget)

        menu_widget = QtGui.QMenuBar(wid)
        menu_widget.setContentsMargins(0, 0, 0, 0)
        orientation_action = QtGui.QAction('Orientation', menu_widget)
        orientation_action.triggered.connect(self.selectViewOrientation)

        menu_widget.addAction(orientation_action)

        sceneSelect = SceneSelectGroup(wid)
        for scene_name in self.scenes.keys():
            sceneSelect.addAction(scene_name)
        sceneSelect.scene_selected.connect(self.selectScene)
        self.scene_selected.connect(sceneSelect.sceneSelected)
        for action in sceneSelect.actions():
            menu_widget.addAction(action)

        view_layout.addWidget(menu_widget)

        view_layout.addWidget(self.graphicsview)
        wid.setLayout(main_layout)

#        sub_layout = QtGui.QVBoxLayout()
        return wid

    @QtCore.pyqtSlot()
    def selectViewOrientation(self):
        self.current_view_orientation += 1
        self.current_view_orientation %= 3
        self.scenes[self.current_scene].setViewOrientation(self.current_view_orientation)
        self.graphicsview.fitInView(self.scenes[self.current_scene].sceneRect(), QtCore.Qt.KeepAspectRatio)

    @QtCore.pyqtSlot(str)
    def selectScene(self, scene_name):
        if scene_name in self.scenes:
            self.current_scene = scene_name
            self.graphicsview.setScene(self.scenes[self.current_scene])
            self.update_scene_data(scene_name)

    def update_scene_data(self, name):
        if self.current_simulation is None:
            return
        if not name in self.scenes:
            return

        if name == 'planning':
            self.scenes[name].update_data(self.current_simulation)
#            if self.current_simulation.ctarray is not None:
#                self.scenes[name].setCtArray(self.current_simulation.ctarray,
#                                             self.current_simulation.spacing,
#                                             self.current_simulation.exposure_modulation)
#
#            elif self.current_simulation.organ is not None:
#                self.scenes[name].setBitArray(self.current_simulation.organ,
#                                             self.current_simulation.spacing,
#                                             self.current_simulation.exposure_modulation)

        elif name == 'running':
            if self.current_simulation.energy_imparted is not None:
                self.scenes[name].setArray(self.current_simulation.energy_imparted,
                                           self.current_simulation.spacing,
                                           self.current_simulation.scaling)
            else:
                self.scenes[name].setNoData()
        elif name == 'energy_imparted':
            self.scenes[name].update_data(self.current_simulation)
#            if self.current_simulation.energy_imparted is not None:
#
##                self.scenes[name].setCtDoseArrays(self.current_simulation.ctarray,
##                                                  self.current_simulation.energy_imparted,
##                                                  self.current_simulation.spacing,
##                                                  self.current_simulation.scaling)
#            else:
#                self.scenes[name].setNoData()
        elif name == 'material':
            self.scenes[name].update_data(self.current_simulation)
#            if self.current_simulation.material is not None and self.current_simulation.material_map is not None:
#                self.scenes[name].setMaterialArray(self.current_simulation.material,
#                                                     self.current_simulation.material_map,
#                                                     self.current_simulation.spacing,
#                                                     self.current_simulation.scaling)
#            else:
#                self.scenes[name].setNoData()

        elif name == 'dose':
            try:
                dose = self.current_simulation.dose
            except ValueError:
                self.scenes[name].setNoData()
            else:
                if self.current_simulation.ctarray is not None:
                    background = self.current_simulation.ctarray
                elif self.current_simulation.organ is not None:
                    background = self.current_simulation.organ
                else:
                    background=None
                self.scenes[name].setCtDoseArrays(background,
                                                  dose,
                                                  self.current_simulation.spacing,
                                                  self.current_simulation.scaling)
                self.organ_dose_widget.set_data(dose,
                                                self.current_simulation.organ,
                                                self.current_simulation.organ_map)


        self.scenes[self.current_scene].setViewOrientation(self.current_view_orientation)
        self.graphicsview.fitInView(self.scenes[self.current_scene].sceneRect(),
                                    QtCore.Qt.KeepAspectRatio)
        self.properties_widget.setVisible(name == 'planning')
        self.organ_dose_widget.setVisible(name == 'dose')


    @QtCore.pyqtSlot(Simulation)
    def applySimulation(self, sim):
        self.current_simulation = sim
        logger.debug('Got signal request to view Simulation {}'.format(sim.name))


        if sim.MC_running: ##############################################!!!!!!!!
            scene = 'running'
            self.selectScene(scene)
        else:
            self.update_scene_data(self.current_scene)



    @QtCore.pyqtSlot(dict, dict)
    def updateSimulation(self, description, volatiles):
        if self.current_simulation is None:
            return
        if description.get('name', None) == self.current_simulation.name:
            for key, value in itertools.chain(description.items(), volatiles.items()):
                setattr(self.current_simulation, key, value)
            if self.current_simulation.MC_running and (self.current_simulation.energy_imparted is not None):
                self.selectScene('running')



def intColor(index, hues=9, values=1, maxValue=255, minValue=150, maxHue=360,
             minHue=0, sat=255, alpha=255, firstBlack=True):
    """
    Creates a QColor from a single index. Useful for stepping through a
    predefined list of colors.

    The argument *index* determines which color from the set will be returned.
    All other arguments determine what the set of predefined colors will be

    Colors are chosen by cycling across hues while varying the value
    (brightness).
    By default, this selects from a list of 9 hues."""
    if firstBlack and index == 0:
        return QtGui.QColor(QtCore.Qt.black)
    hues = int(hues)
    values = int(values)
    ind = int(index) % (hues * values)
    indh = ind % hues
    indv = ind / hues
    if values > 1:
        v = minValue + indv * ((maxValue-minValue) / (values-1))
    else:
        v = maxValue
    h = minHue + (indh * (maxHue-minHue)) / hues

    c = QtGui.QColor()
    c.setHsv(h, sat, v)
    c.setAlpha(alpha)
    return c

def blendArrayToQImage(front_array, back_array, front_level, back_level,
                       front_lut, back_lut):
    """Convert the 2D numpy array `gray` into a 8-bit QImage with a gray
    colormap.  The first dimension represents the vertical image axis.
    ATTENTION: This QImage carries an attribute `ndimage` with a
    reference to the underlying numpy array that holds the data. On
    Windows, the conversion into a QPixmap does not copy the data, so
    that you have to take care that the QImage does not get garbage
    collected (otherwise PyQt will throw away the wrapper, effectively
    freeing the underlying memory - boom!)."""

    np.clip(front_array, front_level[0]-front_level[1],
            front_level[0]+front_level[1],
            out=front_array)

    front_array -= (front_level[0]-front_level[1])
    front_array *= 255./(front_level[1]*2.)

    np.clip(back_array, back_level[0]-back_level[1],
            back_level[0]+back_level[1],
            out=back_array)
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

    p.drawImage(QtCore.QRectF(back_qim.rect()), front_qim)

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
        return QtCore.QRectF(0, 0, 2000, 2000)

    def paint(self, painter, style, widget=None):
        painter.setPen(QtGui.QPen(QtCore.Qt.white))

#        h = self.fontMetrics.boundingRect('A').height()
#        painter.drawText(0, h ,self.msg)

        painter.drawText(QtCore.QRectF(0, 0, 2000, 2000), QtCore.Qt.AlignCenter, self.msg)
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
        self.front_alpha = 127
        self.back_lut = get_lut('gray', self.back_alpha)
        self.front_lut = get_lut('pet', self.front_alpha)

        self.qimage = None

    def qImage(self):
        if self.qimage is None:
            self.render()
        return self.qimage

    def boundingRect(self):
        return QtCore.QRectF(self.qImage().rect())

    def setImage(self, front_image=None, back_image=None):
        if front_image is not None:
            self.front_image = front_image
        if back_image is not None:
            self.back_image = back_image
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


class PlanningScene(QtGui.QGraphicsScene):
    request_reload_slice = QtCore.pyqtSignal(str, str, int, int)
    def __init__(self, parent=None, lut='pet'):
        super().__init__(parent)
        self.image_item = ImageItem()
        self.image_item_bit = BitImageItem()
        self.addItem(self.image_item)
        self.addItem(self.image_item_bit)

        self.name = ''
        self.aec_item = AecItem()
        self.addItem(self.aec_item)
        self.shape = np.array((1., 1., 1.))
        self.array_name = 'ctarray'
        self.spacing = np.array((1., 1., 1.))
        self.index = 0
        self.view_orientation = 2

        self.image_item.setLevels((0, 500))
        self.is_bit_array = False
        self.lut = get_lut(lut)

    def update_data(self, sim):
        self.is_bit_array = sim.is_phantom
        self.image_item.setVisible(not self.is_bit_array)
        self.image_item_bit.setVisible(self.is_bit_array)
        self.name = sim.name
        if self.is_bit_array:
            self.array_name = 'organ'
            organ_max_value = sim.organ_map['key'].max()
            lut =  [self.lut[i*255 // organ_max_value] for i in range(organ_max_value + 1)]
            self.image_item_bit.set_lut(lut)
        else:
            self.array_name = 'ctarray'
        self.spacing = sim.spacing
        self.shape = sim.shape
        self.index = self.index % self.shape[self.view_orientation]
        self.aec_item.set_aec(sim.exposure_modulation, self.view_orientation, self.shape)
        self.updateSceneTransform()
        self.request_reload_slice.emit(self.name, self.array_name, self.index, self.view_orientation)

    @QtCore.pyqtSlot(int)
    def setViewOrientation(self, view_orientation):
        self.view_orientation = view_orientation
        self.aec_item.setViewOrientation(view_orientation)
        self.request_reload_slice.emit(self.name, self.array_name, self.index, self.view_orientation)
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
        if array_name not in ['ctarray', 'organ']:
            self.index = index
            return
        if self.is_bit_array:
            self.image_item_bit.setImage(arr)
        else:
            self.image_item.setImage(arr)
        self.aec_item.setIndex(self.index)

    def wheelEvent(self, ev):
        if ev.delta() > 0:
            self.index += 1
        elif ev.delta() < 0:
            self.index -= 1
        self.index %= self.shape[self.view_orientation]
        self.request_reload_slice.emit(self.name, self.array_name, self.index, self.view_orientation)
        ev.accept()

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


class RunningScene(QtGui.QGraphicsScene):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_item = ImageItem()
        self.addItem(self.image_item)
        self.image_item.setLut(get_lut('hot_iron'))
        self.nodata_item = NoDataItem()
        self.addItem(self.nodata_item)
        self.nodata_item.setVisible(True)
        self.array = np.random.uniform(size=(8, 8, 8))
        self.shape = np.array((8, 8, 8))
        self.spacing = np.array((1., 1., 1.))
        self.view_orientation = 2

    def setNoData(self):
        self.array = np.zeros((2, 2, 2))
        self.shape = self.array.shape
        self.image_item.setVisible(False)
        self.nodata_item.setVisible(True)

    def defaultLevels(self, array):
        p = array.max() - array.min()
        return (p/2., p / 2. )

    def setArray(self, energy_imparted, spacing, scaling):
        self.array = energy_imparted
        self.shape = energy_imparted.shape
        self.spacing = spacing * scaling
        self.nodata_item.setVisible(False)
        self.image_item.setVisible(True)
        self.reloadImages()
        self.image_item.setLevels(self.defaultLevels(self.array))
        self.updateSceneTransform()

    @QtCore.pyqtSlot(int)
    def setViewOrientation(self, view_orientation):
        self.view_orientation = view_orientation
        self.reloadImages()
        self.updateSceneTransform()

    def updateSceneTransform(self):
        sx, sy = [self.spacing[i] for i in range(3) if i != self.view_orientation]
        transform = QtGui.QTransform.fromScale(sy / sx, 1.)
        self.image_item.setTransform(transform)
        if self.nodata_item.isVisible():
            self.setSceneRect(self.nodata_item.sceneBoundingRect())
        else:
            self.setSceneRect(self.image_item.sceneBoundingRect())

    def reloadImages(self):
        self.image_item.setImage(self.array.max(axis=self.view_orientation))



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
            self.map.append((colors[ind], str(mapping['value'][ind], encoding='utf-8')))

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


class MaterialScene(QtGui.QGraphicsScene):
    request_reload_slice = QtCore.pyqtSignal(str, str, int, int)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.lut = get_lut('pet')
        self.image_item = BitImageItem()
        self.addItem(self.image_item)
        self.map_item = MaterialMapItem()
        self.addItem(self.map_item)
        self.nodata_item = NoDataItem()
        self.addItem(self.nodata_item)
        self.array_name = 'material'
#        self.array = np.random.uniform(size=(8, 8, 8))
        self.shape = np.array((8, 8, 8))
        self.spacing = np.array((1., 1., 1.))
        self.index = 0
        self.view_orientation = 2
        self.name = ''

    def setNoData(self):
#        self.array = np.zeros((2, 2, 2))
        self.spacing = np.array((1., 1., 1.))
        self.shape = np.array((8, 8, 8))
        self.nodata_item.setVisible(True)
        self.map_item.setVisible(False)
        self.image_item.setVisible(False)
        self.setSceneRect(self.nodata_item.sceneBoundingRect())

    def update_data(self, sim):
        if sim.material_map is None:
            self.setNoData()
            return
        self.nodata_item.setVisible(False)
        self.map_item.setVisible(True)
        self.image_item.setVisible(True)

        self.name = sim.name
        self.spacing = sim.spacing * sim.scaling
        self.shape = sim.shape / sim.scaling
        self.index = self.index % self.shape[self.view_orientation]
        organ_max_value = sim.material_map['key'].max()
        lut =  [self.lut[i*255 // organ_max_value] for i in range(organ_max_value + 1)]
        self.image_item.set_lut(lut)

        self.map_item.set_map(sim.material_map, lut)
        self.request_reload_slice.emit(self.name, self.array_name, self.index, self.view_orientation)
        self.updateSceneTransform()

    @QtCore.pyqtSlot(int)
    def setViewOrientation(self, view_orientation):
        self.view_orientation = view_orientation
        self.updateSceneTransform()
        self.request_reload_slice.emit(self.name, self.array_name, self.index, self.view_orientation)

    def updateSceneTransform(self):
        sx, sy = [self.spacing[i] for i in range(3) if i != self.view_orientation]
        transform = QtGui.QTransform.fromScale(sy / sx, 1.)
        self.image_item.setTransform(transform)

        self.map_item.prepareGeometryChange()

        shape = tuple(sh for ind, sh in enumerate(self.shape) if ind != self.view_orientation)
        rect = QtCore.QRectF(0, 0, shape[1], shape[0])
        self.map_item.setScale(rect.height() / self.map_item.boundingRect().height())
        self.map_item.setPos(self.image_item.mapRectToScene(rect).topRight())
        self.setSceneRect(self.image_item.mapRectToScene(rect).united(self.map_item.sceneBoundingRect()))

    @QtCore.pyqtSlot(str, np.ndarray, str, int, int)
    def reload_slice(self, simulation_name, arr, array_name, index, orientation):
        if simulation_name != self.name:
            return
        if array_name != 'material':
            self.index = index
            return
        self.image_item.setImage(arr)

    def wheelEvent(self, ev):
        if ev.delta() > 0:
            self.index += 1
        elif ev.delta() < 0:
            self.index -= 1
        self.index %= self.shape[self.view_orientation]
        self.request_reload_slice.emit(self.name, self.array_name, self.index, self.view_orientation)
        ev.accept()



class DoseScene(QtGui.QGraphicsScene):
    request_reload_slice = QtCore.pyqtSignal(str, str, int, int)
    def __init__(self, parent=None, front_array='energy_imparted'):
        super().__init__(parent)
        self.name = ''
        self.image_item = BlendImageItem()
        self.addItem(self.image_item)
        self.nodata_item = NoDataItem()
        self.addItem(self.nodata_item)
        self.name1 = ''
        self.name2 = 'energy_imparted'
        self.max2_value = 0
        self.shape = np.array((8, 8, 8))
        self.spacing = np.array((1., 1., 1.))
        self.scaling = np.array((1., 1., 1.))
        self.index = 0
        self.view_orientation = 2
        self.nodata_item.setVisible(True)
        self.image_item.setVisible(False)

    def setNoData(self):
        self.dose_array = np.zeros((2, 2, 2))
        self.ct_array = np.zeros((2, 2, 2))
        self.shape = self.dose_array.shape
        self.nodata_item.setVisible(True)
        self.image_item.setVisible(False)

    def update_data(self, sim):
        if sim.is_phantom:
            self.name1 = 'organ'
            max_level = max(sim.organ_map['key'])
            self.image_item.setLevels(back=(max_level/2, max_level/2))
        else:
            self.name1 = 'ctarray'
            self.image_item.setLevels(back=(0, 500))
        self.nodata_item.setVisible(False)
        self.image_item.setVisible(True)
        self.max2_value = 0
        self.name = sim.name
        self.shape = sim.shape
        self.scaling = sim.scaling
        self.spacing = sim.spacing
        self.updateSceneTransform()
        self.request_reload_slice.emit(self.name, self.name1, self.index, self.view_orientation)
        index2 = self.index % self.shape[self.view_orientation]
        index2 /= self.scaling[self.view_orientation]
        self.request_reload_slice.emit(self.name, self.name2, index2, self.view_orientation)

#    def setCtDoseArrays(self, ct, dose, spacing, scaling):
#        self.dose_array = gaussian_filter(dose, 1.)
##        self.dose_array = dose
#        if ct is None:
#            self.ct_array = np.zeros(dose.shape, dtype=np.int16)
#
#        else:
#            self.ct_array = ct
#        self.shape = np.array(self.ct_array.shape)
#        self.spacing = spacing
#        self.dose_scale = scaling
#        self.nodata_item.setVisible(False)
#        self.image_item.setVisible(True)
#        self.index = self.index % self.shape[self.view_orientation]
#        self.reloadImages()
#        self.updateSceneTransform()
#        self.image_item.setLevels(front=self.defaultLevels(self.dose_array),
#                                  back=self.defaultLevels(self.ct_array))

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




    @QtCore.pyqtSlot(int)
    def setViewOrientation(self, view_orientation):
        self.view_orientation = view_orientation % 3
        self.request_reload_slice.emit(self.name, self.name1, self.index, self.view_orientation)
        index2 = self.index % self.shape[self.view_orientation]
        index2 /= self.scaling[self.view_orientation]
        self.request_reload_slice.emit(self.name, self.name2, index2, self.view_orientation)
        self.updateSceneTransform()

    @QtCore.pyqtSlot(str, np.ndarray, str, int, int)
    def reload_slice(self, simulation_name, arr, array_name, index, orientation):
        if simulation_name != self.name:
            return
        if array_name in ['ctarray', 'organ']:
            self.image_item.setImage(back_image=arr)

        elif array_name == 'energy_imparted':
            m = arr.max()
            if self.max2_value < m:

                self.image_item.setLevels(front=(m/2., m/2))
                self.max2_value = m
            self.image_item.setImage(front_image=arr)
        else:
            self.index = index

#    def getSlice(self, array, index):
#        if self.view_orientation == 2:
#            return np.copy(np.squeeze(array[: ,: ,index % self.shape[2]]))
#        elif self.view_orientation == 1:
#            return np.copy(np.squeeze(array[:, index % self.shape[1], :]))
#        elif self.view_orientation == 0:
#            return np.copy(np.squeeze(array[index % self.shape[0], :, :]))
#        raise ValueError('view must select one of 0,1,2 dimensions')
#
#    def reloadImages(self):
#        n = self.shape[self.view_orientation]
#        dose_index = np.floor((self.index % self.shape[self.view_orientation]) / n * self.dose_array.shape[self.view_orientation])
#
#        self.image_item.setImage(self.getSlice(self.dose_array, dose_index),
#                                 self.getSlice(self.ct_array, self.index))

    def wheelEvent(self, ev):
        if ev.delta() > 0:
            self.index += 1
        elif ev.delta() < 0:
            self.index -= 1
        self.request_reload_slice.emit(self.name, self.name1, self.index, self.view_orientation)
        index2 = self.index % self.shape[self.view_orientation]
        index2 /= self.scaling[self.view_orientation]
        self.request_reload_slice.emit(self.name, self.name2, index2, self.view_orientation)
        ev.accept()




class Scene(QtGui.QGraphicsScene):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.shape = np.array([1, 1, 1], dtype=np.int)
        self.spacing = np.array([1, 1, 1])
        self.array_name = ''
        self.view_orientation = 2
        self.index = 0

        self.image_items= {'ctarray': ImageItem(),
                           }

        self.image_item = ImageItem()
        self.image_item = BlendImageItem()
        self.addItem(self.image_item)




class View(QtGui.QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setContentsMargins(0, 0, 0, 0)
        self.setBackgroundBrush(QtGui.QBrush(QtCore.Qt.black))

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self.fitInView(self.sceneRect(), QtCore.Qt.KeepAspectRatio)

    def setScene(self, scene):
        super().setScene(scene)
        self.fitInView(self.sceneRect(), QtCore.Qt.KeepAspectRatio)

class PropertiesModel(QtCore.QAbstractTableModel):
#    error_setting_value = QtCore.pyqtSignal(str)
#    request_simulation_update = QtCore.pyqtSignal(dict)
#    request_simulation_start = QtCore.pyqtSignal()
    request_update_simulation = QtCore.pyqtSignal(dict, dict, bool, bool)
    unsaved_data_changed = QtCore.pyqtSignal(bool)
    properties_is_set = QtCore.pyqtSignal(bool)

    def __init__(self, interface, parent=None):
        super().__init__(parent)
        self.__data = copy.copy(SIMULATION_DESCRIPTION)
        self.unsaved_data = {}
        self.__indices = list(self.__data.keys())
        self.__indices.sort()
        interface.request_simulation_view.connect(self.set_data)
        interface.simulation_updated.connect(self.update_data)
        self.request_update_simulation.connect(interface.update_simulation_properties)
        self.__simulation = Simulation('None')


    def properties_data(self):
        return self.__data, self.__indices

    @QtCore.pyqtSlot()
    def reset_properties(self):
        self.unsaved_data = {}
        self.dataChanged.emit(self.createIndex(0,0), self.createIndex(len(self.__indices)-1 , 1))
        self.test_for_unsaved_changes()

    @QtCore.pyqtSlot()
    def apply_properties(self):
        self.__init_data = self.__data
        self.unsaved_data['name'] = self.__data['name'][0]
        self.unsaved_data['MC_ready'] = True
        self.unsaved_data['MC_finished'] = False
        self.unsaved_data['MC_running'] = False
        self.test_for_unsaved_changes()
        self.request_update_simulation.emit(self.unsaved_data, {}, True, True)
        self.properties_is_set.emit(True)
#        self.request_simulation_update.emit({key: value[0] for key, value in self.__data.items()})
        self.unsaved_data = {}
        self.test_for_unsaved_changes()

#    @QtCore.pyqtSlot()
#    def run_simulation(self):
#        self.__data['MC_running'][0] = True
#        self.__data['MC_ready'][0] = True
##        self.request_simulation_update.emit({key: value[0] for key, value in self.__data.items()})
#        self.unsaved_data_changed.emit(False)
##        self.request_simulation_start.emit()

    def test_for_unsaved_changes(self):
        for key, value in self.__simulation.description.items():
            if self.__data[key][3]:
                if isinstance(self.__data[key][0], np.ndarray):
                    if (value - self.__data[key][0]).sum() != 0.0:
                        self.unsaved_data[key] = value
                elif self.__data[key][0] != value:
                    self.unsaved_data[key] = value
        self.unsaved_data_changed.emit(len(self.unsaved_data) > 0)
        self.layoutAboutToBeChanged.emit()
        self.layoutChanged.emit()

    @QtCore.pyqtSlot(Simulation)
    def set_data(self, sim):
        sim_description = sim.description
        self.update_data(sim_description, {})

    @QtCore.pyqtSlot(dict, dict)
    def update_data(self, sim_description, array_dict):
        self.unsaved_data = {}
        self.layoutAboutToBeChanged.emit()
        self.__simulation = Simulation('None', sim_description)
        for key, value in sim_description.items():
            self.__data[key][0] = value

        self.dataChanged.emit(self.createIndex(0,0), self.createIndex(len(self.__indices)-1 , 1))
        self.layoutChanged.emit()
        self.test_for_unsaved_changes()
        self.properties_is_set.emit(self.__data['MC_running'][0])

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

        var = self.__indices[row]
        if column == 0:
            value = self.__data[var][4]
        else:
            value = self.unsaved_data.get(var, self.__data[var][0])

        if role == QtCore.Qt.DisplayRole:
            if (column == 1) and isinstance(value, np.ndarray):
                return ' '.join([str(round(p, 3)) for p in value])
            elif (column == 1) and isinstance(value, bool):
                return ''
            return value
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
            if (column == 1) and isinstance(value, bool):
                if value:
                    return QtCore.Qt.Checked
                else:
                    return QtCore.Qt.Unchecked
        return None

    def setData(self, index, value, role):
        if not index.isValid():
            return False
        if index.column() != 1:
            return False
        if role == QtCore.Qt.DisplayRole or role == QtCore.Qt.EditRole:
#            var = self.__indices[index.row()]
#            self.unsaved_data[var] = value
#            self.dataChanged.emit(index, index)
#            return True
#        elif role == QtCore.Qt.EditRole:
            var = self.__indices[index.row()]
            try:
                setattr(self.__simulation, var, value)
            except Exception as e:
                logger.error(str(e))
                return False
            else:
                if value != self.__data[var][0]:
                    self.unsaved_data[var] = value
                else:
                    try:
                        del self.unsaved_data[var]
                    except KeyError:
                        pass

            self.dataChanged.emit(index, index)
            self.test_for_unsaved_changes()
            return True
        elif role == QtCore.Qt.CheckStateRole:
            var = self.__indices[index.row()]
            if self.__data[var][0] != bool(value == QtCore.Qt.Checked):
                self.unsaved_data[var] = bool(value == QtCore.Qt.Checked)
            else:
                if var in self.unsaved_data:
                    del self.unsaved_data[var]
            self.test_for_unsaved_changes()
            self.dataChanged.emit(index, index)
            return True

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        return str(section)

    def flags(self, index):
        if index.isValid():
            if self.__data[self.__indices[index.row()]][3] and index.column() == 1:
                if self.unsaved_data.get('MC_running', False):
                    return QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled
                if isinstance(self.__data[self.__indices[index.row()]][0], bool):
                    return  QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsUserCheckable# | QtCore.Qt.ItemIsEditable
                return QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsEditable
            return QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled
        return QtCore.Qt.NoItemFlags

class ArrayEdit(QtGui.QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent)

    def set_data(self, value):
        self.setText(' '.join([str(r) for r in value]))


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
        elif isinstance(data[var][0], np.ndarray):
            return ArrayEdit(parent)
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
    def __init__(self, properties_model, parent=None):
        super().__init__(parent)
        self.setModel(properties_model)
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
    def __init__(self, properties_model, parent=None):
        super().__init__(parent)
        self.setLayout(QtGui.QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        view = PropertiesView(properties_model)
        self.layout().addWidget(view)

        apply_button = QtGui.QPushButton()
        apply_button.setText('Reset')
        apply_button.clicked.connect(properties_model.reset_properties)
        apply_button.setEnabled(False)
        properties_model.unsaved_data_changed.connect(apply_button.setEnabled)

        run_button = QtGui.QPushButton()
        run_button.setText('Apply and Run')
        run_button.clicked.connect(properties_model.apply_properties)
        properties_model.properties_is_set.connect(run_button.setDisabled)


#        run_button = QtGui.QPushButton()
#        run_button.setText('Run')
#        run_button.clicked.connect(properties_model.request_simulation_start)

        button_layout = QtGui.QHBoxLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.addWidget(apply_button)
        button_layout.addWidget(run_button)

        self.layout().addLayout(button_layout)

class OrganListModelPopulator(QtCore.QThread):
    new_dose_item = QtCore.pyqtSignal(str, float)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.organ = None
        self.dose = None
        self.organ_map = None

    def run(self):
        if self.organ is None or self.dose is None or self.organ_map is None:
            return
        shape_scale = np.array(self.organ.shape, dtype=np.double) / np.array(self.dose.shape, dtype=np.double)
        interp = RegularGridInterpolator(tuple(np.arange(self.dose.shape[i]) * shape_scale[i] for i in range(3)),
                                         self.dose,
                                         method='nearest',
                                         bounds_error=False,
                                         fill_value=0)
        for key, value in zip(self.organ_map['key'], self.organ_map['value']):
            points = np.array(np.nonzero(self.organ == key), dtype=np.int).T
            dose = np.mean(interp(points))
            if np.isnan(dose):
                dose = 0.
            self.new_dose_item.emit(str(value, encoding='utf-8'), round(dose, 2))


class OrganListModel(QtGui.QStandardItemModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.populator = OrganListModelPopulator()
        self.populator.new_dose_item.connect(self.add_dose_item)


    @QtCore.pyqtSlot(str, float)
    def add_dose_item(self, organ, dose):
        item1 = QtGui.QStandardItem(organ)
        item2 = QtGui.QStandardItem(str(dose))
        self.layoutAboutToBeChanged.emit()

        self.appendRow([item1, item2])
        self.sort(0)
        item2.setData(dose, QtCore.Qt.DisplayRole)
        self.layoutChanged.emit()

    def set_data(self, dose, organ, organ_map):
        self.clear()
        self.setHorizontalHeaderLabels(['Organ', 'Dose [mGy/100mAs]'])
        if self.populator.isRunning():
            self.populator.wait()
        self.populator.dose = dose
        self.populator.organ = organ
        self.populator.organ_map = organ_map
        self.populator.start()



class OrganDoseWidget(QtGui.QTableView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setModel(OrganListModel(self))
        self.model().layoutChanged.connect(self.resizeColumnToContents)
        self.name = ""
        self.setWordWrap(False)
#        self.setTextElideMode(QtCore.Qt.ElideMiddle)
#        self.verticalHeader().setResizeMode(0, QtGui.QHeaderView.ResizeToContents)
        self.horizontalHeader().setStretchLastSection(True)
#        self.horizontalHeader().setMinimumSectionSize(-1)
#        self.horizontalHeader().setResizeMode(1, QtGui.QHeaderView.Stretch)
#        self.verticalHeader().setResizeMode(QtGui.QHeaderView.Stretch)
        self.setSortingEnabled(True)

    @QtCore.pyqtSlot()
    def resizeColumnToContents(self, col=0):
        super().resizeColumnToContents(col)

    def set_data(self, dose, organ, organ_map):
        self.model().set_data(dose, organ, organ_map)
