# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 10:38:15 2015

@author: erlean
"""
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from PyQt4 import QtGui, QtCore
from opendxmc.study import Simulation
from .dicom_lut import get_lut


import logging
logger = logging.getLogger('OpenDXMC')

class SceneSelectButton(QtGui.QPushButton):
    scene_selected = QtCore.pyqtSignal(str)
    def __init__(self, name, parent=None):
        super().__init__(parent)
        self.name = name
        self.setFlat(True)
        self.setText(self.name)

    @QtCore.pyqtSlot()
    def relay_clicked(self):
        self.scene_selected.emit(self.name)


class ViewController(QtCore.QObject):

    def __init__(self, database_interface, parent=None):
        super().__init__(parent)
        database_interface.request_simulation_view.connect(self.applySimulation)

        self.scenes = {'planning': PlanningScene(self),
                       'energy_imparted': DoseScene(self),}
#                       'material': PlanningScene()}
        self.current_scene = 'planning'
        self.current_view_orientation = 2
        self.graphicsview = View()
        self.graphicsview.setScene(self.scenes['planning'])

        self.scene_change_buttons = {}
        for key, value in self.scenes.items():
            self.scene_change_buttons[key] = SceneSelectButton(key)
            self.scene_change_buttons[key].scene_selected.connect(self.selectScene)


    def view_widget(self):
        wid = QtGui.QWidget()
        main_layout = QtGui.QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)

        menu_widget = QtGui.QMenuBar(wid)
        menu_widget.setContentsMargins(0, 0, 0, 0)
        orientation_action = QtGui.QAction('Orientation', menu_widget)
        orientation_action.triggered.connect(self.selectViewOrientation)
        menu_widget.addAction(orientation_action)
        main_layout.addWidget(menu_widget)

        main_layout.addWidget(self.graphicsview)
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
        self.current_scene = scene_name
        self.graphicsview.setScene(self.scenes[self.current_scene])

    @QtCore.pyqtSlot(Simulation)
    def applySimulation(self, sim):
        logger.debug('Got signal request to view Simulation {}'.format(sim.name))
        if sim.energy_imparted is not None:

            logger.debug('View request for Simulation {} denied: No dose array available'.format(sim.name))
        if sim.ctarray is not None:
            self.scenes['planning'].setCtArray(sim.ctarray, sim.spacing, sim.exposure_modulation)
            logger.debug('Populating planning scene'.format(sim.name))
        else:
            pass

        scene = self.scenes['planning']
        self.graphicsview.setScene(scene)
        self.graphicsview.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)


def blendArrayToQImage(front_array, back_array, front_level, back_level, front_lut, back_lut):
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

    front_qim = QtGui.QImage(front_array.data, front_w, front_h,
                             QtGui.QImage.Format_Indexed8)
    back_qim = QtGui.QImage(back_array.data, back_w, back_h,
                            QtGui.QImage.Format_Indexed8)
    front_qim.setColorTable(front_lut)
    back_qim.setColorTable(back_lut)
#    front_qim = front_qim.convertToFormat(QtGui.QImage.Format_ARGB32_Premultiplied, front_lut, flags=QtCore.Qt.DiffuseAlphaDither)
    back_qim = back_qim.convertToFormat(QtGui.QImage.Format_ARGB32_Premultiplied, back_lut)#, flags=QtCore.Qt.DiffuseAlphaDither)


    p = QtGui.QPainter(back_qim)

#    p.drawImage(QtCore.QPointF(0., 0.), back_qim)
#    p.setCompositionMode(QtGui.QPainter.CompositionMode_Plus)
#    p.setCompositionMode(QtGui.QPainter.CompositionMode_Xor)
    p.drawImage(QtCore.QPointF(0., 0.), front_qim)

    return back_qim


def arrayToQImage(array, level, lut):
    """Convert the 2D numpy array `gray` into a 8-bit QImage with a gray
    colormap.  The first dimension represents the vertical image axis.
    ATTENTION: This QImage carries an attribute `ndimage` with a
    reference to the underlying numpy array that holds the data. On
    Windows, the conversion into a QPixmap does not copy the data, so
    that you have to take care that the QImage does not get garbage
    collected (otherwise PyQt will throw away the wrapper, effectively
    freeing the underlying memory - boom!)."""
    WC, WW = level[0], level[1]
    np.clip(array, WC-WW, WC+WW, out=array)
    array -= (WC - WW)
    array *= 255./(WW*2)

#    array = (np.clip(array, WC - 0.5 - (WW-1) / 2, WC - 0.5 + (WW - 1) / 2) -
#             (WC - 0.5 - (WW - 1) / 2)) * 255 / ((WC - 0.5 + (WW - 1) / 2) -
#                                                 (WC - 0.5 - (WW - 1) / 2))
    array = np.require(array, np.uint8, 'C')
    h, w = array.shape
    result = QtGui.QImage(array.data, w, h, QtGui.QImage.Format_Indexed8)
#    result.ndarray = array
#    result.setColorTable(lut)
    result = result.convertToFormat(QtGui.QImage.Format_ARGB32, lut)
    result.ndarray = array
    return result


class BlendImageItem(QtGui.QGraphicsItem):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.back_image = np.zeros((512, 512))
        self.back_level = (500, 1000)
        self.front_image = np.zeros((512, 512))
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

    def setImage(self, front_image, back_image):
        self.front_image = front_image
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
        painter.drawImage(QtCore.QPointF(self.pos()), self.qimage)


class ImageItem(QtGui.QGraphicsItem):
    def __init__(self, parent=None, image=None, level=None, shape=None):
        super().__init__(parent)
        self.setFlag(QtGui.QGraphicsItem.ItemSendsGeometryChanges)
        if image is None:
            if shape is None:
                shape = (512, 512)
            self.image = np.random.uniform(-500, 500,size=shape)
        else:
            self.image = image.view(np.ndarray)
        if image is not None and level is None:
            mi = image.min()
            ma = image.max() + 1
            self.level = ((ma - mi) / 2, ) * 2
        elif level is None:
            self.level = (0, 700)
        else:
            self.level = level

        self.prepareGeometryChange()
        self.qimage = None

        self.lut = get_lut('gray')

        self.setImage(np.random.normal(size=(512, 512)) * 500.)
#        self.setLevels((0., 1.))

    def qImage(self):
        if self.qimage is None:
            self.render()
        return self.qimage

    def boundingRect(self):
        return QtCore.QRectF(self.qImage().rect())

    def setImage(self, image):
        self.image = image.view(np.ndarray)
        self.prepareGeometryChange()
        self.qimage = None
        self.update(self.boundingRect())

    def setLevels(self, level):
        self.level = level
        self.qimage = None
        self.update(self.boundingRect())

    def render(self):
        self.qimage = arrayToQImage(self.image, self.level,
                                    self.lut)

    def shape(self):
        path = QtGui.QPainterPath()
        path.addEllipse(self.boundingRect())
        return path

    def paint(self, painter, style, widget=None):
        painter.drawImage(QtCore.QPointF(self.pos()), self.qImage())
#        super(ImageItem, self).paint(painter, style, widget)

class AecItem(QtGui.QGraphicsItem):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.aec = np.zeros((5, 2))
        self.view_orientation = 2
        self.shape = (1, 1, 1)
        self.index = 0
        self.__path = None
        self.__path_pos = None

    def set_aec(self, aec, view_orientation, shape):
        self.shape = shape
        self.view_orientation = view_orientation

        self.aec = aec
        self.aec[:, 0] -= self.aec[:, 0].min()
        self.aec[:, 0] /= self.aec[:, 0].max()

        self.aec[:, 1] -= self.aec[:, 1].min()
        self.aec[:, 1] /= self.aec[:, 1].max()

        self.__path = None
        self.__path_pos = None
        self.prepareGeometryChange()

        self.update(self.boundingRect())

    def boundingRect(self):
        shape = tuple(self.shape[i] for i in [0, 1, 2] if i != self.view_orientation)
        return QtCore.QRectF(0, 0, shape[1], shape[0]*.1)

    def setIndex(self, index):
        self.index = index % self.shape[self.view_orientation]
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
            shape = tuple(self.shape[i] for i in [0, 1, 2] if i != self.view_orientation)
            self.__path = QtGui.QPainterPath()

            x = self.aec[:, 0] * shape[1]
            y = (1.-self.aec[:, 1]) * shape[0] *.1
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
                shape = tuple(self.shape[i] for i in [0, 1, 2] if i != self.view_orientation)
                self.__path_pos = QtGui.QPainterPath()
                x = self.aec[:, 0] * shape[1]
                y = (1.-self.aec[:, 1]) * shape[0] *.1
                x_c = self.index / self.shape[2]
                y_c = 1. - np.interp(x_c, self.aec[:, 0], self.aec[:, 1])
                self.__path_pos.addEllipse(QtCore.QPointF(x_c * shape[1], y_c * shape[0] *.1), shape[0]*.005, shape[0]*.01)

        p = QtGui.QPainterPath()
        p.addPath(self.__path)
        p.addPath(self.__path_pos)
        return p

    def paint(self, painter, style, widget=None):
#        if self.view_orientation == 2:
#            return
#        shape = tuple(self.shape[i] for i in [0, 1, 2] if i != self.view_orientation)

        painter.setPen(QtGui.QPen(QtCore.Qt.white))
        painter.setRenderHint(painter.Antialiasing, True)
        painter.drawPath(self.aec_path())


class PlanningScene(QtGui.QGraphicsScene):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_item = ImageItem()
        self.addItem(self.image_item)
        self.aec_item = AecItem()
        self.addItem(self.aec_item)

        self.array = np.random.uniform(size=(8, 8, 8))
        self.shape = np.array((8, 8, 8))
        self.spacing = np.array((1., 1., 1.))
        self.index = 0
        self.view_orientation = 2

    @QtCore.pyqtSlot(np.ndarray, np.ndarray)
    def setCtArray(self, ct, spacing, aec):
        #setting transform


#        self.aec_item.setPos(self.image_item.boundingRect().center())


        self.array = ct
        self.shape = ct.shape
        self.spacing = spacing
        self.index = self.index % self.shape[self.view_orientation]
        self.aec_item.set_aec(aec, self.view_orientation, ct.shape)
        self.reloadImages()
        self.updateSceneTransform()



        logger.debug('Planningscene is setting image data')
#        self.reloadImages()


    @QtCore.pyqtSlot(int)
    def setViewOrientation(self, view_orientation):
        self.view_orientation = view_orientation
        self.aec_item.setViewOrientation(view_orientation)
        self.reloadImages()
        self.updateSceneTransform()

    def updateSceneTransform(self):
        sx, sy = [self.spacing[i] for i in range(3) if i != self.view_orientation]
        transform = QtGui.QTransform.fromScale(sy / sx, 1.)
        self.image_item.setTransform(transform)
        self.aec_item.setTransform(transform)

        self.aec_item.prepareGeometryChange()
        self.aec_item.setPos(self.image_item.mapToScene(self.image_item.boundingRect().bottomLeft()))

#        self.image_item.prepareGeometryChange()



        self.setSceneRect(self.itemsBoundingRect())

    def getSlice(self, array, index):
        if self.view_orientation == 2:
            return np.copy(np.squeeze(array[: ,: ,index % self.shape[self.view_orientation]]))
        elif self.view_orientation == 1:
            return np.copy(np.squeeze(array[:, index % self.shape[self.view_orientation], :]))
        elif self.view_orientation == 0:
            return np.copy(np.squeeze(array[index % self.shape[self.view_orientation], :, :]))
        raise ValueError('view must select one of 0,1,2 dimensions')


    def reloadImages(self):
        self.image_item.setImage(self.getSlice(self.array, self.index))
        self.aec_item.setIndex(self.index)

    def wheelEvent(self, ev):
        if ev.delta() > 0:
            self.index += 1
        elif ev.delta() < 0:
            self.index -= 1
        self.index %= self.shape[self.view_orientation]
        self.reloadImages()
        ev.accept()

class DoseScene(QtGui.QGraphicsScene):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.image_item = BlendImageItem()
        self.addItem(self.image_item)

        self.dose_array = None
        self.ct_array = None
        self.view_orientation = 2  # value 0, 1, 2
        self.shape = (0, 0, 0)
        self.index = 0
        self.spacing = (1., 1., 1.)

    @QtCore.pyqtSlot(np.ndarray, np.ndarray, np.ndarray)
    def setCtDoseArrays(self, ct, dose, spacing):
        self.dose_array = gaussian_filter(dose, 1.)
        #setting transform
        sx, sy = [spacing[i] for i in range(3) if i != self.view_orientation]
        transform = QtGui.QTransform.fromScale(sy / sx, 1.)
        self.image_item.setTransform(transform)
        self.image_item.setLevels(front=(self.dose_array.max()/2.,self.dose_array.max()/2.))
        self.ct_array = ct
        self.shape = ct.shape
        self.spacing = spacing
        self.index = 0

        rect = transform.mapRect(self.image_item.boundingRect())

        self.setSceneRect(rect)
        for view in self.views():
            view.fitInView(rect, QtCore.Qt.KeepAspectRatio)
        logger.debug('Dosescene is setting image data')
        self.reloadImages()

    @QtCore.pyqtSlot(np.ndarray)
    def setDoseArray(self, dose):
        self.dose_array = dose
        self.reloadImages()

    def getSlice(self, array, index):
        if self.view_orientation == 2:
            return np.copy(np.squeeze(array[: ,: ,index % self.shape[self.view_orientation]]))
        elif self.view_orientation == 1:
            return np.copy(np.squeeze(array[:, index % self.shape[self.view_orientation], :]))
        elif self.view_orientation == 0:
            return np.copy(np.squeeze(array[index % self.shape[self.view_orientation], :, :]))
        raise ValueError('view must select one of 0,1,2 dimensions')


    def reloadImages(self):
        self.image_item.setImage(self.getSlice(self.dose_array, self.index),
                                 self.getSlice(self.ct_array, self.index))

    def wheelEvent(self, ev):
        if ev.delta() > 0:
            self.index += 1
        elif ev.delta() < 0:
            self.index -= 1
        self.reloadImages()
        ev.accept()

#    def mouseMoveEvent(self, ev):
#        if ev.button() == QtCore.Qt.LeftButton:
#        elif ev.button() == QtCore.Qt.RightButton:

class Scene(QtGui.QGraphicsScene):
    def __init__(self, parent=None):
        super().__init__(parent)
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