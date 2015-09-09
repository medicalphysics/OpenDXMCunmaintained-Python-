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


class ViewController(QtCore.QObject):
    viewCtDoseArray = QtCore.pyqtSignal(np.ndarray, np.ndarray, np.ndarray)
    def __init__(self, database_interface, view, parent=None):
        super().__init__(parent)
        database_interface.request_simulation_view.connect(self.applySimulation)

        self.view = view
        self.dosescene = DoseScene()
        self.viewCtDoseArray.connect(self.dosescene.setCtDoseArrays)
        self.view.setScene(self.dosescene)

    @QtCore.pyqtSlot(Simulation)
    def applySimulation(self, sim):
        logger.debug('Got signal request to view Simulation {}'.format(sim.name))
        if sim.energy_imparted is None:
            logger.debug('View request for Simulation {} denied: No dose array available'.format(sim.name))
        elif sim.ctarray is None:
            logger.debug('View request for Simulation {} denied: No CT array available.'.format(sim.name))
        else:
            self.viewCtDoseArray.emit(sim.ctarray, sim.energy_imparted, sim.spacing)

    @QtCore.pyqtSlot(np.ndarray)
    def updateDoseArray(self, array):
        pass


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
    array *= 256./(WW*2)

#    array = (np.clip(array, WC - 0.5 - (WW-1) / 2, WC - 0.5 + (WW - 1) / 2) -
#             (WC - 0.5 - (WW - 1) / 2)) * 255 / ((WC - 0.5 + (WW - 1) / 2) -
#                                                 (WC - 0.5 - (WW - 1) / 2))
    array = np.require(array, np.uint8, 'C')
    h, w = array.shape
    result = QtGui.QImage(array.data, w, h, QtGui.QImage.Format_Indexed8)
#    result.ndarray = array
    result.setColorTable(lut)
#    result = result.convertToFormat(QtGui.QImage.Format_ARGB32, LUT_TABLE[lut])
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
#        self.setFlag(QtGui.QGraphicsItem.ItemSendsGeometryChanges)
        if image is None:
            if shape is None:
                shape = (512, 512)
            self.image = np.zeros(shape)
        else:
            self.image = image.view(np.ndarray)
        if image is not None and level is None:
            mi = image.min()
            ma = image.max() + 1
            self.level = ((ma - mi) / 2, ) * 2
        elif level is None:
            self.level = (-100, 100)
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
        x, y = self.image.shape
        return QtCore.QRectF(self.x(), self.y(), y, x)

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
        self.qimage = arrayToQImage(self.image, self.level[0],
                                    self.level[1])

    def shape(self):
        path = QtGui.QPainterPath()
        path.addEllipse(self.boundingRect())
        return path

    def paint(self, painter, style, widget=None):
        if self.qimage is None:
            self.render()
        painter.drawImage(QtCore.QPointF(self.pos()), self.qimage)
#        super(ImageItem, self).paint(painter, style, widget)


class DoseScene(QtGui.QGraphicsScene):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.image_item = BlendImageItem()
        self.addItem(self.image_item)

        self.dose_array = None
        self.ct_array = None
        self.view_slice = 1  # value 0, 1, 2
        self.shape = (0, 0, 0)
        self.index = 0
        self.spacing = (1., 1., 1.)

    @QtCore.pyqtSlot(np.ndarray, np.ndarray, np.ndarray)
    def setCtDoseArrays(self, ct, dose, spacing):
        self.dose_array = gaussian_filter(dose, .5)
        #setting transform
        sx, sy = [spacing[i] for i in range(3) if i != self.view_slice]
        transform = QtGui.QTransform.fromScale(sy / sx, 1.)
        self.image_item.setTransform(transform)
        self.image_item.setLevels(front=(self.dose_array.max()/2.,self.dose_array.max()/2.))
        self.ct_array = ct
        self.shape = ct.shape
        self.spacing = spacing
        self.index = 0
        self.reloadImages()
        rect = transform.mapRect(self.image_item.boundingRect())

        self.setSceneRect(rect)
        for view in self.views():
            view.fitInView(rect)
        logger.debug('Dosescene is setting image data')

    @QtCore.pyqtSlot(np.ndarray)
    def setDoseArray(self, dose):
        self.dose_array = dose
        self.reloadImages()

    def getSlice(self, array, index):
        if self.view_slice == 2:
            return np.copy(np.squeeze(array[: ,: ,index % self.shape[self.view_slice]]))
        elif self.view_slice == 1:
            return np.copy(np.squeeze(array[:, index % self.shape[self.view_slice], :]))
        elif self.view_slice == 0:
            return np.copy(np.squeeze(array[index % self.shape[self.view_slice], :, :]))
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
