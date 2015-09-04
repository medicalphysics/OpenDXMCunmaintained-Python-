# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 10:38:15 2015

@author: erlean
"""
import numpy as np
from PyQt4 import QtGui, QtCore

from gui_widgets.dicom_lut import get_lut


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
        self.back_level = (.5, .5)
        self.front_image = np.zeros((512, 512))
        self.front_level = (.5, .5)

        self.back_alpha = 255
        self.front_alpha = 128
        self.back_lut = get_lut('gray', self.back_alpha)
        self.front_lut = get_lut('hot_metal_blue', self.front_alpha)

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

class Scene(QtGui.QGraphicsScene):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_item = BlendImageItem()
        self.addItem(self.image_item)

class View(QtGui.QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = Scene()
        self.setScene(self._scene)
        self.setBackgroundBrush(QtGui.QBrush(QtCore.Qt.black))

    @QtCore.pyqtSlot()
    def set_random(self):
        a1 = np.zeros((256, 256))
#        a1[128:138, :] = 1
        a2 = np.zeros((256, 256))
        for i in range(256):
            if i % 3:
                a1[i, :] = i / 256.
            a2[:, i] = i / 256.
        self.scene().image_item.setImage(a2, a1)
        self.scene().setSceneRect(self.scene().image_item.boundingRect())
        self.fitInView(self.sceneRect(), QtCore.Qt.KeepAspectRatio)
#        self.setImage(array)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self.fitInView(self.sceneRect(), QtCore.Qt.KeepAspectRatio)
