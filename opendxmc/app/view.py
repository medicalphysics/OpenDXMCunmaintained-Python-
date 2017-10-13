# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 10:38:15 2015

@author: erlean
"""

import numpy as np
from PyQt4 import QtGui, QtCore
import os
import pyqtgraph
import pyqtgraph.opengl as gl
from OpenGL.GL import glReadPixels, GL_RGBA, GL_UNSIGNED_BYTE, GL_RGB
from .dicom_lut import get_lut, get_lut_raw
from scipy.ndimage.filters import gaussian_filter
from opendxmc.app.ffmpeg_writer import FFMPEG_VideoWriter
import logging
logger = logging.getLogger('OpenDXMC')


## @file
#
# Qt Widget Range slider widget. This is a slider with two bars, one of
# which is used to set the maximum and the other of which is used to
# set the minimum.
#
# Qt Double Spin Box slider widget. This is the range slider grouped
# with two spin boxes to make the range editable.
#
# Hazen 07/14
#

class QRangeSlider(QtGui.QWidget):
    doubleClick = QtCore.pyqtSignal(bool)
    rangeChanged = QtCore.pyqtSignal(float, float)

    ## __init__
    #
    # @param slider_range [min, max, step size].
    # @param values [initial minimum setting, initial maximum setting].
    # @param parent (Optional) The PyQt parent of this widget.
    #
    def __init__(self, slider_range, values, parent = None):
        super().__init__(parent)
        self.bar_width = 10
        self.emit_while_moving = False
        self.moving = "none"
        self.old_scale_min = 0.0
        self.old_scale_max = 0.0
        self.scale = 0
        self.setMouseTracking(False)
        self.single_step = 0.0

        if slider_range:
            self.setRange(slider_range)
        else:
            self.setRange([0.0, 1.0, 0.01])
        if values:
            self.setValues(values)
        else:
            self.setValues([0.3, 0.6])

        self.setFocusPolicy(QtCore.Qt.ClickFocus)

    def emitRange(self):
        if (self.old_scale_min != self.scale_min) or (self.old_scale_max != self.scale_max):
            self.rangeChanged.emit(self.scale_min, self.scale_max)
            self.old_scale_min = self.scale_min
            self.old_scale_max = self.scale_max

    def getValues(self):
        return [self.scale_min, self.scale_max]

    def keyPressEvent(self, event):
        key = event.key()

        # move bars based on arrow keys
        moving_max = False
        if key == QtCore.Qt.Key_Up:
            self.scale_max += self.single_step
            moving_max = True
        elif key == QtCore.Qt.Key_Down:
            self.scale_max -= self.single_step
            moving_max = True
        elif key == QtCore.Qt.Key_Left:
            self.scale_min -= self.single_step
        elif key == QtCore.Qt.Key_Right:
            self.scale_min += self.single_step

        # update (if necessary) based on allowed range
        if moving_max:
            if (self.scale_max < self.scale_min):
                self.scale_min = self.scale_max
        else:
            if (self.scale_min > self.scale_max):
                self.scale_max = self.scale_min

        if (self.scale_min < self.start):
            self.scale_min = self.start
        if (self.scale_max < self.start):
            self.scale_max = self.start

        slider_max = self.start + self.scale
        if (self.scale_min > slider_max):
            self.scale_min = slider_max
        if (self.scale_max > slider_max):
            self.scale_max = slider_max

        self.emitRange()
        self.updateDisplayValues()
        self.update()


    def mouseDoubleClickEvent(self, event):
        self.doubleClick.emit(True)

    def mouseMoveEvent(self, event):
        size = self.rangeSliderSize()
        diff = self.start_pos - self.getPos(event)
        if self.moving == "min":
            temp = self.start_display_min - diff
            if (temp >= self.bar_width) and (temp < size - self.bar_width):
                self.display_min = temp
                if self.display_max < self.display_min:
                    self.display_max = self.display_min
                self.updateScaleValues()
                if self.emit_while_moving:
                    self.emitRange()
        elif self.moving == "max":
            temp = self.start_display_max - diff
            if (temp >= self.bar_width) and (temp < size - self.bar_width):
                self.display_max = temp
                if self.display_max < self.display_min:
                    self.display_min = self.display_max
                self.updateScaleValues()
                if self.emit_while_moving:
                    self.emitRange()
        elif self.moving == "bar":
            temp = self.start_display_min - diff
            if (temp >= self.bar_width) and (temp < size - self.bar_width - (self.start_display_max - self.start_display_min)):
                self.display_min = temp
                self.display_max = self.start_display_max - diff
                self.updateScaleValues()
                if self.emit_while_moving:
                    self.emitRange()

    def mousePressEvent(self, event):
        pos = self.getPos(event)
        if abs(self.display_min - 0.5 * self.bar_width - pos) < (0.5 * self.bar_width):
            self.moving = "min"
        elif abs(self.display_max + 0.5 * self.bar_width - pos) < (0.5 * self.bar_width):
            self.moving = "max"
        elif (pos > self.display_min) and (pos < self.display_max):
            self.moving = "bar"
        self.start_display_min = self.display_min
        self.start_display_max = self.display_max
        self.start_pos = pos

    def mouseReleaseEvent(self, event):
        if not (self.moving == "none"):
            self.emitRange()
        self.moving = "none"

    def resizeEvent(self, event):
        self.updateDisplayValues()

    def setEmitWhileMoving(self, flag):
        if flag:
            self.emit_while_moving = True
        else:
            self.emit_while_moving = False

    def setRange(self, slider_range):
        self.start = slider_range[0]
        self.scale = slider_range[1] - slider_range[0]
        self.single_step = slider_range[2]

        # Check that the range is a multiple of the step size.
        steps = self.scale / self.single_step
        if (abs(steps - round(steps)) > 0.01 * self.single_step):
            self.single_step = steps


    def setValues(self, values):
        self.scale_min = values[0]
        self.scale_max = values[1]
        self.emitRange()
        self.updateDisplayValues()
        self.update()

    def updateDisplayValues(self):
        size = float(self.rangeSliderSize() - 2 * self.bar_width - 1)
        self.display_min = int(size * (self.scale_min - self.start)/self.scale) + self.bar_width
        self.display_max = int(size * (self.scale_max - self.start)/self.scale) + self.bar_width

    def updateScaleValues(self):
        size = float(self.rangeSliderSize() - 2 * self.bar_width - 1)
        if (self.moving == "min") or (self.moving == "bar"):
            self.scale_min = self.start + (self.display_min - self.bar_width)/float(size) * self.scale
            self.scale_min = float(round(self.scale_min/self.single_step))*self.single_step
        if (self.moving == "max") or (self.moving == "bar"):
            self.scale_max = self.start + (self.display_max - self.bar_width)/float(size) * self.scale
            self.scale_max = float(round(self.scale_max/self.single_step))*self.single_step
        self.updateDisplayValues()
        self.update()



class QHRangeSlider(QRangeSlider):

    ## __init__
    #
    # @param slider_range (Optional) [min, max, step size].
    # @param values (Optional) [initial minimum setting, initial maximum setting].
    # @param parent (Optional) The PyQt parent of this widget.
    #
    def __init__(self, slider_range = None, values = None, parent = None):
        super().__init__(slider_range, values, parent)
        if (not parent):
            self.setGeometry(200, 200, 200, 100)

    def getPos(self, event):
        return event.x()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        w = self.width()
        h = self.height()

        # background
        painter.setPen(QtCore.Qt.gray)
        painter.setBrush(QtCore.Qt.lightGray)
        painter.drawRect(2, 2, w-4, h-4)

        # range bar
        painter.setPen(QtCore.Qt.darkGray)
        painter.setBrush(QtCore.Qt.darkGray)
        painter.drawRect(self.display_min-1, 5, self.display_max-self.display_min+2, h-10)

        # min & max tabs
        painter.setPen(QtCore.Qt.black)
        painter.setBrush(QtCore.Qt.gray)
        painter.drawRect(self.display_min-self.bar_width, 1, self.bar_width, h-2)

        painter.setPen(QtCore.Qt.black)
        painter.setBrush(QtCore.Qt.gray)
        painter.drawRect(self.display_max, 1, self.bar_width, h-2)

    def rangeSliderSize(self):
        return self.width()

class QVRangeSlider(QRangeSlider):

    ## __init__
    #
    # @param slider_range (Optional) [min, max, step size].
    # @param values (Optional) [initial minimum setting, initial maximum setting].
    # @param parent (Optional) The PyQt parent of this widget.
    #
    def __init__(self, slider_range = None, values = None, parent = None):
        super().__init__(slider_range, values, parent)
        if (not parent):
            self.setGeometry(200, 200, 100, 200)
    def getPos(self, event):
        return self.height() - event.y()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        w = self.width()
        h = self.height()

        # background
        painter.setPen(QtCore.Qt.gray)
        painter.setBrush(QtCore.Qt.lightGray)
        painter.drawRect(2, 2, w-4, h-4)

        # range bar
        painter.setPen(QtCore.Qt.darkGray)
        painter.setBrush(QtCore.Qt.darkGray)
        painter.drawRect(5, h-self.display_max-1, w-10, self.display_max-self.display_min+1)

        # min & max tabs
        painter.setPen(QtCore.Qt.black)
        painter.setBrush(QtCore.Qt.gray)
        painter.drawRect(1, h-self.display_max-self.bar_width-1, w-2, self.bar_width)

        painter.setPen(QtCore.Qt.black)
        painter.setBrush(QtCore.Qt.gray)
        painter.drawRect(1, h-self.display_min-1, w-2, self.bar_width)

    ## rangeSliderSize
    #
    # @return The current height of the slider widget.
    #
    def rangeSliderSize(self):
        return self.height()


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

class VolumeViewManager(QtGui.QWidget):
    def __init__(self, volume_scenes, parent=None):
        super().__init__(parent)
        self.setContentsMargins(0, 0, 0, 0)
        main_layout = QtGui.QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setAlignment(QtCore.Qt.AlignTop)

        menu_layout = QtGui.QHBoxLayout()
        menu_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addLayout(menu_layout)

        enable_vrt_button = QtGui.QPushButton('Enable VRTs')
        enable_vrt_button.setCheckable(True)
        enable_vrt_button.setFlat(True)
        enable_vrt_button.setToolTip('Enable VRTs if your graphics card is up for it')
        enable_vrt_button.setStatusTip('Enable VRTs if your graphics card is up for it')
        menu_layout.addWidget(enable_vrt_button)

        splitter = QtGui.QSplitter(QtCore.Qt.Vertical)
        main_layout.addWidget(splitter)

        button_icon = QtGui.QIcon(os.path.join(os.path.dirname(__file__), 'copy_icon.png'))

        for name, wid3d in volume_scenes.items():
            enable_vrt_button.toggled.connect(wid3d.set_showing)
            wid = QtGui.QWidget()
            wid.setVisible(False)
            wid3d.is_showing.connect(wid.setVisible)

            v_lay = QtGui.QHBoxLayout()
            v_lay.setAlignment(QtCore.Qt.AlignRight)
            v_lay.setContentsMargins(0, 0, 0, 0)
            v_lay.addWidget(wid3d)

            ranges = list(wid3d.custom_data_range)
            ranges += [(ranges[1]-ranges[0])/25,]

            slider = QVRangeSlider(ranges, [ranges[0], ranges[1]])

            slider.setMinimumWidth(20)
            slider.setMaximumWidth(20)
            slider.rangeChanged.connect(wid3d.set_custom_data_range)
            l_lay = QtGui.QVBoxLayout()
            l_lay.addWidget(slider)

            copy_button = QtGui.QPushButton()
            copy_button.setFlat(True)
            copy_button.setToolTip('Copy VRT image to clipboard')
            copy_button.setIcon(button_icon)
            copy_button.clicked.connect(wid3d.save_image_to_clipboard)
            l_lay.addWidget(copy_button)

            v_lay.addLayout(l_lay)
            wid.setLayout(v_lay)
            wid.setContentsMargins(0, 0, 0, 0)
            splitter.addWidget(wid)
        enable_vrt_button.clicked.connect(self.update)
        self.setLayout(main_layout)


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
        database_interface.send_material_for_viewing.connect(self.view_material)


        self.scenes = {'planning': PlanningScene(),
#                       'running':  RunningScene(),
                       'material': MaterialScene(),
                       'energy imparted': DoseScene(),
                       'dose': DoseScene(front_array='dose'),
                       }
        self.glwidgets = {#'dose': View3D2(arrays=['ctarray','dose'], lut_names=['gist_earth','jet'], dim_scale=False, custom_data_range=[(0, 500), None]),
                          'doseCT': View3D(array='dose', lut_name='jet', dim_scale=True, smoothness=.5, custom_data_range=(0, .15), custom_data_range_is_modifier=True),
#                          'energy_imparted': View3D(array='energy_imparted', lut_name='pet', dim_scale=True),
#                          'planning': View3D(array='ctarray', lut_name='hot_metal_green', dim_scale=False, custom_data_range=(0, 500))}
                          'dens': View3D(array='density', lut_name='gist_earth', dim_scale=True, custom_data_range=(.5, 2.4), smoothness=.5, magic_number = 50),
                          'planning': View3D(array='ctarray', lut_name='gist_earth', dim_scale=False, custom_data_range=(-200, 700), smoothness=.5, magic_number=50)}

        for name, scene in self.scenes.items():
            # connecting scenes to request array slot
            scene.update_index.connect(self.update_index)
            scene.request_array.connect(database_interface.request_view_array)
            database_interface.send_view_array.connect(scene.set_requested_array)

        for name, glwidget in self.glwidgets.items():
            # connecting scenes to request array slot
            glwidget.request_array.connect(database_interface.request_view_array_bytescaled)
            database_interface.send_view_array_bytescaled.connect(glwidget.set_requested_array)


        self.request_array_slice.connect(database_interface.request_view_array_slice)
        database_interface.send_view_array_slice.connect(self.reload_slice)

        self.request_metadata.connect(database_interface.request_simulation_properties)
        database_interface.send_view_sim_propeties.connect(self.set_simulation_properties)
        self.graphicsview.setScene(self.scenes['planning'])

    def set_simulation_editor(self, propertieseditmodel):
        self.scenes['planning'].update_simulation_properties.connect(propertieseditmodel.set_simulation_properties)


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
        if data_dict['is_phantom']:
            for glwid in self.glwidgets.values():
                glwid.set_metadata(data_dict)
        else:
            for glwid in [value for key, value in self.glwidgets.items() if key != 'dens']:
                glwid.set_metadata(data_dict)
        self.selectScene('planning')
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
#            self.graphicsview.show()
            self.current_scene = scene_name
            self.graphicsview.setScene(self.scenes[self.current_scene])
            self.update_index(self.current_index)
            self.graphicsview.fitInView(self.graphicsview.sceneRect(), QtCore.Qt.KeepAspectRatio)
#        else:
#            self.graphicsview.hide()
#        for wids in self.glwidgets.values():
#                wids.hide()
#        if scene_name in self.glwidgets:
##            self.graphicsview.hide()
#            self.glwidgets[scene_name].show()
#            print('showing')


    def view_widget(self):

#        wid = QtGui.QSplitter(QtCore.Qt.Vertical)
        wid2 = QtGui.QWidget()
        main_layout = QtGui.QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        view_layout = QtGui.QVBoxLayout()
        view_layout.setContentsMargins(0, 0, 0, 0)

        menu_widget = QtGui.QMenuBar(wid2)
        menu_widget.setContentsMargins(0, 0, 0, 0)

        orientation_action = QtGui.QAction('Orientation', menu_widget)
        orientation_action.triggered.connect(self.selectViewOrientation)
        menu_widget.addAction(orientation_action)

        sceneSelect = SceneSelectGroup(wid2)
        for scene_name in self.scenes.keys():
            sceneSelect.addAction(scene_name)
        sceneSelect.scene_selected.connect(self.selectScene)

        main_layout.addWidget(menu_widget)

        main_layout.addWidget(self.graphicsview)

#        for name in self.glwidgets:
#            wid.addWidget(self.glwidgets[name])

        for action in sceneSelect.actions():
            menu_widget.addAction(action)

        cine_action = QtGui.QAction('Cine', menu_widget)
        cine_action.triggered.connect(self.graphicsview.request_cine_film_creation)
        menu_widget.addAction(cine_action)

        wid2.setLayout(main_layout)
        wid = VolumeViewManager(self.glwidgets)
        return wid2, wid



    def make_material_plot_widget(self, material):
        mw = QtGui.QMainWindow()
        mw.setWindowTitle(material.name)

        pw = pyqtgraph.PlotWidget(name=material.name, title=material.name, background='w')
        mw.setCentralWidget(pw)
        pw.addLegend()
#        p = win.addPlot(title='plot_title', background='w')
        atts = material.attinuation
        e_ind = atts['energy'] < 500000
        for j, key in enumerate(['total', 'rayleigh', 'photoelectric', 'compton']):
            if key == 'total':
                pen = pyqtgraph.mkPen('k', width=4)
            else:
                pen = pyqtgraph.mkPen(pyqtgraph.intColor(j, hues=4, minHue=240), width=2)
            pw.plot(x=atts['energy'][e_ind], y=atts[key][e_ind], pen=pen, name=key, antialias=True)

        pw.setLabel('left', "Attenuation", units='cm^2/g')
        pw.setLabel('bottom', "Energy", units='eV')
        pw.setLogMode(x=False, y=True)
        pw.setXRange(0, 50000) # 500 keV
        return mw

    @QtCore.pyqtSlot(object)
    def view_material(self, material):
        win = self.make_material_plot_widget(material)
        win.show()
        self.__plot = win


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
                         back_level[0]+back_level[1]).astype(np.float)
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
#    p.setCompositionMode(QtGui.QPainter.CompositionMode_DestinationOut)
#    p.drawImage(QtCore.QRectF(back_qim.rect()), front_qim)
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
    array = np.clip(array_un, WC-WW, WC+WW).astype(np.float)

    array -= (WC - WW)
    array *= 255. / (WW * 2)


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

def qImageToArray(img, copy=False, transpose=True):
    """
    Convert a QImage into numpy array. The image must have format RGB32, ARGB32, or ARGB32_Premultiplied.
    By default, the image is not copied; changes made to the array will appear in the QImage as well (beware: if
    the QImage is collected before the array, there may be trouble).
    The array will have shape (width, height, (b,g,r,a)).
    """
    fmt = img.format()
    ptr = img.bits()
    ptr.setsize(img.byteCount())
    arr = np.asarray(ptr)
    if img.byteCount() != arr.size * arr.itemsize:
        # Required for Python 2.6, PyQt 4.10
        # If this works on all platforms, then there is no need to use np.asarray..
        arr = np.frombuffer(ptr, np.ubyte, img.byteCount())

    if fmt == img.Format_RGB32:
        arr = arr.reshape(img.height(), img.width(), 3)
    elif fmt == img.Format_ARGB32 or fmt == img.Format_ARGB32_Premultiplied:
        arr = arr.reshape(img.height(), img.width(), 4)

    if copy:
        arr = arr.copy()

    if transpose:
        return arr.transpose((1,0,2))
    else:
        return arr

class NoDataItem(QtGui.QGraphicsItem):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.fontMetrics = QtGui.qApp.fontMetrics()
        self.msg = "Sorry, no data here yet.\nRun a simulation to compute."
        self.rect = QtCore.QRectF(self.fontMetrics.boundingRect(self.msg))
        self.rect.setHeight(self.rect.height()*2.5)
        self.setFlag(self.ItemIgnoresTransformations, True)
    def boundingRect(self):
#        return  QtCore.QRectF(self.fontMetrics.boundingRect(self.msg))
        return self.rect
#        return QtCore.QRectF(0, 0, 200, 200)

    def paint(self, painter, style, widget=None):
        painter.setPen(QtGui.QPen(QtCore.Qt.white))

        painter.drawText(self.boundingRect(), self.msg)



class BlendImageItem(QtGui.QGraphicsItem):
    def __init__(self, parent=None, overlay_adjust_max_only=True):
        super().__init__(parent)

        self.back_image = np.zeros((8, 8))
        self.back_level = (0, 500)
        self.front_image = np.zeros((8, 8))
        self.front_level = (1000000, 10000)
        if overlay_adjust_max_only:
            self.adjust_front_max_only = True
        else:
            self.adjust_front_max_only = False


        self.back_alpha = 255
        self.front_alpha = 255
        self.back_lut = get_lut('gray', self.back_alpha)
        self.front_lut = get_lut('pet', self.front_alpha)

        self.qimage = None
        self.setAcceptedMouseButtons(QtCore.Qt.RightButton | QtCore.Qt.MiddleButton)

        self.cbar = ColorBarItem(self)
        self.cbar.setParentItem(self)

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
            self.cbar.set_levels(front)
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
            lut_raw = get_lut(front_lut, alpha)
            self.front_lut = lut_raw
            self.cbar.set_lut(lut_raw)
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
            dp = event.pos()- event.lastPos()
            x, y = self.front_level
            if self.adjust_front_max_only:
                x += dp.x() *.01*abs(x)
                if x < 0:
                    x=0
                y = x
            else:
                x += dp.x()*.01*abs(x)
                y += dp.y()*.01*abs(y)

                if y < 0:
                    y=0
            self.setLevels(front=(x, y))
        elif event.buttons() == QtCore.Qt.MiddleButton:
            event.accept()
            dp = event.pos()- event.lastPos()
            x, y = self.back_level
            x += dp.x()
            y += dp.y()
            if y < 1:
                y=1
            self.setLevels(back=(x, y))

    def setVisible(self, visible):
        self.cbar.setVisible(visible)
        super().setVisible(visible)


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

    def mousePressEvent(self, event):
        if event.buttons() == QtCore.Qt.MiddleButton:
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == QtCore.Qt.MiddleButton:
            event.accept()
#            print(event.pos()- event.lastPos())
            dp = event.pos()- event.lastPos()
            x, y = self.level
            x += dp.x()
            y += dp.y()
            if y < 1:
                y=1
            self.setLevels((x, y))

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

class ColorBarItem(QtGui.QGraphicsItem):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._array = np.outer(np.linspace(0, 1, 128)[::-1], np.ones(20))
        self.setFlag(self.ItemIgnoresTransformations, True)
        self.fontMetrics = QtGui.qApp.fontMetrics()
        self.box_size = self.fontMetrics.boundingRect('A').height()

        self._qim = None

        self._text = ['test', 'lower']
        self._units = 'mGy/100mAs'
        self.setScale(.75)

    def setUnits(self, units):
        self._units = units

    def set_levels(self, level):
        wc, ww = level
        self._text = ["{:.2G} {}".format(wc+ww, self._units), "{:.2G} {}".format(wc-ww, self._units)]
        self.update()

    def boundingRect(self):
        w = max([self._array.shape[1],
                 self.fontMetrics.boundingRect(self._text[0]).width(),
                 self.fontMetrics.boundingRect(self._text[1]).width()])
        h = sum([self._array.shape[0],
                 self.fontMetrics.boundingRect(self._text[0]).height(),
                 self.fontMetrics.boundingRect(self._text[1]).height()])

        return QtCore.QRectF(0, 0, w, h)

    def set_lut(self, lut):
        self._qim = arrayToQImage(self._array, (.5, .5), lut)

    def paint(self, painter, style, widget=None):
        if self._qim is not None:
            im_rect = QtCore.QRectF(0, self.fontMetrics.boundingRect(self._text[0]).height(), self._array.shape[1], self._array.shape[0])
            painter.drawImage(im_rect, self._qim)
            painter.setPen(QtGui.QPen(QtCore.Qt.white))
            painter.drawText(im_rect.topLeft(), self._text[0])
            painter.drawText(self.boundingRect().bottomLeft(), self._text[1])

class PositionBarItem(QtGui.QGraphicsItem):
    def __init__(self, parent=None, callback=None):
        super().__init__(parent)
        self.axes = np.eye(3)
        self.orientation = 2
        self.shape = np.ones(3)
        self.spacing = np.ones(3)
        self.pos = (0, 0)
        self.callback = callback

    def set_data(self, sim):
        self.shape = sim['shape']
        self.spacing = sim['spacing']#*sim['scaling']
        self.set_cosines(sim['image_orientation'])
        self.pos = [sim['start']/self.spacing[2], sim['stop']/self.spacing[2]]
        self.pos.sort()
        self.update()

    def set_orientation(self, orientation):
        self.orientation = orientation
        self.update()

    def set_cosines(self, cosines):
        self.axes[0, :] = cosines[:3]
        self.axes[1, :] = cosines[3:6]
        self.axes[2, :] = np.cross(cosines[:3], cosines[3:6])

    def boundingRect(self):
        x, y = [i for i in range(3) if i != self.orientation]
        return QtCore.QRectF(0, 0, y, x)

    def mousePressEvent(self, event):
        if event.button()==QtCore.Qt.LeftButton and self.callback:
            event.accept()
            pos = event.pos()
            intersect = sum(self.pos) / 2
            if pos.x() < intersect:
                self.pos[0] = pos.x()
                self.callback(start=self.pos[0])
            else:
                self.pos[1] = pos.x()
                self.callback(stop=self.pos[1])

    def paint(self, painter, style, widget=None):
        if self.orientation == 2:
            return
        i, j = [i for i in range(3) if i != self.orientation]

        painter.setPen(QtCore.Qt.white)
        x1 = 0
        x2 = self.shape[i]
        for pos in self.pos:
            y1 = pos
            y2 = pos + self.axes[i, 2] / self.spacing[j] * self.shape[j]
            painter.drawLine(y1, x1, y2, x2)


class PlanningScene(Scene):
    update_simulation_properties = QtCore.pyqtSignal(dict)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_item = ImageItem()
        self.image_item_bit = BitImageItem()
        self.position_item = PositionBarItem()
        self.addItem(self.image_item)
        self.addItem(self.image_item_bit)
        self.addItem(self.position_item)
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
        self.position_item.set_data(sim)
        if self.is_bit_array:
            self.array_names = ['organ']
            self.request_array.emit(self.name, 'organ_map')
        else:
            self.array_names = ['ctarray']
        self.request_array.emit(self.name, 'exposure_modulation')

        self.updateSceneTransform()

    def position_callback(self, start=None, stop=None):
        update = {}
        if start:
            update['start'] = start
        if stop:
            update['stop'] = stop
        if len(update) > 0:
            update['name'] = self.name
            self.update_simulation_properties.emit(update)

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
        self.position_item.set_orientation(self.view_orientation)
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
        self.position_item.setTransform(transform)
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


#class RunningScene(Scene):
#    def __init__(self, parent=None):
#        super().__init__(parent)
#        self.image_item = ImageItem()
#        self.addItem(self.image_item)
#        self.image_item.setLut(get_lut('hot_iron'))
#        self.nodata_item = NoDataItem()
#        self.addItem(self.nodata_item)
#        self.nodata_item.setVisible(True)
#
#        self.progress_item = self.addText('0 %')
#        self.progress_item.setDefaultTextColor(QtCore.Qt.white)
#
#        self.n_exposures = 1
#        self.array = None
#        self.view_orientation = 1
#
#
#    def set_metadata(self, props, index=0):
#        if props['name'] != self.name:
#            self.setNoData()
#        super().set_metadata(props, index)
#        collimation = props['detector_rows']*props['detector_width']
#        if props['is_spiral']:
#            self.n_exposures = props['exposures'] * (1 + abs(props['start'] - props['stop']) / collimation / props['pitch'])
#        else:
#            self.n_exposures = props['exposures'] * np.ceil(abs(props['start'] - props['stop']) / props['step'])
#            if self.n_exposures < 1:
#                self.n_exposures = 1
#        self.updateSceneTransform()
#
#
#    def setNoData(self):
#        self.array = None
#        self.image_item.setVisible(False)
#        self.progress_item.setVisible(False)
#        self.nodata_item.setVisible(True)
#
#    def defaultLevels(self, array):
#
#        p = array.max() - array.min()
#        return (p/2., p / 2. )
#
#    @QtCore.pyqtSlot(dict, dict)
#    def set_running_data(self, props, arr_dict):
#        self.array = arr_dict.get('energy_imparted', None)
#
#        self.nodata_item.setVisible(False)
#        self.image_item.setVisible(True)
#        self.progress_item.setVisible(True)
#        msg = ""
#        if len(props.get('eta', '')) > 0:
#            if 'start_at_exposure_no' in props:
#                msg += "{} %".format(round(props['start_at_exposure_no'] / self.n_exposures *100, 1))
#            if 'eta' in props:
#                msg += " ETA: {}".format(props['eta'])
#        self.progress_item.setPlainText(msg)
#        if self.array is not None:
#            self.image_item.setLevels(self.defaultLevels(self.array))
#        self.reloadImages()
#        self.updateSceneTransform()
#
#    def set_view_orientation(self, orientation, index):
#        self.view_orientation = orientation
#        self.reloadImages()
#        self.updateSceneTransform()
#
#    def updateSceneTransform(self):
#        sx, sy = [self.spacing[i]*self.scaling[i] for i in range(3) if i != self.view_orientation]
#        transform = QtGui.QTransform.fromScale(sy / sx, 1.)
#        self.image_item.setTransform(transform)
#        if self.nodata_item.isVisible():
#            self.setSceneRect(self.nodata_item.sceneBoundingRect())
#        else:
#            self.setSceneRect(self.image_item.sceneBoundingRect())
#
#    def reloadImages(self):
#        if self.array is not None:
#            self.image_item.setImage(self.array.max(axis=self.view_orientation))
#
#    def wheelEvent(self, ev):
#        pass


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
        self.setSceneRect(self.nodata_item.boundingRect())

    def set_metadata(self, sim, index=0):
        super().set_metadata(sim, index)
        self.setNoData()
        self.index = self.index % self.shape[self.view_orientation]
        self.request_array.emit(self.name, 'material_map')
#        self.updateSceneTransform()

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
        if not self.nodata_item.isVisible():
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
#        self.addItem(self.image_item.cbar)
        self.nodata_item = NoDataItem()
        self.addItem(self.nodata_item)
        self.array_names = ['ctarray', 'organ']

        self.front_array_name = front_array
        self.image_item.cbar.setUnits('mGy/100mAs' if front_array == 'dose' else 'eV')

        self.front_array = None

        alpha = 1. -np.exp(-np.linspace(0, 6, 256))
        alpha *= 255./alpha.max()
        self.image_item.setLut(front_lut='jet', front_alpha=alpha.astype(np.int))
        self.setNoData()

    def setNoData(self):
        self.front_array = None
        self.nodata_item.setVisible(True)
        self.image_item.setVisible(False)

    def set_metadata(self, sim, index=0):
        super().set_metadata(sim, index)
        self.setNoData()
#        self.front_array = None
#        self.nodata_item.setVisible(False)
#        self.image_item.setVisible(False)
        self.request_array.emit(self.name, self.front_array_name)
        if sim['is_phantom']:
            self.request_array.emit(self.name, 'organ_map')
        else:
            self.image_item.setLevels(back=(0, 500))

        self.updateSceneTransform()


#    def updateSceneTransform(self):
#        if not self.nodata_item.isVisible():
#            sx, sy = [self.spacing[i] for i in range(3) if i != self.view_orientation]
#            transform = QtGui.QTransform.fromScale(sy / sx, 1.)
#            self.image_item.setTransform(transform)
#
#            shape = tuple(sh for ind, sh in enumerate(self.shape) if ind != self.view_orientation)
#            rect = self.image_item.mapRectToScene(QtCore.QRectF(0, 0, shape[1], shape[0]))
#            c_rect = self.image_item.cbar.boundingRect()
#
#            if rect.height() >= rect.width():
##                self.image_item.cbar.setScale(rect.height() / (10 * c_rect.height()))
#                c_rect = self.image_item.cbar.sceneBoundingRect()
##                self.image_item.cbar.setPos(-c_rect.width(), 0)
#            else:
#
##                self.image_item.cbar.setScale(rect.width()  / (10 * c_rect.height()))
#                c_rect = self.image_item.cbar.sceneBoundingRect()
##                self.image_item.cbar.setPos(0, -c_rect.height())
#
#            self.setSceneRect(rect.united(self.image_item.cbar.sceneBoundingRect()))
#
#
#        else:
#            self.setSceneRect(self.nodata_item.sceneBoundingRect())

    @QtCore.pyqtSlot(str, np.ndarray, str)
    def set_requested_array(self, name, array, array_name):
        if name != self.name:
            return
        if array_name == 'organ_map':
            max_level = array['organ'].max()
            self.image_item.setLevels(back=(max_level/2, max_level/2))
#            self.image_item.setVisible(True)
#            self.nodata_item.setVisible(False)
#            shape = tuple(sh for ind, sh in enumerate(self.shape) if ind != self.view_orientation)
#            rect = self.image_item.mapRectToScene(QtCore.QRectF(0, 0, shape[1], shape[0]))
#            self.setSceneRect(rect)
        elif array_name == self.front_array_name:
            self.front_array = gaussian_filter(array, 1)
            max_level = array.max()/ 4
            min_level = max_level / 4
            self.image_item.setLevels(front=(min_level/2. + max_level/2.,min_level/2. + max_level/2.))
            self.image_item.setVisible(True)
            self.nodata_item.setVisible(False)
            shape = tuple(sh for ind, sh in enumerate(self.shape) if ind != self.view_orientation)
            rect = self.image_item.mapRectToScene(QtCore.QRectF(0, 0, shape[1], shape[0]))
            self.setSceneRect(rect)

    def updateSceneTransform(self):

        sx, sy = [self.spacing[i] for i in range(3) if i != self.view_orientation]
        transform = QtGui.QTransform.fromScale(sy / sx, 1.)
        self.image_item.setTransform(transform)

        if not self.nodata_item.isVisible():
            shape = tuple(sh for ind, sh in enumerate(self.shape) if ind != self.view_orientation)
            rect = self.image_item.mapRectToScene(QtCore.QRectF(0, 0, shape[1], shape[0]))
            self.setSceneRect(rect)
        else:
            self.setSceneRect(self.nodata_item.boundingRect())


    @QtCore.pyqtSlot(str, np.ndarray, str, int, int)
    def reload_slice(self, name, arr, array_name, index, orientation):
        if name != self.name:
            return
        if array_name in self.array_names:
            if self.front_array is not None:
                index_front = (index % self.shape[self.view_orientation]) // self.scaling[self.view_orientation]
                index_front %= self.front_array.shape[self.view_orientation]
                index_front = int(index_front)
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


class RunningScene(QtGui.QGraphicsScene):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_item = ImageItem()
        self.addItem(self.image_item)
        self.image_item.setLut(get_lut('hot_iron'))
#        self.nodata_item = NoDataItem()
#        self.addItem(self.nodata_item)
#        self.nodata_item.setVisible(True)

        self.progress_item = self.addText('0 %')
        self.progress_item.setDefaultTextColor(QtCore.Qt.white)
        self.progress_item.setFlag(self.progress_item.ItemIgnoresTransformations, True)
        self.progress_item
    def defaultLevels(self, array):

        p = array.max() - array.min()
        return (p/2., p / 2. )

    def set_running_data(self, array, sx, sy, msg):
        self.progress_item.setPlainText(msg)

        self.image_item.setLevels(self.defaultLevels(array))
        self.image_item.setImage(array)
        transform = QtGui.QTransform.fromScale(sy / sx, 1.)
        self.image_item.setTransform(transform)
        self.setSceneRect(self.image_item.sceneBoundingRect())


class RunnerView(QtGui.QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setContentsMargins(0, 0, 0, 0)
        self.setBackgroundBrush(QtGui.QBrush(QtCore.Qt.black))

        self.setRenderHints(QtGui.QPainter.Antialiasing |
                            QtGui.QPainter.TextAntialiasing)
        self.setScene(RunningScene())
        self.hide()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if self.isVisible():
            self.fitInView(self.sceneRect(), QtCore.Qt.KeepAspectRatio)

    def setScene(self, scene):
        super().setScene(scene)
        self.fitInView(self.sceneRect(), QtCore.Qt.KeepAspectRatio)

    def fitInView(self, *args):
        if self.isVisible():
            super().fitInView(*args)

    @QtCore.pyqtSlot(np.ndarray, float, float, str, bool)
    def set_data(self, array, sx, sy, msg, visible):
        if visible:
            self.show()
            self.scene().set_running_data(array, sx, sy, msg)
            self.fitInView(self.sceneRect(), QtCore.Qt.KeepAspectRatio)
        else:
            self.hide()

class View(QtGui.QGraphicsView):
    request_array = QtCore.pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setContentsMargins(0, 0, 0, 0)
        self.setBackgroundBrush(QtGui.QBrush(QtCore.Qt.black))

        self.setRenderHints(QtGui.QPainter.Antialiasing |
                            QtGui.QPainter.TextAntialiasing)

        self.mouse_down_pos = QtCore.QPoint(0, 0)
        self.cine_film_data = {}


    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if self.isVisible():
            self.fitInView(self.sceneRect(), QtCore.Qt.KeepAspectRatio)

    def setScene(self, scene):
        super().setScene(scene)
        self.fitInView(self.sceneRect(), QtCore.Qt.KeepAspectRatio)

    def fitInView(self, *args):
        if self.isVisible():
            super().fitInView(*args)

    def mouseMoveEvent(self, e):
        if e.buttons() == QtCore.Qt.LeftButton:
            dist = self.mouse_down_pos - e.globalPos()
            if dist.manhattanLength() > QtGui.QApplication.startDragDistance():
                e.accept()
                drag = QtGui.QDrag(self)
                # creating mimedata
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

        rect = self.toQImage(square=False).rect()
        height, width = rect.height(), rect.width()

        filename = QtGui.QFileDialog.getSaveFileName(self,
                                                     "Save cineloop",
                                                     "/cine.mp4",
                                                     "Movie (*.mp4)")
        try:
            writer = FFMPEG_VideoWriter(filename,
                                        (width, height),
                                        15, threads=1)
        except FileNotFoundError:
            logger.warning("FFMPEG executable not found")
            return
        else:
            logger.debug('Writing cine movie')

        for index in range(array.shape[self.cine_film_data['view_orientation']]):
            if self.cine_film_data['view_orientation'] == 1:
                arr_slice = np.ascontiguousarray(np.squeeze(array[:, index, :]))
            elif self.cine_film_data['view_orientation'] == 0:
                arr_slice = np.ascontiguousarray(np.squeeze(array[index, :, :]))
            else:
                arr_slice = np.ascontiguousarray(np.squeeze(array[:,:,index]))

            self.scene().reload_slice(self.cine_film_data['name'],
                                      arr_slice,
                                      array_name,
                                      index,
                                      self.cine_film_data['view_orientation']
                                      )
            QtGui.qApp.processEvents(flags=QtCore.QEventLoop.ExcludeUserInputEvents)
            qim = self.toQImage(square=False)#.convertToFormat(QtGui.QImage.Format_RGB888)

            arr = qImageToArray(qim, transpose=False)
#            ptr = qim.bits()
#            ptr.setsize(qim.byteCount())
#            ptr.setsize(width*height*3)

#            arr = np.array(ptr).reshape(height, -1, 3)
#            arr = np.array(ptr).reshape(width, height, 3)
#            arr = np.ascontiguousarray(arr[:,:,:3])
            arr = np.ascontiguousarray(arr[:,:,[2, 1, 0]])
            writer.write_frame(arr)
        writer.close()
        logger.debug('Done writing cine movie')
        self.cine_film_data = {}

    def toQImage(self, scale=True, square=False):
        rect = self.scene().sceneRect()
        h = rect.height()
        h -= h % 4
        w = rect.width()
        w -= w % 4
        b = max([h, w])
        if scale:
            b = max([h, w])
            if b >= 1024:
                s = 1
            else:
                s = int(round(1024/b))
        else:
            s = 1
        if square:
            qim = QtGui.QImage(b, b,
                               QtGui.QImage.Format_ARGB32_Premultiplied)
        else:
            qim = QtGui.QImage(int(w * s), int(h * s),
                               QtGui.QImage.Format_ARGB32_Premultiplied)
        qim.fill(0)
        painter = QtGui.QPainter()
        painter.begin(qim)
        painter.setRenderHints(painter.Antialiasing | painter.TextAntialiasing)
        self.scene().render(painter, source=rect)
        painter.end()
        return qim

class View3Dworker(QtCore.QThread):
    opengl_array = QtCore.pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.data = []
        self.mutex = QtCore.QMutex()

    @QtCore.pyqtSlot(list)
    def generate_tf(self, args):
        self.mutex.lock()
        self.data.append(args)
        self.mutex.unlock()
        self.start()
    def generate_lut(self, array, magic_value=None):
        data=np.arange(256)

        if magic_value is not None:
            corr = (array == int(magic_value)).sum() / (array > 0).sum()
            lut =(np.exp(-(data-magic_value)**2/(15**2))*.1*(1-corr)**2 + .7*np.exp(-(data-255)**2/64**2))*255

#            lut = np.roll(np.kaiser(256, 13), 96) * np.exp(-(255-data)/255) * 128
#            lut[[0, 1, 2]] = 0

        else:
            lut = np.roll(np.kaiser(256, 13), 96) * np.exp(-(255-data)/255) * 50
            lut[[0, 1, 2]] = 0

#            lut = np.roll(np.kaiser(256, 9), 0)*50# * np.exp(-(255-data)/255) * 50
#            lut[[0, 1, 2]] =0


        lut[0]=0
        return lut.astype(np.ubyte)

    def run(self):
        while len(self.data) > 0:
            self.mutex.lock()
            data, magic_value, lut_name, smoothness = self.data.pop(0)
            self.mutex.unlock()

            if smoothness:
                data = gaussian_filter(data, smoothness)
            lut = [np.array(l) for l in get_lut_raw(lut_name)]
            d2 = np.empty(data.shape + (4,), dtype=np.ubyte)

            d2[...,0] = lut[0][data]
            d2[...,1] = lut[1][data]
            d2[...,2] = lut[2][data]

            d2[...,3] = self.generate_lut(data, magic_value)[data]

            self.opengl_array.emit(d2)

class View3D(gl.GLViewWidget):
    request_array = QtCore.pyqtSignal(str, str, float, float, bool)
    request_opengl_array = QtCore.pyqtSignal(list)
    is_showing = QtCore.pyqtSignal(bool)
    def __init__(self, *args, array=None, lut_name=None, dim_scale=False, magic_number=None, custom_data_range=None, smoothness=None, custom_data_range_is_modifier=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.__is_showing = False
        self.setMinimumWidth(100)
        self.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        self.name = ''
        if custom_data_range is None:
            self.custom_data_range = (0, 0)
        else:
            self.custom_data_range = custom_data_range
        if custom_data_range_is_modifier:
            self.custom_data_range_is_modifier = True
        else:
            self.custom_data_range_is_modifier = False
        self.magic_number = magic_number
        if smoothness:
            self.smoothness = smoothness
        else:
            self.smoothness = None

        if array:
            self.array_name = array
        if lut_name:
            self.lut_name = lut_name
        else:
            self.lut_name = 'gray'
        self.dim_scaling = dim_scale

        self.shape = np.ones(3, np.int)
        self.spacing = np.ones(3, np.double)
        self.scaling = np.ones(3, np.double)


        self.worker = View3Dworker()
        self.request_opengl_array.connect(self.worker.generate_tf)
        self.worker.opengl_array.connect(self.set_gl_array)

        self.__glitem = None
        self.hide()


    @QtCore.pyqtSlot()
    def save_image_to_clipboard(self):
        w = self.width()
        h = self.height()
        self.repaint()
        pixels = np.zeros((h, w, 4), dtype=np.ubyte)

        glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, pixels)

        # swap B,R channels for Qt
        tmp = pixels[...,0].copy()
        pixels[...,0] = pixels[...,2]
        pixels[...,2] = tmp
        pixels = pixels[::-1] # flip vertical
        pixels[:, :, 3] = 255 #setting correct alpha for premultiplied RGB
        pixels = np.require(pixels, np.ubyte, ['C', 'A'])
        result = QtGui.QImage(pixels.data, w, h, QtGui.QImage.Format_ARGB32_Premultiplied).convertToFormat(QtGui.QImage.Format_RGB32)
        result.np_data = pixels

        QtGui.qApp.clipboard().setImage(result)

    @QtCore.pyqtSlot(float, float)
    def set_custom_data_range(self, minimum, maximum):
        self.custom_data_range = [minimum, maximum]
        if self.__glitem is not None:
            self.removeItem(self.__glitem)
            self.__glitem = None
        self.showing()

    def showing(self):
        if self.__is_showing and self.__glitem == None:
            self.request_array.emit(self.name, self.array_name, self.custom_data_range[0], self.custom_data_range[1], self.custom_data_range_is_modifier)
        self.setVisible(self.__is_showing and self.__glitem != None)
        self.is_showing.emit(self.__is_showing and self.__glitem != None)

    @QtCore.pyqtSlot(bool)
    def set_showing(self, enabled):
        self.__is_showing = enabled
        self.showing()

    def set_metadata(self, sim, index=0):
        if self.name != sim.get('name', ''):
            if self.__glitem is not None:
                self.removeItem(self.__glitem)
                self.__glitem = None

        self.name = sim.get('name', '')
        self.spacing = sim.get('spacing', np.ones(3, np.double))
        self.shape = sim.get('shape', np.ones(3, np.int))
        if self.dim_scaling:
            self.scaling = sim.get('scaling', np.ones(3, np.double))
        else:
            self.scaling = np.ones(3, np.double)

        self.opts['distance'] = np.sum((self.shape*self.spacing/self.scaling)**2)**.5 * 4

#        if self.__glitem is not None:
#            self.removeItem(self.__glitem)
#            self.__glitem = None
        self.showing()
#        self.index = index % self.shape[self.view_orientation]

    @QtCore.pyqtSlot(str, np.ndarray, str)
    def set_requested_array(self, name, data, array_name):
        if name != self.name:
            if self.__glitem is not None:
                self.removeItem(self.__glitem)
                self.__glitem = None
            return
        if array_name != self.array_name:
            return

        if self.__glitem is None:
            self.request_opengl_array.emit([data, self.magic_number, self.lut_name, self.smoothness])


    @QtCore.pyqtSlot(np.ndarray)
    def set_gl_array(self, d2):
        if self.__glitem is not None:
            self.removeItem(self.__glitem)
            self.__glitem = None
        try:
            #self.__glitem = gl.GLVolumeItem(d2, glOptions='additive')
            self.__glitem = gl.GLVolumeItem(d2, glOptions='translucent', smooth=True, sliceDensity=1)
            #self.__glitem = gl.GLVolumeItem(d2, glOptions='opaque')
        except Exception as e:
            print(e)
            return
        S = self.spacing * self.scaling
        scaling =((S/(np.sum(S*S))**.5))
#        scaling = scaling[[0, 2, 1]]
#        self.__glitem.translate(*(sh/2 for sh in d2.shape))
        self.__glitem.scale(*scaling)
#        self.__glitem.translate(*(-self.shape[[2, 0, 1]] / 2 * scaling))
        self.__glitem.translate(*(-self.shape * .5 * scaling / self.scaling))
        try:
            self.addItem(self.__glitem)
        except Exception as e:
            print(e)
            return
        self.showing()

