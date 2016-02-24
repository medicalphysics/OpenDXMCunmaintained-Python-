# -*- coding: utf-8 -*-
"""
Demonstrates GLVolumeItem for displaying volumetric data.
"""

## Add path to library (just for examples; you do not need this)

import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
from opendxmc.database.dicom_importer import import_ct_series
import os


def import_s():
    pat = import_ct_series([os.path.abspath("C://Users//ander//Documents//GitHub//OpenDXMC//opendxmc//raytracer//thorax")], import_scaling=(2, 2, 1))
        
    for props, arrays in pat:
        arr = arrays['ctarray'].astype('float32')
        return arr, props   

try:
    with np.load('foo.npz') as data:
        arr = data['arr']
        S = data['S']
except:    
    arr, props = import_s()
    S=np.array(props['spacing'])
    np.savez('foo.npz', arr=arr, S=S)

N = np.array(arr.shape)



app = QtGui.QApplication([])
w = gl.GLViewWidget()
w.opts['distance'] = 400
w.show()
w.setWindowTitle('pyqtgraph example: GLVolumeItem')

#b = gl.GLBoxItem()
#w.addItem(b)
#g = gl.GLGridItem()
#g.scale(10, 10, 1)
#w.addItem(g)


## Hydrogen electron probability density
def psi(i, j, k, offset=(50,50,100)):
    x = i-offset[0]
    y = j-offset[1]
    z = k-offset[2]
    th = np.arctan2(z, (x**2+y**2)**0.5)
    phi = np.arctan2(y, x)
    r = (x**2 + y**2 + z **2)**0.5
    a0 = 2
    #ps = (1./81.) * (2./np.pi)**0.5 * (1./a0)**(3/2) * (6 - r/a0) * (r/a0) * np.exp(-r/(3*a0)) * np.cos(th)
    ps = (1./81.) * 1./(6.*np.pi)**0.5 * (1./a0)**(3/2) * (r/a0)**2 * np.exp(-r/(3*a0)) * (3 * np.cos(th)**2 - 1)
    
    return ps
    
    #return ((1./81.) * (1./np.pi)**0.5 * (1./a0)**(3/2) * (r/a0)**2 * (r/a0) * np.exp(-r/(3*a0)) * np.sin(th) * np.cos(th) * np.exp(2 * 1j * phi))**2 


#data = np.fromfunction(psi, (100,100,200))
data = arr 
positive = np.log(np.clip(data, 0, data.max())**2)
negative = np.log(np.clip(data, 0, data.min())**2)

positive = np.log((np.clip(data, 500, .5*data.max())-500)**2 )
negative = np.log((np.clip(data, -100, 300) +100)**2) 



d2 = np.zeros(data.shape + (4,), dtype=np.ubyte)
d2[..., 0] = positive * (255./positive.max())
d2[..., 1] = negative * (128./(negative.max()))
d2[..., 2] = d2[...,1]
d2[..., 3] = d2[..., 0]*0.6 + d2[..., 1]*0.4
d2[..., 3] = (d2[..., 3].astype(float) / 255.) **2 * 255

#d2[:, 0, 0] = [255,0,0,100]
#d2[0, :, 0] = [0,255,0,100]
#d2[0, 0, :] = [0,0,255,100]
print(d2.max(), d2.min())
#v = gl.GLVolumeItem(d2, glOptions='additive')
v = gl.GLVolumeItem(d2, glOptions='translucent', smooth=True, sliceDensity=1)
#v = gl.GLVolumeItem(d2, glOptions='opaque')
scaling =((S/(np.sum(S*S))**.5))
v.scale(*scaling)
v.translate(*(-N/2*scaling))
w.addItem(v)

#ax = gl.GLAxisItem()
#w.addItem(ax)

## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()