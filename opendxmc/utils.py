# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 13:47:11 2015

@author: ERLEAN
"""

import os
import math
import numpy as np


def interpolate3d(array, scaling):
    sh = np.array(array.shape, dtype=np.int)
    sh_n = np.floor(sh / np.array(scaling)).astype(np.int)
    
    x, y, z = (np.linspace(0, sh[i] -2, sh_n[i]) for i in range(3))

    lx, ly, lz = (np.floor(i).astype(np.int) for i in (x, y, z))
    ux, uy, uz = (i + 1 for i in (lx, ly, lz))


    res = np.empty(tuple(sh_n), dtype=array.dtype)


    c00 = np.empty(sh_n[2])
    c01 = np.empty(sh_n[2])
    c10 = np.empty(sh_n[2])
    c11 = np.empty(sh_n[2])
    c0 = np.empty(sh_n[2])
    c1 = np.empty(sh_n[2])

    zd = (z - lz) / z[1]
    
    for indx, xi in enumerate(x):
        xd = (xi - lx[indx]) / x[1]
        for indy, yi in enumerate(y):
            yd = (yi - ly[indy]) / y[1]
            c00[:] = array[lx[indx], ly[indy], lz] * (1. - xd) + array[ux[indx], ly[indy], lz] * xd
            c01[:] = array[lx[indx], ly[indy], uz] * (1. - xd) + array[ux[indx], ly[indy], uz] * xd
            c10[:] = array[lx[indx], uy[indy], lz] * (1. - xd) + array[ux[indx], uy[indy], lz] * xd
            c11[:] = array[lx[indx], uy[indy], uz] * (1. - xd) + array[ux[indx], uy[indy], uz] * xd
            
            c0[:] = c00 * (1-yd) + c10 * yd
            c1[:] = c01 * (1-yd) + c11 * yd
            res[indx, indy, :] = c0 * (1-zd) + c1*zd
            
            
#            for indz, zi in enumerate(z):
#                zd = (zi - lz[indz]) / z[1]
#                c00 = array[lx[indx], ly[indy], lz[indz]] * (1. - xd) + array[ux[indx], ly[indy], lz[indz]] * xd
#                c01 = array[lx[indx], ly[indy], uz[indz]] * (1. - xd) + array[ux[indx], ly[indy], uz[indz]] * xd
#                c10 = array[lx[indx], uy[indy], lz[indz]] * (1. - xd) + array[ux[indx], uy[indy], lz[indz]] * xd
#                c11 = array[lx[indx], uy[indy], uz[indz]] * (1. - xd) + array[ux[indx], uy[indy], uz[indz]] * xd
#                
#                c0 = c00 * (1-yd) + c10 * yd
#                c1 = c01 * (1-yd) + c11 * yd
#                res[indx, indy, indz] = c0 * (1-zd) + c1*zd
#                
                
                
    return res
    
def rebin( a, newshape ):
    '''Rebin an array to a new shape.
    '''
    assert len(a.shape) == len(newshape)

    slices = [ slice(0,old, float(old)/new) for old,new in zip(a.shape,newshape) ]
    coordinates = np.mgrid[slices]
    indices = coordinates.astype('i')   #choose the biggest smaller integer index
    return a[tuple(indices)]    
    
def test_interpolate():
    n = 512
    a = np.arange(n**3).reshape((n,)*3)
    mask = circle_mask((n, n), n//4)*-1 +1
    for i in range(n):
        a[:,:,i] *= mask
    scaling = np.array([1.5, 1.5, 1.5])
    i = interpolate3d(a, scaling)
    import pylab as plt
    plt.imshow(i[:, :, 0])
    plt.show()



def human_time(sec):
    seconds = [86400., 3600., 60., 1.]
    names = ['days', 'hours', 'minutes', 'seconds']

    if sec <= seconds[-1]:
        return 'less than a {0}'.format(names[-1][:-1])
    times = []
    labels = []
    for i, s in enumerate(seconds):
        if sec >= s:
            times.append(int(math.floor(sec/s)))
            labels.append(names[i])
            if times[0] == 1:
                labels[0] = labels[0][:-1]
            if i < len(seconds) - 1:
                times.append(int(math.floor((sec - s * times[0])/seconds[i+1])))
                labels.append(names[i+1])
                if times[1] == 1:
                    labels[1] = labels[1][:-1]

            break
    else:
        return 'less than a {0}'.format(names[-1][:-1])
    if len(times) > 1:
        if times[-1] == 0:
            return 'about {0} {1}'.format(times[0], labels[0])
        return 'about {0} {1} and {2} {3}'.format(times[0], labels[0], times[1], labels[1])
    return 'about {0} {1}'.format(times[0], labels[0])


def circle_mask(array_shape, radius, center=None):
    a = np.zeros(array_shape, np.int)
    if not center:
        cx = array_shape[0] / 2
        cy = array_shape[1] / 2
    else:
        cx, cy = center
    y, x = np.ogrid[-radius: radius, -radius: radius]
    index = x**2 + y**2 <= radius**2
    a[cx-radius:cx+radius+1, cy-radius:cy+radius+1][index] = 1
    return a

def sphere_mask(array_shape, radius, center=None):
    a = np.zeros(array_shape, np.int)
    if not center:
        cx = array_shape[0] / 2
        cy = array_shape[1] / 2
        cz = array_shape[2] / 2
    else:
        cx, cy, cz= center
    x, y, z = np.ogrid[-radius: radius, -radius: radius, -radius: radius]
    index = x**2 + y**2 + z**2 <= radius**2
    a[cx-radius:cx+radius+1, cy-radius:cy+radius+1, cz-radius:cz+radius+1][index] = 1
    return a


def find_all_files(pathList):
    for p in pathList:
        path = os.path.abspath(p)
        if os.path.isdir(path):
            for dirname, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    yield os.path.normpath(os.path.join(dirname, filename))
        elif os.path.isfile(path):
            yield os.path.normpath(path)
            
if __name__ == '__main__':
    test_interpolate()
