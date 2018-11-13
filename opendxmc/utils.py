# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 13:47:11 2015

@author: ERLEAN
"""

import os
import math
import numpy as np

def rebin_scaling(a, scaling):
    factor_shape = tuple( s - (s % scaling[i])  for i, s in enumerate(a.shape))
    a_n = a[:factor_shape[0], :factor_shape[1], :factor_shape[2]]
    return rebin(a_n, scaling)


def rebin(a, factor):
    '''rebin ndarray data into a smaller ndarray of the same rank whose dimensions
    are factors of the original dimensions. eg. An array with 6 columns and 4 rows
    can be reduced to have 6,3,2 or 1 columns and 4,2 or 1 rows.
    example usages:
    >>> a=rand(6,4); b=rebin(a,3,2)
    >>> a=rand(6); b=rebin(a,2)
    '''
    shape = a.shape
    lenShape = len(shape)
    factor = np.asarray(factor, dtype=np.int)
    args = (np.asarray(shape) / factor).astype(np.int)
    evList = ['a.reshape('] + \
             ['args[{}],factor[{}],'.format(i,i) for i in range(lenShape)] + \
             [')'] + ['.sum({})'.format(i+1) for i in range(lenShape)] + \
             ['//factor[{}]'.format(i) for i in range(lenShape)]
    return eval(''.join(evList))


def human_time(sec):
    if sec < 30:
        return "a moment"
    elif sec < 120:
        return 'a minute or so'
    elif sec < 3600:
        return "about {} minutes".format(int(sec) // 60)
    elif sec < 3600*2:
        minutes = (int(sec) - 3600) // 60
        if minutes < 5:
            return "about an hour"
        return "one hour and {} minutes".format(minutes)
    elif sec < 3600*3:
        minutes = (int(sec) - 3600*2) // 60
        if minutes < 5:
            return "about two hours"
        return "two hours and {} minutes".format(minutes)
    
    hours = int(sec) // 3600
    return "about {} hours".format(hours)        

#def human_time(sec):
#    seconds = [86400., 3600., 60., 1.]
#    names = ['days', 'hours', 'minutes', 'seconds']
#
#    if sec <= seconds[-1]:
#        return 'less than a {0}'.format(names[-1][:-1])
#    times = []
#    labels = []
#    for i, s in enumerate(seconds):
#        if sec >= s:
#            times.append(int(math.floor(sec/s)))
#            labels.append(names[i])
#            if times[0] == 1:
#                labels[0] = labels[0][:-1]
#            if i < len(seconds) - 1:
#                times.append(int(math.floor((sec - s * times[0])/seconds[i+1])))
#                labels.append(names[i+1])
#                if times[1] == 1:
#                    labels[1] = labels[1][:-1]
#
#            break
#    else:
#        return 'less than a {0}'.format(names[-1][:-1])
#    if len(times) > 1:
#        if times[-1] == 0:
#            return 'about {0} {1}'.format(times[0], labels[0])
#        return 'about {0} {1} and {2} {3}'.format(times[0], labels[0], times[1], labels[1])
#    return 'about {0} {1}'.format(times[0], labels[0])


def circle_mask(array_shape, radius, center=None):
    a = np.zeros(array_shape, np.int)
    if not center:
        cx = array_shape[0] // 2
        cy = array_shape[1] // 2
    else:
        cx, cy = (int(c) for c in center)
    rint = int(np.ceil(radius))
    y, x = np.ogrid[-rint: rint+1, -rint: rint+1]
    index = x**2 + y**2 <= radius**2
    a[cx-rint:cx+rint+1, cy-rint:cy+rint+1][index] = 1
    return a


def sphere_mask(array_shape, radius, center=None):
    a = np.zeros(array_shape, np.int)
    if not center:
        cx = array_shape[0] / 2
        cy = array_shape[1] / 2
        cz = array_shape[2] / 2
    else:
        cx, cy, cz = center
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


