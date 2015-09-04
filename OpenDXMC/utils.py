# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 13:47:11 2015

@author: ERLEAN
"""

import os
import math
import numpy as np


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


def find_all_files(pathList):
    for path in pathList:
        if os.path.isdir(path):
            for dirname, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    yield os.path.normpath(os.path.join(dirname, filename))
        elif os.path.isfile(path):
            yield os.path.normpath(path)
