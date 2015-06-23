# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 13:47:11 2015

@author: ERLEAN
"""

import os


def find_all_files(pathList):
    for path in pathList:
        if os.path.isdir(path):
            for dirname, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    yield os.path.normpath(os.path.join(dirname, filename))
        elif os.path.isfile(path):
            yield os.path.normpath(path)
