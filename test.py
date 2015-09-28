# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 15:17:40 2015

@author: erlean
"""
import logging
logger = logging.getLogger('OpenDXMC')
logger.addHandler(logging.StreamHandler())
logger.setLevel(10)

def test_import():
    from opendxmc.study import import_ct_series
    p = "C://GitHub//thorax//DICOM//00000058//AAE1C604//AAF19E09//0000AA17"
    for pat in import_ct_series([p]):
        pass


if __name__ == '__main__':
    test_import()