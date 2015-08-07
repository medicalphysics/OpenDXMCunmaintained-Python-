# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 10:07:58 2015

@author: erlean
"""

import numpy as np
import dicom
import os
import sys
import utils
import logging
from database import Patient, CTProtocol, Simulation
logger = logging.getLogger('OpenDXMC')
logger.setLevel(10)

def array_from_dicom_list(dc_list):
    r = int(dc_list[0][0x28, 0x10].value)
    c = int(dc_list[0][0x28, 0x11].value)
    n = len(dc_list)
    arr = np.empty((r, c, n), dtype=np.int16)

    for i, dc in enumerate(dc_list):
       arr[:, :, i] = dc.pixel_array * int(dc[0x28, 0x1053].value) + int(dc[0x28, 0x1052].value)
    return arr

def import_ct(directory_path):
    series = {}
    for p in utils.find_all_files([os.path.abspath(directory_path)]):
        try:
            dc = dicom.read_file(p)
        except dicom.filereader.InvalidDicomError:
            logger.debug("Not imported: {} -Invalid dicom file".format(p))
        else:
            #test for ct image
            sop_class_uid = str(dc[0x8, 0x16].value)
            if sop_class_uid == "1.2.840.10008.5.1.4.1.1.2":
                axial_image = str(dc[0x8, 0x8].value[2]).lower()
                if axial_image == 'axial':
                    series_uid = str(dc[0x20, 0xe].value)
                    if series_uid in series:
                        series[series_uid].append(dc)
                    else:
                        series[series_uid] = [dc]
                    logger.debug("Imported {}".format(p))
                else:
                    logger.debug("Not imported: {} -Image not axial".format(p))
            else:
                logger.debug("Not imported: {} -Not a CT image".format(p))

    logger.info("Imported {0} CT series with total {1} images".format(
                len(series), sum([len(x) for x in series.values()])))


    for key, value in series.items():
        value.sort(key=lambda x: x[0x20, 0x32].value[2])
        pat = Patient(key)
        pat.ct_array = array_from_dicom_list(value)







