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
from simulation import Simulation 

import matplotlib.pyplot as plt
import pdb

logger = logging.getLogger('OpenDXMC')
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

def array_from_dicom_list(dc_list):
    r = int(dc_list[0][0x28, 0x10].value)
    c = int(dc_list[0][0x28, 0x11].value)
    n = len(dc_list)
    arr = np.empty((r, c, n), dtype=np.int16)

    for i, dc in enumerate(dc_list):
        arr[:, :, i] = (dc.pixel_array * int(dc[0x28, 0x1053].value) +
                        int(dc[0x28, 0x1052].value))
    return arr

def aec_from_dicom_list(dc_list):
    n_im = len(dc_list)
    exp = np.empty((n_im, 2), dtype=np.float)
    for i, dc in enumerate(dc_list):
        exp[i, 1] = float(dc[0x18, 0x1152].value) 
        exp[i, 0] = float(dc[0x20, 0x32].value[2])
    return exp


def import_ct(directory_path):
    series = {}
    for p in utils.find_all_files([os.path.abspath(directory_path)]):

        try:
            dc = dicom.read_file(p)
        except dicom.filereader.InvalidDicomError:
            logger.debug("Not imported: {} -Invalid dicom file".format(p))
        else:
            # test for ct image
            sop_class_uid = str(dc[0x8, 0x16].value)

            if sop_class_uid == "CT Image Storage":
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
                logger.debug("Not imported: {0} -Not a CT image:{1}".format(p, sop_class_uid))

    logger.info("Imported {0} CT series with total {1} images".format(
                len(series), sum([len(x) for x in series.values()])))

    for name, value in series.items():
        dc = value[0]
        value.sort(key=lambda x: x[0x20, 0x32].value[2])

        patient = Simulation(name)
        patient.exposure_modulation = aec_from_dicom_list(value)
        patient.ctarray = array_from_dicom_list(value)
        spacing = np.empty(3, dtype=np.float)
        spacing[:2] = np.array(dc[0x28, 0x30].value)
        spacing[2] = np.sum((np.array(value[1][0x20, 0x32].value) -
                            np.array(value[0][0x20, 0x32].value))**2)**.5
        patient.spacing = spacing

        tag_key = {'pitch': (0x18, 0x9311), 
                   'scan_fov': (0x18, 0x60),
                   'sdd': (0x18, 0x1110),
                   'detector_width': (0x18, 0x9306),
                   'region': (0x18, 0x15)
                   }
        for key, tag in tag_key.items():
            try:
                dc[tag].value
            except KeyError:
                pass
            else:
                setattr(patient, key, dc[tag].value)
        if patient.pitch == 0.:
            patient.is_spiral = False
        else:
            patient.is_spiral = True
        try:
            total_collimation = dc[0x18, 0x9307]
        except KeyError:
            pass
        else:
            patient.detector_rows = (total_collimation /
                                      patient.detector_width)
        try:                                      
            exposure = float(dc[0x18, 0x1152].value)
            ctdi = float(dc[0x18, 0x9345].value)
        except KeyError:
            pass
        else:
            if exposure > 0:
                if patient.is_spiral:
                    patient.ctdi_w100 = ctdi / exposure / patient.pitch * 100.
                else:
                    patient.ctdi_w100 = ctdi / exposure * 100.

        patient.start = patient.exposure_modulation[0, 0]
        patient.stop = patient.exposure_modulation[-1, 0]
        yield patient
        





