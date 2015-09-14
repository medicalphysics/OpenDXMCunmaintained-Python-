# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 10:07:58 2015

@author: erlean
"""

import numpy as np
import dicom
import os

import logging
from scipy.ndimage.interpolation import affine_transform
from opendxmc.utils import find_all_files
from opendxmc.study.simulation import Simulation

logger = logging.getLogger('OpenDXMC')

def array_from_dicom_list(dc_list, scaling):
    r = int(dc_list[0][0x28, 0x10].value)
    c = int(dc_list[0][0x28, 0x11].value)
    n = len(dc_list)
    arr = np.empty((r, c, n), dtype=np.int16)




    for i, dc in enumerate(dc_list):
        arr[:, :, i] = (dc.pixel_array * int(dc[0x28, 0x1053].value) +
                        int(dc[0x28, 0x1052].value))
    out_shape = np.round(np.array([r, c, n]) * scaling).astype(np.int)
    return affine_transform(arr, 1./scaling.astype(np.float), output_shape=out_shape, cval=-1000)


def aec_from_dicom_list(dc_list):
    n_im = len(dc_list)
    exp = np.empty((n_im, 2), dtype=np.float)
    for i, dc in enumerate(dc_list):
        exp[i, 1] = float(dc[0x18, 0x1152].value)
        exp[i, 0] = float(dc[0x20, 0x32].value[2]) / 10.
    return exp


def import_ct_series(paths, scaling=(.5, .5, 1)):
    series = {}
    for p in find_all_files(paths):
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
        logger.debug('Setting up data for simulation {}'.format(name))
        dc = value[0]
        value.sort(key=lambda x: x[0x20, 0x32].value[2])

        patient = Simulation(name)
        patient.scaling = scaling
        patient.exposure_modulation = aec_from_dicom_list(value)
        patient.ctarray = array_from_dicom_list(value, patient.scaling).astype(np.int16)
        spacing = np.empty(3, dtype=np.float)
        spacing[:2] = np.array(dc[0x28, 0x30].value)
        spacing[2] = np.sum((np.array(value[1][0x20, 0x32].value) -
                            np.array(value[0][0x20, 0x32].value))**2)**.5
        patient.spacing = spacing / 10. / patient.scaling

        tag_key = {'pitch': (0x18, 0x9311),
                   'scan_fov': (0x18, 0x60),
                   'sdd': (0x18, 0x1110),
                   'detector_width': (0x18, 0x9306),
                   'region': (0x18, 0x15)
                   }
        units_in_mm = ['scan_fov',
                       'sdd',
                       'detector_width',
                       ]
        for key, tag in tag_key.items():
            try:
                dc[tag].value
            except KeyError:
                pass
            else:
                if key in units_in_mm:
                    setattr(patient, key, float(dc[tag].value) / 10.)
                else:
                    setattr(patient, key, dc[tag].value)

        patient.is_spiral = patient.pitch != 0.
        if not patient.is_spiral:
            patient.step = abs(value[0][0x20, 0x32].value[2] -
                               value[1][0x20, 0x32].value[2])

        try:
            total_collimation = dc[0x18, 0x9307].value / 10.
        except KeyError:
            pass
        else:
            patient.detector_rows = (total_collimation /
                                      patient.detector_width) / 10.
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






