# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 10:07:58 2015

@author: erlean
"""

import numpy as np
import dicom
import os
import itertools
import logging
from scipy.ndimage.interpolation import affine_transform, spline_filter, zoom
from opendxmc.runner.ct_study_runner import ct_runner_validate_simulation
from opendxmc.utils import find_all_files
from opendxmc.database.h5database import Validator
from opendxmc.utils import rebin, rebin_scaling


logger = logging.getLogger('OpenDXMC')


def matrix_scaled(orientation, spacing, spacing_scan):
    iop = np.array(orientation, dtype=np.float).reshape(2, 3).T
    s_norm = np.cross(*iop.T[:])
    R = np.eye(3)
    R[:, :2] = np.fliplr(iop)
    R[:, 2] = s_norm
    M = np.eye(3)
    M[:3, :3] = R*spacing
    return (np.eye(3)/spacing_scan).dot(M)


def matrix(orientation):
    iop = np.array(orientation, dtype=np.float).reshape(2, 3).T
    s_norm = np.cross(*iop.T[:])
    R = np.eye(3)
    R[:, :2] = np.fliplr(iop)
    R[:, 2] = s_norm
    return R


def image_to_world_transform(image_vector, position, orientation, spacing):
    iop = np.array(orientation, dtype=np.float).reshape(2, 3).T
    s_norm = np.cross(*iop.T[:])
    R = np.eye(3)
    R[:, :2] = np.fliplr(iop)
    R[:, 2] = s_norm
    R[:3, :3] *= spacing
    return np.dot(R, image_vector) + position


#def array_from_dicom_list_low_memory(dc_list, scaling):
#    scaling = np.array(scaling)
#    r = int(dc_list[0][0x28, 0x10].value)
#    c = int(dc_list[0][0x28, 0x11].value)
#    n = len(dc_list)
#    shape = np.ceil(np.array([r, c, n]) / scaling).astype(np.int)
#    stride = int(np.ceil(scaling[2]))
#
#    arr = np.empty(shape, dtype=np.int16)
#
#    sub_arr = np.empty((r, c, stride), dtype=np.int16)
#
#    n_ind = 0
#    while (n_ind+1)*stride < n:
#        for i, dc in enumerate(dc_list[n_ind*stride:(n_ind+1)*stride]):
#            sub_arr[:, :, i] = (dc.pixel_array * int(dc[0x28, 0x1053].value) +
#                                int(dc[0x28, 0x1052].value))
##        import pdb
##        pdb.set_trace()
#        arr[:, : , n_ind] = np.squeeze(zoom(sub_arr, 1./scaling))
#
#        n_ind += 1
#
#    if n_ind*stride < n:
#
#        for i, dc in enumerate(dc_list[n_ind*stride:]):
#            sub_arr[:, :, i] = (dc.pixel_array * int(dc[0x28, 0x1053].value) +
#                                int(dc[0x28, 0x1052].value))
#        for j in range(i, stride):
#            sub_arr[:, :, j] = sub_arr[:, :, i]
#
#        arr[:, : , n_ind] = np.squeeze(zoom(sub_arr, 1./scaling))
#
#    return arr


def array_from_dicom_list(dc_list, scaling):
    r = int(dc_list[0][0x28, 0x10].value)
    c = int(dc_list[0][0x28, 0x11].value)
    n = len(dc_list)

    arr = np.empty((r, c, scaling[2]), dtype=np.int16)
    for i in range(min((n, scaling[2]))):
        dc = dc_list[i]
        arr[:, :, i] = (dc.pixel_array * int(dc[0x28, 0x1053].value) +
                        int(dc[0x28, 0x1052].value))
    if n <= scaling[2]:
        return arr[:, :, :n].mean(axis=2)

    arr_r = rebin_scaling(arr, scaling)
    teller = 0
    for i in range(scaling[2], n):
        dc = dc_list[i]
        arr[:, :, teller] = (dc.pixel_array * int(dc[0x28, 0x1053].value) +
                        int(dc[0x28, 0x1052].value))
        teller += 1
        if teller >= scaling[2]:
            teller = 0
            arr_r = np.concatenate((arr_r, rebin_scaling(arr, scaling)), axis=2)

    return arr_r


def aec_from_dicom_list(dc_list, iop, spacing):
    n_im = len(dc_list)
    exp = np.empty((n_im, 2), dtype=np.float)
    pos = np.zeros(3)

    for i, dc in enumerate(dc_list):
        exp[i, 1] = float(dc[0x18, 0x1152].value)
        exp[i, 0] = image_to_world_transform(np.array([0, 0, i]), pos, iop, spacing)[2]
    exp[:,0] /= 10.0 # for cm units
    return exp


def dc_slice_indicator(dc):
    """Returns a number indicating slce z position"""
    pos = np.array(dc[0x20, 0x32].value)
    iop = np.array(dc[0x20, 0x37].value).reshape((2, 3)).T
    return np.inner(pos, np.cross(*iop.T[:]))


def z_stop_estimator(iop, spacing, shape):
    choices = []
    pos = np.zeros(3)
    for i in [0, shape[0]]:
        for j in [0, shape[1]]:
            for k in [0, shape[2]]:
                choices.append(image_to_world_transform(np.array([i, j, k]), pos, iop, spacing)[2])
    return min(choices), max(choices)



def import_ct_series(paths, import_scaling=(2, 2, 2)):

    import_scaling=np.rint(np.array(import_scaling)).astype(np.int)
    import_scaling[import_scaling < 1] = 1

    series = {}
    for p in find_all_files(paths):
        try:
            dc = dicom.read_file(p)
        except dicom.filereader.InvalidDicomError:
            logger.debug("Not imported: {} -Invalid dicom file".format(p))
        else:
            # test for ct image
            try:
                sop_class_uid = str(dc[0x8, 0x16].value)
            except KeyError:
                logger.debug('Not imported: {} -No SOP UID for file.'.format(p))
                continue
            if sop_class_uid == "CT Image Storage":
                axial_image = str(dc[0x8, 0x8].value[2]).lower()
                if axial_image == 'axial':
                    series_uid = str(dc[0x20, 0xe].value)
                    if series_uid in series:
                        series[series_uid].append(p)
                    else:
                        series[series_uid] = [p]
                    logger.debug("Imported {}".format(p))
                else:
                    logger.debug("Not imported: {} -Image not axial, possible a scout".format(p))
            else:
                logger.debug("Not imported: {0} -Not a CT image:{1}".format(p, sop_class_uid))

    logger.info("Importing {0} CT series with total {1} images".format(
                len(series), sum([len(x) for x in series.values()])))

    for name, series_paths in series.items():

        if len(series_paths) < 2:
            logger.info('Image series {} is skipped since it contains less than 2 images.'.format(name))
            continue
        logger.debug('Setting up data for simulation {}'.format(name))
        dc_list = [dicom.read_file(p) for p in series_paths]
#        dc_list.sort(key=lambda x: x[0x20, 0x13].value)
        dc = dc_list[0]
        dc_list.sort(key=lambda x: dc_slice_indicator(x))

        spacing = np.empty(3, dtype=np.float)
        spacing[:2] = np.array(dc[0x28, 0x30].value)
        spacing[2] = np.sum((np.array(dc_list[1][0x20, 0x32].value) -
                            np.array(dc_list[0][0x20, 0x32].value))**2)**.5

        #Creating transforrmation matrix
        patient = Validator()
        patient.name = name
        patient.import_scaling = import_scaling

        patient.exposure_modulation = aec_from_dicom_list(dc_list,
                                                          np.array(dc[0x20, 0x37].value),
                                                          spacing)

#        try:
        patient.ctarray = array_from_dicom_list(dc_list, import_scaling)
#        except MemoryError:
#            logger.warning('Memory error when importing {}. Attempting low memory import method.'.format(name))
#            patient.ctarray = array_from_dicom_list_low_memory(dc_list, import_scaling)
        patient.shape = np.array(patient.ctarray.shape, dtype=np.int)
        patient.spacing = spacing / 10. * np.array(import_scaling)
        patient.image_position = np.array(dc[0x20, 0x32].value) / 10.
        patient.image_orientation = np.array(dc[0x20, 0x37].value)



        try:
            patient.data_center = np.array(dc[0x18, 0x9313].value) / 10. - patient.image_position
        except KeyError:
            patient.data_center = image_to_world_transform(np.array(patient.shape) / 2.,
                                                           patient.image_position,
                                                           patient.image_orientation,
                                                           patient.spacing)

        tag_key = {'pitch': (0x18, 0x9311),
                   'scan_fov': (0x18, 0x90),
                   'sdd': (0x18, 0x1110),
                   'aquired_kV': (0x18, 0x60),
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
        if patient.aquired_kV > 0:
            patient.kV = patient.aquired_kV
        # correction for sdd
        try:
            patient.sdd = dc[0x18, 0x1111].value * 2 / 10.
        except KeyError:
            pass

        patient.is_spiral = patient.pitch != 0.
        if not patient.is_spiral:
            patient.step = abs(dc_list[0][0x20, 0x32].value[2] -
                               dc_list[1][0x20, 0x32].value[2])

        try:
            total_collimation = dc[0x18, 0x9307].value / 10.
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

        start, stop = z_stop_estimator(patient.image_orientation, patient.spacing, patient.ctarray.shape)
        patient.start_scan = start
        patient.stop_scan = stop
        patient.start = start
        patient.stop = stop


        yield patient.get_data()

