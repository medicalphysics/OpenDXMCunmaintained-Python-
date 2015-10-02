# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 10:07:58 2015

@author: erlean
"""

import numpy as np
import dicom
import os

import logging
from scipy.ndimage.interpolation import affine_transform, spline_filter
from opendxmc.utils import find_all_files
from opendxmc.study.simulation import Simulation

logger = logging.getLogger('OpenDXMC')


#def find_scan_spacing(orientation, spacing, shape):
#    x = np.array(orientation[:3], dtype=np.float)
#    y = np.array(orientation[3:], dtype=np.float)
#    z = np.cross(x, y)
#    M = np.matrix(np.array([x, y, z]))
#    dim = np.dot(spacing * shape, M)

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

#def matrix_scaled(orientation, spacing, spacing_scan):
#    y = np.array(orientation[:3], dtype=np.float)
#    x = np.array(orientation[3:], dtype=np.float)
#    z = np.cross(x, y)
#
##    M = np.array([x * spacing[0], y * spacing[1], z * spacing[2]]).T
#    M = np.array([x * spacing, y * spacing, z * spacing])
#    M = np.array([x, y, z])
#    M=np.zeros((3, 3))
#    for i, v in enumerate([x, y, z]):
#        M[:, i] = v[:]
#    return M.dot(np.eye(3)/spacing_scan)



def array_from_dicom_list_affine(dc_list, spacing, scan_spacing=(2, 2, 2)):
    sh = dc_list[0].pixel_array.shape
    n = len(dc_list)
#    arr = np.empty((sh[1], sh[0], n), dtype=np.int16)
    arr = np.empty((sh[0], sh[1], n), dtype=np.int16)
    for i, dc in enumerate(dc_list):
        try:
            arr[:, :, i] = dc.pixel_array * int(dc[0x28, 0x1053].value) + int(dc[0x28, 0x1052].value)
        except ValueError:
            arr[:, :, i] = int(dc[0x28, 0x1052].value)
            logger.info('Error in slice number {}. Slice is filled with air.'.format(i))
#    arr=np.swapaxes(arr, 0, 1)
    M = matrix_scaled(dc_list[0][0x20, 0x37].value, spacing, scan_spacing)
    out_dimension = M.dot(np.array(arr.shape))
    offset = np.linalg.inv(M).dot(out_dimension * (out_dimension < 0))
    out_shape = tuple(np.abs(np.rint(out_dimension).astype(np.int)))

    logger.info('Align and scale CT series from {0} to {1} voxels. '
                'Voxel spacing changed from {2} to {3}'.format(sh + (n,),
                                                               out_shape,
                                                               spacing,
                                                               scan_spacing))
    k = np.empty(out_shape, dtype=np.int16)
    arr = spline_filter(arr, order=3, output=np.int16)
    affine_transform(arr, np.linalg.inv(M), output_shape=out_shape, cval=-1000,
                     offset=offset, output=k, order=3, prefilter=False)
    k = np.swapaxes(k, 0, 1)
#    if offset[0] != 0:
#        k = k[::-1,:,:]
#    if offset[1] != 0:
#        k = k[:, ::-1, :]
#    if offset[2] != 0:
#        k = k[:, :, ::-1]
    return k





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
    M = matrix(dc_list[0][0x20, 0x37].value)
    n_im = len(dc_list)
    exp = np.empty((n_im, 2), dtype=np.float)
    for i, dc in enumerate(dc_list):
        exp[i, 1] = float(dc[0x18, 0x1152].value)
        exp[i, 0] = M.dot(np.array(dc[0x20, 0x32].value))[2] / 10.
#    import pylab as plt
#    plt.plot(exp[:,0], exp[:, 1])
#    plt.show(block=True)
    return exp


def import_ct_series(paths, scan_spacing=(.15, .15, .15)):
    series = {}
    scan_spacing = np.array(scan_spacing)
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

    logger.info("Imported {0} CT series with total {1} images".format(
                len(series), sum([len(x) for x in series.values()])))

    for name, series_paths in series.items():

        if len(series_paths) < 2:
            logger.info('Image series {} is skipped since it contains less than 2 images.'.format(name))
            continue
        logger.debug('Setting up data for simulation {}'.format(name))
        dc_list = [dicom.read_file(p) for p in series_paths]
        dc_list.sort(key=lambda x: x[0x20, 0x13].value)
        dc = dc_list[0]
        dc_list.sort(key=lambda x: np.sum((np.array(x[0x20, 0x32].value)-np.array(dc[0x20, 0x32].value))**2))
        M = matrix(dc_list[0][0x20, 0x37].value)
        dc_list.sort(key=lambda x: (M.dot(np.array(x[0x20, 0x32].value)))[2])

        spacing = np.empty(3, dtype=np.float)
        spacing[:2] = np.array(dc[0x28, 0x30].value)
        spacing[2] = np.sum((np.array(dc_list[1][0x20, 0x32].value) -
                            np.array(dc_list[0][0x20, 0x32].value))**2)**.5

        #Creating transforrmation matrix
        patient = Simulation(name)
#        patient.spacing = scaling
        patient.exposure_modulation = aec_from_dicom_list(dc_list)
        patient.ctarray = array_from_dicom_list_affine(dc_list, spacing, scan_spacing*10).astype(np.int16)

        patient.spacing = scan_spacing
        patient.spacing_native = spacing

        tag_key = {'pitch': (0x18, 0x9311),
                   'scan_fov': (0x18, 0x90),
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
        patient.start_scan = patient.exposure_modulation[0, 0]
        patient.stop_scan = patient.exposure_modulation[-1, 0]
        patient.start = patient.exposure_modulation[0, 0]
        patient.stop = patient.exposure_modulation[-1, 0]
        yield patient






