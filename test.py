# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 15:17:40 2015

@author: erlean
"""
import logging
logger = logging.getLogger('OpenDXMC')
logger.addHandler(logging.StreamHandler())
logger.setLevel(10)
import pylab as plt
import numpy as np
from scipy.ndimage.interpolation import affine_transform, map_coordinates

import pdb

def test_import():
    from opendxmc.study import import_ct_series
    p = "C://GitHub//thorax//DICOM//00000058//AAE1C604//AAF19E09//0000A918"
    p = "C://GitHub//thorax//DICOM//00000058//AAE1C604//AAF19E09//00005B5E"
#    p = "C://Users//ander//Documents//GitHub//test_abdomen"
    for pat in import_ct_series([p]):
        pass

def matrix(orientation, spacing):
    x = np.array(orientation[:3], dtype=np.float)
    y = np.array(orientation[3:], dtype=np.float)
    z = np.cross(x, y)
#    print('z', z)
    M = np.array([x * spacing, y * spacing, z * spacing])
#    test = M.dot(np.ones(3))
#    print('test', test)
#
#    if test[2] < 0:
#        M = M.dot(np.array([[1, 0, 0],[0,-1, 0],[0, 0, 1]]))
#    if test[1] < 0:
#        M = M.dot(np.array([[1, 0, 0],[0, 1, 0],[0, 0, -1]]))
#    if test[0] < 0:
#        M = M.dot(np.array([[-1, 0, 0],[0, 1, 0],[0, 0, 1]]))
#    print('test2', M.dot(np.ones(3)))
##    M = np.array([x, y, z])
##    M[np.diag_indices(3)] *= spacing
##    while True:
##        test = M.dot(np.ones(3))
##        if
#    pdb.set_trace()
    return M
#    return np.dot(M, np.array([[0, 1, 0], [1, 0, 0],[0, 0, 1]]))



def array_from_dicom_list_affine():

    arr = np.zeros((50, 50, 50), dtype=np.float)
    arr[20:26, 23:26, 23:29] = 1
    orient = [0, 1, 0, 0, 0, -1]
#    orient = [1,0,0,0,1,0]
    spacing = np.array([2, 4, 2])
#    spacing = np.ones(3)
#    spacing[0] = 2

#    scan_spacing = find_scan_spacing(dc_list[0][0x20, 0x37].value, spacing, shape)
    M = matrix(orient, spacing)
    print('M', M)
    print('M.I', np.linalg.inv(M))
    out_dimension = M.dot(np.array(arr.shape))
    print('out dim', out_dimension)
    offset = np.linalg.inv(M).dot(out_dimension * (out_dimension < 0))
    print('offset', offset)

    x = np.array(orient[:3], dtype=np.float)
    y = np.array(orient[3:], dtype=np.float)
    z = np.cross(x, y)
    print(x, y, z)

#    offset = np.array(arr.shape) * (out_dimension < 0)
#    offset *= -1 * (offset < 0)
#    print('raw', offset)
#    offset[2] = 50
#    offset=25

    out_shape = tuple(np.abs(np.rint(out_dimension).astype(np.int)))
    print(out_shape, offset)

    k = affine_transform(arr, np.linalg.inv(M), output_shape=out_shape, cval=-1, offset=offset)
    print(k.max(), k.min())

    plt.subplot(2,3,1)
    plt.imshow(k[:,:,k.shape[2] // 2])
    plt.subplot(2,3,2)
    plt.imshow(k[:,k.shape[1] // 2, :])
    plt.subplot(2,3,3)
    plt.imshow(k[k.shape[0] // 2, :, :])

    plt.subplot(2,3,4)
    plt.imshow(arr[:, :, arr.shape[2] // 2])
    plt.subplot(2,3,5)
    plt.imshow(arr[:,arr.shape[1] // 2, :])
    plt.subplot(2,3,6)
    plt.imshow(arr[arr.shape[0] // 2, :, :])
    plt.show()
    pdb.set_trace()
def test_affine():
    array_from_dicom_list_affine()

if __name__ == '__main__':
    test_import()
#    test_affine()