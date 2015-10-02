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
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.ndimage.interpolation import affine_transform

import pdb


def test_spiral_transformation():
    sdd = 1200.
    pos = [-77.36, -374.4, -119.33]
#    pos = [0, 0, 0]
    orientation = [.99355, .087918, .071490, -.076963, .986621, -.143718]
    spacing = np.array([0.367, .367, 4.])
    shape = np.array([626, 512, 45])
    pitch = 1.
    collimation = 40.
    data_collection_center = [.2, -240.8, -93.6]
    
    #generating rotation matrix from voxels to world
    iop = np.array(orientation, dtype=np.float).reshape(2, 3).T
    s_norm = np.cross(*iop.T[:])
    R = np.eye(3)
    R[:, :2] = np.fliplr(iop)
    R[:, 2] = s_norm
   
    #matrix from voxel indces to world
    M = np.eye(3)
    M[:3, :3] = R*spacing

    box = np.zeros((10, 3))
    box[0, :] = np.dot(M, np.zeros(3)) + np.array(pos)
    box[1, :] = np.dot(M, np.array([shape[0], 0, 0])) + np.array(pos)
    box[2, :] = np.dot(M, np.array([shape[0], shape[1], 0])) + np.array(pos)
    box[3, :] = np.dot(M, np.array([0, shape[1], 0])) + np.array(pos)
    box[4, :] = np.dot(M, np.array([0, 0, 0])) + np.array(pos)
    box[5, :] = np.dot(M, np.array([0, 0, shape[2]])) + np.array(pos)
    box[6, :] = np.dot(M, np.array([shape[0], 0, shape[2]])) + np.array(pos)
    box[7, :] = np.dot(M, np.array([shape[0], shape[1], shape[2]])) + np.array(pos)
    box[8, :] = np.dot(M, np.array([0, shape[1], shape[2]])) + np.array(pos)
    box[9, :] = np.dot(M, np.array([0, 0, shape[2]])) + np.array(pos)



    

    start = np.array(pos)
    stop = start + np.dot(M, shape)
    n_spiral = 500
    spiral = np.zeros((3, n_spiral))
    
    spiral[2, :] = np.linspace(start[2]-collimation/2, stop[2]+collimation/2, n_spiral)
    angle = spiral[2, :] / (pitch * collimation) * np.pi * 2.
    spiral[0, :] = -sdd/2 * np.cos(angle)
    spiral[1, :] = -sdd/2 * np.sin(angle)
    
    spiral_center = np.array(data_collection_center)
    for i in range(2):
        spiral[i, :] += spiral_center[i]
   
   
    #image_space
    RI = np.linalg.inv(R)   
    im_space = lambda x: np.dot(RI, x.ravel() - np.array(pos))    
    
    for i in range(n_spiral):
        spiral[:, i] = im_space(spiral[:, i])
    for i in range(box.shape[0]):
        box[i, :] = im_space(box[i, :])
    
    
#    
#    
#    spiral_center = np.array(pos) - 
##    spiral_center = (np.array(pos) + np.dot(M, shape)) / 2.
#    spiral_center[2] = start[2]
#    spiral = np.zeros((3, n_spiral))
#    spiral[0, :] = sdd/2
#    spiral[2,:] = spiral_z[:]
#    RI = np.linalg.inv(R)   
#
#    for i in range(n_spiral):
#        spiral[:, i] = np.dot(R_a(angle[i]), np.squeeze(spiral[:, i]))+spiral_center
#  
#    for i in range(n_spiral):
#        spiral[:, i] = np.dot(RI, spiral[:, i] - spiral_center)
#    for i in range(box.shape[0]):
#        box[i, :] = np.dot(RI, box[i, :].ravel() - np.array(pos))
#    
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(box[:, 0], box[:, 1],box[:, 2], label='box')
    ax.plot(box[:, 0], box[:, 1],box[:, 2], 'o')
    ax.plot(spiral[0, :], spiral[1, :],spiral[2, :], label='spiral')
    ax.legend()
#    pdb.set_trace()
    plt.show()
    pdb.set_trace()
    


  
    


def test_import():
    from opendxmc.study import import_ct_series
    p = "C://test//thorax//DICOM//00000058//AAE1C604//AAF19E09//0000A918"
    p = "C://test//thorax//DICOM//00000058//AAE1C604//AAF19E09//0000AA17"
#    p = "C://test//thorax//DICOM//00000058//AAE1C604//AAF19E09//0000CB29"
#    p = "C://test//thorax//DICOM//00000058//AAE1C604//AAF19E09//00007706"
#    p = "C://test//thorax//DICOM//00000058//AAE1C604//AAF19E09//00005B5E"
    p = "C://test//thorax"
#    p = "C://test//abdomen"


    for pat in import_ct_series([p], scan_spacing=(.3, .3, .3)):
        plt.plot(pat.exposure_modulation[:,0], pat.exposure_modulation[:,1])
    plt.show()

#        k = pat.ctarray
#        plt.subplot(1,3,1)
#        plt.imshow(k[:,:,k.shape[2] // 2])
#        plt.subplot(1,3,2)
#        plt.imshow(k[:,k.shape[1] // 2, :])
#        plt.subplot(1,3,3)
#        plt.imshow(k[k.shape[0] // 2, :, :])
#        plt.show(block=True)

#def matrix(orientation, spacing):
#    y = np.array(orientation[:3], dtype=np.float)
#    x = np.array(orientation[3:], dtype=np.float)
#    z = np.cross(x, y)
##    print('z', z)
#    M = np.array([x * spacing, y * spacing, z * spacing])
#    M = np.array([x, y, z])
##    test = M.dot(np.ones(3))
##    print('test', test)
##
##    if test[2] < 0:
##        M = M.dot(np.array([[1, 0, 0],[0,-1, 0],[0, 0, 1]]))
##    if test[1] < 0:
##        M = M.dot(np.array([[1, 0, 0],[0, 1, 0],[0, 0, -1]]))
##    if test[0] < 0:
##        M = M.dot(np.array([[-1, 0, 0],[0, 1, 0],[0, 0, 1]]))
##    print('test2', M.dot(np.ones(3)))
###    M = np.array([x, y, z])
###    M[np.diag_indices(3)] *= spacing
###    while True:
###        test = M.dot(np.ones(3))
###        if
##    pdb.set_trace()
#    return M
##    return np.dot(M, np.array([[0, 1, 0], [1, 0, 0],[0, 0, 1]]))



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
    k=np.swapaxes(k, 0, 1)
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
    test_spiral_transformation()
#    test_import()
#    test_affine()