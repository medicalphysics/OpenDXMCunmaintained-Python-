# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 13:49:06 2015

@author: ander
"""

import numpy as np
from matplotlib import pylab as plt
def world_image_matrix(orientation):
    iop = np.array(orientation, dtype=np.float).reshape(2, 3).T
    s_norm = np.cross(*iop.T[:])
    R = np.eye(3)
    R[:, :2] = np.fliplr(iop)
    R[:, 2] = s_norm
    return np.linalg.inv(R)


def rotation_z_matrix(alpha):
    return np.array([[np.cos(alpha), - np.sin(alpha), 0],
                     [np.sin(alpha), np.cos(alpha), 0],
                     [0, 0, 1]], dtype=np.double)

def test():
    batch_size= 40
    rotation_plane_cosines = np.array([1, 0, 0, 0, 1, 0], dtype='float64')
    rotation_center = np.array([0, 0, 0], dtype='float64')
    ang = np.pi/2. 
    sdd = 100.
    scan_fov = 50.
    d_col = 2.
    histories = batch_size
    
    
    ret = np.zeros(( batch_size, 8), dtype='float64')
    M = world_image_matrix(rotation_plane_cosines)
    rotation_center_image = np.dot(M, rotation_center[[1, 0, 2]])

    R = np.dot(M, rotation_z_matrix(ang))
    
    ret[:, 0] = -sdd/2.
    ret[:, 1] = 0
    ret[:, 2] = 0
#        print('t', t[i])
    ret[:, 0:3] = np.dot(ret[:, 0:3], R)
    
    for j in range(2):
        ret[:, j] += rotation_center_image[j]

    ret[:, 3] = sdd / 2.
    ret[:, 4] = scan_fov /2 * np.random.uniform(-1., 1., histories)
    ret[:, 5] = d_col * np.random.uniform(-1., 1.,
                                                    histories)
    ret[:, 3:6] = np.dot(ret[:, 3:6], R)

    lenght = np.sqrt(np.sum(ret[:, 3:6]**2, axis=0))
    ret[:, 3:6] /= lenght

    plt.plot(ret[:, 0], ret[:, 1], 'o')
    plt.plot(ret[:, 0]+ ret[:, 3]*10, ret[:, 1]+ret[:, 4]*10, 'o')
    plt.show()
    print(ret)

    


if __name__ == '__main__':
    test()