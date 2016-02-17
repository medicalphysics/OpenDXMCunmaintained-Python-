# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 20:24:55 2016

@author: ander
"""

import numpy as np

from opendxmc.database import import_ct_series



def get_data(path):
    props, arrays = import_ct_series(path)
    return 
    
"""
N-D Bresenham line algo
Created by Vikas Dhiman on Wed, 25 Apr 2012 (MIT)
"""
import numpy as np
def _bresenhamline_nslope(slope):
    """
    Normalize slope for Bresenham's line algorithm.

    >>> s = np.array([[-2, -2, -2, 0]])
    >>> _bresenhamline_nslope(s)
    array([[-1., -1., -1.,  0.]])

    >>> s = np.array([[0, 0, 0, 0]])
    >>> _bresenhamline_nslope(s)
    array([[ 0.,  0.,  0.,  0.]])

    >>> s = np.array([[0, 0, 9, 0]])
    >>> _bresenhamline_nslope(s)
    array([[ 0.,  0.,  1.,  0.]])
    """
    scale = np.amax(np.abs(slope), axis=1).reshape(-1, 1)
    zeroslope = (scale == 0).all(1)
    scale[zeroslope] = np.ones(1)
    normalizedslope = np.array(slope, dtype=np.double) / scale
    normalizedslope[zeroslope] = np.zeros(slope[0].shape)
    return normalizedslope

def _bresenhamlines(start, end, max_iter):
    """
    Returns npts lines of length max_iter each. (npts x max_iter x dimension) 

    >>> s = np.array([[3, 1, 9, 0],[0, 0, 3, 0]])
    >>> _bresenhamlines(s, np.zeros(s.shape[1]), max_iter=-1)
    array([[[ 3,  1,  8,  0],
            [ 2,  1,  7,  0],
            [ 2,  1,  6,  0],
            [ 2,  1,  5,  0],
            [ 1,  0,  4,  0],
            [ 1,  0,  3,  0],
            [ 1,  0,  2,  0],
            [ 0,  0,  1,  0],
            [ 0,  0,  0,  0]],
    <BLANKLINE>
           [[ 0,  0,  2,  0],
            [ 0,  0,  1,  0],
            [ 0,  0,  0,  0],
            [ 0,  0, -1,  0],
            [ 0,  0, -2,  0],
            [ 0,  0, -3,  0],
            [ 0,  0, -4,  0],
            [ 0,  0, -5,  0],
            [ 0,  0, -6,  0]]])
    """
    if max_iter == -1:
        max_iter = np.amax(np.amax(np.abs(end - start), axis=1))
    npts, dim = start.shape
    nslope = _bresenhamline_nslope(end - start)

    # steps to iterate on
    stepseq = np.arange(1, max_iter + 1)
    stepmat = np.tile(stepseq, (dim, 1)).T

    # some hacks for broadcasting properly
    bline = start[:, np.newaxis, :] + nslope[:, np.newaxis, :] * stepmat

    # Approximate to nearest int
    return np.array(np.rint(bline), dtype=start.dtype)

def bresenhamline(start, end, max_iter=5):
    """
    Returns a list of points from (start, end] by ray tracing a line b/w the
    points.
    Parameters:
        start: An array of start points (number of points x dimension)
        end:   An end points (1 x dimension)
            or An array of end point corresponding to each start point
                (number of points x dimension)
        max_iter: Max points to traverse. if -1, maximum number of required
                  points are traversed

    Returns:
        linevox (n x dimension) A cumulative array of all points traversed by
        all the lines so far.

    >>> s = np.array([[3, 1, 9, 0],[0, 0, 3, 0]])
    >>> bresenhamline(s, np.zeros(s.shape[1]), max_iter=-1)
    array([[ 3,  1,  8,  0],
           [ 2,  1,  7,  0],
           [ 2,  1,  6,  0],
           [ 2,  1,  5,  0],
           [ 1,  0,  4,  0],
           [ 1,  0,  3,  0],
           [ 1,  0,  2,  0],
           [ 0,  0,  1,  0],
           [ 0,  0,  0,  0],
           [ 0,  0,  2,  0],
           [ 0,  0,  1,  0],
           [ 0,  0,  0,  0],
           [ 0,  0, -1,  0],
           [ 0,  0, -2,  0],
           [ 0,  0, -3,  0],
           [ 0,  0, -4,  0],
           [ 0,  0, -5,  0],
           [ 0,  0, -6,  0]])
    """
    # Return the points as a single array
    return _bresenhamlines(start, end, max_iter).reshape(-1, start.shape[-1])

def test(arr, inclination, azimuth):
    N = np.array(arr.shape)
    
    radii = (np.sum(N*N)**.5) / 2    
    
    P = radii * np.array([np.sin(inclination) * np.cos(azimuth),
                          np.sin(inclination) * np.sin(azimuth),
                          np.cos(inclination)]) 
    P = np.rint(P).astype(np.int)
    
    norm = - P / np.sum(P*P)**.5
    cotx = np.array([np.sin(inclination + np.pi/2) * np.cos(azimuth),
                     np.sin(inclination + np.pi/2) * np.sin(azimuth),
                     np.cos(inclination + np.pi/2)])
    cotx = cotx / np.sum(cotx**2)**.5                     
    coty = np.array([np.sin(inclination) * np.cos(azimuth + np.pi/2),
                     np.sin(inclination) * np.sin(azimuth + np.pi/2),
                     np.cos(inclination)])                     
    coty = coty / np.sum(coty**2)**.5
    
    
    
    grid_size = int(np.rint(radii))
    
   
    
    
    gridx, gridy = np.meshgrid(np.arange(-grid_size, grid_size), np.arange(-grid_size, grid_size))
    start = np.outer(np.ones(np.prod(gridx.shape)), P) 
    start += np.outer(gridx.ravel(), cotx) 
    start += np.outer(gridy.ravel(), coty)
    
    end = start + np.outer(np.ones(np.prod(gridx.shape)), norm * radii*2) 
    
    k = bresenhamline(start, end, max_iter=-1)
    k.shape
    
    
    
    


def test_dum():
    start = np.array([[-6.5, -6.5, -6.5]])
    print(start.shape)
    end = np.array([9, 9, 9])
    k = bresenhamline(start, end, max_iter=-1)
    print(k)
    

if __name__ == '__main__':
    test_dum()
    test(np.empty((9, 9, 9)), 0, 0)    
    

