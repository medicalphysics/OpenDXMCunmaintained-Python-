# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 14:52:07 2015

@author: erlean
"""


import matplotlib
#matplotlib.use("Agg")
import matplotlib.animation as manimation
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy import ndimage
import tables as tb
import random
import pdb

def read_arrays(path):
    h5 = tb.open_file(path, 'r')
#    for name, obje in zip(['dose_s', 'dose', 'ctarray', 'tissue_map'],[dose_s, dose, ctarray, tissue_map]):
    arr = h5.root.density_array.read()
    ds = h5.root.dose.read()
    spac = h5.root.spacing.read()

    h5.close()
    return arr, ds, spac

def mix(im, do, ax1, ax2, norm=90000., cut=10000.):
    ax2.imshow(np.squeeze(im), cmap=plt.cm.gist_gray, alpha=1, vmin=-100, vmax=300)
    mask = np.squeeze(do < cut)
    masked_data = np.ma.masked_array(np.squeeze(do)/norm, mask)
    ax1.imshow(np.squeeze(im), cmap=plt.cm.gist_gray, alpha=1, vmin=-100, vmax=300)
    ax1.pcolormesh(masked_data, cmap=plt.cm.jet, alpha =1, antialiased=True, vmin =0., vmax=1., edgecolor=(1.0, 1.0, 1.0, 0.), linewidth=0.0015625, rasterized=True)
#    ax1.colorbar()

def mix2(image, dose, ax1, ax2, norm=900000., cut=30000., immin=-500.,
         immax=1000., spacing=None):
    k1, k2 = -500., 1000.
    k1 = immin
    k2 = immax

    if spacing is None:
        aspect = 1
    else:
        aspect = spacing[1]*image.shape[1] / float(spacing[0]*image.shape[0])


    im_n = (image-k1) / (k2-k1)
    im_n[im_n > 1.] = 1.
    im_n[im_n < 0] = 0
    do_n = (dose) / (norm-cut)
    do_n[do_n > 1.] = 1.
    do_n[do_n < 0.] = 0
    n = cut / (norm - cut)
    mask = np.repeat(do_n < n, 4).reshape(do_n.shape + (4,))

    im_a = plt.cm.gist_gray(np.squeeze(im_n))
    ax2.imshow(im_a, aspect=aspect)#, vmin=-100, vmax=300)
    do_a = plt.cm.jet(np.squeeze(do_n))

    do_a[mask] = im_a[mask]
    ax1.imshow(do_a, aspect=aspect)

def interpolate_z(arr, i, left=True, method='linear'):
    sh = arr.shape
    if not (0 < i < sh[2]):
        return np.zeros(sh[:2])

    if method == 'nearest':
        k = int(np.rint(i))
        return arr[:,:, k]

    else:
        x0 = np.floor(i)
        if left:
            x1 = x0 + 1.
        else:
            x1 = x0 - 1.
        return arr[:,:,int(x0)] + (arr[:,:,int(x1)] - arr[:,:,int(x0)]) * float(i - x0)

def interpolate_y(arr, i, scale=1., left=True, method='linear'):
    sh = arr.shape
    if not (0 < i < sh[1]):
        return np.zeros((sh[0], sh[2]))

    if method == 'nearest':
        k = int(np.rint(i))
        return np.squeeze(arr[:, k ,:])
    else:

        x0 = np.floor(i)
        if left:
            x1 = x0 + 1.
        else:
            x1 = x0 - 1.
        return np.squeeze(arr[:,int(x0),:] + (arr[:,int(x1),:] - arr[:,int(x0),:]) * float(i - x0))



def make_animation(array, dose, name='mov', spacing=None):


#
#    i = 20
#    dmin = np.percentile(dose[dose > 0.], 10)
#    dmax = np.percentile(dose[dose > 0.], 99)
#    fig = plt.figure(facecolor='k', edgecolor='k', tight_layout=True)
#
#    ax1 = fig.add_subplot(121, axisbg='k')
#    ax2 = fig.add_subplot(122, axisbg='k')
#    ax1.cla()
#    ax2.cla()
#
##    fig.tight_layout(pad=.010, w_pad=.010, h_pad=.01)
#    fig.subplots_adjust(0.0, 0.0, 1., 1., 0., 0.)
#    arr_slice = np.squeeze(array[:,:,i])
#    dose_slice = np.squeeze(dose[:,:,i])
#    print dose_slice.min(), dose_slice.max(), dose_slice.mean()
#    mix2(arr_slice, dose_slice, ax1, ax2, dmax, dmin)
#    plt.show()
#
#    pdb.set_trace()
#    return

    if spacing is None:
        spacing = (1., 1., 1.)


    FFMpegWriter = manimation.writers['ffmpeg']

    metadata = dict(title='CT dose', artist='Erlend Andersen',
            comment='erlend.andersen@sshf.no')
    writer = FFMpegWriter(fps=30, metadata=metadata, bitrate=-1)

    fig = plt.figure(facecolor='k', edgecolor='k', size=(6, 3))


    ax1 = fig.add_subplot(121, axisbg='k')
    ax2 = fig.add_subplot(122, axisbg='k')
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)


#    fig.tight_layout(rect=(0,0,1,1))
    fig.subplots_adjust(0.0, 0.0, 1., 1., 0., 0.)

    plt.xlim(0, 256)
    plt.ylim(0, 256)

    dmin = float(np.percentile(dose[dose > 0.], 10))
    dmax = float(np.percentile(dose[dose > 0.], 99))
    amax = float(np.percentile(array, 99))
    amin = float(np.percentile(array, 1))


    print 'Creating animations...'

    with writer.saving(fig, "C://test//axial"+name+".mp4", 256):
#        for i in range(array.shape[2]):
        spac = (spacing[0], spacing[1])
        for i in np.linspace(array.shape[2]-2,0,array.shape[2]*6):
#            fig.clear()
            ax1.cla()
            ax2.cla()
            arr_slice = interpolate_z(array, i, method='nearest')
            dose_slice = interpolate_z(dose, i, method='nearest')
#            arr_slice = np.squeeze(array[:,:,i])
#            dose_slice = np.squeeze(dose[:,:,i])
            mix2(arr_slice, dose_slice, ax1, ax2, dmax, dmin, amin, amax, spac)
            writer.grab_frame(facecolor='k', edgecolor='k')

            print '{0} %'.format(int(round(100*i / float(array.shape[2]))))
#
    with writer.saving(fig, "C://test//saggital"+name+".mp4", 256):
#        for i in range(array.shape[1]):
        spac = (spacing[0], spacing[2])
        for i in np.linspace(0,array.shape[1]-2,array.shape[1]*2):
            ax1.cla()
            ax2.cla()
            arr_slice = interpolate_y(array, i, method='nearest')
            dose_slice = interpolate_y(dose, i, method='nearest')
#            arr_slice = np.squeeze(array[:,i,:])
#            dose_slice = np.squeeze(dose[:,i,:])

            mix2(arr_slice, dose_slice, ax1, ax2, dmax, dmin, amin, amax, spac)
#            mix2(arr_slice, dose_slice, ax1, ax2)
            writer.grab_frame(facecolor='k', edgecolor='k')

            print '{0} %'.format(int(round(100 * i / float(array.shape[1]))))

    print 'animations finished'



def make():
    name = 'Golem'
    dens, dose, spacing = read_arrays("C://test//data.h5")

#    dose = sp.ndimage.gaussian_filter(d, 2.)


#    plt.imshow(interpolate_z(arr, 50))
#    plt.show()
#    pdb.set_trace()
    make_animation(dens.astype(np.float), dose.astype(np.float), name=name)

