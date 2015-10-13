# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 13:00:16 2015

@author: erlean
"""

from opendxmc.data.import_phantoms import read_phantoms
import pdb
import pylab as plt


def main():
    for sim in read_phantoms():
        i = 10
        print(sim.organ.min(), sim.organ.max())
        while i < sim.organ.shape[2]:
            plt.imshow(sim.organ[:,:,i])
            plt.show(block=True)
            i += 10

        pdb.set_trace()

if __name__ == '__main__':
    main()

