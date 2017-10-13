# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 10:04:26 2015
@author: erlean
"""

from opendxmc.app import start
import opendxmc
import logging
logger = logging.getLogger('OpenDXMC')
logger.addHandler(logging.StreamHandler())
logger.setLevel(10)
LOG_FORMAT = ("[%(asctime)s %(name)s %(levelname)s]  -  %(message)s  -  in method %(funcName)s line:"
    "%(lineno)d filename: %(filename)s")


def main():
    logger.info('Starting OpenDXMC version {}'.format(opendxmc.VERSION))
    start(version=opendxmc.VERSION)

if __name__ == '__main__':
    main()

