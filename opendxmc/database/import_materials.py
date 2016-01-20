# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 14:41:06 2015

@author: erlean
"""


import os
import sys
import logging
import re
from opendxmc.materials.materials import Material
from opendxmc.utils import find_all_files

logger = logging.getLogger('OpenDXMC')

MATERIAL_DATA_PATH = os.path.join(os.path.dirname(sys.argv[0]), 'opendxmc',
                                  'data', 'materials')

BASE_PATH = os.path.dirname(os.path.dirname(__file__))

MATERIAL_DATA_PATH = os.path.join(BASE_PATH,
                                  'data', 'materials')


def get_stored_materials(material_data_path=None):
    if material_data_path is None:
        material_data_path = MATERIAL_DATA_PATH
    logger.debug('Importing materials from {}'.format(material_data_path))
    density_file = os.path.join(material_data_path, "densities.txt")
    organic_file = os.path.join(material_data_path, "organics.txt")

    for p in find_all_files([os.path.join(material_data_path, 'attinuation')]):
        name = str(os.path.splitext(os.path.basename(p))[0])
        # test for valid material name
        if re.match('^[\w-]+$', name) is None:
            logger.warning(
                "Material file {0} contains illegal characters"
                ", only alphanummeric characters and "
                "dashes are allowed.".format(p)
                )
            continue
        try:
            yield Material(name, att_file=p, density_file=density_file,
                           organic_file=organic_file)
        except Exception as e:
            logger.warning(str(e))
    return
