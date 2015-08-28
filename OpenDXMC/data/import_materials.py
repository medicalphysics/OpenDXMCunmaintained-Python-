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

material_data_path = os.path.join(os.path.dirname(sys.argv[0]), 'opendxmc',
                                  'data', 'materials')


def get_stored_materials():
    density_file = os.path.join(material_data_path, "densities.txt")
    organic_file = os.path.join(material_data_path, "organics.txt")

    for p in find_all_files([os.path.join(material_data_path, 'attinuation')]):
        name = os.path.splitext(os.path.basename(p))[0]
        # test for valid material name
        if re.match('^[\w-]+$', name) is None:
            logger.warning(
                "material file {0} contains illegal characters"
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
