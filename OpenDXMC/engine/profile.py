# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 11:30:21 2015

@author: erlean
"""

import pstats, cProfile

import pyximport
pyximport.install()

from . import _interaction_func

cProfile.runctx()




cProfile.runctx("calc_pi.approx_pi()", globals(), locals(), "Profile.prof")

s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()

