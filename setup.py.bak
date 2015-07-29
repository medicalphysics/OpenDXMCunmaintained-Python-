# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 12:18:26 2015

@author: erlean
"""


print 'python setup.py build_ext --inplace'

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy


numpy_dirs=numpy.get_include()

#ext_modules = [Extension("_siddon", ["_siddon.pyx"], include_dirs=[numpy_dirs])]
#ext_modules += [Extension("_interaction", ["_interaction.pyx"], include_dirs=[numpy_dirs])]
ext_modules = [Extension("engine._siddon_func", ["engine//_siddon_func.pyx"], include_dirs=[numpy_dirs], extra_compile_args=[])]
ext_modules += [Extension("engine._interaction_func", ["engine//_interaction_func.pyx"], include_dirs=[numpy_dirs], extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp'])]
ext_modules += [Extension("engine.cyrandom", ["engine//cyrandom.pyx"], include_dirs=[numpy_dirs], extra_compile_args=[])]
#ext_modules += [Extension("specter.specter", ["specter//specter.pyx"], include_dirs=[numpy_dirs], extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp'])]
#ext_modules = [Extension("engine.test", ["engine//test.pyx"], include_dirs=[numpy_dirs], extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp'])]
setup(
      name = 'OpenDXMC',
      cmdclass = {'build_ext': build_ext},
      ext_modules = cythonize(ext_modules, annotate=True)
    )
