# python setup.py build_ext -i
import numpy as np
from distutils.core import setup
from Cython.Build import cythonize
from setuptools.extension import Extension
from os.path import join

extending = Extension("extending",
                      sources=['extending.pyx'],
                      include_dirs=[np.get_include()])
distributions = Extension("extending_distributions",
                          sources=['extending_distributions.pyx',
                                   join('..', '..', '..', 'randomgen', 'src',
                                        'distributions', 'distributions.c')],
                          include_dirs=[np.get_include()])

extensions = [extending, distributions]

setup(
    ext_modules=cythonize(extensions)
)
