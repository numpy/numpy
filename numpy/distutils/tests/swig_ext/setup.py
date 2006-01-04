
import os
from numpy_distutils.core import setup, Extension

ext_c = Extension('swig_ext._example',['src/example.i','src/example.c'])
ext_cpp = Extension('swig_ext._example2',['src/zoo.i','src/zoo.cc'],
                    depends=['src/zoo.h'],include_dirs=['src'])

setup(
    name = 'swig_ext',
    ext_modules = [ext_c,ext_cpp],
    packages = ['swig_ext.tests','swig_ext'],
    package_dir = {'swig_ext':'.'})

