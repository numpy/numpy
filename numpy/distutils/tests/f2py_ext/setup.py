
import os
from numpy.distutils.core import setup, Extension

ext = Extension('f2py_ext.fib2',['src/fib2.pyf','src/fib1.f'])

setup(
    name = 'f2py_ext',
    ext_modules = [ext],
    packages = ['f2py_ext.tests','f2py_ext'],
    package_dir = {'f2py_ext':'.'})

