
import os
from numpy_distutils.core import setup, Extension

package = 'f2py_f90_ext'

ext = Extension(package+'.foo',['src/foo_free.f90'],
                include_dirs=['include'],
                f2py_options=['--include_paths','include'])

setup(
    name = package,
    ext_modules = [ext],
    packages = [package+'.tests',package],
    package_dir = {package:'.'})

