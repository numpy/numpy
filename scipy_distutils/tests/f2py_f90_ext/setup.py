
import os
from scipy_distutils.core import setup, Extension

package = 'f2py_f90_ext'

ext = Extension(package+'.foo',['src/foo_free.f90'])

setup(
    name = package,
    ext_modules = [ext],
    packages = [package+'.tests',package],
    package_dir = {package:'.'})

