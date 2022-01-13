"""
Provide python-space access to the functions exposed in numpy/__init__.pxd
for testing.
"""

import numpy as np
from distutils.core import setup
from Cython.Build import cythonize
from setuptools.extension import Extension
import os

include_dirs = [np.get_include()]
macros = [("NPY_NO_DEPRECATED_API", 0)]

checks = Extension(
    "checks",
    sources=[os.path.join('.', "checks.pyx")],
    include_dirs=include_dirs,
    define_macros=macros,
)

example_limited_api = Extension(
    "example_limited_api",
    sources=[os.path.join('.', "example_limited_api.c")],
    include_dirs=include_dirs,
    define_macros=macros,
)

extensions = [checks, example_limited_api]

setup(
    ext_modules=cythonize(extensions)
)
