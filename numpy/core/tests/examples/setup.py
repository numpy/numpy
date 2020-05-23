"""
Provide python-space access to the functions exposed in numpy/__init__.pxd
for testing.
"""

import numpy as np
from distutils.core import setup
from Cython.Build import cythonize
from setuptools.extension import Extension
import os

here = os.path.dirname(__file__)
macros = [("NPY_NO_DEPRECATED_API", 0)]

checks = Extension(
    "checks",
    sources=[os.path.join(here, "checks.pyx")],
    include_dirs=[np.get_include()],
    define_macros=macros,
)

extensions = [checks]

setup(
    ext_modules=cythonize(extensions)
)
