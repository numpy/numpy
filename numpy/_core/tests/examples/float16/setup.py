"""
Build an example package using the limited Python C API.
"""

import os

from setuptools import Extension, setup

import numpy as np

macros = [("NPY_TARGET_VERSION", "NPY_2_0_API_VERSION")]


float16_tests = Extension(
    "_float16_tests",
    sources=[os.path.join('.', "_float16_tests.c")],
    include_dirs=[np.get_include()],
    define_macros=macros,
)

extensions = [float16_tests]

setup(
    ext_modules=extensions
)
