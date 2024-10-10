"""
Build an example package using the limited Python C API.
"""

import numpy as np
from setuptools import setup, Extension
import os

macros = [("Py_LIMITED_API", "0x03070000")]

limited_api = Extension(
    "limited_api",
    sources=[os.path.join('.', "limited_api.c")],
    include_dirs=[np.get_include()],
    define_macros=macros,
)

extensions = [limited_api]

setup(
    ext_modules=extensions
)
