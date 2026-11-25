"""
Build an example package using the Half-precision API.
"""

import os

import numpy as np

setup(
    ext_modules=Extension(
    "_float16_tests",
    sources=[os.path.join('.', "_float16_tests.c")],
    include_dirs=[np.get_include()],
    )
)
