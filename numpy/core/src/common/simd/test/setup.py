import os
import sys
from os.path import join, abspath

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('_simd', parent_package, top_path)
    dir_path = os.path.dirname(os.path.realpath(__file__))

    config.add_extension('_intrinsics',
        sources=[
            join('..', '..', 'npy_cpu_features.c'),
            join('src', 'module.cpp'),
            join('src', 'intrinsics.dispatch.cpp'),
            join('src', 'datatypes.dispatch.cpp')
        ],
        depends=[
            join('..', '..', 'npy_cpu_dispatch.h'),
            join('..', 'simd', 'simd.h'),
            join('..', 'simd', 'simd.hpp'),
            join('..', 'simd', 'forward.inc.hpp'),
            join('..', 'simd', 'wrapper', 'wrapper.hpp'),
            join('..', 'simd', 'wrapper', 'datatypes.hpp'),
            join('src', 'module.hpp'),
            join('src', 'intrinsics.hpp'),
            join('src', 'datatypes.hpp')
        ]
    )
    config.add_subpackage('tests')
    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(configuration=configuration)
