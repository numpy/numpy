#!/usr/bin/env python
import os
from os.path import join
from scipy.distutils.core      import setup
from scipy.distutils.misc_util import Configuration, allpath
from scipy.distutils.system_info import get_info
import sys

def configuration(parent_package='',top_path=None):
    config = Configuration('lib',parent_package,top_path)
    local_dir = config.local_path

    # Configure blasdot
    blas_info = get_info('blas_opt')
    #blas_info = {}
    def get_dotblas_sources(ext, build_dir):
        if blas_info:
            return ext.depends[:1]
        return None # no extension module will be built

    config.add_extension('_dotblas',
                         sources = [get_dotblas_sources],
                         depends=[join('blasdot','_dotblas.c'),
                                  join('blasdot','cblas.h'),
                                  ],
                         include_dirs = ['blasdot'],
                         extra_info = blas_info
                         )

    # Configure fftpack_lite
    config.add_extension('fftpack_lite',
                         sources=[join('fftpack_lite', x) for x in \
                                  ['fftpack_litemodule.c', 'fftpack.c']]
                         )

    # Configure mtrand
    config.add_extension('mtrand',
                         sources=[join('mtrand', x) for x in 
                                  ['mtrand.c', 'randomkit.c', 'initarray.c',
                                   'distributions.c']],
                         libraries=['m'],
                         depends = [join('mtrand','*.h'),
                                    join('mtrand','*.pyx'),
                                    join('mtrand','*.pxi'),
                                    ]
                        )

    # Configure lapack_lite
    lapack_info = get_info('lapack_opt')
    def get_lapack_lite_sources(ext, build_dir):
        if not lapack_info:
            print "### Warning:  Using unoptimized lapack ###"
            return ext.depends
        else:
            return ext.depends[:1]

    config.add_extension('lapack_lite',
                         sources = [get_lapack_lite_sources],
                         depends=[join('lapack_lite', x) for x in \
                                  ['lapack_litemodule.c',
                                   'zlapack_lite.c', 'dlapack_lite.c',
                                   'blas_lite.c', 'dlamch.c',
                                   'f2c_lite.c']],
                         extra_info = lapack_info
                         )
            
    return config.todict()

if __name__ == '__main__':
    setup(**configuration(top_path=''))
