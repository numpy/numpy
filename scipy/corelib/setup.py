#!/usr/bin/env python
import os
from os.path import join
from scipy.distutils.core      import setup
from scipy.distutils.misc_util import Configuration, allpath
from scipy.distutils.system_info import get_info

def configuration(parent_package='',top_path=None):
    config = Configuration('lib',parent_package,top_path)
    local_dir = config.local_path

    # Configure blasdot
    blas_info = get_info('blas_opt')
    #blas_info = 0
    if blas_info:
        config.add_extension('_dotblas',
                             sources=[join('blasdot','_dotblas.c')],
                             **blas_info
                             )

    # Configure fftpack_lite
    config.add_extension('fftpack_lite',
                         sources=[join('fftpack_lite', x) for x in \
                                  ['fftpack_litemodule.c', 'fftpack.c']]
                         )

    config.add_extension('mtrand',
                         sources=[join('mtrand', x) for x in 
                                  ['mtrand.c', 'randomkit.c', 'initarray.c',
                                   'distributions.c', 'gamma.c', 'polevl.c', 
                                   'isnan.c', 'const.c']],
                         libraries=['m'],
                        )

    # Configure lapack_lite
    lapack_info = get_info('lapack_opt')
    #lapack_info = 0
    if not lapack_info:
        # use C-sources provided
        print "### Warning:  Using unoptimized lapack ###"
        config.add_extension('lapack_lite',
                             sources=[join('lapack_lite', x) for x in \
                                      ['lapack_litemodule.c',
                                       'zlapack_lite.c', 'dlapack_lite.c',
                                       'blas_lite.c', 'dlamch.c',
                                       'f2c_lite.c']]
                             )
        
    else:
        config.add_extension('lapack_lite',
                             sources=[join('lapack_lite',
                                           'lapack_litemodule.c')],
                             **lapack_info)        
    
    return config.todict()

if __name__ == '__main__':
    setup(**configuration(top_path=''))
