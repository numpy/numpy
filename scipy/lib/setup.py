#!/usr/bin/env python

from os.path import join

def configuration(parent_package='',top_path=None):
    from scipy.distutils.misc_util import Configuration
    from scipy.distutils.system_info import get_info

    config = Configuration('lib',parent_package,top_path)

    # Configure blasdot
    blas_info = get_info('blas_opt',0)
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

    return config

if __name__ == '__main__':
    from scipy.distutils.core import setup
    setup(**configuration(top_path='').todict())
