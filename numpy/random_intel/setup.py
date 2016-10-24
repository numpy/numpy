from __future__ import division, print_function

from os.path import join, split, dirname
import sys
from distutils.msvccompiler import get_build_version as get_msvc_build_version


def needs_mingw_ftime_workaround():
    # We need the mingw workaround for _ftime if the msvc runtime version is
    # 7.1 or above and we build with mingw ...
    # ... but we can't easily detect compiler version outside distutils command
    # context, so we will need to detect in randomkit whether we build with gcc
    msver = get_msvc_build_version()
    if msver and msver >= 8:
        return True

    return False


def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration, get_mathlibs
    from numpy.distutils.system_info import get_info
    config = Configuration('random_intel', parent_package, top_path)

    if not get_info('mkl'):
        return config

    def generate_libraries(ext, build_dir):
        libs = get_mathlibs()
        if sys.platform == 'win32':
            libs.append('Advapi32')
        ext.libraries.extend(libs)
        return None

    # enable unix large file support on 32 bit systems
    # (64 bit off_t, lseek -> lseek64 etc.)
    defs = [('_FILE_OFFSET_BITS', '64'),
            ('_LARGEFILE_SOURCE', '1'),
            ('_LARGEFILE64_SOURCE', '1')]
    if needs_mingw_ftime_workaround():
        defs.append(("NPY_NEEDS_MINGW_TIME_WORKAROUND", None))

    libs = ['mkl_rt', 'mkl_dists']
    # Configure mklrand
    Q = '/Q' if sys.platform.startswith('win') or sys.platform == 'cygwin' else '-'
    config.add_library('mkl_dists',
                         sources=join('mklrand', 'mkl_distributions.cpp'),
                         libraries=libs,
                         extra_compiler_args=[Q + 'std=c++11'],
                         depends=[join('mklrand', '*.h'),],
                         define_macros=defs,
                         )

    config.add_extension('mklrand',
                         sources=[join('mklrand', x) for x in
                                  ['mklrand.c', 'randomkit.c']]+[generate_libraries],
                         libraries=libs,
                         depends=[join('mklrand', '*.h'),
                                  join('mklrand', '*.pyx'),
                                  join('mklrand', '*.pxi'),],
                         define_macros=defs,
                         )

    config.add_data_files(('.', join('mklrand', 'randomkit.h')))
    config.add_data_files(('.', join('mklrand', 'mkl_distributions.h')))
    config.add_data_dir('tests')

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(configuration=configuration)
