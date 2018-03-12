import os
import glob
import struct
import sys
from os.path import join

import Cython.Compiler.Options
import numpy as np
from Cython.Build import cythonize
from setuptools import setup, find_packages, Distribution
from setuptools.extension import Extension

try:
    import Cython.Tempita as tempita
except ImportError:
    try:
        import tempita
    except ImportError:
        raise ImportError('tempita required to install, '
                          'use pip install tempita')

import versioneer

Cython.Compiler.Options.annotate = True

USE_SSE2 = True if not '--no-sse2' in sys.argv else False
MOD_DIR = './core_prng'

DEBUG = False
PCG_EMULATED_MATH = False

EXTRA_INCLUDE_DIRS = []
EXTRA_LINK_ARGS = []
EXTRA_COMPILE_ARGS = [] if os.name == 'nt' else ['-std=c99']
if os.name == 'nt':
    EXTRA_LINK_ARGS = ['/LTCG', '/OPT:REF', 'Advapi32.lib', 'Kernel32.lib']
    PCG_EMULATED_MATH = True
    if DEBUG:
        EXTRA_LINK_ARGS += ['-debug']
        EXTRA_COMPILE_ARGS += ["-Zi", "/Od"]
    if sys.version_info < (3, 0):
        EXTRA_INCLUDE_DIRS += [join(MOD_DIR, 'src', 'common')]

DSFMT_DEFS = [('DSFMT_MEXP', '19937')]
if USE_SSE2:
    if os.name == 'nt':
        EXTRA_COMPILE_ARGS += ['/wd4146', '/GL']
        if struct.calcsize('P') < 8:
            EXTRA_COMPILE_ARGS += ['/arch:SSE2']
    else:
        EXTRA_COMPILE_ARGS += ['-msse2']
    DSFMT_DEFS += [('HAVE_SSE2', '1')]

files = glob.glob('./core_prng/*.in')
for templated_file in files:
    output_file_name = os.path.splitext(templated_file)[0]
    if (os.path.exists(output_file_name) and
            (os.path.getmtime(templated_file) < os.path.getmtime(output_file_name))):
        continue
    with open(templated_file, 'r') as source_file:
        template = tempita.Template(source_file.read())
    with open(output_file_name, 'w') as output_file:
        output_file.write(template.substitute())


extensions = [Extension('core_prng.entropy',
                        sources=[join(MOD_DIR, 'entropy.pyx'),
                                 join(MOD_DIR, 'src', 'entropy', 'entropy.c')],
                        include_dirs=EXTRA_INCLUDE_DIRS + [np.get_include(),
                                                           join(MOD_DIR, 'src',
                                                                'entropy')],
                        extra_compile_args=EXTRA_COMPILE_ARGS,
                        extra_link_args=EXTRA_LINK_ARGS
                        ),
              Extension("core_prng.dsfmt",
                        ["core_prng/dsfmt.pyx",
                         join(MOD_DIR, 'src', 'dsfmt', 'dSFMT.c'),
                         join(MOD_DIR, 'src', 'dsfmt', 'dSFMT-jump.c'),
                         join(MOD_DIR, 'src', 'aligned_malloc', 'aligned_malloc.c')],
                        include_dirs=EXTRA_INCLUDE_DIRS + [np.get_include(),
                                                           join(MOD_DIR, 'src',
                                                                'dsfmt')],
                        extra_compile_args=EXTRA_COMPILE_ARGS,
                        extra_link_args=EXTRA_LINK_ARGS,
                        define_macros=DSFMT_DEFS,
                        ),
              Extension("core_prng.mt19937",
                        ["core_prng/mt19937.pyx",
                         join(MOD_DIR, 'src', 'mt19937', 'mt19937.c'),
                         join(MOD_DIR, 'src', 'mt19937', 'mt19937-jump.c')],
                        include_dirs=EXTRA_INCLUDE_DIRS + [np.get_include(),
                                                           join(MOD_DIR, 'src',
                                                                'mt19937')],
                        extra_compile_args=EXTRA_COMPILE_ARGS,
                        extra_link_args=EXTRA_LINK_ARGS
                        ),
              Extension("core_prng.philox",
                        ["core_prng/philox.pyx",
                         join(MOD_DIR, 'src', 'philox', 'philox.c')],
                        include_dirs=EXTRA_INCLUDE_DIRS + [np.get_include(),
                                                           join(MOD_DIR, 'src',
                                                                'philox')],
                        extra_compile_args=EXTRA_COMPILE_ARGS,
                        extra_link_args=EXTRA_LINK_ARGS
                        ),
              Extension("core_prng.pcg64",
                        ["core_prng/pcg64.pyx",
                         join(MOD_DIR, 'src', 'pcg64',
                              'pcg64.c')],
                        include_dirs=EXTRA_INCLUDE_DIRS + [np.get_include(),
                                                           join(MOD_DIR, 'src',
                                                                'pcg64')],
                        extra_compile_args=EXTRA_COMPILE_ARGS,
                        extra_link_args=EXTRA_LINK_ARGS
                        ),
              Extension("core_prng.threefry",
                        ["core_prng/threefry.pyx",
                         join(MOD_DIR, 'src', 'threefry', 'threefry.c')],
                        include_dirs=EXTRA_INCLUDE_DIRS + [np.get_include(),
                                                           join(MOD_DIR, 'src',
                                                                'threefry')],
                        extra_compile_args=EXTRA_COMPILE_ARGS,
                        extra_link_args=EXTRA_LINK_ARGS
                        ),
              Extension("core_prng.threefry32",
                        ["core_prng/threefry32.pyx",
                         join(MOD_DIR, 'src', 'threefry32', 'threefry32.c')],
                        include_dirs=EXTRA_INCLUDE_DIRS + [np.get_include(),
                                                           join(MOD_DIR, 'src',
                                                                'threefry32')],
                        extra_compile_args=EXTRA_COMPILE_ARGS,
                        extra_link_args=EXTRA_LINK_ARGS
                        ),
              Extension("core_prng.xoroshiro128",
                        ["core_prng/xoroshiro128.pyx",
                         join(MOD_DIR, 'src', 'xoroshiro128',
                              'xoroshiro128.c')],
                        include_dirs=EXTRA_INCLUDE_DIRS + [np.get_include(),
                                                           join(
                                                               MOD_DIR, 'src',
                                                               'xoroshiro128')],
                        extra_compile_args=EXTRA_COMPILE_ARGS,
                        extra_link_args=EXTRA_LINK_ARGS
                        ),
              Extension("core_prng.xorshift1024",
                        ["core_prng/xorshift1024.pyx",
                         join(MOD_DIR, 'src', 'xorshift1024',
                              'xorshift1024.c')],
                        include_dirs=EXTRA_INCLUDE_DIRS + [np.get_include(),
                                                           join(MOD_DIR, 'src',
                                                                'xorshift1024')],
                        extra_compile_args=EXTRA_COMPILE_ARGS,
                        extra_link_args=EXTRA_LINK_ARGS
                        ),
              Extension("core_prng.generator",
                        ["core_prng/generator.pyx",
                         join(MOD_DIR, 'src', 'distributions',
                              'distributions.c')],
                        include_dirs=EXTRA_INCLUDE_DIRS + [np.get_include()],
                        extra_compile_args=EXTRA_COMPILE_ARGS,
                        extra_link_args=EXTRA_LINK_ARGS
                        ),
              Extension("core_prng.common",
                        ["core_prng/common.pyx"],
                        include_dirs=EXTRA_INCLUDE_DIRS + [np.get_include()],
                        extra_compile_args=EXTRA_COMPILE_ARGS,
                        extra_link_args=EXTRA_LINK_ARGS
                        ),
              Extension("core_prng.bounded_integers",
                        ["core_prng/bounded_integers.pyx",
                         join(MOD_DIR, 'src', 'distributions',
                              'distributions.c')],
                        include_dirs=EXTRA_INCLUDE_DIRS + [np.get_include()],
                        extra_compile_args=EXTRA_COMPILE_ARGS,
                        extra_link_args=EXTRA_LINK_ARGS
                        ),
              ]


class BinaryDistribution(Distribution):
    def is_pure(self):
        return False


setup(
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    ext_modules=cythonize(extensions, compile_time_env={
        "PCG_EMULATED_MATH": PCG_EMULATED_MATH}),
    name='core_prng',
    packages=find_packages(),
    package_dir={'core_prng': './core_prng'},
    package_data={'': ['*.c', '*.h', '*.pxi', '*.pyx', '*.pxd']},
    include_package_data=True,
    license='NSCA',
    author='Kevin Sheppard',
    author_email='kevin.k.sheppard@gmail.com',
    distclass=BinaryDistribution,
    description='Next-gen RandomState supporting multiple PRNGs',
    url='https://github.com/bashtage/core-prng',
    keywords=['pseudo random numbers', 'PRNG', 'Python'],
    zip_safe=False
)
