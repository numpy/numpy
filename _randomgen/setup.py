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

try:
    import pypandoc
    # With an input file: it will infer the input format from the filename
    with open('README.rst', 'w') as readme:
        readme.write(pypandoc.convert_file('README.md', 'rst'))
except ImportError:
    import warnings
    warnings.warn(
        'Unable to import pypandoc.  Do not use this as a release build!')

import versioneer

with open('requirements.txt') as f:
    required = f.read().splitlines()

Cython.Compiler.Options.annotate = True

USE_SSE2 = True if not '--no-sse2' in sys.argv else False
MOD_DIR = './randomgen'

DEBUG = False
PCG_EMULATED_MATH = False

EXTRA_INCLUDE_DIRS = []
EXTRA_LINK_ARGS = []
# Undef for manylinux
EXTRA_COMPILE_ARGS = [] if os.name == 'nt' else ['-std=c99', '-U__GNUC_GNU_INLINE__']
if os.name == 'nt':
    EXTRA_LINK_ARGS = ['/LTCG', '/OPT:REF', 'Advapi32.lib', 'Kernel32.lib']
    if DEBUG:
        EXTRA_LINK_ARGS += ['-debug']
        EXTRA_COMPILE_ARGS += ["-Zi", "/Od"]
    if sys.version_info < (3, 0):
        EXTRA_INCLUDE_DIRS += [join(MOD_DIR, 'src', 'common')]
PCG64_DEFS = []
if sys.maxsize < 2 ** 32 or os.name == 'nt':
    # Force emulated mode here
    PCG_EMULATED_MATH = True
    PCG64_DEFS += [('PCG_FORCE_EMULATED_128BIT_MATH', '1')]

DSFMT_DEFS = [('DSFMT_MEXP', '19937')]
if USE_SSE2:
    if os.name == 'nt':
        EXTRA_COMPILE_ARGS += ['/wd4146', '/GL']
        if struct.calcsize('P') < 8:
            EXTRA_COMPILE_ARGS += ['/arch:SSE2']
    else:
        EXTRA_COMPILE_ARGS += ['-msse2']
    DSFMT_DEFS += [('HAVE_SSE2', '1')]
if struct.calcsize('P') < 8:
    PCG_EMULATED_MATH = True

files = glob.glob('./randomgen/*.in')
for templated_file in files:
    output_file_name = os.path.splitext(templated_file)[0]
    if (os.path.exists(output_file_name) and
            (os.path.getmtime(templated_file) < os.path.getmtime(output_file_name))):
        continue
    with open(templated_file, 'r') as source_file:
        template = tempita.Template(source_file.read())
    with open(output_file_name, 'w') as output_file:
        output_file.write(template.substitute())


extensions = [Extension('randomgen.entropy',
                        sources=[join(MOD_DIR, 'entropy.pyx'),
                                 join(MOD_DIR, 'src', 'entropy', 'entropy.c')],
                        include_dirs=EXTRA_INCLUDE_DIRS + [np.get_include(),
                                                           join(MOD_DIR, 'src',
                                                                'entropy')],
                        extra_compile_args=EXTRA_COMPILE_ARGS,
                        extra_link_args=EXTRA_LINK_ARGS
                        ),
              Extension("randomgen.dsfmt",
                        ["randomgen/dsfmt.pyx",
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
              Extension("randomgen.mt19937",
                        ["randomgen/mt19937.pyx",
                         join(MOD_DIR, 'src', 'mt19937', 'mt19937.c'),
                         join(MOD_DIR, 'src', 'mt19937', 'mt19937-jump.c')],
                        include_dirs=EXTRA_INCLUDE_DIRS + [np.get_include(),
                                                           join(MOD_DIR, 'src',
                                                                'mt19937')],
                        extra_compile_args=EXTRA_COMPILE_ARGS,
                        extra_link_args=EXTRA_LINK_ARGS
                        ),
              Extension("randomgen.philox",
                        ["randomgen/philox.pyx",
                         join(MOD_DIR, 'src', 'philox', 'philox.c')],
                        include_dirs=EXTRA_INCLUDE_DIRS + [np.get_include(),
                                                           join(MOD_DIR, 'src',
                                                                'philox')],
                        extra_compile_args=EXTRA_COMPILE_ARGS,
                        extra_link_args=EXTRA_LINK_ARGS
                        ),
              Extension("randomgen.pcg64",
                        ["randomgen/pcg64.pyx",
                         join(MOD_DIR, 'src', 'pcg64',  'pcg64.c')],
                        include_dirs=EXTRA_INCLUDE_DIRS + [np.get_include(),
                                                           join(MOD_DIR, 'src',
                                                                'pcg64')],
                        extra_compile_args=EXTRA_COMPILE_ARGS,
                        define_macros=PCG64_DEFS,
                        extra_link_args=EXTRA_LINK_ARGS
                        ),
              Extension("randomgen.pcg32",
                        ["randomgen/pcg32.pyx",
                         join(MOD_DIR, 'src', 'pcg32', 'pcg32.c')],
                        include_dirs=EXTRA_INCLUDE_DIRS + [np.get_include(),
                                                           join(MOD_DIR, 'src',
                                                                'pcg32')],
                        extra_compile_args=EXTRA_COMPILE_ARGS,
                        extra_link_args=EXTRA_LINK_ARGS
                        ),
              Extension("randomgen.threefry",
                        ["randomgen/threefry.pyx",
                         join(MOD_DIR, 'src', 'threefry', 'threefry.c')],
                        include_dirs=EXTRA_INCLUDE_DIRS + [np.get_include(),
                                                           join(MOD_DIR, 'src',
                                                                'threefry')],
                        extra_compile_args=EXTRA_COMPILE_ARGS,
                        extra_link_args=EXTRA_LINK_ARGS
                        ),
              Extension("randomgen.threefry32",
                        ["randomgen/threefry32.pyx",
                         join(MOD_DIR, 'src', 'threefry32', 'threefry32.c')],
                        include_dirs=EXTRA_INCLUDE_DIRS + [np.get_include(),
                                                           join(MOD_DIR, 'src',
                                                                'threefry32')],
                        extra_compile_args=EXTRA_COMPILE_ARGS,
                        extra_link_args=EXTRA_LINK_ARGS
                        ),
              Extension("randomgen.xoroshiro128",
                        ["randomgen/xoroshiro128.pyx",
                         join(MOD_DIR, 'src', 'xoroshiro128',
                              'xoroshiro128.c')],
                        include_dirs=EXTRA_INCLUDE_DIRS + [np.get_include(),
                                                           join(
                                                               MOD_DIR, 'src',
                                                               'xoroshiro128')],
                        extra_compile_args=EXTRA_COMPILE_ARGS,
                        extra_link_args=EXTRA_LINK_ARGS
                        ),
              Extension("randomgen.xorshift1024",
                        ["randomgen/xorshift1024.pyx",
                         join(MOD_DIR, 'src', 'xorshift1024',
                              'xorshift1024.c')],
                        include_dirs=EXTRA_INCLUDE_DIRS + [np.get_include(),
                                                           join(MOD_DIR, 'src',
                                                                'xorshift1024')],
                        extra_compile_args=EXTRA_COMPILE_ARGS,
                        extra_link_args=EXTRA_LINK_ARGS
                        ),
              Extension("randomgen.generator",
                        ["randomgen/generator.pyx",
                         join(MOD_DIR, 'src', 'distributions',
                              'distributions.c')],
                        include_dirs=EXTRA_INCLUDE_DIRS + [np.get_include()],
                        extra_compile_args=EXTRA_COMPILE_ARGS,
                        extra_link_args=EXTRA_LINK_ARGS
                        ),
              Extension("randomgen.common",
                        ["randomgen/common.pyx"],
                        include_dirs=EXTRA_INCLUDE_DIRS + [np.get_include()],
                        extra_compile_args=EXTRA_COMPILE_ARGS,
                        extra_link_args=EXTRA_LINK_ARGS
                        ),
              Extension("randomgen.bounded_integers",
                        ["randomgen/bounded_integers.pyx",
                         join(MOD_DIR, 'src', 'distributions',
                              'distributions.c')],
                        include_dirs=EXTRA_INCLUDE_DIRS + [np.get_include()],
                        extra_compile_args=EXTRA_COMPILE_ARGS,
                        extra_link_args=EXTRA_LINK_ARGS
                        ),
              ]


classifiers = ['Development Status :: 5 - Production/Stable',
               'Environment :: Console',
               'Intended Audience :: End Users/Desktop',
               'Intended Audience :: Financial and Insurance Industry',
               'Intended Audience :: Information Technology',
               'Intended Audience :: Science/Research',
               'License :: OSI Approved',
               'Operating System :: MacOS :: MacOS X',
               'Operating System :: Microsoft :: Windows',
               'Operating System :: POSIX :: Linux',
               'Operating System :: Unix',
               'Programming Language :: C',
               'Programming Language :: Cython',
               'Programming Language :: Python :: 2.7',
               'Programming Language :: Python :: 3.3',
               'Programming Language :: Python :: 3.4',
               'Programming Language :: Python :: 3.5',
               'Programming Language :: Python :: 3.6',
               'Topic :: Adaptive Technologies',
               'Topic :: Artistic Software',
               'Topic :: Office/Business :: Financial',
               'Topic :: Scientific/Engineering',
               'Topic :: Security :: Cryptography']


class BinaryDistribution(Distribution):
    def is_pure(self):
        return False


setup(
    name='randomgen',
    version=versioneer.get_version(),
    classifiers=classifiers,
    cmdclass=versioneer.get_cmdclass(),
    ext_modules=cythonize(extensions, compile_time_env={
        "PCG_EMULATED_MATH": PCG_EMULATED_MATH}),
    packages=find_packages(),
    package_dir={'randomgen': './randomgen'},
    package_data={'': ['*.h', '*.pxi', '*.pyx', '*.pxd', '*.in'],
                  'randomgen.tests.data': ['*.csv']},
    include_package_data=True,
    license='NSCA',
    author='Kevin Sheppard',
    author_email='kevin.k.sheppard@gmail.com',
    distclass=BinaryDistribution,
    long_description=open('README.rst').read(),
    description='Random generator supporting multiple PRNGs',
    url='https://github.com/bashtage/randomgen',
    keywords=['pseudo random numbers', 'PRNG', 'RNG', 'RandomState', 'random',
              'random numbers', 'parallel random numbers', 'PCG',
              'XorShift', 'dSFMT', 'MT19937', 'Random123', 'ThreeFry',
              'Philox'],
    zip_safe=False,
    install_requires=required
)
