import os
from os.path import join

import numpy as np
from Cython.Build import cythonize
from setuptools import setup, find_packages, Distribution
from setuptools.extension import Extension

MOD_DIR = './core_prng'

EXTRA_LINK_ARGS = []
if os.name == 'nt':
    EXTRA_LINK_ARGS = ['/LTCG', '/OPT:REF', 'Advapi32.lib', 'Kernel32.lib']

extensions = [Extension('core_prng.entropy',
                        sources=[join(MOD_DIR, 'entropy.pyx'),
                                 join(MOD_DIR, 'src', 'entropy', 'entropy.c')],
                        include_dirs=[np.get_include(),
                                      join(MOD_DIR, 'src', 'entropy')],
                        extra_link_args=EXTRA_LINK_ARGS),
              Extension("core_prng.splitmix64",
                        ["core_prng/splitmix64.pyx",
                         join(MOD_DIR, 'src', 'splitmix64', 'splitmix64.c')],
                        include_dirs=[np.get_include(),
                                      join(MOD_DIR, 'src', 'splitmix64')]),
              Extension("core_prng.xoroshiro128",
                        ["core_prng/xoroshiro128.pyx",
                         join(MOD_DIR, 'src', 'xoroshiro128',
                              'xoroshiro128.c')],
                        include_dirs=[np.get_include(),
                                      join(MOD_DIR, 'src', 'xoroshiro128')]),
              Extension("core_prng.generator",
                        ["core_prng/generator.pyx"],
                        include_dirs=[np.get_include()]),
              Extension("core_prng.common",
                        ["core_prng/common.pyx"],
                        include_dirs=[np.get_include()]),
              ]


class BinaryDistribution(Distribution):
    def is_pure(self):
        return False


setup(
    ext_modules=cythonize(extensions),
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
