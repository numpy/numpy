import numpy as np
from Cython.Build import cythonize
from setuptools import setup, find_packages, Distribution
from setuptools.extension import Extension

extensions = [Extension("core_prng.core_prng",
                        ["core_prng/core_prng.pyx"],
                        include_dirs=[np.get_include()]),
              Extension("core_prng.generator",
                        ["core_prng/generator.pyx"],
                        include_dirs=[np.get_include()])]

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
