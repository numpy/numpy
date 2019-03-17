# <img alt="NumPy" src="https://cdn.rawgit.com/numpy/numpy/master/branding/icons/numpylogo.svg" height="60">

[![Travis](https://img.shields.io/travis/numpy/numpy/master.svg?label=Travis%20CI)](
    https://travis-ci.org/numpy/numpy)
[![AppVeyor](https://img.shields.io/appveyor/ci/charris/numpy/master.svg?label=AppVeyor)](
    https://ci.appveyor.com/project/charris/numpy)
[![Azure](https://dev.azure.com/numpy/numpy/_apis/build/status/azure-pipeline%20numpy.numpy)](
    https://dev.azure.com/numpy/numpy/_build/latest?definitionId=5)
[![codecov](https://codecov.io/gh/numpy/numpy/branch/master/graph/badge.svg)](
    https://codecov.io/gh/numpy/numpy)

NumPy is the fundamental package needed for scientific computing with Python.

- **Website (including documentation):** https://www.numpy.org
- **Mailing list:** https://mail.python.org/mailman/listinfo/numpy-discussion
- **Source:** https://github.com/numpy/numpy
- **Bug reports:** https://github.com/numpy/numpy/issues
- **Contributing:** https://www.numpy.org/devdocs/dev/index.html

It provides:

- a powerful N-dimensional array object
- sophisticated (broadcasting) functions
- tools for integrating C/C++ and Fortran code
- useful linear algebra, Fourier transform, and random number capabilities

Testing:

- NumPy versions &ge; 1.15 require `pytest`
- NumPy versions &lt; 1.15 require `nose`

Tests can then be run after installation with:

    python -c 'import numpy; numpy.test()'

[![Powered by NumFOCUS](https://img.shields.io/badge/powered%20by-NumFOCUS-orange.svg?style=flat&colorA=E1523D&colorB=007D8A)](https://numfocus.org)
