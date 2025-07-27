Building and installing NumPy
+++++++++++++++++++++++++++++

**IMPORTANT**: the below notes are about building NumPy, which for most users
is *not* the recommended way to install NumPy.  Instead, use either a complete
scientific Python distribution (recommended) or a binary installer - see
https://scipy.org/install.html.


.. Contents::

Prerequisites
=============

Building NumPy requires the following installed software:

1) Python__ 3.11.x or newer.

   Please note that the Python development headers also need to be installed,
   e.g., on Debian/Ubuntu one needs to install both `python3` and
   `python3-dev`. On Windows and macOS this is normally not an issue.

2) Cython >= 3.0.6

3) pytest__ (optional)

   This is required for testing NumPy, but not for using it.

4) Hypothesis__ (optional) 5.3.0 or later

   This is required for testing NumPy, but not for using it.

Python__ https://www.python.org/
pytest__ https://docs.pytest.org/en/stable/
Hypothesis__ https://hypothesis.readthedocs.io/en/latest/


.. note::

   If you want to build NumPy in order to work on NumPy itself, use
   ``spin``.  For more details, see
   https://numpy.org/devdocs/dev/development_environment.html

.. note::

   More extensive information on building NumPy is maintained at
   https://numpy.org/devdocs/building/#building-numpy-from-source


Basic installation
==================

If this is a clone of the NumPy git repository, then first initialize
the ``git`` submodules::

    git submodule update --init

To install NumPy, run::

    pip install .

This will compile NumPy on all available CPUs and install it into the active
environment.

To run the build from the source folder for development purposes, use the
``spin`` development CLI::

    spin build    # installs in-tree under `build-install/`
    spin ipython  # drop into an interpreter where `import numpy` picks up the local build

Alternatively, use an editable install with::

    pip install -e . --no-build-isolation

See `Requirements for Installing Packages <https://packaging.python.org/tutorials/installing-packages/>`_
for more details.


Choosing compilers
==================

NumPy needs C and C++ compilers, and for development versions also needs
Cython.  A Fortran compiler isn't needed to build NumPy itself; the
``numpy.f2py`` tests will be skipped when running the test suite if no Fortran
compiler is available. 

For more options including selecting compilers, setting custom compiler flags
and controlling parallelism, see
https://scipy.github.io/devdocs/building/compilers_and_options.html

Windows
-------

On Windows, building from source can be difficult (in particular if you need to
build SciPy as well, because that requires a Fortran compiler). Currently, the
most robust option is to use MSVC (for NumPy only). If you also need SciPy,
you can either use MSVC + Intel Fortran or the Intel compiler suite.
Intel itself maintains a good `application note
<https://software.intel.com/en-us/articles/numpyscipy-with-intel-mkl>`_
on this.

If you want to use a free compiler toolchain, our current recommendation is to
use Docker or Windows subsystem for Linux (WSL).  See
https://scipy.github.io/devdocs/dev/contributor/contributor_toc.html#development-environment
for more details.


Building with optimized BLAS support
====================================

Configuring which BLAS/LAPACK is used if you have multiple libraries installed
is done via a ``--config-settings`` CLI flag - if not given, the default choice
is OpenBLAS. If your installed library is in a non-standard location, selecting
that location is done via a pkg-config ``.pc`` file.
See https://scipy.github.io/devdocs/building/blas_lapack.html for more details.

Windows
-------

The Intel compilers work with Intel MKL, see the application note linked above.

For an overview of the state of BLAS/LAPACK libraries on Windows, see
`here <https://mingwpy.github.io/blas_lapack.html>`_.

macOS
-----

On macOS >= 13.3, you can use Apple's Accelerate library. On older macOS versions,
Accelerate has bugs and we recommend using OpenBLAS or (on x86-64) Intel MKL.

Ubuntu/Debian
-------------

For best performance, a development package providing BLAS and CBLAS should be
installed.  Some of the options available are:

- ``libblas-dev``: reference BLAS (not very optimized)
- ``libopenblas-base``: (recommended) OpenBLAS is performant, and used
  in the NumPy wheels on PyPI except where Apple's Accelerate is tuned better for Apple hardware

The package linked to when numpy is loaded can be chosen after installation via
the alternatives mechanism::

    update-alternatives --config libblas.so.3
    update-alternatives --config liblapack.so.3


Build issues
============

If you run into build issues and need help, the NumPy and SciPy
`mailing list <https://scipy.org/scipylib/mailing-lists.html>`_ is the best
place to ask. If the issue is clearly a bug in NumPy, please file an issue (or
even better, a pull request) at https://github.com/numpy/numpy.
