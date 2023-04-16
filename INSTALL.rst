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

1) Python__ 3.9.x or newer.

   Please note that the Python development headers also need to be installed,
   e.g., on Debian/Ubuntu one needs to install both `python3` and
   `python3-dev`. On Windows and macOS this is normally not an issue.

2) Cython >= 0.29.30 but < 3.0

3) pytest__ (optional)

   This is required for testing NumPy, but not for using it.

4) Hypothesis__ (optional) 5.3.0 or later

   This is required for testing NumPy, but not for using it.

Python__ https://www.python.org/
pytest__ https://docs.pytest.org/en/stable/
Hypothesis__ https://hypothesis.readthedocs.io/en/latest/


.. note::

   If you want to build NumPy in order to work on NumPy itself, use
   ``runtests.py``.  For more details, see
   https://numpy.org/devdocs/dev/development_environment.html

.. note::

   More extensive information on building NumPy is maintained at
   https://numpy.org/devdocs/user/building.html#building-from-source


Basic Installation
==================

To install NumPy, run::

    python setup.py build -j 4 install --prefix $HOME/.local

This will compile numpy on 4 CPUs and install it into the specified prefix.
To perform an inplace build that can be run from the source folder run::

    python setup.py build_ext --inplace -j 4

See `Requirements for Installing Packages <https://packaging.python.org/tutorials/installing-packages/>`_
for more details.

The number of build jobs can also be specified via the environment variable
NPY_NUM_BUILD_JOBS.


Choosing compilers
==================

NumPy needs a C compiler, and for development versions also needs Cython.  A Fortran
compiler isn't needed to build NumPy itself; the ``numpy.f2py`` tests will be
skipped when running the test suite if no Fortran compiler is available.  For
building Scipy a Fortran compiler is needed though, so we include some details
on Fortran compilers in the rest of this section.

On OS X and Linux, all common compilers will work. The minimum supported GCC
version is 6.5.

For Fortran, ``gfortran`` works, ``g77`` does not.  In case ``g77`` is
installed then ``g77`` will be detected and used first.  To explicitly select
``gfortran`` in that case, do::

    python setup.py build --fcompiler=gnu95

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

Configuring which BLAS/LAPACK is used if you have multiple libraries installed,
or you have only one installed but in a non-standard location, is done via a
``site.cfg`` file.  See the ``site.cfg.example`` shipped with NumPy for more
details.

Windows
-------

The Intel compilers work with Intel MKL, see the application note linked above.

For an overview of the state of BLAS/LAPACK libraries on Windows, see
`here <https://mingwpy.github.io/blas_lapack.html>`_.

macOS
-----

You will need to install a BLAS/LAPACK library. We recommend using OpenBLAS or
Intel MKL. Apple's Accelerate also still works, however it has bugs and we are
likely to drop support for it in the near future.

Ubuntu/Debian
-------------

For best performance, a development package providing BLAS and CBLAS should be
installed.  Some of the options available are:

- ``libblas-dev``: reference BLAS (not very optimized)
- ``libatlas-base-dev``: generic tuned ATLAS, it is recommended to tune it to
  the available hardware, see /usr/share/doc/libatlas3-base/README.Debian for
  instructions
- ``libopenblas-base``: fast and runtime detected so no tuning required but a
  very recent version is needed (>=0.2.15 is recommended).  Older versions of
  OpenBLAS suffered from correctness issues on some CPUs.

The package linked to when numpy is loaded can be chosen after installation via
the alternatives mechanism::

    update-alternatives --config libblas.so.3
    update-alternatives --config liblapack.so.3

Or by preloading a specific BLAS library with::

    LD_PRELOAD=/usr/lib/atlas-base/atlas/libblas.so.3 python ...


Build issues
============

If you run into build issues and need help, the NumPy and SciPy
`mailing list <https://scipy.org/scipylib/mailing-lists.html>`_ is the best
place to ask. If the issue is clearly a bug in NumPy, please file an issue (or
even better, a pull request) at https://github.com/numpy/numpy.
