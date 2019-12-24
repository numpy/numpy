.. _building-from-source:

Building from source
====================

A general overview of building NumPy from source is given here, with detailed
instructions for specific platforms given separately.

Prerequisites
-------------

Building NumPy requires the following software installed:

1) Python 3.5.x or newer

   Please note that the Python development headers also need to be installed,
   e.g., on Debian/Ubuntu one needs to install both `python3` and
   `python3-dev`. On Windows and macOS this is normally not an issue.

2) Compilers

   To build any extension modules for Python, you'll need a C compiler.
   Various NumPy modules use FORTRAN 77 libraries, so you'll also need a
   FORTRAN 77 compiler installed.

   Note that NumPy is developed mainly using GNU compilers. Compilers from
   other vendors such as Intel, Absoft, Sun, NAG, Compaq, Vast, Portland,
   Lahey, HP, IBM, Microsoft are only supported in the form of community
   feedback, and may not work out of the box. GCC 4.x (and later) compilers
   are recommended.

3) Linear Algebra libraries

   NumPy does not require any external linear algebra libraries to be
   installed. However, if these are available, NumPy's setup script can detect
   them and use them for building. A number of different LAPACK library setups
   can be used, including optimized LAPACK libraries such as OpenBLAS or MKL.

4) Cython

   For building NumPy, you'll need a recent version of Cython.

Basic Installation
------------------

To install NumPy, run::

    pip install .

To perform an in-place build that can be run from the source folder run::

    python setup.py build_ext --inplace

*Note: for build instructions to do development work on NumPy itself, see*
:ref:`development-environment`.

Testing
-------

Make sure to test your builds. To ensure everything stays in shape, see if all tests pass::

    $ python runtests.py -v -m full

For detailed info on testing, see :ref:`testing-builds`.

.. _parallel-builds:

Parallel builds
~~~~~~~~~~~~~~~

It's possible to do a parallel build with::

    python setup.py build -j 4 install --prefix $HOME/.local

This will compile numpy on 4 CPUs and install it into the specified prefix.
to perform a parallel in-place build, run::

    python setup.py build_ext --inplace -j 4

The number of build jobs can also be specified via the environment variable
``NPY_NUM_BUILD_JOBS``.

Choosing the fortran compiler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compilers are auto-detected; building with a particular compiler can be done
with ``--fcompiler``.  E.g. to select gfortran::

    python setup.py build --fcompiler=gnu95

For more information see::

    python setup.py build --help-fcompiler

How to check the ABI of BLAS/LAPACK libraries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One relatively simple and reliable way to check for the compiler used to build
a library is to use ldd on the library. If libg2c.so is a dependency, this
means that g77 has been used (note: g77 is no longer supported for building NumPy).
If libgfortran.so is a dependency, gfortran has been used. If both are dependencies,
this means both have been used, which is almost always a very bad idea.

Accelerated BLAS/LAPACK libraries
---------------------------------

NumPy searches for optimized linear algebra libraries such as BLAS and LAPACK.
There are specific orders for searching these libraries, as described below.

BLAS
~~~~

The default order for the libraries are:

1. MKL
2. BLIS
3. OpenBLAS
4. ATLAS
5. Accelerate (MacOS)
6. BLAS (NetLIB)

If you wish to build against OpenBLAS but you also have BLIS available one
may predefine the order of searching via the environment variable
``NPY_BLAS_ORDER`` which is a comma-separated list of the above names which
is used to determine what to search for, for instance::

      NPY_BLAS_ORDER=ATLAS,blis,openblas,MKL python setup.py build

will prefer to use ATLAS, then BLIS, then OpenBLAS and as a last resort MKL.
If neither of these exists the build will fail (names are compared
lower case).

LAPACK
~~~~~~

The default order for the libraries are:

1. MKL
2. OpenBLAS
3. libFLAME
4. ATLAS
5. Accelerate (MacOS)
6. LAPACK (NetLIB)


If you wish to build against OpenBLAS but you also have MKL available one
may predefine the order of searching via the environment variable
``NPY_LAPACK_ORDER`` which is a comma-separated list of the above names,
for instance::

      NPY_LAPACK_ORDER=ATLAS,openblas,MKL python setup.py build

will prefer to use ATLAS, then OpenBLAS and as a last resort MKL.
If neither of these exists the build will fail (names are compared
lower case).


Disabling ATLAS and other accelerated libraries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Usage of ATLAS and other accelerated libraries in NumPy can be disabled
via::

    NPY_BLAS_ORDER= NPY_LAPACK_ORDER= python setup.py build

or::

    BLAS=None LAPACK=None ATLAS=None python setup.py build


64-bit BLAS and LAPACK
~~~~~~~~~~~~~~~~~~~~~~

You can tell Numpy to use 64-bit BLAS/LAPACK libraries by setting the
environment variable::

    NPY_USE_BLAS_ILP64=1

when building Numpy. The following 64-bit BLAS/LAPACK libraries are
supported:

1. OpenBLAS ILP64 with ``64_`` symbol suffix (``openblas64_``)
2. OpenBLAS ILP64 without symbol suffix (``openblas_ilp64``)

The order in which they are preferred is determined by
``NPY_BLAS_ILP64_ORDER`` and ``NPY_LAPACK_ILP64_ORDER`` environment
variables. The default value is ``openblas64_,openblas_ilp64``.

.. note::

   Using non-symbol-suffixed 64-bit BLAS/LAPACK in a program that also
   uses 32-bit BLAS/LAPACK can cause crashes under certain conditions
   (e.g. with embedded Python interpreters on Linux).

   The 64-bit OpenBLAS with ``64_`` symbol suffix is obtained by
   compiling OpenBLAS with settings::

       make INTERFACE64=1 SYMBOLSUFFIX=64_

   The symbol suffix avoids the symbol name clashes between 32-bit and
   64-bit BLAS/LAPACK libraries.


Supplying additional compiler flags
-----------------------------------

Additional compiler flags can be supplied by setting the ``OPT``,
``FOPT`` (for Fortran), and ``CC`` environment variables.
When providing options that should improve the performance of the code ensure
that you also set ``-DNDEBUG`` so that debugging code is not executed.
