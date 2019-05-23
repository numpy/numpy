.. _building-from-source:

Building from source
====================

A general overview of building NumPy from source is given here, with detailed
instructions for specific platforms given separately.

Prerequisites
-------------

Building NumPy requires the following software installed:

1) Python 2.7.x, 3.4.x or newer

   On Debian and derivatives (Ubuntu): python, python-dev (or python3-dev)

   On Windows: the official python installer at
   `www.python.org <https://www.python.org>`_ is enough

   Make sure that the Python package distutils is installed before
   continuing. For example, in Debian GNU/Linux, installing python-dev
   also installs distutils.

   Python must also be compiled with the zlib module enabled. This is
   practically always the case with pre-packaged Pythons.

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
   can be used, including optimized LAPACK libraries such as ATLAS, MKL or the
   Accelerate/vecLib framework on OS X.

4) Cython

   To build development versions of NumPy, you'll need a recent version of
   Cython.  Released NumPy sources on PyPi include the C files generated from
   Cython code, so for released versions having Cython installed isn't needed.

Basic Installation
------------------

To install NumPy run::

    python setup.py install

To perform an in-place build that can be run from the source folder run::

    python setup.py build_ext --inplace

The NumPy build system uses ``setuptools`` (from numpy 1.11.0, before that it
was plain ``distutils``) and ``numpy.distutils``.
Using ``virtualenv`` should work as expected.

*Note: for build instructions to do development work on NumPy itself, see*
:ref:`development-environment`.

.. _parallel-builds:

Parallel builds
~~~~~~~~~~~~~~~

From NumPy 1.10.0 on it's also possible to do a parallel build with::

    python setup.py build -j 4 install --prefix $HOME/.local

This will compile numpy on 4 CPUs and install it into the specified prefix.
to perform a parallel in-place build, run::

    python setup.py build_ext --inplace -j 4

The number of build jobs can also be specified via the environment variable
``NPY_NUM_BUILD_JOBS``.


FORTRAN ABI mismatch
--------------------

The two most popular open source fortran compilers are g77 and gfortran.
Unfortunately, they are not ABI compatible, which means that concretely you
should avoid mixing libraries built with one with another. In particular, if
your blas/lapack/atlas is built with g77, you *must* use g77 when building
numpy and scipy; on the contrary, if your atlas is built with gfortran, you
*must* build numpy/scipy with gfortran. This applies for most other cases
where different FORTRAN compilers might have been used.

Choosing the fortran compiler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To build with gfortran::

    python setup.py build --fcompiler=gnu95

For more information see::

    python setup.py build --help-fcompiler

How to check the ABI of blas/lapack/atlas
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One relatively simple and reliable way to check for the compiler used to build
a library is to use ldd on the library. If libg2c.so is a dependency, this
means that g77 has been used. If libgfortran.so is a dependency, gfortran
has been used. If both are dependencies, this means both have been used, which
is almost always a very bad idea.

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


Supplying additional compiler flags
-----------------------------------

Additional compiler flags can be supplied by setting the ``OPT``,
``FOPT`` (for Fortran), and ``CC`` environment variables.
When providing options that should improve the performance of the code ensure
that you also set ``-DNDEBUG`` so that debugging code is not executed.


Building with ATLAS support
---------------------------

Ubuntu
~~~~~~

You can install the necessary package for optimized ATLAS with this command::

    sudo apt-get install libatlas-base-dev
