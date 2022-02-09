.. _building-from-source:

Building from source
====================

There are two options for building NumPy- building with Gitpod or locally from
source. Your choice depends on your operating system and familiarity with the
command line.

Gitpod
------------

Gitpod is an open-source platform that automatically creates
the correct development environment right in your browser, reducing the need to
install local development environments and deal with incompatible dependencies.

If you are a Windows user, unfamiliar with using the command line or building
NumPy for the first time, it is often faster to build with Gitpod. Here are the
in-depth instructions for building NumPy with `building NumPy with Gitpod`_.

.. _building NumPy with Gitpod: https://numpy.org/devdocs/dev/development_gitpod.html

Building locally
------------------

Building locally on your machine gives you
more granular control. If you are a MacOS or Linux user familiar with using the
command line, you can continue with building NumPy locally by following the
instructions below.

..
  This page is referenced from numpy/numpy/__init__.py. Please keep its
  location in sync with the link there.

Prerequisites
-------------

Building NumPy requires the following software installed:

1) Python 3.8.x or newer

   Please note that the Python development headers also need to be installed,
   e.g., on Debian/Ubuntu one needs to install both `python3` and
   `python3-dev`. On Windows and macOS this is normally not an issue.

2) Compilers

   Much of NumPy is written in C.  You will need a C compiler that complies
   with the C99 standard.

   While a FORTRAN 77 compiler is not necessary for building NumPy, it is
   needed to run the ``numpy.f2py`` tests. These tests are skipped if the
   compiler is not auto-detected.

   Note that NumPy is developed mainly using GNU compilers and tested on
   MSVC and Clang compilers. Compilers from other vendors such as Intel,
   Absoft, Sun, NAG, Compaq, Vast, Portland, Lahey, HP, IBM are only
   supported in the form of community feedback, and may not work out of the
   box.  GCC 4.x (and later) compilers are recommended. On ARM64 (aarch64)
   GCC 8.x (and later) are recommended.

3) Linear Algebra libraries

   NumPy does not require any external linear algebra libraries to be
   installed. However, if these are available, NumPy's setup script can detect
   them and use them for building. A number of different LAPACK library setups
   can be used, including optimized LAPACK libraries such as OpenBLAS or MKL.
   The choice and location of these libraries as well as include paths and
   other such build options can be specified in a ``site.cfg`` file located in
   the NumPy root repository or a ``.numpy-site.cfg`` file in your home
   directory. See the ``site.cfg.example`` example file included in the NumPy
   repository or sdist for documentation, and below for specifying search
   priority from environmental variables.

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

Make sure to test your builds. To ensure everything stays in shape, see if
all tests pass::

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

One relatively simple and reliable way to check for the compiler used to
build a library is to use ldd on the library. If libg2c.so is a dependency,
this means that g77 has been used (note: g77 is no longer supported for
building NumPy). If libgfortran.so is a dependency, gfortran has been used.
If both are dependencies, this means both have been used, which is almost
always a very bad idea.

.. _accelerated-blas-lapack-libraries:

Accelerated BLAS/LAPACK libraries
---------------------------------

NumPy searches for optimized linear algebra libraries such as BLAS and LAPACK.
There are specific orders for searching these libraries, as described below and
in the ``site.cfg.example`` file.

BLAS
~~~~

Note that both BLAS and CBLAS interfaces are needed for a properly
optimized build of NumPy.

The default order for the libraries are:

1. MKL
2. BLIS
3. OpenBLAS
4. ATLAS
5. BLAS (NetLIB)

The detection of BLAS libraries may be bypassed by defining the environment
variable ``NPY_BLAS_LIBS`` , which should contain the exact linker flags you
want to use (interface is assumed to be Fortran 77).  Also define
``NPY_CBLAS_LIBS`` (even empty if CBLAS is contained in your BLAS library) to
trigger use of CBLAS and avoid slow fallback code for matrix calculations.

If you wish to build against OpenBLAS but you also have BLIS available one
may predefine the order of searching via the environment variable
``NPY_BLAS_ORDER`` which is a comma-separated list of the above names which
is used to determine what to search for, for instance::

      NPY_BLAS_ORDER=ATLAS,blis,openblas,MKL python setup.py build

will prefer to use ATLAS, then BLIS, then OpenBLAS and as a last resort MKL.
If neither of these exists the build will fail (names are compared
lower case).

Alternatively one may use ``!`` or ``^`` to negate all items::

        NPY_BLAS_ORDER='^blas,atlas' python setup.py build

will allow using anything **but** NetLIB BLAS and ATLAS libraries, the order
of the above list is retained.

One cannot mix negation and positives, nor have multiple negations, such
cases will raise an error.

LAPACK
~~~~~~

The default order for the libraries are:

1. MKL
2. OpenBLAS
3. libFLAME
4. ATLAS
5. LAPACK (NetLIB)

The detection of LAPACK libraries may be bypassed by defining the environment
variable ``NPY_LAPACK_LIBS``, which should contain the exact linker flags you
want to use (language is assumed to be Fortran 77).

If you wish to build against OpenBLAS but you also have MKL available one
may predefine the order of searching via the environment variable
``NPY_LAPACK_ORDER`` which is a comma-separated list of the above names,
for instance::

      NPY_LAPACK_ORDER=ATLAS,openblas,MKL python setup.py build

will prefer to use ATLAS, then OpenBLAS and as a last resort MKL.
If neither of these exists the build will fail (names are compared
lower case).

Alternatively one may use ``!`` or ``^`` to negate all items::

        NPY_LAPACK_ORDER='^lapack' python setup.py build

will allow using anything **but** the NetLIB LAPACK library, the order of
the above list is retained.

One cannot mix negation and positives, nor have multiple negations, such
cases will raise an error.

.. deprecated:: 1.20
  The native libraries on macOS, provided by Accelerate, are not fit for use
  in NumPy since they have bugs that cause wrong output under easily
  reproducible conditions. If the vendor fixes those bugs, the library could
  be reinstated, but until then users compiling for themselves should use
  another linear algebra library or use the built-in (but slower) default,
  see the next section.


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
When providing options that should improve the performance of the code
ensure that you also set ``-DNDEBUG`` so that debugging code is not
executed.

Cross compilation
-----------------

Although ``numpy.distutils`` and ``setuptools`` do not directly support cross
compilation, it is possible to build NumPy on one system for different
architectures with minor modifications to the build environment. This may be
desirable, for example, to use the power of a high-performance desktop to
create a NumPy package for a low-power, single-board computer. Because the
``setup.py`` scripts are unaware of cross-compilation environments and tend to
make decisions based on the environment detected on the build system, it is
best to compile for the same type of operating system that runs on the builder.
Attempting to compile a Mac version of NumPy on Windows, for example, is likely
to be met with challenges not considered here.

For the purpose of this discussion, the nomenclature adopted by `meson`_ will
be used: the "build" system is that which will be running the NumPy build
process, while the "host" is the platform on which the compiled package will be
run. A native Python interpreter, the setuptools and Cython packages and the
desired cross compiler must be available for the build system. In addition, a
Python interpreter and its development headers as well as any external linear
algebra libraries must be available for the host platform. For convenience, it
is assumed that all host software is available under a separate prefix
directory, here called ``$CROSS_PREFIX``.

.. _meson: https://mesonbuild.com/Cross-compilation.html#cross-compilation

When building and installing NumPy for a host system, the ``CC`` environment
variable must provide the path the cross compiler that will be used to build
NumPy C extensions. It may also be necessary to set the ``LDSHARED``
environment variable to the path to the linker that can link compiled objects
for the host system. The compiler must be told where it can find Python
libraries and development headers. On Unix-like systems, this generally
requires adding, *e.g.*, the following parameters to the ``CFLAGS`` environment
variable::

    -I${CROSS_PREFIX}/usr/include
    -I${CROSS_PREFIX}/usr/include/python3.y

for Python version 3.y. (Replace the "y" in this path with the actual minor
number of the installed Python runtime.) Likewise, the linker should be told
where to find host libraries by adding a parameter to the ``LDFLAGS``
environment variable::

    -L${CROSS_PREFIX}/usr/lib

To make sure Python-specific system configuration options are provided for the
intended host and not the build system, set::

    _PYTHON_SYSCONFIGDATA_NAME=_sysconfigdata_${ARCH_TRIPLET}

where ``${ARCH_TRIPLET}`` is an architecture-dependent suffix appropriate for
the host architecture. (This should be the name of a ``_sysconfigdata`` file,
without the ``.py`` extension, found in the host Python library directory.)

When using external linear algebra libraries, include and library directories
should be provided for the desired libraries in ``site.cfg`` as described
above and in the comments of the ``site.cfg.example`` file included in the
NumPy repository or sdist. In this example, set::

    include_dirs = ${CROSS_PREFIX}/usr/include
    library_dirs = ${CROSS_PREFIX}/usr/lib

under appropriate sections of the file to allow ``numpy.distutils`` to find the
libraries.

As of NumPy 1.22.0, a vendored copy of SVML will be built on ``x86_64`` Linux
hosts to provide AVX-512 acceleration of floating-point operations. When using
an ``x86_64`` Linux build system to cross compile NumPy for hosts other than
``x86_64`` Linux, set the environment variable ``NPY_DISABLE_SVML`` to prevent
the NumPy build script from incorrectly attempting to cross-compile this
platform-specific library::

    NPY_DISABLE_SVML=1

With the environment configured, NumPy may be built as it is natively::

    python setup.py build

When the ``wheel`` package is available, the cross-compiled package may be
packed into a wheel for installation on the host with::

    python setup.py bdist_wheel

It may be possible to use ``pip`` to build a wheel, but ``pip`` configures its
own environment; adapting the ``pip`` environment to cross-compilation is
beyond the scope of this guide.

The cross-compiled package may also be installed into the host prefix for
cross-compilation of other packages using, *e.g.*, the command::

    python setup.py install --prefix=${CROSS_PREFIX}

When cross compiling other packages that depend on NumPy, the host
npy-pkg-config file must be made available. For further discussion, refer to
`numpy distutils documentation`_.

.. _numpy distutils documentation: https://numpy.org/devdocs/reference/distutils.html#numpy.distutils.misc_util.Configuration.add_npy_pkg_config
