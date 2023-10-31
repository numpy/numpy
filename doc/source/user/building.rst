.. _building-from-source:

Building from source
====================

Building locally on your machine gives you complete control over build options.
If you are a MacOS or Linux user familiar with using the
command line, you can continue with building NumPy locally by following the
instructions below.

.. note:: If you want to build NumPy for development purposes, please refer to 
   :ref:`development-environment` for additional information.

..
  This page is referenced from numpy/numpy/__init__.py. Please keep its
  location in sync with the link there.

Prerequisites
-------------

Building NumPy requires the following software installed:

1) Python 3.9.x or newer

   Please note that the Python development headers also need to be installed,
   e.g., on Debian/Ubuntu one needs to install both `python3` and
   `python3-dev`. On Windows and macOS this is normally not an issue.

2) Compilers

   Much of NumPy is written in C and C++.  You will need a C compiler that
   complies with the C99 standard, and a C++ compiler that complies with the
   C++17 standard.

   While a FORTRAN 77 compiler is not necessary for building NumPy, it is
   needed to run the ``numpy.f2py`` tests. These tests are skipped if the
   compiler is not auto-detected.

   Note that NumPy is developed mainly using GNU compilers and tested on
   MSVC and Clang compilers. Compilers from other vendors such as Intel,
   Absoft, Sun, NAG, Compaq, Vast, Portland, Lahey, HP, IBM are only
   supported in the form of community feedback, and may not work out of the
   box.  GCC 6.5 (and later) compilers are recommended. On ARM64 (aarch64)
   GCC 8.x (and later) are recommended.

3) Linear Algebra libraries

   NumPy does not require any external linear algebra libraries to be
   installed. However, if these are available, NumPy's setup script can detect
   them and use them for building. A number of different LAPACK library setups
   can be used, including optimized LAPACK libraries such as OpenBLAS or MKL.
   The choice and location of these libraries as well as include paths and
   other such build options can be specified in a ``.pc`` file, as documented in
   :ref:`scipy:building-blas-and-lapack`.

4) Cython

   For building NumPy, you'll need a recent version of Cython.

5) The NumPy source code

   Clone the repository following the instructions in :doc:`/dev/index`.

.. note::

    Starting on version 1.26, NumPy will adopt Meson as its build system (see
    :ref:`distutils-status-migration` and
    :doc:`scipy:building/understanding_meson` for more details.)

Basic installation
------------------

To build and install NumPy from a local copy of the source code, run::

    pip install .

This will install all build dependencies and use Meson to compile and install
the NumPy C-extensions and Python modules. If you need more control of build
options and commands, see the following sections.

To perform an in-place build that can be run from the source folder run::

    pip install -r build_requirements.txt
    pip install -e . --no-build-isolation

*Note: for build instructions to do development work on NumPy itself, see*
:ref:`development-environment`.


Advanced building with Meson
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Meson supports the standard environment variables ``CC``, ``CXX`` and ``FC`` to
select specific C, C++ and/or Fortran compilers. These environment variables are
documented in `the reference tables in the Meson docs
<https://mesonbuild.com/Reference-tables.html#compiler-and-linker-flag-environment-variables>`_.

Note that environment variables only get applied from a clean build, because
they affect the configure stage (i.e., meson setup). An incremental rebuild does
not react to changes in environment variables - you have to run
``git clean -xdf`` and do a full rebuild, or run ``meson setup --reconfigure``.

For more options including selecting compilers, setting custom compiler flags
and controlling parallelism, see :doc:`scipy:building/compilers_and_options`
(from the SciPy documentation) and `the Meson FAQ
<https://mesonbuild.com/howtox.html#set-extra-compiler-and-linker-flags-from-the-outside-when-eg-building-distro-packages>`_.


Testing
-------

Make sure to test your builds. To ensure everything stays in shape, see if
all tests pass.

The test suite requires additional dependencies, which can easily be 
installed with::

    python -m pip install -r test_requirements.txt

Run the full test suite with::

    cd ..  # avoid picking up the source tree
    pytest --pyargs numpy

For detailed info on testing, see :ref:`testing-builds`.

.. _accelerated-blas-lapack-libraries:

Accelerated BLAS/LAPACK libraries
---------------------------------

NumPy searches for optimized linear algebra libraries such as BLAS and LAPACK.
There are specific orders for searching these libraries, as described below and
in the
`meson_options.txt <https://github.com/numpy/numpy/blob/main/meson_options.txt>`_
file.

Cross compilation
-----------------

For cross compilation instructions, see :doc:`scipy:building/cross_compilation`
and the `Meson documentation <meson>`_.

.. _meson: https://mesonbuild.com/Cross-compilation.html#cross-compilation
