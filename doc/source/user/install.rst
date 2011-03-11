*****************************
Building and installing NumPy
*****************************

Binary installers
=================

In most use cases the best way to install NumPy on your system is by using an
installable binary package for your operating system.

Windows
-------

Good solutions for Windows are, The Enthought Python Distribution `(EPD)
<http://www.enthought.com/products/epd.php>`_ (which provides binary
installers for Windows, OS X and Redhat) and `Python (x, y)
<http://www.pythonxy.com>`_. Both of these packages include Python, NumPy and
many additional packages.

A lightweight alternative is to download the Python
installer from `www.python.org <http://www.python.org>`_ and the NumPy
installer for your Python version from the Sourceforge `download site <http://
sourceforge.net/project/showfiles.php?group_id=1369&package_id=175103>`_

The NumPy installer includes binaries for different CPU's (without SSE
instructions, with SSE2 or with SSE3) and installs the correct one
automatically. If needed, this can be bypassed from the command line with ::

  numpy-<1.y.z>-superpack-win32.exe /arch nosse

or 'sse2' or 'sse3' instead of 'nosse'.

Linux
-----

Most of the major distributions provide packages for NumPy, but these can lag
behind the most recent NumPy release. Pre-built binary packages for Ubuntu are
available on the `scipy ppa
<https://edge.launchpad.net/~scipy/+archive/ppa>`_. Redhat binaries are
available in the `EPD <http://www.enthought.com/products/epd.php>`_.

Mac OS X
--------

A universal binary installer for NumPy is available from the `download site
<http://sourceforge.net/project/showfiles.php?group_id=1369&
package_id=175103>`_. The `EPD <http://www.enthought.com/products/epd.php>`_
provides NumPy binaries.

Building from source
====================

A general overview of building NumPy from source is given here, with detailed
instructions for specific platforms given seperately.

Prerequisites
-------------

Building NumPy requires the following software installed:

1) Python 2.4.x, 2.5.x or 2.6.x

   On Debian and derivative (Ubuntu): python, python-dev

   On Windows: the official python installer at
   `www.python.org <http://www.python.org>`_ is enough

   Make sure that the Python package distutils is installed before
   continuing. For example, in Debian GNU/Linux, distutils is included
   in the python-dev package.

   Python must also be compiled with the zlib module enabled.

2) Compilers

   To build any extension modules for Python, you'll need a C compiler.
   Various NumPy modules use FORTRAN 77 libraries, so you'll also need a
   FORTRAN 77 compiler installed.

   Note that NumPy is developed mainly using GNU compilers. Compilers from
   other vendors such as Intel, Absoft, Sun, NAG, Compaq, Vast, Porland,
   Lahey, HP, IBM, Microsoft are only supported in the form of community
   feedback, and may not work out of the box. GCC 3.x (and later) compilers
   are recommended.

3) Linear Algebra libraries

   NumPy does not require any external linear algebra libraries to be
   installed. However, if these are available, NumPy's setup script can detect
   them and use them for building. A number of different LAPACK library setups
   can be used, including optimized LAPACK libraries such as ATLAS, MKL or the
   Accelerate/vecLib framework on OS X.

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

To build with g77::

    python setup.py build --fcompiler=gnu

To build with gfortran::

    python setup.py build --fcompiler=gnu95

For more information see::

    python setup.py build --help-fcompiler

How to check the ABI of blas/lapack/atlas
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One relatively simple and reliable way to check for the compiler used to build
a library is to use ldd on the library. If libg2c.so is a dependency, this
means that g77 has been used. If libgfortran.so is a a dependency, gfortran
has been used. If both are dependencies, this means both have been used, which
is almost always a very bad idea.

Disabling ATLAS and other accelerated libraries
-----------------------------------------------

Usage of ATLAS and other accelerated libraries in Numpy can be disabled
via::

    BLAS=None LAPACK=None ATLAS=None python setup.py build


Supplying additional compiler flags
-----------------------------------

Additional compiler flags can be supplied by setting the ``OPT``,
``FOPT`` (for Fortran), and ``CC`` environment variables.


Building with ATLAS support
---------------------------

Ubuntu 8.10 (Intrepid) and 9.04 (Jaunty)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can install the necessary packages for optimized ATLAS with this command::

    sudo apt-get install libatlas-base-dev

If you have a recent CPU with SIMD suppport (SSE, SSE2, etc...), you should
also install the corresponding package for optimal performances. For example,
for SSE2::

    sudo apt-get install libatlas3gf-sse2

This package is not available on amd64 platforms.

*NOTE*: Ubuntu changed its default fortran compiler from g77 in Hardy to
gfortran in Intrepid. If you are building ATLAS from source and are upgrading
from Hardy to Intrepid or later versions, you should rebuild everything from
scratch, including lapack.

Ubuntu 8.04 and lower
~~~~~~~~~~~~~~~~~~~~~

You can install the necessary packages for optimized ATLAS with this command::

    sudo apt-get install atlas3-base-dev

If you have a recent CPU with SIMD suppport (SSE, SSE2, etc...), you should
also install the corresponding package for optimal performances. For example,
for SSE2::

    sudo apt-get install atlas3-sse2
