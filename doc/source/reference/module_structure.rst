.. _module-structure:

************************
NumPy's module structure
************************

NumPy has a large number of submodules. Most regular usage of NumPy requires
only the main namespace and a smaller set of submodules. The rest either either
special-purpose or niche namespaces.

Main namespaces
===============

Regular/recommended user-facing namespaces for general use:

.. Note: there is no single doc page that covers all of the main namespace as
   of now. It's hard to create without introducing duplicate references. For
   now, just link to the "Routines and objects by topic" page.

- :ref:`numpy <routines>`
- :ref:`numpy.exceptions <routines.exceptions>`
- :ref:`numpy.fft <routines.fft>`
- :ref:`numpy.linalg <routines.linalg>`
- :ref:`numpy.polynomial <numpy-polynomial>`
- :ref:`numpy.random <numpyrandom>`
- :ref:`numpy.strings <routines.strings>`
- :ref:`numpy.testing <routines.testing>`
- :ref:`numpy.typing <typing>`

Special-purpose namespaces
==========================

- :ref:`numpy.ctypeslib <routines.ctypeslib>` - interacting with NumPy objects with `ctypes`
- :ref:`numpy.dtypes <routines.dtypes>` - dtype classes (typically not used directly by end users)
- :ref:`numpy.emath <routines.emath>` - mathematical functions with automatic domain
- :ref:`numpy.lib <routines.lib>` - utilities & functionality which do not fit the main namespace
- :ref:`numpy.rec <routines.rec>` - record arrays (largely superseded by dataframe libraries)
- :ref:`numpy.version <routines.version>` - small module with more detailed version info

Legacy namespaces
=================

Prefer not to use these namespaces for new code. There are better alternatives
and/or this code is deprecated or isn't reliable.

- :ref:`numpy.char <routines.char>` - legacy string functionality, only for fixed-width strings
- :ref:`numpy.distutils <numpy-distutils-refguide>` (deprecated) - build system support
- :ref:`numpy.f2py <python-module-numpy.f2py>` - Fortran binding generation (usually used from the command line only)
- :ref:`numpy.ma <routines.ma>` - masked arrays (not very reliable, needs an overhaul)
- :ref:`numpy.matlib <routines.matlib>` (pending deprecation) - functions supporting ``matrix`` instances


.. This will appear in the left sidebar on this page. Keep in sync with the contents above!

.. toctree::
   :hidden:

   numpy.exceptions <routines.exceptions>
   numpy.fft <routines.fft>
   numpy.linalg <routines.linalg>
   numpy.polynomial <routines.polynomials-package>
   numpy.random <random/index>
   numpy.strings <routines.strings>
   numpy.testing <routines.testing>
   numpy.typing <typing>
   numpy.ctypeslib <routines.ctypeslib>
   numpy.dtypes <routines.dtypes>
   numpy.emath <routines.emath>
   numpy.lib <routines.lib>
   numpy.rec <routines.rec>
   numpy.version <routines.version>
   numpy.char <routines.char>
   numpy.distutils <distutils>
   numpy.f2py <../f2py/index>
   numpy.ma <routines.ma>
   numpy.matlib <routines.matlib>
