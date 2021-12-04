.. _f2py:

=====================================
F2PY user guide and reference manual
=====================================

The purpose of the ``F2PY`` --*Fortran to Python interface generator*-- utility
is to provide a connection between Python and Fortran
languages.  F2PY is a part of NumPy_ (``numpy.f2py``) and also available as a
standalone command line tool ``f2py`` when ``numpy`` is installed that
facilitates creating/building Python C/API extension modules that make it
possible

* to call Fortran 77/90/95 external subroutines and Fortran 90/95
  module subroutines as well as C functions;
* to access Fortran 77 ``COMMON`` blocks and Fortran 90/95 module data,
  including allocatable arrays

from Python.

.. toctree::
   :maxdepth: 2

   usage
   f2py.getting-started
   python-usage
   signature-file
   buildtools/index
   advanced

.. _Python: https://www.python.org/
.. _NumPy: https://www.numpy.org/
