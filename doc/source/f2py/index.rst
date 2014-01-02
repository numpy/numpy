.. -*- rest -*-

//////////////////////////////////////////////////////////////////////
                  F2PY Users Guide and Reference Manual
//////////////////////////////////////////////////////////////////////

:Author: Pearu Peterson
:Contact: pearu@cens.ioc.ee
:Web site: http://cens.ioc.ee/projects/f2py2e/
:Date: 2005/04/02 10:03:26

================
 Introduction
================

The purpose of the F2PY_ --*Fortran to Python interface generator*--
project is to provide a connection between Python and Fortran
languages.  F2PY is a Python_ package (with a command line tool
``f2py`` and a module ``f2py2e``) that facilitates creating/building
Python C/API extension modules that make it possible

* to call Fortran 77/90/95 external subroutines and Fortran 90/95
  module subroutines as well as C functions;
* to access Fortran 77 ``COMMON`` blocks and Fortran 90/95 module data,
  including allocatable arrays

from Python. See F2PY_ web site for more information and installation
instructions.

.. toctree::
   :maxdepth: 2

   getting-started
   signature-file
   python-usage
   usage
   distutils
   advanced

.. _F2PY: http://cens.ioc.ee/projects/f2py2e/
.. _Python: http://www.python.org/
.. _NumPy: http://www.numpy.org/
.. _SciPy: http://www.numpy.org/
