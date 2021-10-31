=======================
F2PY and Build Systems
=======================

In this chapter we will cover the various popular build systems and their usage
with ``f2py``.

.. note::
   **As of November 2021**

   The default build system for ``F2PY`` has traditionally been the through the
   enhanced ``numpy.distutils`` module. This module is based on ``distutils`` which
   will be removed in ``Python 3.12.0`` in **October 2023**; ``setuptools`` does not
   have support for Fortran or ``F2PY`` and it is unclear if it will be supported
   in the future. Alternative methods are thus increasingly more important.


Build Systems
==============

.. toctree::
   :maxdepth: 2

   distutils
   meson
   cmake
   skbuild
