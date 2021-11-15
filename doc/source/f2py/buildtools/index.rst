=======================
F2PY and Build Systems
=======================

In this section we will cover the various popular build systems and their usage
with ``f2py``.

.. note::
   **As of November 2021**

   The default build system for ``F2PY`` has traditionally been the through the
   enhanced ``numpy.distutils`` module. This module is based on ``distutils`` which
   will be removed in ``Python 3.12.0`` in **October 2023**; ``setuptools`` does not
   have support for Fortran or ``F2PY`` and it is unclear if it will be supported
   in the future. Alternative methods are thus increasingly more important.


Basic Concepts
===============

Building an extension module which includes Python and Fortran consists of:

- Fortran source(s)
- One or more generated files from ``f2py``

  + A ``C`` wrapper file is always created
  + Code with modules require an additional ``.f90`` wrapper

- ``fortranobject.{c,h}``

  + Distributed with ``numpy``
  + Can be queried via ``python -c "import numpy.f2py; print(numpy.f2py.get_include())"``

- NumPy headers

  + Can be queried via ``python -c "import numpy; print(numpy.get_include())"``

- Python libraries and development headers

Broadly speaking there are two cases which arise when considering the outputs of ``f2py``:

Fortran 77 programs
   - Input file ``blah.f``
   - Generates ``blahmodule.c`` **default name**

   In this instance, only a ``C`` wrapper file is generated and only one file needs to be kept track of.

Fortran 90 programs
   - Input file ``blah.f``
   - Generates:
     + ``blahmodule.c``
     + ``blah-f2pywrappers2.f90``

   The secondary wrapper is used to handle code which is subdivided into modules.

In theory keeping the above requirements in hand, any build system can be
adapted to generate ``f2py`` extension modules. Here we will cover a subset of
the more popular systems.

Build Systems
==============

.. toctree::
   :maxdepth: 2

   distutils
   meson
   cmake
   skbuild

.. note::
   ``make`` has no place in a modern multi-language setup, and so is not
   discussed further.
