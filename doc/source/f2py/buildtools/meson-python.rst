.. _f2py-meson-python:

=====================================================
Distributing F2PY extensions with ``meson-python``
=====================================================

The :ref:`f2py-meson` page covers building F2PY extensions using raw ``meson``
commands. This page shows how to package those extensions into installable
Python distributions (sdists and wheels) using `meson-python
<https://meson-python.readthedocs.io/>`_ as the PEP 517 build backend.

This is the recommended approach for distributing F2PY-wrapped Fortran code as a
Python package on PyPI or for local ``pip install`` workflows.

.. note::

   ``meson-python`` replaced ``setuptools`` / ``numpy.distutils`` as the
   standard way to build and distribute compiled extensions in the NumPy and
   SciPy ecosystem. See :ref:`distutils-status-migration` for background.

Prerequisites
=============

You need:

* A Fortran compiler (``gfortran``, ``ifort``, ``ifx``, ``flang-new``, etc.)
* A C compiler
* Python >= 3.10
* ``meson``, ``meson-python``, and ``numpy`` (installed automatically during the
  build when listed in ``build-system.requires``)

Minimal example
===============

The project below wraps a Fortran ``fib`` subroutine into an importable Python
package called ``fib_wrapper``.

Project layout::

    fib_wrapper/
        fib.f90
        fib_wrapper/
            __init__.py
        meson.build
        pyproject.toml

Fortran source
--------------

Save the following as ``fib.f90``:

.. literalinclude:: ../code/fib_mesonpy.f90
   :language: fortran

``pyproject.toml``
------------------

.. literalinclude:: ../code/pyproj_mesonpy.toml
   :language: toml

The key entries are:

* ``build-backend = "mesonpy"`` tells ``pip`` to use ``meson-python``.
* ``requires`` lists build-time dependencies. ``numpy`` is required so that
  ``f2py`` and the NumPy headers are available during compilation.

``meson.build``
---------------

.. literalinclude:: ../code/meson_mesonpy.build

The ``meson.build`` file does three things:

1. Runs ``f2py`` via ``custom_target`` to generate the C wrapper sources.
2. Compiles the generated C code together with the Fortran source into a Python
   extension module using ``py.extension_module``.
3. Installs ``__init__.py`` into the package directory so the result is a proper
   Python package.

``subdir: 'fib_wrapper'`` on the extension module ensures the shared library
lands inside the ``fib_wrapper`` package directory next to ``__init__.py``.

``__init__.py``
---------------

A minimal ``__init__.py`` re-exports the wrapped function:

.. code-block:: python

   from .fib import fib

Building and installing
=======================

Editable install (development)
------------------------------

.. code-block:: bash

   pip install --no-build-isolation --editable .

``--no-build-isolation`` reuses the current environment, which is useful when
iterating. This requires ``meson-python``, ``meson``, ``ninja``, and ``numpy``
to already be installed.

Standard install
----------------

.. code-block:: bash

   pip install .

This creates an isolated build environment, installs build dependencies from
``pyproject.toml``, compiles the extension, and installs the package.

Building a wheel
----------------

.. code-block:: bash

   pip wheel . --no-deps -w dist/

Or, equivalently, using ``build``:

.. code-block:: bash

   pip install build
   python -m build

The resulting ``.whl`` file in ``dist/`` can be uploaded to PyPI or installed
elsewhere with ``pip install dist/fib_wrapper-0.1.0-*.whl``.

Verifying the install
---------------------

.. code-block:: python

   >>> from fib_wrapper import fib
   >>> fib(10)
   array([ 0,  1,  1,  2,  3,  5,  8, 13, 21, 34], dtype=int32)

Customizing the Fortran compiler
================================

``meson-python`` delegates compiler selection to ``meson``. Set the ``FC``
environment variable before building:

.. code-block:: bash

   FC=ifx pip install .

For more control, use a `Meson native file
<https://mesonbuild.com/Native-environments.html>`_:

.. code-block:: ini

   ; native.ini
   [binaries]
   fortran = 'ifx'
   c = 'icx'

.. code-block:: bash

   pip install . -Csetup-args="--native-file=native.ini"

Adding dependencies (BLAS, LAPACK, etc.)
========================================

Use ``dependency()`` in ``meson.build`` to link against system libraries:

.. code-block:: none

   lapack_dep = dependency('lapack')

   py.extension_module('mymod',
     [sources, generated],
     incdir_f2py / 'fortranobject.c',
     include_directories: inc_np,
     dependencies : [py_dep, lapack_dep],
     install : true,
   )

``meson`` resolves dependencies through ``pkg-config``, CMake, or its own
detection logic. See the `Meson dependency documentation
<https://mesonbuild.com/Dependencies.html>`_ for details.

Differences from the ``scikit-build`` workflow
==============================================

The ``scikit-build`` approach documented in :ref:`f2py-skbuild` uses CMake under
the hood and is now considered legacy for new projects in the NumPy/SciPy
ecosystem. ``meson-python`` provides:

* Native Fortran compiler support in ``meson`` (no CMake layer).
* Direct integration with ``pip`` / ``build`` via PEP 517.
* The same build system used by NumPy and SciPy themselves.

Further reading
===============

* `meson-python documentation <https://meson-python.readthedocs.io/>`_
* `Meson build system <https://mesonbuild.com/>`_
* `SciPy's meson build configuration <https://github.com/scipy/scipy/blob/main/meson.build>`_ (real-world F2PY usage)
* :ref:`f2py-meson` (raw meson build without ``meson-python``)
* :ref:`f2py-meson-distutils` (migration from ``distutils``)
