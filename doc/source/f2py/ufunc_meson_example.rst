.. _ufunc-meson-example:

===============================
Building a NumPy Ufunc with Meson
===============================

This page provides a concrete example of how to build a C-based NumPy ufunc extension using Meson.

Project layout::

  ufunc_example/
    meson.build
    your_ufunc_module.c
    __init__.py  (can be empty)

The C source file ``your_ufunc_module.c`` should implement your ufunc logic.
See ``doc/source/f2py/examples/your_ufunc_module.c`` for a minimal example.

To build a NumPy ufunc extension, you need to include the NumPy headers. You can find the include path using Python:

.. code-block:: python

   import numpy
   print(numpy.get_include())

Example ``meson.build``:

.. code-block:: meson

   project('ufunc_example', 'c')

   py = import('python').find_installation()
   numpy_include = run_command(py.full_path(), '-c', 'import numpy; print(numpy.get_include())').stdout().strip()

   sources = files('your_ufunc_module.c')

   extension_module = py.extension_module(
     'your_ufunc_module',
     sources,
     include_directories: [numpy_include],
     install: true
   )

To build and install the extension:

.. code-block:: bash

   meson setup builddir
   meson compile -C builddir
   meson install -C builddir

.. note::
   This example assumes you have a minimal C ufunc implementation. For more advanced usage, see the official NumPy documentation:
   https://numpy.org/doc/stable/reference/c-api.ufunc.html
   