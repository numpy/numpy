.. _c-api.limited-api:

=================================================
Using the NumPy C-API with the Python limited API
=================================================

.. versionadded:: 2.0

Starting from NumPy 2.0, it is possible to build C extension modules that use
the NumPy C-API while targeting the Python limited API. This allows you to
build a single binary that works across multiple Python versions supported by
the chosen ``Py_LIMITED_API`` value (the stable CPython ABI). NumPy
compatibility is still determined separately by the NumPy C-API used by the
extension and the NumPy version available at runtime.

Currently, only the ``abi3`` target is supported. The ``abi3t`` (freethreading)
target is under active development in CPython but is not yet supported for use
with NumPy.

What is the Limited API?
========================

The Python limited API is a subset of the CPython C-API that is guaranteed to be
stable across Python versions. By defining ``Py_LIMITED_API``, you restrict
your extension to this stable subset. The resulting shared library (often with
an ``.abi3.so`` or ``.abi3.pyd`` suffix) can be loaded by any version of
CPython equal to or newer than the version it was compiled for, without
recompilation.

This ABI stability guarantee applies to CPython itself. It does not imply that
the NumPy C-API is stable across NumPy releases, so extensions must still be
built and tested against NumPy versions they intend to support.

*Opaque* means the C compiler can see that the type exists but cannot see its
internal layout; you cannot read or write its fields directly, only call
functions that accept a pointer to it.

Key Constraints
===============

The main constraint when using the limited API is that **direct access to
Python object internals is forbidden**. Many structures that are "fully defined"
in the full C-API become "opaque" in the limited API.

For NumPy extension authors, this means:

*   You cannot directly access fields of :c:type:`PyArrayObject`,
    ``PyDescrObject``, or other NumPy structs.
*   You must use the provided accessor functions (e.g., :c:func:`PyArray_DATA`,
    :c:func:`PyArray_NDIM`, :c:func:`PyArray_STRIDES`) instead of direct struct
    access.
*   In regular builds, macros such as :c:func:`PyArray_DATA` may expand to
    direct struct field access. When ``Py_LIMITED_API`` is defined, NumPy
    redirects these macros to function-call forms.
*   Code that relies on struct layout details, including
    ``sizeof(PyArrayObject)``, will not work with the limited API.

Using NumPy Headers with ``Py_LIMITED_API``
===========================================

To use the NumPy C-API in limited-API mode, you must define ``Py_LIMITED_API``
before including any Python or NumPy headers.

.. code-block:: c

    #define Py_LIMITED_API 0x030a0000 /* Target Python 3.10+ */
    #include <Python.h>
    #define PY_ARRAY_UNIQUE_SYMBOL MY_EXTENSION_ARRAY_API
    #include <numpy/arrayobject.h>

    /* In your module init function */
    PyMODINIT_FUNC PyInit_my_module(void) {
        import_array();  /* Still required under limited API */
        /* ... */
    }

.. note::

   NumPy exposes most of its C-API through a runtime API table. Calling
   :c:func:`import_array` initializes that table for the current process before
   the C-API is used. This requirement is unchanged when building against the
   Python limited API.

NumPy's headers detect ``Py_LIMITED_API`` at compile time and conditionally
redefine macros such as :c:func:`PyArray_DATA`, :c:func:`PyArray_NDIM`, and
:c:func:`PyArray_STRIDES` to use function-call forms rather than direct struct
field access.

The value assigned to ``Py_LIMITED_API`` (e.g., ``0x030a0000``) represents the
*minimum Python version* your extension targets, not a NumPy-imposed
requirement.

Supported and Unsupported Patterns Under Limited API
----------------------------------------------------

*   **Supported**: Function calls like :c:func:`PyArray_SimpleNew`,
    :c:func:`PyArray_FROM_OTF`, and ufunc creation.
*   **Supported**: Macros like :c:func:`PyArray_DATA`, :c:func:`PyArray_NDIM`,
    :c:func:`PyArray_STRIDES`, and :c:func:`PyArray_TYPE`.
*   **Unsupported**: Casting a ``PyObject *`` to ``PyArrayObject *`` and accessing
    fields like ``->data`` or ``->nd`` directly. This will fail to compile
    with an "incomplete type" error.
*   **Unsupported**: Static initialization of types that inherit from NumPy types
    using C struct literals.
*   **Unsupported**: Subclassing NumPy array types using C-API static types.

Build Examples
==============

Meson
-----

When building with Meson 1.1 or later, you can use the ``limited_api`` keyword
argument.

.. code-block:: meson

    project('my-extension', 'c')
    py = import('python').find_installation()
    dep_py = py.dependency(embed: false)
    # Requires NumPy 2.0+
    dep_numpy = dependency('numpy')

    py.extension_module('my_module',
      'my_module.c',
      dependencies: [dep_py, dep_numpy],
      limited_api: '3.10',   # Sets Py_LIMITED_API and abi3 suffix
      install: true
    )

Setuptools
----------

For ``setuptools`` discovery, you can specify the limited API in ``pyproject.toml``.

.. code-block:: toml

    [tool.setuptools]
    # ...

    [[tool.setuptools.ext-modules]]
    name = "my_module"
    sources = ["my_module.c"]
    define-macros = [["Py_LIMITED_API", "0x030a0000"]]
    py-limited-api = true

Caveats
=======

*   **Version Requirements**: The NumPy limited API support requires NumPy 2.0
    or later at compile time.
*   **Deprecated API**: Extension modules may also define
    ``NPY_NO_DEPRECATED_API`` to reduce reliance on older NumPy C-API entry
    points.
*   **Cython Support**: As of the time of writing, Cython support for the
    limited API combined with NumPy is still evolving.
*   **Performance**: Accessing array properties through function calls instead
    of direct struct access may have a small performance overhead in
    tight loops.
*   **Opaque Structs**: Note that while :c:type:`PyArrayObject` is opaque, the
    underlying data pointer returned by :c:func:`PyArray_DATA` is still a
    raw pointer to the array's memory.
*   **Free-threaded support**: ``abi3t`` support is under active development in
    CPython but is not yet available with NumPy. Track progress in `NumPy issue
    #26157 <https://github.com/numpy/numpy/issues/26157>`_.

Testing and Verification
========================

To verify that your module is correctly built as an ``abi3`` wheel, you can check
the shared library suffix and metadata. On Linux/macOS, the ``file`` command
can help ensure it's not pinned to a specific Python version:

.. code-block:: bash

    file my_module.abi3.so
    # Should show: ELF shared object (or Mach-O), not mentioning a specific python version
