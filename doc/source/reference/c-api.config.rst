System configuration
====================

.. sectionauthor:: Travis E. Oliphant

When NumPy is built, information about system configuration is
recorded, and is made available for extension modules using Numpy's C
API.  These are mostly defined in ``numpyconfig.h`` (included in
``ndarrayobject.h``). The public symbols are prefixed by ``NPY_*``.
Numpy also offers some functions for querying information about the
platform in use.

For private use, Numpy also constructs a ``config.h`` in the NumPy
include directory, which is not exported by Numpy (that is a python
extension which use the numpy C API will not see those symbols), to
avoid namespace pollution.


Data type sizes
---------------

The :cdata:`NPY_SIZEOF_{CTYPE}` constants are defined so that sizeof
information is available to the pre-processor.

.. cvar:: NPY_SIZEOF_SHORT

    sizeof(short)

.. cvar:: NPY_SIZEOF_INT

    sizeof(int)

.. cvar:: NPY_SIZEOF_LONG

    sizeof(long)

.. cvar:: NPY_SIZEOF_LONG_LONG

    sizeof(longlong) where longlong is defined appropriately on the
    platform (A macro defines **NPY_SIZEOF_LONGLONG** as well.)

.. cvar:: NPY_SIZEOF_PY_LONG_LONG


.. cvar:: NPY_SIZEOF_FLOAT

    sizeof(float)

.. cvar:: NPY_SIZEOF_DOUBLE

    sizeof(double)

.. cvar:: NPY_SIZEOF_LONG_DOUBLE

    sizeof(longdouble) (A macro defines **NPY_SIZEOF_LONGDOUBLE** as well.)

.. cvar:: NPY_SIZEOF_PY_INTPTR_T

    Size of a pointer on this platform (sizeof(void \*)) (A macro defines
    NPY_SIZEOF_INTP as well.)


Platform information
--------------------

.. cvar:: NPY_CPU_X86
.. cvar:: NPY_CPU_AMD64
.. cvar:: NPY_CPU_IA64
.. cvar:: NPY_CPU_PPC
.. cvar:: NPY_CPU_PPC64
.. cvar:: NPY_CPU_SPARC
.. cvar:: NPY_CPU_SPARC64
.. cvar:: NPY_CPU_S390
.. cvar:: NPY_CPU_PARISC

    .. versionadded:: 1.3.0

    CPU architecture of the platform; only one of the above is
    defined.

    Defined in ``numpy/npy_cpu.h``

.. cvar:: NPY_LITTLE_ENDIAN

.. cvar:: NPY_BIG_ENDIAN

.. cvar:: NPY_BYTE_ORDER

    .. versionadded:: 1.3.0

    Portable alternatives to the ``endian.h`` macros of GNU Libc.
    If big endian, :cdata:`NPY_BYTE_ORDER` == :cdata:`NPY_BIG_ENDIAN`, and
    similarly for little endian architectures.

    Defined in ``numpy/npy_endian.h``.

.. cfunction:: PyArray_GetEndianness()

    .. versionadded:: 1.3.0

    Returns the endianness of the current platform.
    One of :cdata:`NPY_CPU_BIG`, :cdata:`NPY_CPU_LITTLE`,
    or :cdata:`NPY_CPU_UNKNOWN_ENDIAN`.
