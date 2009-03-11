Configuration defines
=====================

.. sectionauthor:: Travis E. Oliphant

When NumPy is built, a configuration file is constructed and placed as config.h
in the NumPy include directory. This configuration file ensures that specific
macros are defined and defines other macros based on whether or not your system
has certain features. This file is private, and is not exported by numpy (that
is a python extension which use the numpy C API will not see those symbols), to
avoid namespace pollution.

Some of those defines have a public equivalent, which are defined in
numpyconfig.h (included in ndarrayobject.h). The public symbols are prefixed by
NPY_*.

Guaranteed to be defined
------------------------

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
