.. currentmodule:: numpy.random

.. _extending:

Extending
---------
The BitGenerators have been designed to be extendable using standard tools for
high-performance Python -- numba and Cython.  The `~Generator` object can also
be used with user-provided BitGenerators as long as these export a small set of
required functions.

Numba
=====
Numba can be used with either CTypes or CFFI.  The current iteration of the
BitGenerators all export a small set of functions through both interfaces.

This example shows how numba can be used to produce gaussian samples using
a pure Python implementation which is then compiled.  The random numbers are
provided by ``ctypes.next_double``.

.. literalinclude:: ../../../../numpy/random/_examples/numba/extending.py
    :language: python
    :end-before: example 2

Both CTypes and CFFI allow the more complicated distributions to be used
directly in Numba after compiling the file distributions.c into a ``DLL`` or
``so``.  An example showing the use of a more complicated distribution is in
the `examples` section below.

.. _random_cython:

Cython
======

Cython can be used to unpack the ``PyCapsule`` provided by a BitGenerator.
This example uses `PCG64` and the example from above.  The usual caveats
for writing high-performance code using Cython -- removing bounds checks and
wrap around, providing array alignment information -- still apply.

.. literalinclude:: ../../../../numpy/random/_examples/cython/extending_distributions.pyx
    :language: cython
    :end-before: example 2

The BitGenerator can also be directly accessed using the members of the basic
RNG structure.

.. literalinclude:: ../../../../numpy/random/_examples/cython/extending_distributions.pyx
    :language: cython
    :start-after: example 2

See :ref:`extending_cython_example` for a complete working example including a
minimal setup and cython files.

CFFI
====

CFFI can be used to directly access the functions in
``include/numpy/random/distributions.h``. Some "massaging" of the header
file is required:

.. literalinclude:: ../../../../numpy/random/_examples/cffi/extending.py
    :language: python
    :end-before: dlopen

Once the header is parsed by ``ffi.cdef``, the functions can be accessed
directly from the ``_generator`` shared object, using the `BitGenerator.cffi` interface.

.. literalinclude:: ../../../../numpy/random/_examples/cffi/extending.py
    :language: python
    :start-after: dlopen


New Basic RNGs
==============
`~Generator` can be used with other user-provided BitGenerators. The simplest
way to write a new BitGenerator is to examine the pyx file of one of the
existing BitGenerators. The key structure that must be provided is the
``capsule`` which contains a ``PyCapsule`` to a struct pointer of type
``bitgen_t``,

.. code-block:: c

  typedef struct bitgen {
    void *state;
    uint64_t (*next_uint64)(void *st);
    uint32_t (*next_uint32)(void *st);
    double (*next_double)(void *st);
    uint64_t (*next_raw)(void *st);
  } bitgen_t;

which provides 5 pointers. The first is an opaque pointer to the data structure
used by the BitGenerators.  The next three are function pointers which return
the next 64- and 32-bit unsigned integers, the next random double and the next
raw value.  This final function is used for testing and so can be set to
the next 64-bit unsigned integer function if not needed. Functions inside
``Generator`` use this structure as in

.. code-block:: c

  bitgen_state->next_uint64(bitgen_state->state)

Examples
========

.. toctree::
    Numba <examples/numba>
    CFFI + Numba <examples/numba_cffi> 
    Cython <examples/cython/index>
    CFFI <examples/cffi>
