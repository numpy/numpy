.. currentmodule:: numpy.random

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

This example shows how numba can be used to produce Box-Muller normals using
a pure Python implementation which is then compiled.  The random numbers are
provided by ``ctypes.next_double``.

.. code-block:: python

    from numpy.random import PCG64
    import numpy as np
    import numba as nb

    x = PCG64()
    f = x.ctypes.next_double
    s = x.ctypes.state
    state_addr = x.ctypes.state_address

    def normals(n, state):
        out = np.empty(n)
        for i in range((n+1)//2):
            x1 = 2.0*f(state) - 1.0
            x2 = 2.0*f(state) - 1.0
            r2 = x1*x1 + x2*x2
            while r2 >= 1.0 or r2 == 0.0:
                x1 = 2.0*f(state) - 1.0
                x2 = 2.0*f(state) - 1.0
                r2 = x1*x1 + x2*x2
            g = np.sqrt(-2.0*np.log(r2)/r2)
            out[2*i] = g*x1
            if 2*i+1 < n:
                out[2*i+1] = g*x2
        return out

    # Compile using Numba
    print(normals(10, s).var())
    # Warm up
    normalsj = nb.jit(normals, nopython=True)
    # Must use state address not state with numba
    normalsj(1, state_addr)
    %timeit normalsj(1000000, state_addr)
    print('1,000,000 Box-Muller (numba/PCG64) randoms')
    %timeit np.random.standard_normal(1000000)
    print('1,000,000 Box-Muller (NumPy) randoms')


Both CTypes and CFFI allow the more complicated distributions to be used
directly in Numba after compiling the file distributions.c into a DLL or so.
An example showing the use of a more complicated distribution is in the
examples folder.

.. _randomgen_cython:

Cython
======

Cython can be used to unpack the ``PyCapsule`` provided by a BitGenerator.
This example uses `~pcg64.PCG64` and
``random_gauss_zig``, the Ziggurat-based generator for normals, to fill an
array.  The usual caveats for writing high-performance code using Cython --
removing bounds checks and wrap around, providing array alignment information
-- still apply.

.. code-block:: cython

    import numpy as np
    cimport numpy as np
    cimport cython
    from cpython.pycapsule cimport PyCapsule_IsValid, PyCapsule_GetPointer
    from numpy.random.common cimport *
    from numpy.random.distributions cimport random_gauss_zig
    from numpy.random import PCG64


    @cython.boundscheck(False)
    @cython.wraparound(False)
    def normals_zig(Py_ssize_t n):
        cdef Py_ssize_t i
        cdef bitgen_t *rng
        cdef const char *capsule_name = "BitGenerator"
        cdef double[::1] random_values

        x = PCG64()
        capsule = x.capsule
        if not PyCapsule_IsValid(capsule, capsule_name):
            raise ValueError("Invalid pointer to anon_func_state")
        rng = <bitgen_t *> PyCapsule_GetPointer(capsule, capsule_name)
        random_values = np.empty(n)
        # Best practice is to release GIL and acquire the lock
        with x.lock, nogil:
            for i in range(n):
                random_values[i] = random_gauss_zig(rng)
        randoms = np.asarray(random_values)
        return randoms

The BitGenerator can also be directly accessed using the members of the basic
RNG structure.

.. code-block:: cython

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def uniforms(Py_ssize_t n):
        cdef Py_ssize_t i
        cdef bitgen_t *rng
        cdef const char *capsule_name = "BitGenerator"
        cdef double[::1] random_values

        x = PCG64()
        capsule = x.capsule
        # Optional check that the capsule if from a BitGenerator
        if not PyCapsule_IsValid(capsule, capsule_name):
            raise ValueError("Invalid pointer to anon_func_state")
        # Cast the pointer
        rng = <bitgen_t *> PyCapsule_GetPointer(capsule, capsule_name)
        random_values = np.empty(n)
        with x.lock, nogil:
            for i in range(n):
                # Call the function
                random_values[i] = rng.next_double(rng.state)
        randoms = np.asarray(random_values)
        return randoms

These functions along with a minimal setup file are included in the
examples folder.

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
