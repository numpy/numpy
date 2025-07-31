.. _NEP49:

===================================
NEP 49 — Data allocation strategies
===================================

:Author: Matti Picus
:Status: Final
:Type: Standards Track
:Created: 2021-04-18
:Resolution: `NumPy Discussion <https://mail.python.org/archives/list/numpy-discussion@python.org/thread/YZ3PNTXZUT27B6ITFAD3WRSM3T3SRVK4/#PKYXCTG4R5Q6LIRZC4SEWLNBM6GLRF26>`_

Abstract
--------

The ``numpy.ndarray`` requires additional memory allocations
to hold ``numpy.ndarray.strides``, ``numpy.ndarray.shape`` and
``numpy.ndarray.data`` attributes. These attributes are specially allocated
after creating the python object in ``__new__`` method.

This NEP proposes a mechanism to override the memory management strategy used
for ``ndarray->data`` with user-provided alternatives. This allocation holds
the data and can be very large. As accessing this data often becomes
a performance bottleneck, custom allocation strategies to guarantee data
alignment or pinning allocations to specialized memory hardware can enable
hardware-specific optimizations. The other allocations remain unchanged.

Motivation and scope
--------------------

Users may wish to override the internal data memory routines with ones of their
own. Two such use-cases are to ensure data alignment and to pin certain
allocations to certain NUMA cores. This desire for alignment was discussed
multiple times on the mailing list `in 2005`_,  and in `issue 5312`_ in 2014,
which led to `PR 5457`_ and more mailing list discussions here_ `and here`_. In
a comment on the issue `from 2017`_, a user described how 64-byte alignment
improved performance by 40x.

Also related is `issue 14177`_ around the use of ``madvise`` and huge pages on
Linux.

Various tracing and profiling libraries like filprofiler_ or `electric fence`_
override ``malloc``.

The long CPython discussion of `BPO 18835`_  began with discussing the need for
``PyMem_Alloc32`` and ``PyMem_Alloc64``.  The early conclusion was that the
cost (of wasted padding) vs. the benefit of aligned memory is best left to the
user, but then evolves into a discussion of various proposals to deal with
memory allocations, including `PEP 445`_ `memory interfaces`_ to
``PyTraceMalloc_Track`` which apparently was explicitly added for NumPy.

Allowing users to implement different strategies via the NumPy C-API will
enable exploration of this rich area of possible optimizations. The intention
is to create a flexible enough interface without burdening normative users.

.. _`issue 5312`: https://github.com/numpy/numpy/issues/5312
.. _`from 2017`: https://github.com/numpy/numpy/issues/5312#issuecomment-315234656
.. _`in 2005`: https://numpy-discussion.scipy.narkive.com/MvmMkJcK/numpy-arrays-data-allocation-and-simd-alignement
.. _`here`: https://mail.python.org/archives/list/numpy-discussion@python.org/thread/YPC5BGPUMKT2MLBP6O3FMPC35LFM2CCH/#YPC5BGPUMKT2MLBP6O3FMPC35LFM2CCH
.. _`and here`: https://mail.python.org/archives/list/numpy-discussion@python.org/thread/IQK3EPIIRE3V4BPNAMJ2ZST3NUG2MK2A/#IQK3EPIIRE3V4BPNAMJ2ZST3NUG2MK2A
.. _`issue 14177`: https://github.com/numpy/numpy/issues/14177
.. _`filprofiler`: https://github.com/pythonspeed/filprofiler/blob/master/design/allocator-overrides.md
.. _`electric fence`: https://github.com/boundarydevices/efence
.. _`memory interfaces`: https://docs.python.org/3/c-api/memory.html#customize-memory-allocators
.. _`BPO 18835`: https://bugs.python.org/issue18835
.. _`PEP 445`: https://www.python.org/dev/peps/pep-0445/

Usage and impact
----------------

The new functions can only be accessed via the NumPy C-API. An example is
included later in this NEP. The added ``struct`` will increase the size of the
``ndarray`` object. It is a necessary price to pay for this approach. We
can be reasonably sure that the change in size will have a minimal impact on
end-user code because NumPy version 1.20 already changed the object size.

The implementation preserves the use of ``PyTraceMalloc_Track`` to track
allocations already present in NumPy.

Backward compatibility
----------------------

The design will not break backward compatibility. Projects that were assigning
to the ``ndarray->data`` pointer were already breaking the current memory
management strategy and should restore
``ndarray->data`` before calling ``Py_DECREF``. As mentioned above, the change
in size should not impact end-users.

Detailed description
--------------------

High level design
=================

Users who wish to change the NumPy data memory management routines will use
:c:func:`PyDataMem_SetHandler`, which uses a :c:type:`PyDataMem_Handler`
structure to hold pointers to functions used to manage the data memory. In
order to allow lifetime management of the ``context``, the structure is wrapped
in a ``PyCapsule``.

Since a call to ``PyDataMem_SetHandler`` will change the default functions, but
that function may be called during the lifetime of an ``ndarray`` object, each
``ndarray`` will carry with it the ``PyDataMem_Handler``-wrapped PyCapsule used
at the time of its instantiation, and these will be used to reallocate or free
the data memory of the instance. Internally NumPy may use ``memcpy`` or
``memset`` on the pointer to the data memory.

The name of the handler will be exposed on the python level via a
``numpy.core.multiarray.get_handler_name(arr)`` function. If called as
``numpy.core.multiarray.get_handler_name()`` it will return the name of the
handler that will be used to allocate data for the next new `ndarrray`.

The version of the handler will be exposed on the python level via a
``numpy.core.multiarray.get_handler_version(arr)`` function. If called as
``numpy.core.multiarray.get_handler_version()`` it will return the version of the
handler that will be used to allocate data for the next new `ndarrray`.

The version, currently 1, allows for future enhancements to the
``PyDataMemAllocator``. If fields are added, they must be added to the end.


NumPy C-API functions
=====================

.. c:type:: PyDataMem_Handler

    A struct to hold function pointers used to manipulate memory

    .. code-block:: c

        typedef struct {
            char name[127];  /* multiple of 64 to keep the struct aligned */
            uint8_t version; /* currently 1 */
            PyDataMemAllocator allocator;
        } PyDataMem_Handler;

    where the allocator structure is

    .. code-block:: c

        /* The declaration of free differs from PyMemAllocatorEx */
        typedef struct {
            void *ctx;
            void* (*malloc) (void *ctx, size_t size);
            void* (*calloc) (void *ctx, size_t nelem, size_t elsize);
            void* (*realloc) (void *ctx, void *ptr, size_t new_size);
            void (*free) (void *ctx, void *ptr, size_t size);
        } PyDataMemAllocator;

    The use of a ``size`` parameter in ``free`` differentiates this struct from
    the :c:type:`PyMemAllocatorEx` struct in Python. This call signature is
    used internally in NumPy currently, and also in other places for instance
    `C++98 <https://en.cppreference.com/w/cpp/memory/allocator/deallocate>`,
    `C++11 <https://en.cppreference.com/w/cpp/memory/allocator_traits/deallocate>`, and
    `Rust (allocator_api) <https://doc.rust-lang.org/std/alloc/trait.Allocator.html#tymethod.deallocate>`.

    The consumer of the `PyDataMemAllocator` interface must keep track of ``size`` and make sure it is
    consistent with the parameter passed to the ``(m|c|re)alloc``  functions.

    NumPy itself may violate this requirement when the shape of the requested
    array contains a ``0``, so authors of PyDataMemAllocators should relate to
    the ``size`` parameter as a best-guess. Work to fix this is ongoing in PRs
    15780_ and 15788_ but has not yet been resolved. When it is this NEP should
    be revisited.

.. c:function:: PyObject * PyDataMem_SetHandler(PyObject *handler)

   Sets a new allocation policy. If the input value is ``NULL``, will reset
   the policy to the default. Return the previous policy, or
   return NULL if an error has occurred. We wrap the user-provided
   so they will still call the Python and NumPy memory management callback
   hooks. All the function pointers must be filled in, ``NULL`` is not
   accepted.

.. c:function:: const PyObject * PyDataMem_GetHandler()

   Return the current policy that will be used to allocate data for the
   next ``PyArrayObject``. On failure, return ``NULL``.

``PyDataMem_Handler`` thread safety and lifetime
================================================
The active handler is stored in the current :py:class:`~contextvars.Context`
via a :py:class:`~contextvars.ContextVar`. This ensures it can be configured both
per-thread and per-async-coroutine.

There is currently no lifetime management of ``PyDataMem_Handler``.
The user of `PyDataMem_SetHandler` must ensure that the argument remains alive
for as long as any objects allocated with it, and while it is the active handler.
In practice, this means the handler must be immortal.

As an implementation detail, currently this ``ContextVar`` contains a ``PyCapsule``
object storing a pointer to a ``PyDataMem_Handler`` with no destructor,
but this should not be relied upon.

Sample code
===========

This code adds a 64-byte header to each ``data`` pointer and stores information
about the allocation in the header. Before calling ``free``, a check ensures
the ``sz`` argument is correct.

.. code-block:: c

    #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
    #include <numpy/arrayobject.h>
    NPY_NO_EXPORT void *

    typedef struct {
        void *(*malloc)(size_t);
        void *(*calloc)(size_t, size_t);
        void *(*realloc)(void *, size_t);
        void (*free)(void *);
    } Allocator;

    NPY_NO_EXPORT void *
    shift_alloc(Allocator *ctx, size_t sz) {
        char *real = (char *)ctx->malloc(sz + 64);
        if (real == NULL) {
            return NULL;
        }
        snprintf(real, 64, "originally allocated %ld", (unsigned long)sz);
        return (void *)(real + 64);
    }

    NPY_NO_EXPORT void *
    shift_zero(Allocator *ctx, size_t sz, size_t cnt) {
        char *real = (char *)ctx->calloc(sz + 64, cnt);
        if (real == NULL) {
            return NULL;
        }
        snprintf(real, 64, "originally allocated %ld via zero",
                 (unsigned long)sz);
        return (void *)(real + 64);
    }

    NPY_NO_EXPORT void
    shift_free(Allocator *ctx, void * p, npy_uintp sz) {
        if (p == NULL) {
            return ;
        }
        char *real = (char *)p - 64;
        if (strncmp(real, "originally allocated", 20) != 0) {
            fprintf(stdout, "uh-oh, unmatched shift_free, "
                    "no appropriate prefix\\n");
            /* Make C runtime crash by calling free on the wrong address */
            ctx->free((char *)p + 10);
            /* ctx->free(real); */
        }
        else {
            npy_uintp i = (npy_uintp)atoi(real +20);
            if (i != sz) {
                fprintf(stderr, "uh-oh, unmatched shift_free"
                        "(ptr, %ld) but allocated %ld\\n", sz, i);
                /* This happens when the shape has a 0, only print */
                ctx->free(real);
            }
            else {
                ctx->free(real);
            }
        }
    }

    NPY_NO_EXPORT void *
    shift_realloc(Allocator *ctx, void * p, npy_uintp sz) {
        if (p != NULL) {
            char *real = (char *)p - 64;
            if (strncmp(real, "originally allocated", 20) != 0) {
                fprintf(stdout, "uh-oh, unmatched shift_realloc\\n");
                return realloc(p, sz);
            }
            return (void *)((char *)ctx->realloc(real, sz + 64) + 64);
        }
        else {
            char *real = (char *)ctx->realloc(p, sz + 64);
            if (real == NULL) {
                return NULL;
            }
            snprintf(real, 64, "originally allocated "
                     "%ld  via realloc", (unsigned long)sz);
            return (void *)(real + 64);
        }
    }

    static Allocator new_handler_ctx = {
        malloc,
        calloc,
        realloc,
        free
    };

    static PyDataMem_Handler new_handler = {
        "secret_data_allocator",
        1,
        {
            &new_handler_ctx,
            shift_alloc,      /* malloc */
            shift_zero, /* calloc */
            shift_realloc,      /* realloc */
            shift_free       /* free */
        }
    };

Related work
------------

This NEP is being tracked by the pnumpy_ project and a `comment in the PR`_
mentions use in orchestrating FPGA DMAs.

Implementation
--------------

This NEP has been implemented in `PR  17582`_.

Alternatives
------------

These were discussed in `issue 17467`_. `PR 5457`_  and `PR 5470`_ proposed a
global interface for specifying aligned allocations.

``PyArray_malloc_aligned`` and friends were added to NumPy with the
`numpy.random` module API refactor. and are used there for performance.

`PR 390`_ had two parts: expose ``PyDataMem_*`` via the NumPy C-API, and a hook
mechanism. The PR was merged with no example code for using these features.

Discussion
----------

The discussion on the mailing list led to the ``PyDataMemAllocator`` struct
with a ``context`` field like :c:type:`PyMemAllocatorEx` but with a different
signature for ``free``.


References and footnotes
------------------------

.. [1] Each NEP must either be explicitly labeled as placed in the public domain (see
   this NEP as an example) or licensed under the `Open Publication License`_.

.. _Open Publication License: https://www.opencontent.org/openpub/

.. _`PR 17582`: https://github.com/numpy/numpy/pull/17582
.. _`PR 5457`: https://github.com/numpy/numpy/pull/5457
.. _`PR 5470`: https://github.com/numpy/numpy/pull/5470
.. _`15780`: https://github.com/numpy/numpy/pull/15780
.. _`15788`: https://github.com/numpy/numpy/pull/15788
.. _`PR 390`: https://github.com/numpy/numpy/pull/390
.. _`issue 17467`: https://github.com/numpy/numpy/issues/17467
.. _`comment in the PR`: https://github.com/numpy/numpy/pull/17582#issuecomment-809145547
.. _pnumpy: https://quansight.github.io/pnumpy/stable/index.html

Copyright
---------

This document has been placed in the public domain. [1]_
