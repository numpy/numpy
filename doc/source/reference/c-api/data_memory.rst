.. _data_memory:

Memory management in NumPy
==========================

The `numpy.ndarray` is a python class. It requires additional memory allocations
to hold `numpy.ndarray.strides`, `numpy.ndarray.shape` and
`numpy.ndarray.data` attributes. These attributes are specially allocated
after creating the python object in :meth:`~object.__new__`. The ``strides``
and ``shape`` are stored in a piece of memory allocated internally.

The ``data`` allocation used to store the actual array values (which could be
pointers in the case of ``object`` arrays) can be very large, so NumPy has
provided interfaces to manage its allocation and release. This document details
how those interfaces work.

Historical overview
-------------------

Since version 1.7.0, NumPy has exposed a set of ``PyDataMem_*`` functions
(:c:func:`PyDataMem_NEW`, :c:func:`PyDataMem_FREE`, :c:func:`PyDataMem_RENEW`)
which are backed by `alloc`, `free`, `realloc` respectively.

Since those early days, Python also improved its memory management
capabilities, and began providing
various :ref:`management policies <memoryoverview>` beginning in version
3.4. These routines are divided into a set of domains, each domain has a
:c:type:`PyMemAllocatorEx` structure of routines for memory management. Python also
added a `tracemalloc` module to trace calls to the various routines. These
tracking hooks were added to the NumPy ``PyDataMem_*`` routines.

NumPy added a small cache of allocated memory in its internal
``npy_alloc_cache``, ``npy_alloc_cache_zero``, and ``npy_free_cache``
functions. These wrap ``alloc``, ``alloc-and-memset(0)`` and ``free``
respectively, but when ``npy_free_cache`` is called, it adds the pointer to a
short list of available blocks marked by size. These blocks can be re-used by
subsequent calls to ``npy_alloc*``, avoiding memory thrashing.

Configurable memory routines in NumPy (NEP 49)
----------------------------------------------

Users may wish to override the internal data memory routines with ones of their
own. Since NumPy does not use the Python domain strategy to manage data memory,
it provides an alternative set of C-APIs to change memory routines. There are
no Python domain-wide strategies for large chunks of object data, so those are
less suited to NumPy's needs. User who wish to change the NumPy data memory
management routines can use :c:func:`PyDataMem_SetHandler`, which uses a
:c:type:`PyDataMem_Handler` structure to hold pointers to functions used to
manage the data memory. The calls are still wrapped by internal routines to
call :c:func:`PyTraceMalloc_Track`, :c:func:`PyTraceMalloc_Untrack`. Since the
functions may change during the lifetime of the process, each ``ndarray``
carries with it the functions used at the time of its instantiation, and these
will be used to reallocate or free the data memory of the instance.

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

.. c:function:: PyObject * PyDataMem_SetHandler(PyObject *handler)

   Set a new allocation policy. If the input value is ``NULL``, will reset the
   policy to the default. Return the previous policy, or
   return ``NULL`` if an error has occurred. We wrap the user-provided functions
   so they will still call the python and numpy memory management callback
   hooks.
    
.. c:function:: PyObject * PyDataMem_GetHandler()

   Return the current policy that will be used to allocate data for the
   next ``PyArrayObject``. On failure, return ``NULL``.

For an example of setting up and using the PyDataMem_Handler, see the test in
:file:`numpy/_core/tests/test_mem_policy.py`


What happens when deallocating if there is no policy set
--------------------------------------------------------

A rare but useful technique is to allocate a buffer outside NumPy, use
:c:func:`PyArray_NewFromDescr` to wrap the buffer in a ``ndarray``, then switch
the ``OWNDATA`` flag to true. When the ``ndarray`` is released, the
appropriate function from the ``ndarray``'s ``PyDataMem_Handler`` should be
called to free the buffer. But the ``PyDataMem_Handler`` field was never set,
it will be ``NULL``. For backward compatibility, NumPy will call ``free()`` to
release the buffer. If ``NUMPY_WARN_IF_NO_MEM_POLICY`` is set to ``1``, a
warning will be emitted. The current default is not to emit a warning, this may
change in a future version of NumPy.

A better technique would be to use a ``PyCapsule`` as a base object:

.. code-block:: c

    /* define a PyCapsule_Destructor, using the correct deallocator for buff */
    void free_wrap(void *capsule){
        void * obj = PyCapsule_GetPointer(capsule, PyCapsule_GetName(capsule));
        free(obj); 
    };

    /* then inside the function that creates arr from buff */
    ...
    arr = PyArray_NewFromDescr(... buf, ...);
    if (arr == NULL) {
        return NULL;
    }
    capsule = PyCapsule_New(buf, "my_wrapped_buffer",
                            (PyCapsule_Destructor)&free_wrap);
    if (PyArray_SetBaseObject(arr, capsule) == -1) {
        Py_DECREF(arr);
        return NULL;
    }
    ...

Example of memory tracing with ``np.lib.tracemalloc_domain``
------------------------------------------------------------

The builtin ``tracemalloc`` module can be used to track allocations inside NumPy.
NumPy places its CPU memory allocations into the  ``np.lib.tracemalloc_domain`` domain.
For additional information, check: https://docs.python.org/3/library/tracemalloc.html.

Here is an example on how to use ``np.lib.tracemalloc_domain``:

.. code-block:: python

    """
       The goal of this example is to show how to trace memory
       from an application that has NumPy and non-NumPy sections.
       We only select the sections using NumPy related calls.
    """
    
    import tracemalloc
    import numpy as np
    
    # Flag to determine if we select NumPy domain
    use_np_domain = True
    
    nx = 300
    ny = 500
    
    # Start to trace memory
    tracemalloc.start()
    
    # Section 1
    # ---------
    
    # NumPy related call
    a = np.zeros((nx,ny))
    
    # non-NumPy related call
    b = [i**2 for i in range(nx*ny)]
    
    snapshot1 = tracemalloc.take_snapshot()
    # We filter the snapshot to only select NumPy related calls
    np_domain = np.lib.tracemalloc_domain
    dom_filter = tracemalloc.DomainFilter(inclusive=use_np_domain,
                                          domain=np_domain)
    snapshot1 = snapshot1.filter_traces([dom_filter])
    top_stats1 = snapshot1.statistics('traceback')
    
    print("================ SNAPSHOT 1 =================")
    for stat in top_stats1:
        print(f"{stat.count} memory blocks: {stat.size / 1024:.1f} KiB")
        print(stat.traceback.format()[-1])
    
    # Clear traces of memory blocks allocated by Python
    # before moving to the next section.
    tracemalloc.clear_traces()
    
    # Section 2
    #----------
    
    # We are only using NumPy
    c = np.sum(a*a)
    
    snapshot2 = tracemalloc.take_snapshot()
    top_stats2 = snapshot2.statistics('traceback')

    print()
    print("================ SNAPSHOT 2 =================")
    for stat in top_stats2:
        print(f"{stat.count} memory blocks: {stat.size / 1024:.1f} KiB")
        print(stat.traceback.format()[-1])
    
    tracemalloc.stop()
    
    print()
    print("============================================")
    print("\nTracing Status : ", tracemalloc.is_tracing())
    
    try:
        print("\nTrying to Take Snapshot After Tracing is Stopped.")
        snap = tracemalloc.take_snapshot()
    except Exception as e:
        print("Exception : ", e)
    
