Memory management in NumPy
==========================

The `numpy.ndarray` is a python class. It requires additional memory allocations
to hold `numpy.ndarray.strides`, `numpy.ndarray.shape` and
`numpy.ndarray.data` attributes. These attributes are specially allocated
after creating the python object in `__new__`. The ``strides`` and
``shape`` are stored in a piece of memory allocated internally.

These allocations are small relative to the ``data``, the homogeneous chunk of
memory used to store the actual array values (which could be pointers in the
case of ``object`` arrays). Since that memory can be significantly large, NumPy
has provided interfaces to manage it. This document details how those
interfaces work.

Historical overview
-------------------

Since version 1.7.0, NumPy has exposed a set of ``PyDataMem_*`` functions
(:c:func:`PyDataMem_NEW`, :c:func:`PyDataMem_FREE`, :c:func:`PyDataMem_RENEW`)
which are backed by `alloc`, `free`, `realloc` respectively. In that version
NumPy also exposed the `PyDataMem_EventHook` function described below, which
wrap the OS-level calls.

Python also improved its memory management capabilities, and began providing
various :ref:`management policies <memoryoverview>` beginning in version
3.4. These routines are divided into a set of domains, each domain has a
:c:type:`PyMemAllocatorEx` structure of routines for memory management. Python also
added a `tracemalloc` module to trace calls to the various routines. These
tracking hooks were added to the NumPy ``PyDataMem_*`` routines.

Configurable memory routines in NumPy
-------------------------------------

Users may wish to override the internal data memory routines with ones of their
own. Since NumPy does not use the Python domain strategy to manage data memory,
it provides an alternative set of C-APIs to change memory routines. There are
no Python domain-wide strategies for large chunks of object data, so those are
less suited to NumPy's needs. User who wish to change the NumPy data memory
management routines can use :c:func:`PyDataMem_SetHandler`, which uses a
:c:type:`PyDataMem_Handler` structure to hold pointers to functions used to
manage the data memory. The calls are still wrapped by internal routines to
call :c:func:`PyTraceMalloc_Track`, :c:func:`PyTraceMalloc_Untrack`, and will
use the :c:func:`PyDataMem_EventHookFunc` mechanism. Since the functions may
change during the lifetime of the process, each ``ndarray`` carries with it the
functions used at the time of its instantiation, and these will be used to
reallocate or free the data memory of the instance. As of NumPy version 1.21,
the copy functions are not yet implemented, all memory copies are handled by
calls to ``memcpy``.

.. c:type:: PyDataMem_Handler

    A struct to hold function pointers used to manipulate memory

    .. code-block:: c

        typedef struct {
            char name[128];  /* multiple of 64 to keep the struct unaligned */
            PyDataMem_AllocFunc *alloc;
            PyDataMem_ZeroedAllocFunc *zeroed_alloc;
            PyDataMem_FreeFunc *free;
            PyDataMem_ReallocFunc *realloc;
            PyDataMem_CopyFunc *host2obj;  /* copy from the host python */
            PyDataMem_CopyFunc *obj2host;  /* copy to the host python */
            PyDataMem_CopyFunc *obj2obj;  /* copy between two objects */
        } PyDataMem_Handler;

    where the function's signatures are

    .. code-block:: c

        typedef void *(PyDataMem_AllocFunc)(size_t size);
        typedef void *(PyDataMem_ZeroedAllocFunc)(size_t nelems, size_t elsize);
        typedef void (PyDataMem_FreeFunc)(void *ptr, size_t size);
        typedef void *(PyDataMem_ReallocFunc)(void *ptr, size_t size);
        typedef void *(PyDataMem_CopyFunc)(void *dst, const void *src, size_t size);

.. c:function:: const PyDataMem_Handler * PyDataMem_SetHandler(PyDataMem_Handler *handler)

   Sets a new allocation policy. If the input value is NULL, will reset
   the policy to the default. Returns the previous policy, NULL if the
   previous policy was the default. We wrap the user-provided functions
   so they will still call the python and numpy memory management callback
   hooks.
    
.. c:function:: const char * PyDataMem_GetHandlerName(PyArrayObject *obj)

   Return the const char name of the PyDataMem_Handler used by the
   PyArrayObject. If NULL, return the name of the current global policy that
   will be used to allocate data for the next PyArrayObject

For an example of setting up and using the PyDataMem_Handler, see the test in
:file:`numpy/core/tests/test_mem_policy.py`

.. c:function:: void PyDataMem_EventHookFunc(void *inp, void *outp, size_t size, void *user_data);

    This function will be called on NEW,FREE,RENEW calls in data memory
    manipulation



.. c:function:: PyDataMem_EventHookFunc * PyDataMem_SetEventHook(PyDataMem_EventHookFunc *newhook, void *user_data, void **old_data)

    Sets the allocation event hook for numpy array data.
  
    Returns a pointer to the previous hook or NULL.  If old_data is
    non-NULL, the previous user_data pointer will be copied to it.
  
    If not NULL, hook will be called at the end of each PyDataMem_NEW/FREE/RENEW:

    .. code-block:: c
   
        result = PyDataMem_NEW(size)        -> (*hook)(NULL, result, size, user_data)
        PyDataMem_FREE(ptr)                 -> (*hook)(ptr, NULL, 0, user_data)
        result = PyDataMem_RENEW(ptr, size) -> (*hook)(ptr, result, size, user_data)
  
    When the hook is called, the GIL will be held by the calling
    thread.  The hook should be written to be reentrant, if it performs
    operations that might cause new allocation events (such as the
    creation/destruction numpy objects, or creating/destroying Python
    objects which might cause a gc)

