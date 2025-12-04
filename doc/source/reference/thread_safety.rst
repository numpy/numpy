.. _thread_safety:

*************
Thread Safety
*************

NumPy supports use in a multithreaded context via the `threading` module in the
standard library. Many NumPy operations release the :term:`python:GIL`, so unlike many
situations in Python, it is possible to improve parallel performance by
exploiting multithreaded parallelism in Python.

The easiest performance gains happen when each worker thread owns its own array
or set of array objects, with no data directly shared between threads. Because
NumPy releases the :term:`python:GIL` for many low-level operations, threads
that spend most of the time in low-level code will run in parallel.

It is possible to share NumPy arrays between threads, but extreme care must be
taken to avoid creating thread safety issues when mutating arrays that are
shared between multiple threads. If two threads simultaneously read from and
write to the same array, they will at best produce inconsistent, racey results that
are not reproducible, let alone correct. It is also possible to crash the Python
interpreter by, for example, resizing an array while another thread is reading
from it to compute a ufunc operation.

In the future, we may add locking to :class:`~numpy.ndarray` to make writing multithreaded
algorithms using NumPy arrays safer, but for now we suggest focusing on
read-only access of arrays that are shared between threads, or adding your own
locking if you need to mutation and multithreading.

Note that operations that *do not* release the :term:`python:GIL` will see no performance gains
from use of the `threading` module, and instead might be better served with
`multiprocessing`. In particular, operations on arrays with ``dtype=object`` do
not release the :term:`python:GIL`.

Thread-local state
------------------

NumPy maintains some state on a per-thread basis, which means each thread has
its own independent configuration:

* **Error handling** - The floating-point error handling mode (configured via
  :func:`numpy.seterr`, :func:`numpy.geterr`, and :func:`numpy.errstate`) is
  thread-local. Each thread can have different settings for how to handle
  invalid operations, division by zero, overflow, and underflow. See
  :ref:`misc-error-handling` for more details.

* **Buffer size** - The internal buffer size used by ufuncs for processing
  misaligned or type-converted data is configurable per-thread.

Free-threaded Python
--------------------

.. versionadded:: 2.1

Starting with NumPy 2.1 and CPython 3.13, NumPy also has experimental support
for python runtimes with the GIL disabled. See
https://py-free-threading.github.io for more information about installing and
using :py:term:`free-threaded <python:free threading>` Python, as well as
information about supporting it in libraries that depend on NumPy.

Because :py:term:`free-threaded <python:free threading>` Python does not have a
global interpreter lock to serialize access to Python objects, there are more
opportunities for threads to mutate shared state and create thread safety
issues. In addition to the limitations about locking of the
:class:`~numpy.ndarray` object noted above, this also means that arrays with
``dtype=object`` are not protected by the GIL, creating data races for python
objects that are not possible outside :py:term:`free-threaded <python:free
threading>` python.

C-API Threading Support
-----------------------

For developers writing C extensions that interact with NumPy, several parts of
the :doc:`C-API array documentation </reference/c-api/array>` provide detailed
information about multithreading considerations:

* :ref:`Threading support macros <array.ndarray.capi.threading>` - Macros for
  releasing and acquiring the GIL when calling compiled functions, including
  :c:macro:`NPY_BEGIN_ALLOW_THREADS`, :c:macro:`NPY_END_ALLOW_THREADS`,
  :c:func:`NPY_BEGIN_THREADS_DESCR`, and related utilities.

* :c:enumerator:`NPY_METH_REQUIRES_PYAPI
  <NPY_ARRAYMETHOD_FLAGS.NPY_METH_REQUIRES_PYAPI>` - This ArrayMethod flag
  indicates when a method must hold the :term:`python:GIL`. If this flag is not
  set, the :term:`python:GIL` is released before the loop is called.

* :c:func:`PyArray_RegisterDataType` - Note that registering new data types is
  not thread-safe. The ``NPY_NUMUSERTYPES`` global variable is not protected,
  so adding new data types in a multithreaded context requires external
  locking.

* :c:type:`NpyAuxData_CloneFunc` - Clone functions for ``NpyAuxData`` may be
  called from a multi-threaded context and should never set Python exceptions.

See Also
--------

* :doc:`/reference/random/multithreading` - Practical example of using NumPy's
  random number generators in a multithreaded context with
  :mod:`concurrent.futures`.

* :doc:`/reference/routines.err` - API documentation for error handling
  functions (:func:`numpy.seterr`, :func:`numpy.errstate`, etc.).
