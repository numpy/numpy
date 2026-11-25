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
NumPy releases the GIL for many low-level operations, threads that spend most of
the time in low-level code will run in parallel.

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

Note that operations that *do not* release the GIL will see no performance gains
from use of the `threading` module, and instead might be better served with
`multiprocessing`. In particular, operations on arrays with ``dtype=np.object_``
do not release the GIL.

.. _context_local:

Context Local State
-------------------

NumPy stores user-adjustable configuration options in :py:mod:`context variables
<python:contextvars>`. Context variables allow *context-local state*, which
means each thread in a multithreaded program or task in an asyncio program has
its own independent configuration. This includes the following state:

* The `numpy.errstate`, which storesl floating-point error handling configuration
  options and the ufunc buffer size settable via `numpy.setbufsize`. See
  :doc:`/reference/routines.err` and :ref:`use-of-internal-buffers` for
  more details.
* The `numpy.printoptions`  and all text-formatting configuration options.
  See :ref:`text_formatting_options` for more details.
* The memory allocator, see :ref:`data_memory` and :ref:`NEP 49 <NEP49>` for
  more details.

State stored in a context variable is set syntactically using the ``with``
statement. For example, you can update the numpy printing options state like so:

.. code-block:: pycon

   >>> with np.printoptions(precision=2):
   ...    np.array([2.0]) / 3
   array([0.67])
   >>> np.array([2.0]) / 3
   array([0.66666667])


This property applies to all context-local state, not just `numpy.printoptions`.

Interaction with the `threading` module
+++++++++++++++++++++++++++++++++++++++

Before Python 3.14, new threads always start with newly initialized context
state. For example:

.. code-block:: pycon

     $ python3.12
     >>> import numpy, threading
     >>> def print_printoptions():
     ...     print(numpy.get_printoptions()['precision'])
     >>> with numpy.printoptions(precision=2):
     ...     threading.Thread(target=print_printoptions).start()
     8

Starting in Python 3.14 a new ``thread_inherit_context`` startup configuration
option for Python allows opting into a new behavior where context state for
spawned threads behaves syntactically as one would expect the above code to
behave:

.. code-block:: pycon

     $ python3.14 -Xthread_inherit_context=1
     >>> import numpy, threading
     >>> def print_printoptions():
     ...     print(numpy.get_printoptions()['precision'])
     >>> with numpy.printoptions(precision=2):
     ...     threading.Thread(target=print_printoptions).start()
     2

See `the CPython documentation
<https://docs.python.org/3/using/cmdline.html#envvar-PYTHON_THREAD_INHERIT_CONTEXT>`_
for more details.

Free-threaded Python
--------------------

.. versionadded:: 2.1

Starting with NumPy 2.1 and CPython 3.13, NumPy also has experimental support
for python runtimes with the GIL disabled. See
https://py-free-threading.github.io for more information about installing and
using :py:term:`free-threaded <python:free threading>` Python, as well as
information about supporting it in libraries that depend on NumPy.

Because free-threaded Python does not have a
global interpreter lock to serialize access to Python objects, there are more
opportunities for threads to mutate shared state and create thread safety
issues. In addition to the limitations about locking of the
:class:`~numpy.ndarray` object noted above, this also means that arrays with
``dtype=np.object_`` are not protected by the GIL, creating data races for python
objects that are not possible outside free-threaded python.

Free-threaded Python has ``thread_inherit_context`` turned on by default
starting in Python 3.14. See :ref:`context_local` for more information.

C-API Threading Support
-----------------------

For developers writing C extensions that interact with NumPy, several parts of
the :doc:`C-API array documentation </reference/c-api/array>` provide detailed
information about multithreading considerations.

See Also
--------

* :doc:`/reference/random/multithreading` - Practical example of using NumPy's
  random number generators in a multithreaded context with
  :mod:`concurrent.futures`.
