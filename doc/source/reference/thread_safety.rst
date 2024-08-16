.. _thread_safety:

*************
Thread Safety
*************

NumPy supports use in a multithreaded context via the `threading` module in the
standard library. Many NumPy operations release the GIL, so unlike many
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

In the future, we may add locking to ndarray to make writing multithreaded
algorithms using NumPy arrays safer, but for now we suggest focusing on
read-only access of arrays that are shared between threads, or adding your own
locking if you need to mutation and multithreading.

Note that operations that *do not* release the GIL will see no performance gains
from use of the `threading` module, and instead might be better served with
`multiprocessing`. In particular, operations on arrays with ``dtype=object`` do
not release the GIL.

Free-threaded Python
--------------------

.. versionadded:: 2.1

Starting with NumPy 2.1 and CPython 3.13, NumPy also has experimental support
for python runtimes with the GIL disabled. See
https://py-free-threading.github.io for more information about installing and
using free-threaded Python, as well as information about supporting it in
libraries that depend on NumPy.

Because free-threaded Python does not have a global interpreter lock to
serialize access to Python objects, there are more opportunities for threads to
mutate shared state and create thread safety issues. In addition to the
limitations about locking of the ndarray object noted above, this also means
that arrays with ``dtype=object`` are not protected by the GIL, creating data
races for python objects that are not possible outside free-threaded python.
