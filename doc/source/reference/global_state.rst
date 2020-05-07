.. _global_state:

************
Global State
************

NumPy has a few import-time, compile-time, or runtime options
which change the global behaviour.
Most of these are related to performance or for debugging
purposes and will not be interesting to the vast majority
of users.


Performance-Related Options
===========================

Number of Threads used for Linear Algebra
-----------------------------------------

NumPy itself is normally intentionally limited to a single thread
during function calls, however it does support multiple Python
threads running at the same time.
Note that for performant linear algebra NumPy uses a BLAS backend
such as OpenBLAS or MKL, which may use multiple threads that may
be controlled by environment variables such as ``OMP_NUM_THREADS``
depending on what is used.
One way to control the number of threads is the package
`threadpoolctl <https://pypi.org/project/threadpoolctl/>`_


Madvise Hugepage on Linux
-------------------------

When working with very large arrays on modern Linux kernels,
you can experience a significant speedup when
`transparent hugepage <https://www.kernel.org/doc/html/latest/admin-guide/mm/transhuge.html>`_
is used.
The current system policy for transparent hugepages can be seen by::

    cat /sys/kernel/mm/transparent_hugepage/enabled

When set to ``madvise`` NumPy will typically use hugepages for a performance
boost. This behaviour can be modified by setting the environment variable::

    NUMPY_MADVISE_HUGEPAGE=0

or setting it to ``1`` to always enable it. When not set, the default
is to use madvise on Kernels 4.6 and newer. These kernels presumably
experience a large speedup with hugepage support.
This flag is checked at import time.


Interoperability-Related Options
================================

The array function protocol which allows array-like objects to
hook into the NumPy API is currently enabled by default.
This option exists since NumPy 1.16 and is enabled by default since
NumPy 1.17. It can be disabled using::

    NUMPY_EXPERIMENTAL_ARRAY_FUNCTION=0

See also :py:meth:`numpy.class.__array_function__` for more information.
This flag is checked at import time.


Debugging-Related Options
=========================

Relaxed Strides Checking
------------------------

The *compile-time* environment variables::

    NPY_RELAXED_STRIDES_DEBUG=0
    NPY_RELAXED_STRIDES_CHECKING=1

control how NumPy reports contiguity for arrays.
The default that it is enabled and the debug mode is disabled.
This setting should always be enabled. Setting the
debug option can be interesting for testing code written
in C which iterates through arrays that may or may not be
contiguous in memory.
Most users will have no reason to change these, for details
please see the `memory layout <memory-layout>`_ documentation.
