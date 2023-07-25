.. _global_state:

************
Global state
************

NumPy has a few import-time, compile-time, or runtime options
which change the global behaviour.
Most of these are related to performance or for debugging
purposes and will not be interesting to the vast majority
of users.


Performance-related options
===========================

Number of threads used for Linear Algebra
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

SIMD feature selection
----------------------

Setting ``NPY_DISABLE_CPU_FEATURES`` will exclude simd features at runtime.
See :ref:`runtime-simd-dispatch`.


Debugging-related options
=========================

Relaxed strides checking
------------------------

The *compile-time* environment variable::

    NPY_RELAXED_STRIDES_DEBUG=0

can be set to help debug code written in C which iteraters through arrays
manually.  When an array is contiguous and iterated in a contiguous manner,
its ``strides`` should not be queried.  This option can help find errors where
the ``strides`` are incorrectly used.
For details see the :ref:`memory layout <memory-layout>` documentation.


Warn if no memory allocation policy when deallocating data
----------------------------------------------------------

Some users might pass ownership of the data pointer to the ``ndarray`` by
setting the ``OWNDATA`` flag. If they do this without setting (manually) a
memory allocation policy, the default will be to call ``free``. If
``NUMPY_WARN_IF_NO_MEM_POLICY`` is set to ``"1"``, a ``RuntimeWarning`` will
be emitted. A better alternative is to use a ``PyCapsule`` with a deallocator
and set the ``ndarray.base``.


Testing planned future behavior
===============================

NumPy has some code paths which are planned to be activated in the future
but are not yet the default behavior.
You can try testing some of these which may be shipped with a new "major"
release (NumPy 2.0) by setting an environment before importing NumPy:

    NPY_NUMPY_2_BEHAVIOR=1

By default this will also activate the :ref:`NEP 50 <NEP50>` related setting
``NPY_PROMOTION_STATE`` (please see the NEP for details on this).

.. versionchanged:: 1.25.2
    This variable is only checked on the first import.
