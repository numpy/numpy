.. _global_state:

****************************
Global Configuration Options
****************************

NumPy has a few import-time, compile-time, or runtime configuration
options which change the global behaviour.  Most of these are related to
performance or for debugging purposes and will not be interesting to the
vast majority of users.


Performance-related options
===========================

Number of threads used for linear algebra
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


madvise hugepage on Linux
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

Setting ``NPY_DISABLE_CPU_FEATURES`` will exclude SIMD features at runtime.

This environment variable lets you turn off specific CPU features that NumPy
would normally use to speed things up. You might need this for debugging,
testing, or if some CPU instructions are unstable on your system.

To use it, set the variable to a list of CPU feature names separated by commas.
For example::

    export NPY_DISABLE_CPU_FEATURES="AVX2,AVX512F"

You must set this variable before you import NumPy for it to work.

To see which CPU features are available on your system, run::

    python -c "import numpy as np; print(np.__cpu_features__)"

For more details, see :ref:`runtime-simd-dispatch`.
