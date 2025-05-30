Multithreaded generation
========================

The four core distributions (:meth:`~.Generator.random`,
:meth:`~.Generator.standard_normal`, :meth:`~.Generator.standard_exponential`,
and :meth:`~.Generator.standard_gamma`) all allow existing arrays to be filled
using the ``out`` keyword argument. Existing arrays need to be contiguous and
well-behaved (writable and aligned). Under normal circumstances, arrays
created using the common constructors such as :meth:`numpy.empty` will satisfy
these requirements.

This example makes use of:mod:`concurrent.futures` to fill an array using
multiple threads.  Threads are long-lived so that repeated calls do not
require any additional overheads from thread creation.

The random numbers generated are reproducible in the sense that the same
seed will produce the same outputs, given that the number of threads does not
change.

.. code-block:: ipython

    from numpy.random import default_rng, SeedSequence
    import multiprocessing
    import concurrent.futures
    import numpy as np

    class MultithreadedRNG:
        def __init__(self, n, seed=None, threads=None):
            if threads is None:
                threads = multiprocessing.cpu_count()
            self.threads = threads

            seq = SeedSequence(seed)
            self._random_generators = [default_rng(s)
                                       for s in seq.spawn(threads)]

            self.n = n
            self.executor = concurrent.futures.ThreadPoolExecutor(threads)
            self.values = np.empty(n)
            self.step = np.ceil(n / threads).astype(np.int_)

        def fill(self):
            def _fill(random_state, out, first, last):
                random_state.standard_normal(out=out[first:last])

            futures = {}
            for i in range(self.threads):
                args = (_fill,
                        self._random_generators[i],
                        self.values,
                        i * self.step,
                        (i + 1) * self.step)
                futures[self.executor.submit(*args)] = i
            concurrent.futures.wait(futures)

        def __del__(self):
            self.executor.shutdown(False)



The multithreaded random number generator can be used to fill an array.
The ``values`` attributes shows the zero-value before the fill and the
random value after.

.. code-block:: ipython

    In [2]: mrng = MultithreadedRNG(10000000, seed=12345)
       ...: print(mrng.values[-1])
    Out[2]: 0.0

    In [3]: mrng.fill()
       ...: print(mrng.values[-1])
    Out[3]: 2.4545724517479104

The time required to produce using multiple threads can be compared to
the time required to generate using a single thread.

.. code-block:: ipython

    In [4]: print(mrng.threads)
       ...: %timeit mrng.fill()

    Out[4]: 4
       ...: 32.8 ms ± 2.71 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

The single threaded call directly uses the BitGenerator.

.. code-block:: ipython

    In [5]: values = np.empty(10000000)
       ...: rg = default_rng()
       ...: %timeit rg.standard_normal(out=values)

    Out[5]: 99.6 ms ± 222 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

The gains are substantial and the scaling is reasonable even for arrays that
are only moderately large. The gains are even larger when compared to a call
that does not use an existing array due to array creation overhead.

.. code-block:: ipython

    In [6]: rg = default_rng()
       ...: %timeit rg.standard_normal(10000000)

    Out[6]: 125 ms ± 309 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

Note that if ``threads`` is not set by the user, it will be determined by
``multiprocessing.cpu_count()``.

.. code-block:: ipython

    In [7]: # simulate the behavior for `threads=None`, if the machine had only one thread
       ...: mrng = MultithreadedRNG(10000000, seed=12345, threads=1)
       ...: print(mrng.values[-1])
    Out[7]: 1.1800150052158556
