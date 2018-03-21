Multithreaded Generation
========================

The four core distributions all allow existing arrays to be filled using the
``out`` keyword argument.  Existing arrays need to be contiguous and
well-behaved (writable and aligned).  Under normal circumstances, arrays
created using the common constructors such as :meth:`numpy.empty` will satisfy
these requirements.

This example makes use of Python 3 :mod:`concurrent.futures` to fill an array
using multiple threads.  Threads are long-lived so that repeated calls do not
require any additional overheads from thread creation. The underlying PRNG is
xorshift2014 which is fast, has a long period and supports using ``jump`` to
advance the state. The random numbers generated are reproducible in the sense
that the same seed will produce the same outputs.

.. code-block:: ipython

    from randomgen import Xorshift1024
    import multiprocessing
    import concurrent.futures
    import numpy as np
    
    class MultithreadedRNG(object):
        def __init__(self, n, seed=None, threads=None):
            rg = Xorshift1024(seed)
            if threads is None:
                threads = multiprocessing.cpu_count()
            self.threads = threads
    
            self._random_generators = []
            for _ in range(0, threads-1):
                _rg = Xorshift1024()
                _rg.state = rg.state
                self._random_generators.append(_rg.generator)
                rg.jump()
            self._random_generators.append(rg.generator)
    
            self.n = n
            self.executor = concurrent.futures.ThreadPoolExecutor(threads)
            self.values = np.empty(n)
            self.step = np.ceil(n / threads).astype(np.int)
    
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

    In [2]: mrng = MultithreadedRNG(10000000, seed=0)
    ...: print(mrng.values[-1])
    0.0

    In [3]: mrng.fill()
        ...: print(mrng.values[-1])
    3.296046120254392

The time required to produce using multiple threads can be compared to
the time required to generate using a single thread.

.. code-block:: ipython

    In [4]: print(mrng.threads)
        ...: %timeit mrng.fill()
    4
    42.9 ms ± 2.55 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

The single threaded call directly uses the PRNG.

.. code-block:: ipython

    In [5]: values = np.empty(10000000)
        ...: rg = Xorshift1024().generator
        ...: %timeit rg.standard_normal(out=values)
    220 ms ± 27.3 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

The gains are substantial and the scaling is reasonable even for large that
are only moderately large.  The gains are even larger when compared to a call
that does not use an existing array due to array creation overhead.

.. code-block:: ipython

    In [6]: rg = Xorshift1024().generator
        ...: %timeit rg.standard_normal(10000000)
        ...: 
    256 ms ± 41.8 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
