.. _basics.parallel:

*****************************
Parallel Programming in NumPy
*****************************

NumPy is designed to be fast, but sometimes single-threaded performance is not enough. 
This guide covers how to use multiple CPU cores with NumPy.

Vectorization vs Parallelism
----------------------------
Before using threads, ensure your code is "Vectorized". NumPy's internal loops 
(ufuncs) are often faster than any manual parallel loop in Python.

The GIL and NumPy
-----------------
NumPy releases the Global Interpreter Lock (GIL) for most computational tasks. 
This means you can use ``concurrent.futures.ThreadPoolExecutor`` to run NumPy 
code on multiple cores simultaneously without the usual Python bottlenecks.

Using ThreadPoolExecutor
------------------------
For CPU-bound tasks like large array exponentiation, threading is efficient:

.. code-block:: python

    import numpy as np
    from concurrent.futures import ThreadPoolExecutor

    def process_chunk(chunk):
        return np.exp(chunk).sum()

    data = np.random.rand(10_000, 10_000)
    chunks = np.array_split(data, 4)

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_chunk, chunks))

When to use Multiprocessing
---------------------------
If your code involves heavy Python-level logic (not just NumPy calls), 
use ``multiprocessing``. However, be aware of the overhead of copying 
large arrays between processes.