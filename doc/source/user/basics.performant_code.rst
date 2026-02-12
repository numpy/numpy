.. _basics.performant_code:


***************************************************
Writing Performant NumPy Code with Multi-Core CPUs
***************************************************


Introduction
================

NumPy is designed for high performance numerical computing in Python by leveraging vectorized operations.
However, vectorization does not always fully utilize the capabilities of multi-core processors. To realize multiprocessing and multithreading, additional strategies are necessary.

In this section, we cover the following topics:

* :ref:`General concepts for using multi-core processors in Python <basics.performant_code.general_concepts_for_multi_core_processors>`
* :ref:`Using multi-core processors with Python standard libraries <basics.performant_code.multi_core_with_standard_libraries>`
* :ref:`Other tips for writing performant NumPy code <basics.performant_code.other_tips>`
* :ref:`Third party libraries for multi-core processing <basics.performant_code.third_party_libraries>`


.. _basics.performant_code.general_concepts_for_multi_core_processors:

General concepts for multi-core processors in Python
=====================================================


Multiprocessing
----------------

Multiprocessing is a technique that allows the execution of multiple processes simultaneously, each with its own Python interpreter and memory space.
As a high-level API, Python provides the `concurrent.futures.ProcessPoolExecutor` class to facilitate multiprocessing.

Firstly, we introduce brief Pros and Cons of multiprocessing:

Pros
++++
* Bypasses the Global Interpreter Lock (GIL), allowing true parallelism
* Avoids accidental data sharing due to separate memory spaces


Cons
++++
* Higher memory usage due to separate memory spaces for each process
* Difficulty in sharing data between processes, requiring serialization (pickling) of objects


General tips
++++++++++++

The following are general tips for utilizing multiprocessing.

Reduce creation overhead
~~~~~~~~~~~~~~~~~~~~~~~~~

Process creation has a higher overhead compared to thread creation due to the need to initialize a new Python interpreter and memory space.
To mitigate this overhead, consider the following strategies:

* Use process pools to reuse existing processes instead of creating new ones for each task. `concurrent.futures.ProcessPoolExecutor` provides this feature.
* Select appropriate startup methods. On Unix-like systems, the ``fork`` method can be faster than ``spawn`` or ``forkserver`` because it duplicates the parent process's memory space. However, ``fork`` may cause potential issues when using combined with threads or certain libraries. See the `multiprocessing documentation <https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods>`__ for more details.


Reduce communication overhead
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Inter-process communication (IPC) can introduce significant overhead due to data serialization and transfer between processes. In Python, only picklable objects are allowed to be passed between processes.
Due to this nature, multiprocessing is not suitable for the program which needs to synthesize data between processes frequently.

To reduce communication overhead, consider the following strategies:

* Minimize the amount of data transferred between processes.
* Use shared memory constructs such as `multiprocessing.shared_memory`, `multiprocessing.Array` or `multiprocessing.Value` for large data that needs to be accessed by multiple processes.
* Set appropriate ``chunksize`` when using methods like ``map`` or ``imap`` in process pools. ``chunksize`` determines the number of tasks assigned to each process at a time, which can help balance the trade-off between task granularity and communication overhead.


Balance processing load
~~~~~~~~~~~~~~~~~~~~~~~~

If the processing load is not evenly distributed among processes, some processes may finish their tasks earlier and remain idle while others are still working. It leads to inefficient use of resources and longer overall execution time.

To achieve better load balancing, consider the following strategies:

* Use dynamic task allocation where tasks are assigned to processes as they become available, rather than pre-allocating tasks.
* Check ``chunksize`` parameter to ensure that tasks are neither too small (causing excessive overhead) nor too large (leading to load imbalance).


Multithreading
-----------------

Multithreading allows multiple threads to run within the same process, sharing the same memory space.
Starting with Python 3.13, a free-threaded build of Python is available. When combined with libraries that are explicitly designed to be thread-safe, this can enable true parallel execution with threads.

As a high-level API, Python provides the `concurrent.futures.ThreadPoolExecutor` class  for thread-based parallelism.
Python also provides the `concurrent.futures.InterpreterPoolExecutor`, which uses multiple interpreters running in separate threads and avoids sharing Python objects between them. However, it is not yet available in NumPy. (See `gh-24755 <https://github.com/numpy/numpy/issues/24755>`__ for details.)


The main pros and cons of multithreading are as follows:

Pros
++++
* Lower memory usage since threads share the same memory space
* Easier communication between threads

Cons
++++
* Possibility of race conditions when mutating shared data simultaneously with reads.
* Limited performance improvement if using Python libraries are not thread-safe or have limited support for free-threaded Python builds

General tips
++++++++++++

The following are general tips for utilizing multithreading.

Avoid race conditions
~~~~~~~~~~~~~~~~~~~~~

Race conditions occur when multiple threads update shared data simultaneously, leading to unpredictable results.
To avoid race conditions, consider the following strategies:

* Use thread-safe data structures or synchronization primitives like locks, semaphores, or condition variables to manage access to shared data.
* Minimize the amount of shared data between threads by designing your program to use thread-local storage or by passing data explicitly to threads. 
* Prefer immutable NumPy arrays or read-only access patterns when possible, since they reduce the need for explicit synchronization.


.. _basics.performant_code.multi_core_with_standard_libraries:

Using multi-core processors with Python standard libraries
=============================================================

In this section, we demonstrate how to use Python's standard libraries to leverage multi-core processors with NumPy.

As an example, we use `Mandelbrot set <https://en.wikipedia.org/wiki/Mandelbrot_set>`__ generation. 
Mandelbrot set is defined as the set of complex numbers ``c`` for which the sequence defined by the iterative function does not diverge to infinity:

.. math::

    z_{n+1} = z_n^2 + c, \quad z_0 = 0


If the absolute value of :math:`z_n` remains bounded (i.e., does not exceed a certain threshold, typically ``2`` ) after a fixed number of iterations, then ``c`` is considered to be in the Mandelbrot set.

Following to this definition, we can calculate each point in the complex plane independently, making it suited for parallel computation.

The hot colors in the image below represent the number of iterations it took for the sequence to diverge for each point in the complex plane.


.. image:: images/np_mandelbrot.png
    :alt: Mandelbrot set
    :align: center
    :width: 500px



Multiprocessing Example
------------------------

The following code demonstrates how to use `concurrent.futures.ProcessPoolExecutor` to parallelize the Mandelbrot set generation across multiple processes.

This example prioritizes clarity over efficiency. In practice, transferring large NumPy arrays between processes can be expensive. Defining shared-memory arrays or creating arrays within each process may be more efficient implementation.


.. code-block:: python

    from concurrent.futures import ProcessPoolExecutor

    import numpy as np
    from numpy.typing import NDArray


    def mandelbrot_block(
        c_block: NDArray[np.complex128], max_iter: int
    ) -> NDArray[np.int64]:
        z = np.zeros(c_block.shape, dtype=np.complex128)
        steps = np.zeros(c_block.shape, dtype=np.int64)

        for _ in range(max_iter):
            mask = np.abs(z) <= 2
            z[mask] = z[mask] * z[mask] + c_block[mask]
            steps[mask] += 1
        return steps


    def mandelbrot_set(
        arr: NDArray[np.complex128],
        max_iter: int,
        n_workers: int,
    ) -> NDArray[np.int64]:
        n_workers = min(n_workers, arr.size)
        arrs = np.array_split(arr, n_workers)

        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = [
                pool.submit(mandelbrot_block, _arr, max_iter) for _arr in arrs
            ]
            results = [future.result() for future in futures]

        return np.concatenate(results)


    if __name__ == '__main__':

        xmin, xmax, ymin, ymax = -2.0, 1.0, -1.5, 1.5
        nx, ny = 800, 800
        max_iter = 10000
        n_workers = 10

        real = np.linspace(xmin, xmax, nx, dtype=np.float64)
        imag = np.linspace(ymin, ymax, ny, dtype=np.float64)

        arr = (real[:, np.newaxis] + 1j * imag[np.newaxis, :]).ravel()
        mandelbrot_image = mandelbrot_set(arr, max_iter, n_workers)
        mandelbrot_image = mandelbrot_image.reshape((nx, ny))


Multithreading Example
----------------------

Setup
+++++

Install a free-threaded build Python
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before running the multithreading example, ensure you have a free-threaded build of Python 3.13 or later.
For example, if you use ``pyenv``, you can install it and create a virtual environment naming ``numpy-multithreading`` as follows:

.. code-block:: bash

    pyenv install 3.14t-dev
    pyenv virtualenv 3.14t-dev numpy-multithreading
    pyenv activate numpy-multithreading


According to the `Python documentation <https://docs.python.org/3/howto/free-threading-python.html>`__, there are several ways to verify if your Python build is free-threaded. 

* Run ``python -VV`` in your terminal and check ``free-threading build`` is shown
* Check the value of `sys._is_gil_enabled()` in a Python shell, which should return `False`.


Install NumPy from Source
~~~~~~~~~~~~~~~~~~~~~~~~~~

NumPy includes ongoing work to improve compatibility and thread-safety when used with free-threaded Python builds.
(See https://github.com/numpy/numpy/issues/30494 for details.)

Thus, to utilize multithreading parallelism fully, we may need to build NumPy from source.
Please refer to the :ref:`Building from source to use NumPy <building-from-source>` section for detailed instructions.


Code Example
++++++++++++

The following code demonstrates how to use `concurrent.futures.ThreadPoolExecutor` to parallelize the Mandelbrot set generation across multiple threads.

This implementation shares several arrays between threads. For example, ``SHARED_readonly_arr`` is a read-only array that holds the complex numbers to be evaluated, and ``SHARED_updating_steps`` is an array that holds the number of iterations for each point.



.. code-block:: python

    import sys
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np


    def mandelbrot_block(start: int, stop: int, max_iter: int) -> None:
        z_target = np.zeros(stop - start, dtype=np.complex128)

        indexes = slice(start, stop)
        arr_target = SHARED_readonly_arr[indexes]
        steps_target = SHARED_updating_steps[indexes]

        threshold = 2.0
        for _ in range(max_iter):
            mask = np.abs(z_target) <= threshold
            z_target[mask] = z_target[mask] * z_target[mask] + arr_target[mask]
            steps_target[mask] += 1
        SHARED_updating_steps[indexes] = steps_target
        return None


    def mandelbrot_set(
        total_size: int,
        max_iter: int,
        n_workers: int,
    ) -> None:

        chunksize = total_size // n_workers
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = [
                pool.submit(
                    mandelbrot_block, start, min(start + chunksize, total_size), max_iter
                )
                for start in range(0, total_size, chunksize)
            ]
            _ = [future.result() for future in futures]


    if __name__ == '__main__':

        print("Python version is free-threaded:", not sys._is_gil_enabled())
        assert not sys._is_gil_enabled()

        xmin, xmax, ymin, ymax = -2.0, 1.0, -1.5, 1.5
        nx, ny = 800, 800
        max_iter = 10000
        n_workers = 10

        real = np.linspace(xmin, xmax, nx, dtype=np.float64)
        imag = np.linspace(ymin, ymax, ny, dtype=np.float64)

        SHARED_readonly_arr = (real[:, np.newaxis] + 1j * imag[np.newaxis, :]).ravel()
        SHARED_readonly_arr.flags.writeable = False

        SHARED_updating_steps = np.zeros(SHARED_readonly_arr.shape, dtype=np.int64)

        mandelbrot_set(SHARED_readonly_arr.size, max_iter, n_workers)
        mandelbrot_image = SHARED_updating_steps.reshape((nx, ny))



.. _basics.performant_code.other_tips:

Other Micro Tips for Writing Performant NumPy Code
==================================================


Use tuples for creating arrays
------------------------------

When creating NumPy arrays, using tuples instead of lists can lead to slight performance improvements.

.. code-block:: python

    # Using a list
    arr_list = np.array([1, 2, 3, 4, 5])

    # Using a tuple
    arr_tuple = np.array((1, 2, 3, 4, 5))


This is reported in https://github.com/numpy/numpy/pull/30514#issuecomment-3716554540

Using tuples avoids critical sections in the array creation code path, resulting in faster execution. Critical sections are portions of code that must be executed by only one thread at a time to prevent data corruption. In NumPy, critical sections are managed using ``NPY_BEGIN_CRITICAL_SECTION_SEQUENCE_FAST`` and ``NPY_END_CRITICAL_SECTION_SEQUENCE_FAST``. You can refer to those macros if necessary for more details.



Avoids module attribute lookups
-------------------------------

Module attribute lookups can introduce overhead. To minimize this overhead, you can assign frequently used functions or attributes to local variables.
This is reported in https://github.com/numpy/numpy/issues/30494#issuecomment-3700169826


.. code-block:: python

    import numpy as np

    # Avoiding module attribute lookups
    sin = np.sin

    arr = array([1, 2, 3, 4, 5])
    value = sin(arr)



.. _basics.performant_code.third_party_libraries:


Third Party Libraries for Multi-Core Processing
===============================================

In many practical scenarios, third-party libraries can provide more convenient and efficient solutions than using Python's standard libraries.
We will briefly introduce some third party libraries at the end of this introduction.


Dask
----

Dask is an open-source library that provides parallel compuing features not only for a single machine but also for a cluster of machines.
It also provides ``DaskArray`` which has a similar API to NumPy's ``ndarray``. If you are familiar with NumPy, you can easily get started with ``DaskArray``.

* Dask Documentaion: https://docs.dask.org/en/stable/
* Dask GitHub Repository: https://github.com/dask/dask
