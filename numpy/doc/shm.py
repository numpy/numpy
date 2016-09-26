"""

====================
Shared memory arrays
====================

Many numerical problems can be decomposed into sub-problems that can be solved
in a parallel manner. Consider an example, when you have an array
of random float values, and you want to convert them to a string
and find out how many 9's are in them.

There are various approaches:

The multitasking-unaware approach
=================================

Normally, you would go for something like

.. code-block:: python

    import numpy as np

    # We define the function
    def howmany9(num):
        string = str(num)
        result = string.count("9")
        return result

    # We create another function from it, aware that the input parameter is an array
    howmany9_vec = np.vectorize(howmany9)

    # Our input data
    in_arr = np.random.random(100)

    # DEFINITIONS END HERE, now comes the computation.
    # Output data follow
    out_arr = howmany9_vec(in_arr)

Although you can pass the array to a corresponding function, only one OS thread
will work on it. That's not what we want, we want true multitasking
(i.e. parallel execution).

The threading approach
======================

Another option is to use threads.

However, as long as you are using the CPython interpreter, it is going
to stay the same even if you use the :mod:`threading` module due to a
CPython feature called `GIL <https://wiki.python.org/moin/GlobalInterpreterLock>`_
(global interpreter lock). GIL ensures that only one thread is active
at a time, so there is no true multitasking.

.. code-block:: python

    import concurrent.futures
    import numpy as np


    def howmany9(num):
        string = str(num)
        result = string.count("9")
        return result

    # Our input data
    in_arr = np.random.random(100)
    # Extra definitions to the multitasking-unaware setup -
    # we have to prepare the output
    out_arr = np.zeros(in_arr.shape, int)

    # And finally the function wrapper.
    # It can be a function defined deeper than in the module's top level.
    def land_howmany9(index):
        result = howmany9(in_arr[index])
        out_arr[index] = result

    # DEFINITIONS END HERE, now comes the computation.

    threads = concurrent.futures.ThreadPoolExecutor()
    threads.map(land_howmany9, range(in_arr.size))
    threads.shutdown()


The multiprocessing approach
============================

The solution could be the :mod:`multiprocessing` module.
Instead of running your calculation in separate threads, you can delegate it
to separate processes.
(A process created solely for the purpose of doing the calculation
is commonly referred to as the worker process.)
Judging by APIs, one could think that one can use processes in the same way
as threads and since there is no GIL,
running multiple worker processes at the same time implies true multitasking.

This is true, but there is a catch.
Processes can't share memory easily, and if your problem is more data-heavy
than CPU-heavy, the overhead of copying the source data to the worker process
and the result data to the main process can easily cripple
the overall performance below single-threaded and single-process calculation.
Not to speak about duplication of data, since in one moment,
the source data has to be copied to all processes.
If your source dataset is large, you can run out of memory, too.

Fortunately, there is a solution to this problem.
Processes can share memory, and this is especially easy to do if
if they are related to each other.
This is the case of worker processes, which are created from the main process
and therefore are its children.
Sharing data between related processes is taken care of
for all supported platforms by Python itself
and this module uses this functionality.

If you examine the mod:`multiprocessing` documentation, you find out
that there are two means of sharing data: The :class:`multiprocessing.Value` and
:class:`multiprocessing.Array`.
The former is a single numeric type and the latter is a vector of numbers
(not to be confused with ``numpy`` arrays --- it is just a dumb data container,
one can't add two Arrays together as one is used to with ``numpy`` arrays).

The :mod:`numpy.shm` adds the numpy array type.
It enables you to create numpy arrays that are readable and writable
both in main and worker processes without major additional overhead.
Moreover, it supports byte-alignment of arrays for cases when you would like to
use them with libraries that benefit from that (e.g. :mod:`pyfftw`).
Shared-memory arrays are handy if you implement a producer-consumer pattern,
when the producer produces (and consumer consumes) a ``numpy`` array.
Code-wise, you have to use same idioms when working with those
as if you were working with ordinary instances of
:class:`multiprocessing.Array`.

.. warning::

    If you are not familiar with the Python :mod:`multiprocessing` module
    and its idioms, pay attention to these points:

    *  The shm numpy array has to exist before the process is created / started?
    *  The same as with threading, be sure that you protect your array
       with multiprocessing locks (if applicable).
    *  Multiprocessing is more difficult to get right on Windows than on Unixes
       (such as Linux, OS X), so don't get carried away by "works for me".
    *  The shm array can't be pickled.
       If you want values it holds to be pickled, pass the shm array
       to the :func:`numpy.array` function and pickle the result.

Threading-like approach
-----------------------

If you want to use the low-level threading-like interface to multiprocessing,
you can just swap the normal numpy arrays with shm numpy arrays and off you go:

.. code-block:: python

    out_arr = np.shm.zeros(in_arr.shape, int)
    # no other extra definitions compared to the threading version
    # except land_howmany9 must be module's top-level function.
    def land_howmany9(input_arr, output_arr, index):
        result = howmany9(input_arr[index])
        output_arr_arr[index] = result

    # DEFINITIONS END HERE, now comes the computation.

    import multiprocessing

    for index in range(in_arr.size):
        process = multiprocessing.Process(target=land_howmany9,
                                          args=(in_arr, out_arr, index,))
        process.start()

    # Ideally, we should join all processes, not just the last one
    process.join()


The above example illustrates the pitfall of the approach:

* You don't want to launch all processes at once, you want to keep
  the number of worker processes below the number of CPU cores.
* In order to be sure, you should check that each process has finished,
  and handle possible exceptions that might have occured.

It is likely that the :class:`multiprocessing.Pool` class has been introduced
exactly for that reason --- you tell it how many jobs can run at once,
you can find out easily when everything got finished and
if there were any problems.

Multiprocessing pool
--------------------

If you want to use the more convenient :class:`multiprocessing.Pool` class
and it's :meth:`multiprocessing.Pool.map_async` method (or another),
things get more complicated.
You need to pass the data to the pool's process, but for reasons
beyond the scope of this tutorial, you can't do it as a parameter
to the :meth:`multiprocessing.Pool.map_async` (or other ``Pool``) method.

The only way is to use the Pool's initializer, which is capable
to place the input/output data in a place where worker processes can read from,
and that place is the module's global namespace.
This is rightfully regarded as a very bad practice, therefore it is
highly advisable that you put the calculation in a dedicated module.

The source code of the module (let's assume it is called ``calculate_9s``)
would look like this:


.. literalinclude:: ../../../numpy/tests/shm_pool.py
    :language: python

Then, you obtain the result by importing it and calling
the ``carry_out_computation`` function:

.. code-block:: python

    import calculate_9s

    in_arr = np.shm.zeros(100)
    in_arr += np.random.random(100)
    out_arr = calculate_9s.carry_out_computation(in_arr)
"""
