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
to stay that way even if you use the threading module due to a CPython feature
called GIL (global interpreter lock). GIL ensures that only one thread is active
at a time, so threre is no true multitasking.

.. code-block:: python

    # Extra definitions to the multitasking-unaware setup -
    # we have to prepare the output
    out_arr = np.zeros(in_arr.shape, int)

    # And finally the function wrapper
    def land_howmany9(index):
        result = howmany9(in_arr[index])
        out_arr[index] = result

    # DEFINITIONS END HERE, now comes the computation.

    import threading

    for index in range(in_arr.size):
        thread = threading.Thread(target=land_howmany9, args=(index,))
        thread.start()

    # Ideally, we should join all threads, not just the last one
    thread.join()

The multiprocessing approach
============================

The solution could be the multiprocessing module.
Instead of running your calculation in separate threads, you can delegate it
to separate processes.
(A process created solely for the purpose of doing the calculation
is commonly reffered to as the worker process.)
Judging by APIs, one could think that one can use processes in the same way
as threads and since there is no GIL,
running multiple worker processes at the same time implies true multitasking.

This is true, but there is a catch.
Processes can't share memory easily, and if your problem is more data-heavy
than CPU-heavy, the overhead of copying the source data to the worker process
and the result data to the main process can easily cripple
the overall performance below single-threaded and single-process calculation.
Not to speak about duplication of data, since in one moment,
the source data has to be known to all processes.
If your source dataset is large, you can run out of memory, too.

Fortunatelly, there is a solution to this problem.
Luckily, processes can somehow share memory.
Moreover it is especially easy if they are related to each other,
which is the case of worker processes
(since they are created from the main process).
Sharing data between related processes is taken care of
for all supported platforms by Python
and numpy's ``shm`` module uses this functionality.
It enables you to create arrays that are readable and writable both in main
and worker processes without additional overhead.
Moreover, it supports byte-alignment of arrays for cases when you would like to
use them with libraries that benefit from that (e.g. pyfftw).

Threading-like approach
-----------------------

If you want to use the threading-like interface to multiprocessing,
you can just swap the normal numpy arrays with shm numpy arrays and off you go:

.. code-block:: python

    # no extra definitions compared to the threading version

    # DEFINITIONS END HERE, now comes the computation.

    import multiprocessing

    for index in range(in_arr.size):
        process = multiprocessing.Process(target=land_howmany9, args=(index,))
        process.start()

    # Ideally, we should join all processes, not just the last one
    process.join()

Multiprocessing pool
--------------------

However, if you want to use the multiprocessing.Pool and it's
``map`` functionality, things get more complicated.
You need to pass the data to the pool's process, but for reasons
beyond the scope of this tutorial, you can't do it as a parameter
to the ``map`` (or other) method.
The best thing to use is the Pool's initializer, which is capable
to place the input/output data in a place where worker processes can read from,
and that place is the module's global namespace.
Therefore, the code would look like this:

.. code-block:: python

    # Extra definitions to the multitasking-unaware setup -
    # we have to prepare the output

    # IN_ARRAY and OUT_ARRAY are module's top-level variables
    # So define the there, not inside a function
    IN_ARRAY = None
    OUT_ARRAY = None

    def pool_init(in_array):
        global IN_ARRAY, OUT_ARRAY
        IN_ARRAY = in_array
        OUT_ARRAY = np.zeros(in_arr.shape, int)

    # different from the threading version since here,
    # referring to local variables wouldn't work
    def land_howmany9(index):
        result = howmany9(IN_ARR[index])
        OUT_ARR[index] = result


    # DEFINITIONS END HERE, now comes the computation.

    import multiprocessing

    pool = multiprocessing.Pool(multiprocessing.cpu_count(),
                                pool_init, (in_array, ))
    pool.map(range(in_array.size))
    pool.close()
    pool.join()

    # look up the result in OUT_ARR

"""
