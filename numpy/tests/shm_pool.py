# Let's assume that the name of this module is "calculate_9s"
import time

import numpy as np
import multiprocessing


# IN_ARRAY and OUT_ARRAY are module's top-level variables
# So define the there, not inside a function
IN_ARRAY = None
OUT_ARRAY = None
COUNTER = None


# We define the function whose execution we  want to parallelize
def howmany9(num):
    """
    Given a number, return how many 9's are in it.
    """
    string = str(num)
    result = string.count("9")
    return result


def _pool_init(in_array, out_array, counter):
    global IN_ARRAY, OUT_ARRAY, COUNTER
    IN_ARRAY = in_array
    OUT_ARRAY = out_array
    COUNTER = counter


# Different from the threading version since here,
# referring to local variables wouldn't work, so the parameter is an
# index and we access the input and output via a global variable.
def _wrapped_for_pool(index):
    """
    Wrapped function, saves the result to the corresponding element of the
    output array and increments the counter.
    """
    result = howmany9(IN_ARRAY[index])
    OUT_ARRAY[index] = result
    # We protect COUNTER by its multiprocessing lock
    with COUNTER:
        # We give a feedback that is reachable from the main process.
        COUNTER.value = COUNTER.value + 1


def carry_out_computation(in_array, out_array=None, update_period=2):
    """
    Given the input, output array and a function, iterate through the input
    using ``multiprocessing.Pool``, save results to the output and
    update the user if needed.
    """
    arr_size = in_array.size
    if out_array is None:
        out_array = np.shm.zeros(arr_size, int)
        # Initialize to -1, so it is obvious if some return values are skipped
        out_array -= 1
        # vvv TODO: This doesn't work vvv
        # out_array = np.shm.zeros(arr_size, int) - 1
    counter = multiprocessing.Value("i")
    pool = multiprocessing.Pool(None, _pool_init,
                                (in_array, out_array, counter))
    result = pool.map_async(_wrapped_for_pool, range(arr_size))
    pool.close()

    while not result.ready():
        time.sleep(update_period)
        # We only peek at the counter, so no lock is needed
        # Also note that we don't use the seemingly global COUNTER, but counter
        percent_complete = 100 * counter.value / float(arr_size)
        print("Done: %02d%%" % percent_complete)

    pool.join()
    if not result.successful():
        raise RuntimeError(
            "The multi-core processing of the task was not successful "
            "(unfortunatelly, it is impossible to recover failure details)."
        )

    return out_array
