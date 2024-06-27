import concurrent.futures
import threading

import numpy as np
import pytest

from numpy.testing import IS_WASM

if IS_WASM:
    pytest.skip(allow_module_level=True, reason="no threading support in wasm")


def run_threaded(func, iters, pass_count=False):
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as tpe:
        if pass_count:
            futures = [tpe.submit(func, i) for i in range(iters)]
        else:
            futures = [tpe.submit(func) for _ in range(iters)]
        for f in futures:
            f.result()


def test_parallel_randomstate_creation():
    # if the coercion cache is enabled and not thread-safe, creating
    # RandomState instances simultaneously leads to a data race
    def func(seed):
        np.random.RandomState(seed)

    run_threaded(func, 500, pass_count=True)

def test_parallel_ufunc_execution():
    # if the loop data cache or dispatch cache are not thread-safe
    # computing ufuncs simultaneously in multiple threads leads
    # to a data race that causes crashes or spurious exceptions
    def func():
        arr = np.random.random((25,))
        np.isnan(arr)

    run_threaded(func, 500)

    # see gh-26690
    NUM_THREADS = 50

    b = threading.Barrier(NUM_THREADS)

    a = np.ones(1000)

    def f():
        b.wait()
        return a.sum()

    threads = [threading.Thread(target=f) for _ in range(NUM_THREADS)]

    [t.start() for t in threads]
    [t.join() for t in threads]

def test_temp_elision_thread_safety():
    amid = np.ones(50000)
    bmid = np.ones(50000)
    alarge = np.ones(1000000)
    blarge = np.ones(1000000)

    def func(count):
        if count % 4 == 0:
            (amid * 2) + bmid
        elif count % 4 == 1:
            (amid + bmid) - 2
        elif count % 4 == 2:
            (alarge * 2) + blarge
        else:
            (alarge + blarge) - 2

    run_threaded(func, 100, pass_count=True)


def test_eigvalsh_thread_safety():
    # if lapack isn't thread safe this will randomly segfault or error
    # see gh-24512
    rng = np.random.RandomState(873699172)
    matrices = (
        rng.random((5, 10, 10, 3, 3)),
        rng.random((5, 10, 10, 3, 3)),
    )

    run_threaded(lambda i: np.linalg.eigvalsh(matrices[i]), 2,
                 pass_count=True)
