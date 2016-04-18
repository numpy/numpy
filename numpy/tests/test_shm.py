import multiprocessing

import numpy as np
import concurrent.futures as futures
from numpy.testing import TestCase, assert_equal, run_module_suite
from nose.exc import SkipTest

import shm_pool as pool


numtypes = [np.float64, np.int32, np.float32, np.uint8, np.complex]


# In order to pass a shm array to a child process, we need to use module-level
# global variables ...
pool_array = None


# ... and global functions


def init_pool(arr):
    global pool_array
    pool_array = arr


def _modify_array_pool(idx):
    pool_array[idx] = idx ** 2


def _modify_array_normal(arr, index):
    arr[index] = index + 1


class TestCreation(TestCase):
    def test_shared_ones(self):
        for typestr in numtypes:
            shape = (10,)
            a = np.shm.ones(shape, dtype=typestr)
            assert_equal(a, np.ones(shape))

    def test_shared_zeros(self):
        for typestr in numtypes:
            shape = (10,)
            a = np.shm.zeros(shape, dtype=typestr)
            assert_equal(a, np.zeros(shape))

    def test_KiB_shared_zeros(self):
        for typestr in numtypes:
            shape = (2 ** 16, 8)
            a = np.shm.zeros(shape, dtype=typestr)
            assert_equal(a, np.zeros(shape))

    def test_MiB_shared_zeros(self):
        shape = (2 ** 17, 8, 8)
        a = np.shm.zeros(shape, dtype='uint8')
        assert_equal(a, np.zeros(shape))


class TestModification(TestCase):
    def test_processpool(self):
        raise SkipTest("This is even not supposed to work.")
        orig = np.zeros(4, float) + 8
        arr = np.shm.copy(orig)

        pool = futures.ProcessPoolExecutor()
        pool.map(_modify_array_normal, (arr,) * 2, (4, 1))
        pool.shutdown()

        assert_equal(arr, np.array([1, 2, 8, 8], dtype=float))

    def test_two_subprocesses_no_pickle(self):
        orig = np.zeros(4, float) + 8
        arr = np.shm.copy(orig)

        p = multiprocessing.Process(target=_modify_array_normal,
                                    args=(arr, 1))
        p.start()
        p.join()

        assert_equal(arr, np.array([8, 2, 8, 8], dtype=float))

    def test_pool(self):
        arr = np.shm.zeros(4, dtype=float)

        try:
            pool = multiprocessing.Pool(2, init_pool, (arr,))
        except OSError as exc:
            msg = ("Couldn't even instantiate 'multiprocessing.Pool' from "
                   "Python standard library ('{}'), no point in continuing."
                   .format(str(exc)))
            raise SkipTest(msg)
        pool.map(_modify_array_pool, range(arr.size))
        pool.close()
        pool.join()

        assert_equal(arr, np.array([0, 1, 4, 9], dtype=float))

    def test_doc_example(self):
        in_array = np.array((9, 19, 88, 1949, 99999, 18, 23, -9, -919, 9e99))
        result = pool.carry_out_computation(in_array, update_period=0.1)
        assert_equal(result, np.array([1, 1, 0, 2, 5, 0, 0, 1, 2, 3], dtype=int))


if __name__ == "__main__":
    run_module_suite()
