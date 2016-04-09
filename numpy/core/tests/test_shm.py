import multiprocessing

import numpy as np
from numpy.testing import TestCase, assert_equal, run_module_suite


numtypes = [np.float64, np.int32, np.float32, np.uint8, np.complex]


# In order to pass a shm array to a child process, we need to use module-level
# global variables ...
pool_array = None


# ... and global functions
def _modify_array(idx):
    pool_array[idx] = idx ** 2


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
    def test_two_subprocesses_no_pickle(self):
        orig = np.zeros(4, float) + 8
        a = np.shm.copy(orig)

        def modify_array(a):
            a[0] = 1
            a[1] = 2
            a[2] = 3

        p = multiprocessing.Process(target=modify_array, args=(a,))
        p.start()
        p.join()

        assert_equal(a, np.array([1, 2, 3, 8], dtype=float))


    def test_pool(self):
        global pool_array
        pool_array = np.shm.zeros(4, dtype=float)
        arr = pool_array

        pool = multiprocessing.Pool(2)
        pool.map(_modify_array, range(pool_array.size))
        pool.close()
        pool.join()

        assert_equal(arr, np.array([0, 1, 4, 9], dtype=float))


if __name__ == "__main__":
    run_module_suite()
