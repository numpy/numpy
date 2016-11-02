from __future__ import division, absolute_import, print_function

import sys
import itertools

import numpy as np
from numpy.testing import run_module_suite, assert_, assert_raises, assert_equal

from numpy.core.multiarray_tests import solve_diophantine, internal_overlap
from numpy.lib.stride_tricks import as_strided
from numpy.compat import long

if sys.version_info[0] >= 3:
    xrange = range


ndims = 2
size = 10
shape = tuple([size] * ndims)

MAY_SHARE_BOUNDS = 0
MAY_SHARE_EXACT = -1


def _indices_for_nelems(nelems):
    """Returns slices of length nelems, from start onwards, in direction sign."""

    if nelems == 0:
        return [size // 2]  # int index

    res = []
    for step in (1, 2):
        for sign in (-1, 1):
            start = size // 2 - nelems * step * sign // 2
            stop = start + nelems * step * sign
            res.append(slice(start, stop, step * sign))

    return res


def _indices_for_axis():
    """Returns (src, dst) pairs of indices."""

    res = []
    for nelems in (0, 2, 3):
        ind = _indices_for_nelems(nelems)

        # no itertools.product available in Py2.4
        res.extend([(a, b) for a in ind for b in ind])  # all assignments of size "nelems"

    return res


def _indices(ndims):
    """Returns ((axis0_src, axis0_dst), (axis1_src, axis1_dst), ... ) index pairs."""

    ind = _indices_for_axis()

    # no itertools.product available in Py2.4

    res = [[]]
    for i in range(ndims):
        newres = []
        for elem in ind:
            for others in res:
                newres.append([elem] + others)
        res = newres

    return res


def _check_assignment(srcidx, dstidx):
    """Check assignment arr[dstidx] = arr[srcidx] works."""

    arr = np.arange(np.product(shape)).reshape(shape)

    cpy = arr.copy()

    cpy[dstidx] = arr[srcidx]
    arr[dstidx] = arr[srcidx]

    assert_(np.all(arr == cpy),
            'assigning arr[%s] = arr[%s]' % (dstidx, srcidx))


def test_overlapping_assignments():
    """Test automatically generated assignments which overlap in memory."""

    inds = _indices(ndims)

    for ind in inds:
        srcidx = tuple([a[0] for a in ind])
        dstidx = tuple([a[1] for a in ind])

        yield _check_assignment, srcidx, dstidx


def test_diophantine_fuzz():
    # Fuzz test the diophantine solver
    rng = np.random.RandomState(1234)

    max_int = np.iinfo(np.intp).max

    for ndim in range(10):
        feasible_count = 0
        infeasible_count = 0

        min_count = 500//(ndim + 1)

        numbers = []
        while min(feasible_count, infeasible_count) < min_count:
            # Ensure big and small integer problems
            A_max = 1 + rng.randint(0, 11, dtype=np.intp)**6
            U_max = rng.randint(0, 11, dtype=np.intp)**6

            A_max = min(max_int, A_max)
            U_max = min(max_int-1, U_max)

            A = tuple(int(rng.randint(1, A_max+1, dtype=np.intp))
                      for j in range(ndim))
            U = tuple(int(rng.randint(0, U_max+2, dtype=np.intp))
                      for j in range(ndim))

            b_ub = min(max_int-2, sum(a*ub for a, ub in zip(A, U)))
            b = rng.randint(-1, b_ub+2, dtype=np.intp)

            if ndim == 0 and feasible_count < min_count:
                b = 0

            X = solve_diophantine(A, U, b)

            if X is None:
                # Check the simplified decision problem agrees
                X_simplified = solve_diophantine(A, U, b, simplify=1)
                assert_(X_simplified is None, (A, U, b, X_simplified))

                # Check no solution exists (provided the problem is
                # small enough so that brute force checking doesn't
                # take too long)
                try:
                    ranges = tuple(xrange(0, a*ub+1, a) for a, ub in zip(A, U))
                except OverflowError:
                    # xrange on 32-bit Python 2 may overflow
                    continue

                size = 1
                for r in ranges:
                    size *= len(r)
                if size < 100000:
                    assert_(not any(sum(w) == b for w in itertools.product(*ranges)))
                    infeasible_count += 1
            else:
                # Check the simplified decision problem agrees
                X_simplified = solve_diophantine(A, U, b, simplify=1)
                assert_(X_simplified is not None, (A, U, b, X_simplified))

                # Check validity
                assert_(sum(a*x for a, x in zip(A, X)) == b)
                assert_(all(0 <= x <= ub for x, ub in zip(X, U)))
                feasible_count += 1


def test_diophantine_overflow():
    # Smoke test integer overflow detection
    max_intp = np.iinfo(np.intp).max
    max_int64 = np.iinfo(np.int64).max

    if max_int64 <= max_intp:
        # Check that the algorithm works internally in 128-bit;
        # solving this problem requires large intermediate numbers
        A = (max_int64//2, max_int64//2 - 10)
        U = (max_int64//2, max_int64//2 - 10)
        b = 2*(max_int64//2) - 10

        assert_equal(solve_diophantine(A, U, b), (1, 1))


def check_may_share_memory_exact(a, b):
    got = np.may_share_memory(a, b, max_work=MAY_SHARE_EXACT)

    assert_equal(np.may_share_memory(a, b),
                 np.may_share_memory(a, b, max_work=MAY_SHARE_BOUNDS))

    a.fill(0)
    b.fill(0)
    a.fill(1)
    exact = b.any()

    err_msg = ""
    if got != exact:
        err_msg = "    " + "\n    ".join([
            "base_a - base_b = %r" % (a.__array_interface__['data'][0] - b.__array_interface__['data'][0],),
            "shape_a = %r" % (a.shape,),
            "shape_b = %r" % (b.shape,),
            "strides_a = %r" % (a.strides,),
            "strides_b = %r" % (b.strides,),
            "size_a = %r" % (a.size,),
            "size_b = %r" % (b.size,)
        ])

    assert_equal(got, exact, err_msg=err_msg)


def test_may_share_memory_manual():
    # Manual test cases for may_share_memory

    # Base arrays
    xs0 = [
        np.zeros([13, 21, 23, 22], dtype=np.int8),
        np.zeros([13, 21, 23*2, 22], dtype=np.int8)[:,:,::2,:]
    ]

    # Generate all negative stride combinations
    xs = []
    for x in xs0:
        for ss in itertools.product(*(([slice(None), slice(None, None, -1)],)*4)):
            xp = x[ss]
            xs.append(xp)

    for x in xs:
        # The default is a simple extent check
        assert_(np.may_share_memory(x[:,0,:], x[:,1,:]))
        assert_(np.may_share_memory(x[:,0,:], x[:,1,:], max_work=None))

        # Exact checks
        check_may_share_memory_exact(x[:,0,:], x[:,1,:])
        check_may_share_memory_exact(x[:,::7], x[:,3::3])

        try:
            xp = x.ravel()
            if xp.flags.owndata:
                continue
            xp = xp.view(np.int16)
        except ValueError:
            continue

        # 0-size arrays cannot overlap
        check_may_share_memory_exact(x.ravel()[6:6],
                                     xp.reshape(13, 21, 23, 11)[:,::7])

        # Test itemsize is dealt with
        check_may_share_memory_exact(x[:,::7],
                                     xp.reshape(13, 21, 23, 11))
        check_may_share_memory_exact(x[:,::7],
                                     xp.reshape(13, 21, 23, 11)[:,3::3])
        check_may_share_memory_exact(x.ravel()[6:7],
                                     xp.reshape(13, 21, 23, 11)[:,::7])

    # Check unit size
    x = np.zeros([1], dtype=np.int8)
    check_may_share_memory_exact(x, x)
    check_may_share_memory_exact(x, x.copy())


def check_may_share_memory_easy_fuzz(get_max_work, same_steps, min_count):
    # Check that overlap problems with common strides are solved with
    # little work.
    x = np.zeros([17,34,71,97], dtype=np.int16)

    rng = np.random.RandomState(1234)

    def random_slice(n, step):
        start = rng.randint(0, n+1, dtype=np.intp)
        stop = rng.randint(start, n+1, dtype=np.intp)
        if rng.randint(0, 2, dtype=np.intp) == 0:
            stop, start = start, stop
            step *= -1
        return slice(start, stop, step)

    feasible = 0
    infeasible = 0

    while min(feasible, infeasible) < min_count:
        steps = tuple(rng.randint(1, 11, dtype=np.intp)
                      if rng.randint(0, 5, dtype=np.intp) == 0 else 1
                      for j in range(x.ndim))
        if same_steps:
            steps2 = steps
        else:
            steps2 = tuple(rng.randint(1, 11, dtype=np.intp)
                           if rng.randint(0, 5, dtype=np.intp) == 0 else 1
                           for j in range(x.ndim))

        t1 = np.arange(x.ndim)
        rng.shuffle(t1)

        t2 = np.arange(x.ndim)
        rng.shuffle(t2)

        s1 = tuple(random_slice(p, s) for p, s in zip(x.shape, steps))
        s2 = tuple(random_slice(p, s) for p, s in zip(x.shape, steps2))
        a = x[s1].transpose(t1)
        b = x[s2].transpose(t2)

        bounds_overlap = np.may_share_memory(a, b)
        may_share_answer = np.may_share_memory(a, b)
        easy_answer = np.may_share_memory(a, b, max_work=get_max_work(a, b))
        exact_answer = np.may_share_memory(a, b, max_work=MAY_SHARE_EXACT)

        if easy_answer != exact_answer:
            # assert_equal is slow...
            assert_equal(easy_answer, exact_answer, err_msg=repr((s1, s2)))

        if may_share_answer != bounds_overlap:
            assert_equal(may_share_answer, bounds_overlap,
                         err_msg=repr((s1, s2)))

        if bounds_overlap:
            if exact_answer:
                feasible += 1
            else:
                infeasible += 1


def test_may_share_memory_easy_fuzz():
    # Check that overlap problems with common strides are always
    # solved with little work.

    check_may_share_memory_easy_fuzz(get_max_work=lambda a, b: 1,
                                     same_steps=True,
                                     min_count=2000)


def test_may_share_memory_harder_fuzz():
    # Overlap problems with not necessarily common strides take more
    # work.
    #
    # The work bound below can't be reduced much. Harder problems can
    # also exist but not be detected here, as the set of problems
    # comes from RNG.

    check_may_share_memory_easy_fuzz(get_max_work=lambda a, b: max(a.size, b.size)//2,
                                     same_steps=False,
                                     min_count=2000)


def test_shares_memory_api():
    x = np.zeros([4, 5, 6], dtype=np.int8)

    assert_equal(np.shares_memory(x, x), True)
    assert_equal(np.shares_memory(x, x.copy()), False)

    a = x[:,::2,::3]
    b = x[:,::3,::2]
    assert_equal(np.shares_memory(a, b), True)
    assert_equal(np.shares_memory(a, b, max_work=None), True)
    assert_raises(np.TooHardError, np.shares_memory, a, b, max_work=1)
    assert_raises(np.TooHardError, np.shares_memory, a, b, max_work=long(1))


def test_may_share_memory_bad_max_work():
    x = np.zeros([1])
    assert_raises(OverflowError, np.may_share_memory, x, x, max_work=10**100)
    assert_raises(OverflowError, np.shares_memory, x, x, max_work=10**100)


def test_internal_overlap_diophantine():
    def check(A, U, exists=None):
        X = solve_diophantine(A, U, 0, require_ub_nontrivial=1)

        if exists is None:
            exists = (X is not None)

        if X is not None:
            assert_(sum(a*x for a, x in zip(A, X)) == sum(a*u//2 for a, u in zip(A, U)))
            assert_(all(0 <= x <= u for x, u in zip(X, U)))
            assert_(any(x != u//2 for x, u in zip(X, U)))

        if exists:
            assert_(X is not None, repr(X))
        else:
            assert_(X is None, repr(X))

    # Smoke tests
    check((3, 2), (2*2, 3*2), exists=True)
    check((3*2, 2), (15*2, (3-1)*2), exists=False)


def test_internal_overlap_slices():
    # Slicing an array never generates internal overlap

    x = np.zeros([17,34,71,97], dtype=np.int16)

    rng = np.random.RandomState(1234)

    def random_slice(n, step):
        start = rng.randint(0, n+1, dtype=np.intp)
        stop = rng.randint(start, n+1, dtype=np.intp)
        if rng.randint(0, 2, dtype=np.intp) == 0:
            stop, start = start, stop
            step *= -1
        return slice(start, stop, step)

    cases = 0
    min_count = 5000

    while cases < min_count:
        steps = tuple(rng.randint(1, 11, dtype=np.intp)
                      if rng.randint(0, 5, dtype=np.intp) == 0 else 1
                      for j in range(x.ndim))
        t1 = np.arange(x.ndim)
        rng.shuffle(t1)
        s1 = tuple(random_slice(p, s) for p, s in zip(x.shape, steps))
        a = x[s1].transpose(t1)

        assert_(not internal_overlap(a))
        cases += 1


def check_internal_overlap(a, manual_expected=None):
    got = internal_overlap(a)

    # Brute-force check
    m = set()
    ranges = tuple(xrange(n) for n in a.shape)
    for v in itertools.product(*ranges):
        offset = sum(s*w for s, w in zip(a.strides, v))
        if offset in m:
            expected = True
            break
        else:
            m.add(offset)
    else:
        expected = False

    # Compare
    if got != expected:
        assert_equal(got, expected, err_msg=repr((a.strides, a.shape)))
    if manual_expected is not None and expected != manual_expected:
        assert_equal(expected, manual_expected)
    return got


def test_internal_overlap_manual():
    # Stride tricks can construct arrays with internal overlap

    # We don't care about memory bounds, the array is not
    # read/write accessed
    x = np.arange(1).astype(np.int8)

    # Check low-dimensional special cases

    check_internal_overlap(x, False) # 1-dim
    check_internal_overlap(x.reshape([]), False) # 0-dim

    a = as_strided(x, strides=(3, 4), shape=(4, 4))
    check_internal_overlap(a, False)

    a = as_strided(x, strides=(3, 4), shape=(5, 4))
    check_internal_overlap(a, True)

    a = as_strided(x, strides=(0,), shape=(0,))
    check_internal_overlap(a, False)

    a = as_strided(x, strides=(0,), shape=(1,))
    check_internal_overlap(a, False)

    a = as_strided(x, strides=(0,), shape=(2,))
    check_internal_overlap(a, True)

    a = as_strided(x, strides=(0, -9993), shape=(87, 22))
    check_internal_overlap(a, True)

    a = as_strided(x, strides=(0, -9993), shape=(1, 22))
    check_internal_overlap(a, False)

    a = as_strided(x, strides=(0, -9993), shape=(0, 22))
    check_internal_overlap(a, False)


def test_internal_overlap_fuzz():
    # Fuzz check; the brute-force check is fairly slow

    x = np.arange(1).astype(np.int8)

    overlap = 0
    no_overlap = 0
    min_count = 100

    rng = np.random.RandomState(1234)

    while min(overlap, no_overlap) < min_count:
        ndim = rng.randint(1, 4, dtype=np.intp)

        strides = tuple(rng.randint(-1000, 1000, dtype=np.intp)
                        for j in range(ndim))
        shape = tuple(rng.randint(1, 30, dtype=np.intp)
                      for j in range(ndim))

        a = as_strided(x, strides=strides, shape=shape)
        result = check_internal_overlap(a)

        if result:
            overlap += 1
        else:
            no_overlap += 1


def test_non_ndarray_inputs():
    # Regression check for gh-5604

    class MyArray(object):
        def __init__(self, data):
            self.data = data

        @property
        def __array_interface__(self):
            return self.data.__array_interface__

    class MyArray2(object):
        def __init__(self, data):
            self.data = data

        def __array__(self):
            return self.data

    for cls in [MyArray, MyArray2]:
        x = np.arange(5)

        assert_(np.may_share_memory(cls(x[::2]), x[1::2]))
        assert_(not np.shares_memory(cls(x[::2]), x[1::2]))

        assert_(np.shares_memory(cls(x[1::3]), x[::2]))
        assert_(np.may_share_memory(cls(x[1::3]), x[::2]))


if __name__ == "__main__":
    run_module_suite()
