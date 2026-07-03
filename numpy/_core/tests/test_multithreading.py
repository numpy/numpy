import concurrent.futures
import inspect
import subprocess
import sys
import textwrap
import threading

import pytest

import numpy as np
from numpy._core import _rational_tests
from numpy._core.tests.test_stringdtype import random_unicode_string_list
from numpy.testing import IS_64BIT, IS_WASM
from numpy.testing._private.utils import run_subprocess, run_threaded

if IS_WASM:
    pytest.skip(allow_module_level=True, reason="no threading support in wasm")

pytestmark = pytest.mark.thread_unsafe(
    reason="tests in this module are already explicitly multi-threaded"
)

def test_parallel_randomstate():
    # if the coercion cache is enabled and not thread-safe, creating
    # RandomState instances simultaneously leads to a data race
    def func(seed):
        np.random.RandomState(seed)

    run_threaded(func, 500, pass_count=True)

    # seeding and setting state shouldn't race with generating RNG samples
    rng = np.random.RandomState()

    def func(seed):
        base_rng = np.random.RandomState(seed)
        state = base_rng.get_state()
        rng.seed(seed)
        rng.random()
        rng.set_state(state)

    run_threaded(func, 8, pass_count=True)

def test_parallel_ufunc_execution():
    # if the loop data cache or dispatch cache are not thread-safe
    # computing ufuncs simultaneously in multiple threads leads
    # to a data race that causes crashes or spurious exceptions
    for dtype in [np.float32, np.float64, np.int32]:
        for op in [np.random.random((25,)).astype(dtype), dtype(25)]:
            for ufunc in [np.isnan, np.sin]:
                run_threaded(lambda: ufunc(op), 500)

    # see gh-26690
    NUM_THREADS = 50

    a = np.ones(1000)

    def f(b):
        b.wait()
        return a.sum()

    run_threaded(f, NUM_THREADS, pass_barrier=True)


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


def _detected_blas():
    blas = np.show_config('dicts').get('Build Dependencies', {}).get('blas', {})
    return blas.get('name', 'unknown'), blas.get('version', 'unknown')


def _openblas_predates_gemm_fix(name, version):
    name = (name or '').lower().strip()
    # Assume failures using wrapper BLAS packages are buggy OpenBLAS.
    # This may ignore a genuine bug but we can't do anything more fine-grained.
    if name in ('blas', 'cblas', 'flexiblas', 'unknown', ''):
        return True
    if 'openblas' not in name:
        # e.g. MKL, accelerate, where we'd want to know about a new failure
        return False
    try:
        parsed = tuple(int(p) for p in version.split('.'))
    except ValueError:
        # unparseable OpenBLAS version so assume an old buggy version
        return True
    return parsed < (0, 3, 33, 112)


def test_blas_gemm_thread_safety():
    # gh-31618: concurrently run transpose and no-transpose GEMM variants to
    # exercise possible thread safety issues due to lock sharding between
    # kernels, see OpenBLAS issue #5836.
    num_threads = 8
    num_iters = 10
    M = 512 * 512

    rng = np.random.default_rng(0x9e3779b9)
    no_trans = rng.random((M, 4))            # C-contiguous -> NoTrans GEMM
    no_trans_w = rng.random((4, 2))
    trans = rng.random((2, M)).T             # F-contiguous -> Trans GEMM
    trans_w = rng.random((2, 2))
    expected_no_trans = no_trans @ no_trans_w
    expected_trans = trans @ trans_w

    mismatches = 0
    lock = threading.Lock()

    def closure(i, b):
        nonlocal mismatches
        count = 0
        for _ in range(num_iters):
            b.wait()
            if i % 2:
                ok = np.array_equal(no_trans @ no_trans_w, expected_no_trans)
            else:
                ok = np.array_equal(trans @ trans_w, expected_trans)
            if not ok:
                count += 1
        with lock:
            mismatches += count

    run_threaded(closure, num_threads, pass_count=True, pass_barrier=True)

    blas_name, blas_version = _detected_blas()
    if mismatches and _openblas_predates_gemm_fix(blas_name, blas_version):
        pytest.xfail(
            f"OpenBLAS version ({blas_version}) predates first OpenBLAS "
            "version with a fix (0.3.33.112) or BLAS metadata is not "
            "sufficient to identify the BLAS implementation."
        )

    assert mismatches == 0, (
        f"{mismatches} concurrent matmul results were corrupted "
        f"({blas_name} {blas_version})"
    )


def test_printoptions_thread_safety():
    # until NumPy 2.1 the printoptions state was stored in globals
    # this verifies that they are now stored in a context variable
    b = threading.Barrier(2)

    def legacy_113():
        np.set_printoptions(legacy='1.13', precision=12)
        b.wait()
        po = np.get_printoptions()
        assert po['legacy'] == '1.13'
        assert po['precision'] == 12
        orig_linewidth = po['linewidth']
        with np.printoptions(linewidth=34, legacy='1.21'):
            po = np.get_printoptions()
            assert po['legacy'] == '1.21'
            assert po['precision'] == 12
            assert po['linewidth'] == 34
        po = np.get_printoptions()
        assert po['linewidth'] == orig_linewidth
        assert po['legacy'] == '1.13'
        assert po['precision'] == 12

    def legacy_125():
        np.set_printoptions(legacy='1.25', precision=7)
        b.wait()
        po = np.get_printoptions()
        assert po['legacy'] == '1.25'
        assert po['precision'] == 7
        orig_linewidth = po['linewidth']
        with np.printoptions(linewidth=6, legacy='1.13'):
            po = np.get_printoptions()
            assert po['legacy'] == '1.13'
            assert po['precision'] == 7
            assert po['linewidth'] == 6
        po = np.get_printoptions()
        assert po['linewidth'] == orig_linewidth
        assert po['legacy'] == '1.25'
        assert po['precision'] == 7

    task1 = threading.Thread(target=legacy_113)
    task2 = threading.Thread(target=legacy_125)

    task1.start()
    task2.start()
    task1.join()
    task2.join()


def test_parallel_reduction():
    # gh-28041
    NUM_THREADS = 50

    x = np.arange(1000)

    def closure(b):
        b.wait()
        np.sum(x)

    run_threaded(closure, NUM_THREADS, pass_barrier=True)


def test_parallel_flat_iterator():
    # gh-28042
    x = np.arange(20).reshape(5, 4).T

    def closure(b):
        b.wait()
        for _ in range(100):
            list(x.flat)

    run_threaded(closure, outer_iterations=100, pass_barrier=True)

    # gh-28143
    def prepare_args():
        return [np.arange(10)]

    def closure(x, b):
        b.wait()
        for _ in range(100):
            y = np.arange(10)
            y.flat[x] = x

    run_threaded(closure, pass_barrier=True, prepare_args=prepare_args)


def test_multithreaded_repeat():
    x0 = np.arange(10)

    def closure(b):
        b.wait()
        for _ in range(100):
            x = np.repeat(x0, 2, axis=0)[::2]

    run_threaded(closure, max_workers=10, pass_barrier=True)


def test_structured_advanced_indexing():
    # Test that copyswap(n) used by integer array indexing is threadsafe
    # for structured datatypes, see gh-15387. This test can behave randomly.

    # Create a deeply nested dtype to make a failure more likely:
    dt = np.dtype([("", "f8")])
    dt = np.dtype([("", dt)] * 2)
    dt = np.dtype([("", dt)] * 2)
    # The array should be large enough to likely run into threading issues
    arr = np.random.uniform(size=(6000, 8)).view(dt)[:, 0]

    rng = np.random.default_rng()

    def func(arr):
        indx = rng.integers(0, len(arr), size=6000, dtype=np.intp)
        arr[indx]

    tpe = concurrent.futures.ThreadPoolExecutor(max_workers=8)
    futures = [tpe.submit(func, arr) for _ in range(10)]
    for f in futures:
        f.result()

    assert arr.dtype is dt


def test_structured_threadsafety2():
    # Nonzero (and some other functions) should be threadsafe for
    # structured datatypes, see gh-15387. This test can behave randomly.
    from concurrent.futures import ThreadPoolExecutor

    # Create a deeply nested dtype to make a failure more likely:
    dt = np.dtype([("", "f8")])
    dt = np.dtype([("", dt)])
    dt = np.dtype([("", dt)] * 2)
    # The array should be large enough to likely run into threading issues
    arr = np.random.uniform(size=(5000, 4)).view(dt)[:, 0]

    def func(arr):
        arr.nonzero()

    tpe = ThreadPoolExecutor(max_workers=8)
    futures = [tpe.submit(func, arr) for _ in range(10)]
    for f in futures:
        f.result()

    assert arr.dtype is dt


def test_stringdtype_multithreaded_access_and_mutation():
    # this test uses an RNG and may crash or cause deadlocks if there is a
    # threading bug
    rng = np.random.default_rng(0x4D3D3D3)

    string_list = random_unicode_string_list()

    def func(arr):
        rnd = rng.random()
        # either write to random locations in the array, compute a ufunc, or
        # re-initialize the array
        if rnd < 0.25:
            num = np.random.randint(0, arr.size)
            arr[num] = arr[num] + "hello"
        elif rnd < 0.5:
            if rnd < 0.375:
                np.add(arr, arr)
            else:
                np.add(arr, arr, out=arr)
        elif rnd < 0.75:
            if rnd < 0.875:
                np.multiply(arr, np.int64(2))
            else:
                np.multiply(arr, np.int64(2), out=arr)
        else:
            arr[:] = string_list

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as tpe:
        arr = np.array(string_list, dtype="T")
        futures = [tpe.submit(func, arr) for _ in range(500)]

        for f in futures:
            f.result()


@pytest.mark.skipif(
    not IS_64BIT,
    reason="Sometimes causes failures or crashes due to OOM on 32 bit runners"
)
@pytest.mark.parametrize("rat_cls", [
    _rational_tests.rational, _rational_tests.rational2])
def test_legacy_usertype_cast_init_thread_safety(rat_cls):
    def closure(b):
        b.wait()
        np.full((10, 10), 1, rat_cls)

    run_threaded(closure, 250, pass_barrier=True)

@pytest.mark.parametrize("dtype", [bool, int, float])
def test_nonzero(dtype):
    # See: gh-28361
    #
    # np.nonzero uses np.count_nonzero to determine the size of the output.
    # array. In a second pass the indices of the non-zero elements are
    # determined, but they can have changed
    #
    # This test triggers a data race which is suppressed in the TSAN CI.
    # The test is to ensure np.nonzero does not generate a segmentation fault
    x = np.random.randint(4, size=100).astype(dtype)
    expected_warning = ('number of non-zero array elements changed'
                        ' during function execution')

    def func(index):
        for _ in range(10):
            if index == 0:
                x[::2] = np.random.randint(2)
            else:
                try:
                    _ = np.nonzero(x)
                except RuntimeError as ex:
                    assert expected_warning in str(ex)

    run_threaded(func, max_workers=10, pass_count=True, outer_iterations=5)


# These are all implemented using PySequence_Fast, which needs locking to be safe
def np_broadcast(arrs):
    for i in range(50):
        np.broadcast(arrs)

def create_array(arrs):
    for i in range(50):
        np.array(arrs)

def create_nditer(arrs):
    for i in range(50):
        np.nditer(arrs)


@pytest.mark.parametrize(
    "kernel, outcome",
    (
        (np_broadcast, "error"),
        (create_array, "error"),
        (create_nditer, "success"),
    ),
)
def test_arg_locking(kernel, outcome):
    # should complete without triggering races but may error

    done = 0
    arrs = [np.array([1, 2, 3]) for _ in range(1000)]

    def read_arrs(b):
        nonlocal done
        b.wait()
        try:
            kernel(arrs)
        finally:
            done += 1

    def contract_and_expand_list(b):
        b.wait()
        while done < 4:
            if len(arrs) > 10:
                arrs.pop(0)
            elif len(arrs) <= 10:
                arrs.extend([np.array([1, 2, 3]) for _ in range(1000)])

    def replace_list_items(b):
        b.wait()
        rng = np.random.RandomState()
        rng.seed(0x4d3d3d3)
        while done < 4:
            data = rng.randint(0, 1000, size=4)
            arrs[data[0]] = data[1:]

    for mutation_func in (replace_list_items, contract_and_expand_list):
        b = threading.Barrier(5)
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as tpe:
                tasks = [tpe.submit(read_arrs, b) for _ in range(4)]
                tasks.append(tpe.submit(mutation_func, b))
                for t in tasks:
                    t.result()
        except RuntimeError as e:
            if outcome == "success":
                raise
            assert "Inconsistent object during array creation?" in str(e)
            msg = "replace_list_items should not raise errors"
            assert mutation_func is contract_and_expand_list, msg
        finally:
            if len(tasks) < 5:
                b.abort()

def test_array__buffer__thread_safety():
    import inspect
    arr = np.arange(1000)
    flags = [inspect.BufferFlags.STRIDED, inspect.BufferFlags.READ]

    def func(b):
        b.wait()
        for i in range(100):
            arr.__buffer__(flags[i % 2])

    run_threaded(func, max_workers=8, pass_barrier=True)

def test_void_dtype__buffer__thread_safety():
    import inspect
    dt = np.dtype([('name', np.str_, 16), ('grades', np.float64, (2,))])
    x = np.array(('ndarray_scalar', (1.2, 3.0)), dtype=dt)[()]
    assert isinstance(x, np.void)
    flags = [inspect.BufferFlags.STRIDES, inspect.BufferFlags.READ]

    def func(b):
        b.wait()
        for i in range(100):
            x.__buffer__(flags[i % 2])

    run_threaded(func, max_workers=8, pass_barrier=True)


def assert_no_deadlock(workload, *, args=(), helpers=(), timeout=30,
                       reason="deadlock"):
    """Run ``workload`` in a fresh subprocess; fail if it does not finish in time.

    ``workload`` is a self-contained function: its source is extracted with
    ``inspect.getsource`` and executed in a clean interpreter. It must do its
    own imports and may only reference arguments and helpers explicitly passed
    to this function.

    For regression tests whose failure mode is a hang. Delegates to
    ``run_subprocess`` (which folds the child's output into the failure on a
    nonzero exit); this only adds the timeout watchdog, reporting a likely
    ``reason`` if the child is killed.

    """
    source = "\n".join(
        textwrap.dedent(inspect.getsource(helper)) for helper in helpers
    )
    source += "\n" + textwrap.dedent(inspect.getsource(workload))
    script = (
        "import faulthandler\n"
        f"faulthandler.dump_traceback_later({timeout}, exit=True)\n"
        f"{source}\n"
        f"{workload.__name__}(*{args!r})\n"
        "faulthandler.cancel_dump_traceback_later()\n"
    )
    try:
        run_subprocess([sys.executable, "-c", script], timeout=timeout + 15)
    except subprocess.TimeoutExpired:
        raise AssertionError(
            f"subprocess did not finish within {timeout}s -- likely {reason}"
        ) from None


def threaded_deadlock_reproducer(operation, nworkers, niters, stall):
    import faulthandler
    import os
    import sys
    import threading
    import time

    progress = [0] * nworkers
    errors = []
    barrier = threading.Barrier(nworkers)

    def worker(idx):
        try:
            barrier.wait()
            for _ in range(niters):
                operation(idx)
                progress[idx] += 1
        except BaseException as exc:
            errors.append(exc)
            raise

    workers = [threading.Thread(target=worker, args=(i,), daemon=True)
               for i in range(nworkers)]
    for t in workers:
        t.start()

    last_total, last_change = 0, time.monotonic()
    while any(t.is_alive() for t in workers):
        time.sleep(0.05)
        if errors:
            raise AssertionError(f"worker failed: {errors[0]!r}") from errors[0]
        total = sum(progress)
        now = time.monotonic()
        if total != last_total:
            last_total, last_change = total, now
        elif now - last_change > stall:
            # dump the wedged threads' tracebacks, then os._exit to avoid a
            # shutdown hang on the daemon threads.
            sys.stderr.write("Probably deadlocked!\n")
            faulthandler.dump_traceback()
            sys.stderr.flush()
            os._exit(1)
    if errors:
        raise AssertionError(f"worker failed: {errors[0]!r}") from errors[0]


def allocator_lock_order_workload(case_):
    import numpy as np

    N = 1_000
    NWORKERS = 8
    NITERS = 1000
    STALL = 2.0

    if case_ == "two-allocator":
        a = np.array(["alpha"] * N, dtype="T")
        b = np.array(["bravo"] * N, dtype="T")

        def operation(idx):
            if idx % 2 == 0:
                np.less(a, b)   # locks alloc(a) then alloc(b)
            else:
                np.less(b, a)   # locks alloc(b) then alloc(a)

    elif case_ == "three-allocator":
        a = np.array(["alpha"] * N, dtype="T")
        b = np.array(["bravo"] * N, dtype="T")
        c = np.array(["charlie"] * N, dtype="T")

        def operation(idx):
            order = idx % 3
            if order == 0:
                np.maximum(a, b, out=c)   # locks alloc(a), alloc(b), alloc(c)
            elif order == 1:
                np.maximum(b, c, out=a)   # locks alloc(b), alloc(c), alloc(a)
            else:
                np.maximum(c, a, out=b)   # locks alloc(c), alloc(a), alloc(b)

    elif case_ == "four-allocator":
        from numpy._core.umath import _replace

        a = np.array(["a"] * N, dtype="T")
        b = np.array(["b"] * N, dtype="T")
        c = np.array(["c"] * N, dtype="T")
        d = np.array(["d"] * N, dtype="T")
        counts = np.ones(N, dtype=np.int64)

        def operation(idx):
            order = idx % 4
            if order == 0:
                _replace(a, b, c, counts, out=d)
            elif order == 1:
                _replace(b, c, d, counts, out=a)
            elif order == 2:
                _replace(c, d, a, counts, out=b)
            else:
                _replace(d, a, b, counts, out=c)

    elif case_ == "duplicate-allocator":
        from numpy._core.umath import _replace

        a = np.array(["same"] * N, dtype="T")
        b = np.array(["other"] * N, dtype="T")
        c = np.array(["out"] * N, dtype="T")
        counts = np.ones(N, dtype=np.int64)

        def operation(idx):
            if idx % 2 == 0:
                np.maximum(a, a, out=b)
            else:
                _replace(a, a, b, counts, out=c)

    else:
        raise ValueError(f"unknown allocator deadlock case: {case_}")

    threaded_deadlock_reproducer(
        operation, NWORKERS, NITERS, stall=STALL,
    )


def test_setitem_reentrant_no_deadlock():
    def workload():
        import numpy as np

        arr = np.empty(2, dtype="T")

        class Reenter:
            def __str__(self):
                # re-enters stringdtype_setitem and re-acquires the allocator
                arr[1] = "inner"
                return "outer"

        arr[0] = Reenter()
        assert arr[0] == "outer"
        assert arr[1] == "inner"

    assert_no_deadlock(
        workload,
        reason="re-entrant allocator acquire in stringdtype_setitem",
    )


@pytest.mark.parametrize(
    "case_",
    ["two-allocator", "three-allocator", "four-allocator",
     "duplicate-allocator"],
)
def test_concurrent_allocator_acquire_no_deadlock(case_):
    # NpyString_acquire_allocators must lock distinct allocators in a
    # canonical order, otherwise deadlocks are possible from ABBA cycles
    assert_no_deadlock(
        allocator_lock_order_workload,
        args=(case_,),
        helpers=(threaded_deadlock_reproducer,),
        reason=f"allocator lock-ordering in {case_}",
    )


def unique_deadlock_workload():
    import numpy as np
    from numpy._core._multiarray_umath import _unique_hash

    NWORKERS = 8
    NITERS = 200
    STALL = 2.0

    # _unique_hash (the engine under np.unique) rather than np.unique itself:
    # it spends almost all its time in the GIL-released, allocator-locked load
    # loop, so workers reliably overlap there. A single shared array with many
    # duplicates concentrates contention on one allocator.
    data = np.array([f"v{i % 128}" for i in range(50_000)], dtype="T")

    def operation(idx):
        _unique_hash(data, equal_nan=False)

    threaded_deadlock_reproducer(operation, NWORKERS, NITERS, stall=STALL)


def test_concurrent_unique_no_deadlock():
    # Concurrent unique on a shared array deadlocks the allocator lock against
    # the GIL unless unique_vstring releases the GIL before locking. Only
    # reproduces on Python <= 3.12; PyMutex detaches on 3.13+.
    assert_no_deadlock(
        unique_deadlock_workload,
        helpers=(threaded_deadlock_reproducer,),
        reason="unique allocator lock vs GIL",
    )
