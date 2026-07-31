import copy
import pickle
import threading

import pytest

import numpy as np
from numpy.random import SeedSequence
from numpy.testing import (
    IS_WASM,
    assert_array_compare,
    assert_array_equal,
    assert_raises,
    assert_raises_regex,
)

_THREAD_TIMEOUT = 5
_SPAWN_LOCK_STRIPES = 257


class _BlockingSeedSequence(SeedSequence):
    entered = None
    release = None
    entry_count = 0
    entry_count_lock = threading.Lock()
    local = threading.local()

    def __init__(self, *args, **kwargs):
        cls = type(self)
        if cls.release is not None and not getattr(cls.local, "entered", False):
            cls.local.entered = True
            with cls.entry_count_lock:
                entry = cls.entry_count
                cls.entry_count += 1
            cls.entered[entry].set()
            if not cls.release.wait(_THREAD_TIMEOUT):
                raise TimeoutError("child construction gate timed out")
        super().__init__(*args, **kwargs)


class _CustomSeedSequence(SeedSequence):
    pass


class _FailingSeedSequence(SeedSequence):
    fail_at = None
    fail_thread = None
    failure_entered = None
    release_failure = None
    success_entered = None

    def __init__(self, *args, **kwargs):
        cls = type(self)
        spawn_key = kwargs.get("spawn_key", ())
        should_fail = (
            spawn_key
            and spawn_key[-1] == cls.fail_at
            and (
                cls.fail_thread is None
                or cls.fail_thread == threading.get_ident()
            )
        )
        if (
            spawn_key
            and cls.success_entered is not None
            and cls.fail_thread != threading.get_ident()
        ):
            cls.success_entered.set()
        if should_fail:
            if cls.failure_entered is not None:
                cls.failure_entered.set()
            if (
                cls.release_failure is not None
                and not cls.release_failure.wait(_THREAD_TIMEOUT)
            ):
                raise TimeoutError("failure gate timed out")
            raise RuntimeError("failed child construction")
        super().__init__(*args, **kwargs)


class _ReentrantSeedSequence(SeedSequence):
    spawn_during_init = None

    def __init__(self, *args, **kwargs):
        cls = type(self)
        spawn_key = kwargs.get("spawn_key", ())
        if spawn_key and cls.spawn_during_init is not None:
            seed_sequence = cls.spawn_during_init
            cls.spawn_during_init = None
            seed_sequence.spawn(1)
        super().__init__(*args, **kwargs)


class _RecordingSeedSequence(SeedSequence):
    constructed_spawn_keys = None

    def __init__(self, *args, **kwargs):
        spawn_key = kwargs.get("spawn_key", ())
        if spawn_key and type(self).constructed_spawn_keys is not None:
            type(self).constructed_spawn_keys.append(tuple(spawn_key))
        super().__init__(*args, **kwargs)


class _HistoricalSeedSequencePickle:
    def __init__(self, seed_sequence, checksum):
        self.seed_sequence = seed_sequence
        self.checksum = checksum

    def __reduce__(self):
        constructor, args, state = self.seed_sequence.__reduce__()
        return constructor, (args[0], self.checksum, state)


def _pickle_roundtrip(value):
    return pickle.loads(pickle.dumps(value))


def _spawn_worker(spawn, n_children, started, results, errors, index):
    started.set()
    try:
        results[index] = spawn(n_children)
    except BaseException as exc:
        errors[index] = exc


def _join_threads(threads):
    for thread in threads:
        if thread.ident is not None:
            thread.join(_THREAD_TIMEOUT)
    assert not [thread.name for thread in threads if thread.is_alive()]


def _bounded_spawn(spawn, n_children=1):
    started = threading.Event()
    results = [None]
    errors = [None]
    thread = threading.Thread(
        target=_spawn_worker,
        args=(spawn, n_children, started, results, errors, 0),
        daemon=True,
    )
    thread.start()
    assert started.wait(_THREAD_TIMEOUT), "spawn worker did not start"
    _join_threads([thread])
    return results[0], errors[0]


def _reset_blocking_seed_sequence():
    _BlockingSeedSequence.release = None
    _BlockingSeedSequence.entered = None
    _BlockingSeedSequence.entry_count = 0
    _BlockingSeedSequence.local = threading.local()


def _find_seed_sequence_on_stripe(seed_sequence, same_stripe):
    stripe = id(seed_sequence) % _SPAWN_LOCK_STRIPES
    candidates = []
    for entropy in range(1, _SPAWN_LOCK_STRIPES * 4):
        candidate = type(seed_sequence)(entropy)
        candidates.append(candidate)
        matches = id(candidate) % _SPAWN_LOCK_STRIPES == stripe
        if matches is same_stripe:
            return candidate, candidates
    pytest.fail("could not allocate a SeedSequence on the requested stripe")


def test_reference_data():
    """ Check that SeedSequence generates data the same as the C++ reference.

    https://gist.github.com/imneme/540829265469e673d045
    """
    inputs = [
        [3735928559, 195939070, 229505742, 305419896],
        [3668361503, 4165561550, 1661411377, 3634257570],
        [164546577, 4166754639, 1765190214, 1303880213],
        [446610472, 3941463886, 522937693, 1882353782],
        [1864922766, 1719732118, 3882010307, 1776744564],
        [4141682960, 3310988675, 553637289, 902896340],
        [1134851934, 2352871630, 3699409824, 2648159817],
        [1240956131, 3107113773, 1283198141, 1924506131],
        [2669565031, 579818610, 3042504477, 2774880435],
        [2766103236, 2883057919, 4029656435, 862374500],
    ]
    outputs = [
        [3914649087, 576849849, 3593928901, 2229911004],
        [2240804226, 3691353228, 1365957195, 2654016646],
        [3562296087, 3191708229, 1147942216, 3726991905],
        [1403443605, 3591372999, 1291086759, 441919183],
        [1086200464, 2191331643, 560336446, 3658716651],
        [3249937430, 2346751812, 847844327, 2996632307],
        [2584285912, 4034195531, 3523502488, 169742686],
        [959045797, 3875435559, 1886309314, 359682705],
        [3978441347, 432478529, 3223635119, 138903045],
        [296367413, 4262059219, 13109864, 3283683422],
    ]
    outputs64 = [
        [2477551240072187391, 9577394838764454085],
        [15854241394484835714, 11398914698975566411],
        [13708282465491374871, 16007308345579681096],
        [15424829579845884309, 1898028439751125927],
        [9411697742461147792, 15714068361935982142],
        [10079222287618677782, 12870437757549876199],
        [17326737873898640088, 729039288628699544],
        [16644868984619524261, 1544825456798124994],
        [1857481142255628931, 596584038813451439],
        [18305404959516669237, 14103312907920476776],
    ]
    for seed, expected, expected64 in zip(inputs, outputs, outputs64):
        expected = np.array(expected, dtype=np.uint32)
        ss = SeedSequence(seed)
        state = ss.generate_state(len(expected))
        assert_array_equal(state, expected)
        state64 = ss.generate_state(len(expected64), dtype=np.uint64)
        assert_array_equal(state64, expected64)


def test_zero_padding():
    """ Ensure that the implicit zero-padding does not cause problems.
    """
    # Ensure that large integers are inserted in little-endian fashion to avoid
    # trailing 0s.
    ss0 = SeedSequence(42)
    ss1 = SeedSequence(42 << 32)
    assert_array_compare(
        np.not_equal,
        ss0.generate_state(4),
        ss1.generate_state(4))

    # Ensure backwards compatibility with the original 0.17 release for small
    # integers and no spawn key.
    expected42 = np.array([3444837047, 2669555309, 2046530742, 3581440988],
                          dtype=np.uint32)
    assert_array_equal(SeedSequence(42).generate_state(4), expected42)

    # Regression test for gh-16539 to ensure that the implicit 0s don't
    # conflict with spawn keys.
    assert_array_compare(
        np.not_equal,
        SeedSequence(42, spawn_key=(0,)).generate_state(4),
        expected42)


def test_seedsequence_rejects_nested_sequence():
    with assert_raises(TypeError):
        SeedSequence(SeedSequence(42))

    # Prevents infinite recursion (Issue #28822) and rejects
    # invalid types (Issue #27380)
    match_str = "SeedSequence does not accept nested sequences."

    # Test standard nested lists
    with assert_raises_regex(TypeError, match_str):
        SeedSequence([[1, 2], [3, 4]])

    # Test self-referencing/cyclic lists
    with assert_raises_regex(TypeError, match_str):
        cyclic_seed = []
        cyclic_seed.append(cyclic_seed)
        SeedSequence(cyclic_seed)


@pytest.mark.skipif(IS_WASM, reason="can't start thread")
@pytest.mark.thread_unsafe(
    reason="uses shared mutable helper class state",
)
def test_spawn_concurrent_unique_ranges():
    seed_sequence = _BlockingSeedSequence(12345)
    entered = [threading.Event(), threading.Event()]
    release = threading.Event()
    started = [threading.Event(), threading.Event()]
    results = [None, None]
    errors = [None, None]
    threads = [
        threading.Thread(
            target=_spawn_worker,
            args=(
                seed_sequence.spawn,
                n_children,
                started[index],
                results,
                errors,
                index,
            ),
            daemon=True,
        )
        for index, n_children in enumerate((2, 3))
    ]

    _BlockingSeedSequence.entered = entered
    _BlockingSeedSequence.release = release

    try:
        threads[0].start()
        assert entered[0].wait(_THREAD_TIMEOUT), "first spawn did not block"
        threads[1].start()
        assert started[1].wait(_THREAD_TIMEOUT), "second spawn did not start"
        # The worker signals immediately before calling spawn. Without
        # exclusion, its first child reaches this event while the other child
        # constructor is still gated.
        assert not entered[1].wait(1), (
            "second spawn entered child construction while the first call "
            "was still active"
        )
    finally:
        release.set()
        try:
            _join_threads(threads)
        finally:
            _reset_blocking_seed_sequence()

    assert errors == [None, None]
    batches = results

    indexes_by_call = [
        [child.spawn_key[-1] for child in batch]
        for batch in batches
    ]
    for indexes in indexes_by_call:
        assert indexes == list(range(indexes[0], indexes[0] + len(indexes)))

    indexes = sorted(index for batch in indexes_by_call for index in batch)
    assert indexes == list(range(5))
    assert seed_sequence.n_children_spawned == 5
    assert seed_sequence.spawn(1)[0].spawn_key == (5,)
    assert seed_sequence.n_children_spawned == 6


@pytest.mark.skipif(IS_WASM, reason="can't start thread")
@pytest.mark.thread_unsafe(
    reason="uses shared mutable helper class state",
)
def test_spawn_independent_seed_sequences_use_different_stripes():
    first = _BlockingSeedSequence(12345)
    second, candidates = _find_seed_sequence_on_stripe(first, False)
    assert candidates

    entered = [threading.Event(), threading.Event()]
    release = threading.Event()
    started = [threading.Event(), threading.Event()]
    results = [None, None]
    errors = [None, None]
    threads = [
        threading.Thread(
            target=_spawn_worker,
            args=(
                seed_sequence.spawn,
                1,
                started[index],
                results,
                errors,
                index,
            ),
            daemon=True,
        )
        for index, seed_sequence in enumerate((first, second))
    ]

    _BlockingSeedSequence.entered = entered
    _BlockingSeedSequence.release = release
    try:
        threads[0].start()
        assert entered[0].wait(_THREAD_TIMEOUT), "first spawn did not block"
        threads[1].start()
        assert entered[1].wait(_THREAD_TIMEOUT), (
            "independent SeedSequences on different stripes were serialized"
        )
    finally:
        release.set()
        try:
            _join_threads(threads)
        finally:
            _reset_blocking_seed_sequence()

    assert errors == [None, None]
    assert [batch[0].spawn_key for batch in results] == [(0,), (0,)]
    assert first.n_children_spawned == second.n_children_spawned == 1


def test_spawn_sequential_ranges():
    seed_sequence = SeedSequence(
        12345,
        spawn_key=(7,),
        n_children_spawned=4,
    )

    first_batch = seed_sequence.spawn(3)
    first_keys = [(7, 4), (7, 5), (7, 6)]
    assert [child.spawn_key for child in first_batch] == first_keys
    assert seed_sequence.n_children_spawned == 7

    assert seed_sequence.spawn(0) == []
    assert seed_sequence.n_children_spawned == 7

    second_batch = seed_sequence.spawn(2)
    second_keys = [(7, 7), (7, 8)]
    assert [child.spawn_key for child in second_batch] == second_keys
    assert seed_sequence.n_children_spawned == 9

    for child, spawn_key in zip(first_batch + second_batch,
                                first_keys + second_keys):
        expected = SeedSequence(12345, spawn_key=spawn_key)
        assert_array_equal(child.generate_state(4), expected.generate_state(4))


@pytest.mark.thread_unsafe(
    reason="uses shared mutable helper class state",
)
def test_spawn_rejects_uint32_counter_overflow_before_construction():
    max_uint32 = np.iinfo(np.uint32).max
    seed_sequence = _RecordingSeedSequence(
        12345,
        n_children_spawned=max_uint32 - 1,
    )
    constructed_spawn_keys = []
    _RecordingSeedSequence.constructed_spawn_keys = constructed_spawn_keys
    try:
        with pytest.raises(
            OverflowError,
            match="n_children_spawned cannot exceed 4294967295",
        ):
            seed_sequence.spawn(2)
        assert constructed_spawn_keys == []
        assert seed_sequence.n_children_spawned == max_uint32 - 1

        child = seed_sequence.spawn(1)[0]
        assert child.spawn_key == (max_uint32 - 1,)
        assert seed_sequence.n_children_spawned == max_uint32

        constructed_spawn_keys.clear()
        with pytest.raises(
            OverflowError,
            match="n_children_spawned cannot exceed 4294967295",
        ):
            seed_sequence.spawn(1)
        assert constructed_spawn_keys == []
        assert seed_sequence.n_children_spawned == max_uint32
    finally:
        _RecordingSeedSequence.constructed_spawn_keys = None


@pytest.mark.thread_unsafe(
    reason="uses shared mutable helper class state",
)
def test_spawn_partial_failure_does_not_advance_counter():
    seed_sequence = _FailingSeedSequence(12345)
    _FailingSeedSequence.fail_at = 2
    _FailingSeedSequence.fail_thread = threading.get_ident()
    try:
        with pytest.raises(RuntimeError, match="failed child construction"):
            seed_sequence.spawn(4)
    finally:
        _FailingSeedSequence.fail_at = None
        _FailingSeedSequence.fail_thread = None

    assert seed_sequence.n_children_spawned == 0
    children = seed_sequence.spawn(4)
    assert [child.spawn_key for child in children] == [
        (0,), (1,), (2,), (3,)
    ]
    assert seed_sequence.n_children_spawned == 4


@pytest.mark.skipif(IS_WASM, reason="can't start thread")
@pytest.mark.thread_unsafe(
    reason="uses shared mutable helper class state",
)
def test_spawn_concurrent_failure_and_success_are_transactional():
    seed_sequence = _FailingSeedSequence(12345)
    failure_entered = threading.Event()
    release_failure = threading.Event()
    success_entered = threading.Event()
    started = [threading.Event(), threading.Event()]
    results = [None, None]
    errors = [None, None]

    def fail_spawn():
        _FailingSeedSequence.fail_thread = threading.get_ident()
        _spawn_worker(
            seed_sequence.spawn, 3, started[0], results, errors, 0
        )

    threads = [
        threading.Thread(target=fail_spawn, daemon=True),
        threading.Thread(
            target=_spawn_worker,
            args=(
                seed_sequence.spawn,
                2,
                started[1],
                results,
                errors,
                1,
            ),
            daemon=True,
        ),
    ]
    _FailingSeedSequence.fail_at = 1
    _FailingSeedSequence.failure_entered = failure_entered
    _FailingSeedSequence.release_failure = release_failure
    _FailingSeedSequence.success_entered = success_entered

    try:
        threads[0].start()
        assert failure_entered.wait(_THREAD_TIMEOUT), (
            "failing spawn did not reach the configured child"
        )
        threads[1].start()
        assert started[1].wait(_THREAD_TIMEOUT), "successful spawn did not start"
        assert not success_entered.wait(1), (
            "successful spawn entered child construction before the failing "
            "call released its range"
        )
    finally:
        release_failure.set()
        try:
            _join_threads(threads)
        finally:
            _FailingSeedSequence.fail_at = None
            _FailingSeedSequence.fail_thread = None
            _FailingSeedSequence.failure_entered = None
            _FailingSeedSequence.release_failure = None
            _FailingSeedSequence.success_entered = None

    assert results[0] is None
    assert isinstance(errors[0], RuntimeError)
    assert errors[1] is None
    assert success_entered.is_set()
    assert [child.spawn_key for child in results[1]] == [(0,), (1,)]
    assert seed_sequence.n_children_spawned == 2


@pytest.mark.skipif(IS_WASM, reason="can't start thread")
@pytest.mark.thread_unsafe(
    reason="uses shared mutable helper class state",
)
def test_spawn_same_object_reentry_is_rejected_and_cleaned_up():
    seed_sequence = _ReentrantSeedSequence(12345)
    _ReentrantSeedSequence.spawn_during_init = seed_sequence
    try:
        result, error = _bounded_spawn(seed_sequence.spawn)
    finally:
        _ReentrantSeedSequence.spawn_during_init = None

    assert result is None
    assert isinstance(error, RuntimeError)
    assert "cannot be re-entered" in str(error)
    assert seed_sequence.n_children_spawned == 0
    assert seed_sequence.spawn(1)[0].spawn_key == (0,)
    assert seed_sequence.n_children_spawned == 1


@pytest.mark.skipif(IS_WASM, reason="can't start thread")
@pytest.mark.thread_unsafe(
    reason="uses shared mutable helper class state",
)
def test_spawn_different_object_reentry_on_same_stripe():
    seed_sequence = _ReentrantSeedSequence(12345)
    nested, candidates = _find_seed_sequence_on_stripe(seed_sequence, True)
    assert candidates
    _ReentrantSeedSequence.spawn_during_init = nested
    try:
        children, error = _bounded_spawn(seed_sequence.spawn)
    finally:
        _ReentrantSeedSequence.spawn_during_init = None

    assert error is None
    assert children[0].spawn_key == (0,)
    assert nested.n_children_spawned == 1
    assert seed_sequence.n_children_spawned == 1


@pytest.mark.parametrize(
    ("copier", "shares_pool"),
    [
        pytest.param(copy.copy, True, id="copy"),
        pytest.param(copy.deepcopy, False, id="deepcopy"),
        pytest.param(_pickle_roundtrip, False, id="pickle"),
    ],
)
def test_seedsequence_copy_and_pickle(copier, shares_pool):
    seed_sequence = _CustomSeedSequence(
        12345,
        spawn_key=(7,),
        pool_size=6,
    )
    seed_sequence.spawn(3)
    seed_sequence.pool[0] ^= 1
    seed_sequence.custom_attribute = "preserved"

    clone = copier(seed_sequence)

    assert type(clone) is _CustomSeedSequence
    assert clone is not seed_sequence
    assert clone.state == seed_sequence.state
    assert_array_equal(clone.pool, seed_sequence.pool)
    assert (clone.pool is seed_sequence.pool) is shares_pool
    assert clone.custom_attribute == "preserved"

    clone_children = clone.spawn(2)
    assert [child.spawn_key for child in clone_children] == [(7, 3), (7, 4)]
    assert clone.n_children_spawned == 5
    assert seed_sequence.n_children_spawned == 3

    original_children = seed_sequence.spawn(2)
    assert [child.spawn_key for child in original_children] == [(7, 3), (7, 4)]
    assert seed_sequence.n_children_spawned == 5


@pytest.mark.parametrize("checksum", [0x3EAA222, 0xC464C19, 0xF88304A])
def test_seedsequence_historical_pickle_checksums(checksum):
    seed_sequence = SeedSequence(
        12345,
        spawn_key=(7,),
        pool_size=6,
        n_children_spawned=3,
    )

    clone = _pickle_roundtrip(
        _HistoricalSeedSequencePickle(seed_sequence, checksum)
    )

    assert type(clone) is SeedSequence
    assert clone.state == seed_sequence.state
    assert_array_equal(clone.pool, seed_sequence.pool)
    assert [child.spawn_key for child in clone.spawn(2)] == [(7, 3), (7, 4)]
