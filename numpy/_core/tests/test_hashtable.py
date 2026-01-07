from threading import Barrier, Thread

import pytest

from numpy._core._multiarray_tests import (
    create_identity_hash,
    identity_hash_get_item,
    identity_hash_set_item_default,
)


@pytest.mark.parametrize("key_length", [1, 3, 6])
@pytest.mark.parametrize("length", [1, 16, 2000])
def test_identity_hashtable_get_set(key_length, length):
    # no collisions expected
    keys_vals = []
    for i in range(length):
        keys = tuple(object() for _ in range(key_length))
        keys_vals.append((keys, object()))

    ht = create_identity_hash(key_length)

    for i in range(length):
        key, value = keys_vals[i]
        assert identity_hash_set_item_default(ht, key, value) is value

    for key, value in keys_vals:
        got = identity_hash_get_item(ht, key)
        assert got is value


@pytest.mark.parametrize("key_length", [1, 3, 6])
def test_identity_hashtable_default(key_length):
    ht = create_identity_hash(key_length)

    key = tuple(object() for _ in range(key_length))
    val1 = object()
    val2 = object()

    # first insertion sets the value as val1
    got1 = identity_hash_set_item_default(ht, key, val1)
    assert got1 is val1

    # second insertion with the same key returns the existing value val1
    got2 = identity_hash_set_item_default(ht, key, val2)
    assert got2 is val1

@pytest.mark.parametrize("key_length", [1, 3, 6])
def test_identity_hashtable_set_thread_safety(key_length):
    ht = create_identity_hash(key_length)
    barrier = Barrier(2)

    key = tuple(object() for _ in range(key_length))
    val1 = object()
    val2 = object()

    def thread_func(value_to_set, results, idx):
        barrier.wait()
        result = identity_hash_set_item_default(ht, key, value_to_set)
        results[idx] = result

    results = [None, None]
    thread1 = Thread(target=thread_func, args=(val1, results, 0))
    thread2 = Thread(target=thread_func, args=(val2, results, 1))

    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()

    # both threads should get the same result and it should be either val1 or val2
    assert results[0] is results[1]
    assert results[0] in (val1, val2)


@pytest.mark.parametrize("key_length", [1, 3, 6])
def test_identity_hashtable_get_thread_safety(key_length):
    ht = create_identity_hash(key_length)
    key = tuple(object() for _ in range(key_length))
    value = object()
    identity_hash_set_item_default(ht, key, value)

    barrier = Barrier(2)

    def thread_func(results, idx):
        barrier.wait()
        result = identity_hash_get_item(ht, key)
        results[idx] = result

    results = [None, None]
    thread1 = Thread(target=thread_func, args=(results, 0))
    thread2 = Thread(target=thread_func, args=(results, 1))

    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()

    assert results[0] is value
    assert results[1] is value
