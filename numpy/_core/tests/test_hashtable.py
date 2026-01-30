import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import pytest

from numpy._core._multiarray_tests import (
    create_identity_hash,
    identity_hash_get_item,
    identity_hash_set_item_default,
)
from numpy.testing import IS_WASM


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

@pytest.mark.skipif(IS_WASM, reason="wasm doesn't have support for threads")
@pytest.mark.parametrize("key_length", [1, 3, 6])
def test_identity_hashtable_default_thread_safety(key_length):
    ht = create_identity_hash(key_length)

    key = tuple(object() for _ in range(key_length))
    val1 = object()
    val2 = object()

    got1 = identity_hash_set_item_default(ht, key, val1)
    assert got1 is val1

    def thread_func(val):
        return identity_hash_set_item_default(ht, key, val)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(thread_func, val2) for _ in range(8)]
        results = [f.result() for f in futures]

    assert all(r is val1 for r in results)


@pytest.mark.skipif(IS_WASM, reason="wasm doesn't have support for threads")
@pytest.mark.parametrize("key_length", [1, 3, 6])
def test_identity_hashtable_set_thread_safety(key_length):
    ht = create_identity_hash(key_length)

    key = tuple(object() for _ in range(key_length))
    val1 = object()

    def thread_func(val):
        return identity_hash_set_item_default(ht, key, val)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(thread_func, val1) for _ in range(100)]
        results = [f.result() for f in futures]

    assert all(r is val1 for r in results)

@pytest.mark.skipif(IS_WASM, reason="wasm doesn't have support for threads")
@pytest.mark.parametrize("key_length", [1, 3, 6])
def test_identity_hashtable_get_thread_safety(key_length):
    ht = create_identity_hash(key_length)
    key = tuple(object() for _ in range(key_length))
    value = object()
    identity_hash_set_item_default(ht, key, value)

    def thread_func():
        return identity_hash_get_item(ht, key)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(thread_func) for _ in range(100)]
        results = [f.result() for f in futures]

    assert all(r is value for r in results)

@pytest.mark.skipif(IS_WASM, reason="wasm doesn't have support for threads")
@pytest.mark.parametrize("key_length", [1, 3, 6])
@pytest.mark.parametrize("length", [1 << 4, 1 << 8, 1 << 12])
def test_identity_hashtable_get_set_concurrent(key_length, length):
    ht = create_identity_hash(key_length)
    keys_vals = []
    for i in range(length):
        keys = tuple(object() for _ in range(key_length))
        keys_vals.append((keys, object()))

    def set_item(kv):
        key, value = kv
        got = identity_hash_set_item_default(ht, key, value)
        assert got is value

    def get_item(kv):
        key, value = kv
        got = identity_hash_get_item(ht, key)
        assert got is None or got is value

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for kv in keys_vals:
            futures.append(executor.submit(set_item, kv))
            futures.append(executor.submit(get_item, kv))
        for future in futures:
            future.result()

@pytest.mark.skipif(IS_WASM, reason="wasm doesn't have support for threads")
@pytest.mark.parametrize("key_length", [3, 6, 10])
@pytest.mark.parametrize("length", [1 << 4, 1 << 8, 1 << 12])
def test_identity_hashtable_get_set_concurrent_collisions(key_length, length):
    ht = create_identity_hash(key_length)
    base_key = tuple(object() for _ in range(key_length - 1))
    keys_vals = defaultdict(list)
    for i in range(length):
        keys = base_key + (random.choice(base_key), )
        keys_vals[keys].append(object())

    set_item_results = defaultdict(set)

    def set_item(kv):
        key, values = kv
        value = random.choice(values)
        got = identity_hash_set_item_default(ht, key, value)
        set_item_results[key].add(got)

    get_item_results = defaultdict(set)

    def get_item(kv):
        key, values = kv
        got = identity_hash_get_item(ht, key)
        get_item_results[key].add(got)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for keys, values in keys_vals.items():
            futures.append(executor.submit(set_item, (keys, values)))
            futures.append(executor.submit(get_item, (keys, values)))
        for future in futures:
            future.result()

    for key in keys_vals.keys():
        assert len(set_item_results[key]) == 1
        set_item_value = set_item_results[key].pop()
        for r in get_item_results[key]:
            assert r is None or r is set_item_value
