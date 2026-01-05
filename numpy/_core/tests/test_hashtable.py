import pytest

from numpy._core._multiarray_tests import (
    create_identity_hash,
    identity_hash_get_item,
    identity_hash_set_item,
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
        identity_hash_set_item(ht, key, value)

    for key, value in keys_vals:
        got = identity_hash_get_item(ht, key)
        assert got is value


@pytest.mark.parametrize("key_length", [1, 3, 6])
def test_identity_hashtable_replace(key_length):
    ht = create_identity_hash(key_length)

    key = tuple(object() for _ in range(key_length))
    val1 = object()
    val2 = object()

    identity_hash_set_item(ht, key, val1)
    got = identity_hash_get_item(ht, key)
    assert got is val1

    with pytest.raises(RuntimeError):
        identity_hash_set_item(ht, key, val2)

    identity_hash_set_item(ht, key, val2, replace=True)
    got = identity_hash_get_item(ht, key)
    assert got is val2
