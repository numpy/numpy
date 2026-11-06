import pytest

import numpy as np
from numpy.random import MT19937, PCG64, PCG64DXSM, SFC64, Generator, Philox
from numpy.testing import assert_array_equal, assert_equal


class _NoBulkBitGenerator:
    """Expose the original interface without the optional bulk capsule."""

    def __init__(self, bit_generator):
        self._bit_generator = bit_generator
        self.capsule = bit_generator.capsule
        self.lock = bit_generator.lock


@pytest.mark.parametrize("bit_generator", [MT19937, PCG64, PCG64DXSM, SFC64, Philox])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("size", [None, 0, 1, 31, 32, 311, 312, 313, 624, 625, 1000])
def test_bulk_random_matches_fallback(bit_generator, dtype, size):
    bulk_bitgen = bit_generator(987654321)
    fallback_bitgen = bit_generator(987654321)
    bulk = Generator(bulk_bitgen)
    fallback = Generator(_NoBulkBitGenerator(fallback_bitgen))
    assert_array_equal(
        bulk.random(size=size, dtype=dtype),
        fallback.random(size=size, dtype=dtype),
    )
    assert_equal(bulk_bitgen.state, fallback_bitgen.state)


@pytest.mark.parametrize("bit_generator", [MT19937, PCG64, PCG64DXSM, SFC64, Philox])
@pytest.mark.parametrize(
    "dtype, low, high",
    [
        (np.uint32, 0, 2**32),
        (np.uint32, 17, 1009),
        (np.uint64, 0, 2**64),
        (np.uint64, 17, 1009),
        (np.int32, -(2**31), 2**31),
        (np.int32, -503, 509),
        (np.int64, -(2**63), 2**63),
        (np.int64, -503, 509),
    ],
)
@pytest.mark.parametrize("size", [None, 0, 1, 31, 32, 311, 312, 313, 624, 625, 1000])
def test_bulk_integers_matches_fallback(bit_generator, size, dtype, low, high):
    bulk_bitgen = bit_generator(987654321)
    fallback_bitgen = bit_generator(987654321)
    bulk = Generator(bulk_bitgen)
    fallback = Generator(_NoBulkBitGenerator(fallback_bitgen))
    assert_array_equal(
        bulk.integers(low, high, size=size, dtype=dtype),
        fallback.integers(low, high, size=size, dtype=dtype),
    )
    assert_equal(bulk_bitgen.state, fallback_bitgen.state)


@pytest.mark.parametrize("bit_generator", [PCG64, PCG64DXSM, SFC64, Philox])
def test_bulk_cached_state(bit_generator):
    bulk_bitgen = bit_generator(987654321)
    fallback_bitgen = bit_generator(987654321)
    bulk = Generator(bulk_bitgen)
    fallback = Generator(_NoBulkBitGenerator(fallback_bitgen))
    assert_equal(
        bulk.integers(0, 2**32, dtype=np.uint32),
        fallback.integers(0, 2**32, dtype=np.uint32),
    )
    assert bulk_bitgen.state["has_uint32"] == 1
    assert_equal(bulk_bitgen.state, fallback_bitgen.state)

    assert_array_equal(
        bulk.random(0, dtype=np.float32),
        fallback.random(0, dtype=np.float32),
    )
    assert bulk_bitgen.state["has_uint32"] == 1
    assert_equal(bulk_bitgen.state, fallback_bitgen.state)

    assert_array_equal(
        bulk.integers(0, 2**64, size=312, dtype=np.uint64),
        fallback.integers(0, 2**64, size=312, dtype=np.uint64),
    )
    assert bulk_bitgen.state["has_uint32"] == 1
    assert_equal(bulk_bitgen.state, fallback_bitgen.state)

    assert_array_equal(bulk.random(313), fallback.random(313))
    assert bulk_bitgen.state["has_uint32"] == 1
    assert_equal(bulk_bitgen.state, fallback_bitgen.state)

    assert_array_equal(
        bulk.integers(0, 2**32, size=313, dtype=np.uint32),
        fallback.integers(0, 2**32, size=313, dtype=np.uint32),
    )
    assert bulk_bitgen.state["has_uint32"] == 0
    assert_equal(bulk_bitgen.state, fallback_bitgen.state)
