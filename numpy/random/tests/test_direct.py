import os
from os.path import join

import numpy as np
from numpy.testing import (assert_equal, assert_allclose, assert_array_equal,
                           assert_raises)
import pytest

from numpy.random import (Generator, MT19937, DSFMT, ThreeFry, PCG32, PCG64,
                          Philox, Xoshiro256, Xoshiro512, RandomState)
from numpy.random.common import interface

try:
    import cffi  # noqa: F401

    MISSING_CFFI = False
except ImportError:
    MISSING_CFFI = True

try:
    import ctypes  # noqa: F401

    MISSING_CTYPES = False
except ImportError:
    MISSING_CTYPES = False

pwd = os.path.dirname(os.path.abspath(__file__))


def assert_state_equal(actual, target):
    for key in actual:
        if isinstance(actual[key], dict):
            assert_state_equal(actual[key], target[key])
        elif isinstance(actual[key], np.ndarray):
            assert_array_equal(actual[key], target[key])
        else:
            assert actual[key] == target[key]


def uniform32_from_uint64(x):
    x = np.uint64(x)
    upper = np.array(x >> np.uint64(32), dtype=np.uint32)
    lower = np.uint64(0xffffffff)
    lower = np.array(x & lower, dtype=np.uint32)
    joined = np.column_stack([lower, upper]).ravel()
    out = (joined >> np.uint32(9)) * (1.0 / 2 ** 23)
    return out.astype(np.float32)


def uniform32_from_uint53(x):
    x = np.uint64(x) >> np.uint64(16)
    x = np.uint32(x & np.uint64(0xffffffff))
    out = (x >> np.uint32(9)) * (1.0 / 2 ** 23)
    return out.astype(np.float32)


def uniform32_from_uint32(x):
    return (x >> np.uint32(9)) * (1.0 / 2 ** 23)


def uniform32_from_uint(x, bits):
    if bits == 64:
        return uniform32_from_uint64(x)
    elif bits == 53:
        return uniform32_from_uint53(x)
    elif bits == 32:
        return uniform32_from_uint32(x)
    else:
        raise NotImplementedError


def uniform_from_uint(x, bits):
    if bits in (64, 63, 53):
        return uniform_from_uint64(x)
    elif bits == 32:
        return uniform_from_uint32(x)


def uniform_from_uint64(x):
    return (x >> np.uint64(11)) * (1.0 / 9007199254740992.0)


def uniform_from_uint32(x):
    out = np.empty(len(x) // 2)
    for i in range(0, len(x), 2):
        a = x[i] >> 5
        b = x[i + 1] >> 6
        out[i // 2] = (a * 67108864.0 + b) / 9007199254740992.0
    return out


def uniform_from_dsfmt(x):
    return x.view(np.double) - 1.0


def gauss_from_uint(x, n, bits):
    if bits in (64, 63):
        doubles = uniform_from_uint64(x)
    elif bits == 32:
        doubles = uniform_from_uint32(x)
    else:  # bits == 'dsfmt'
        doubles = uniform_from_dsfmt(x)
    gauss = []
    loc = 0
    x1 = x2 = 0.0
    while len(gauss) < n:
        r2 = 2
        while r2 >= 1.0 or r2 == 0.0:
            x1 = 2.0 * doubles[loc] - 1.0
            x2 = 2.0 * doubles[loc + 1] - 1.0
            r2 = x1 * x1 + x2 * x2
            loc += 2

        f = np.sqrt(-2.0 * np.log(r2) / r2)
        gauss.append(f * x2)
        gauss.append(f * x1)

    return gauss[:n]


class Base(object):
    dtype = np.uint64
    data2 = data1 = {}

    @classmethod
    def setup_class(cls):
        cls.bit_generator = Xoshiro256
        cls.bits = 64
        cls.dtype = np.uint64
        cls.seed_error_type = TypeError
        cls.invalid_seed_types = []
        cls.invalid_seed_values = []

    @classmethod
    def _read_csv(cls, filename):
        with open(filename) as csv:
            seed = csv.readline()
            seed = seed.split(',')
            seed = [int(s.strip(), 0) for s in seed[1:]]
            data = []
            for line in csv:
                data.append(int(line.split(',')[-1].strip(), 0))
            return {'seed': seed, 'data': np.array(data, dtype=cls.dtype)}

    def test_raw(self):
        bit_generator = self.bit_generator(*self.data1['seed'])
        uints = bit_generator.random_raw(1000)
        assert_equal(uints, self.data1['data'])

        bit_generator = self.bit_generator(*self.data1['seed'])
        uints = bit_generator.random_raw()
        assert_equal(uints, self.data1['data'][0])

        bit_generator = self.bit_generator(*self.data2['seed'])
        uints = bit_generator.random_raw(1000)
        assert_equal(uints, self.data2['data'])

    def test_random_raw(self):
        bit_generator = self.bit_generator(*self.data1['seed'])
        uints = bit_generator.random_raw(output=False)
        assert uints is None
        uints = bit_generator.random_raw(1000, output=False)
        assert uints is None

    def test_gauss_inv(self):
        n = 25
        rs = RandomState(self.bit_generator(*self.data1['seed']))
        gauss = rs.standard_normal(n)
        assert_allclose(gauss,
                        gauss_from_uint(self.data1['data'], n, self.bits))

        rs = RandomState(self.bit_generator(*self.data2['seed']))
        gauss = rs.standard_normal(25)
        assert_allclose(gauss,
                        gauss_from_uint(self.data2['data'], n, self.bits))

    def test_uniform_double(self):
        rs = Generator(self.bit_generator(*self.data1['seed']))
        vals = uniform_from_uint(self.data1['data'], self.bits)
        uniforms = rs.random(len(vals))
        assert_allclose(uniforms, vals)
        assert_equal(uniforms.dtype, np.float64)

        rs = Generator(self.bit_generator(*self.data2['seed']))
        vals = uniform_from_uint(self.data2['data'], self.bits)
        uniforms = rs.random(len(vals))
        assert_allclose(uniforms, vals)
        assert_equal(uniforms.dtype, np.float64)

    def test_uniform_float(self):
        rs = Generator(self.bit_generator(*self.data1['seed']))
        vals = uniform32_from_uint(self.data1['data'], self.bits)
        uniforms = rs.random(len(vals), dtype=np.float32)
        assert_allclose(uniforms, vals)
        assert_equal(uniforms.dtype, np.float32)

        rs = Generator(self.bit_generator(*self.data2['seed']))
        vals = uniform32_from_uint(self.data2['data'], self.bits)
        uniforms = rs.random(len(vals), dtype=np.float32)
        assert_allclose(uniforms, vals)
        assert_equal(uniforms.dtype, np.float32)

    def test_seed_float(self):
        # GH #82
        rs = Generator(self.bit_generator(*self.data1['seed']))
        assert_raises(self.seed_error_type, rs.bit_generator.seed, np.pi)
        assert_raises(self.seed_error_type, rs.bit_generator.seed, -np.pi)

    def test_seed_float_array(self):
        # GH #82
        rs = Generator(self.bit_generator(*self.data1['seed']))
        assert_raises(self.seed_error_type, rs.bit_generator.seed,
                      np.array([np.pi]))
        assert_raises(self.seed_error_type, rs.bit_generator.seed,
                      np.array([-np.pi]))
        assert_raises(ValueError, rs.bit_generator.seed,
                      np.array([np.pi, -np.pi]))
        assert_raises(TypeError, rs.bit_generator.seed, np.array([0, np.pi]))
        assert_raises(TypeError, rs.bit_generator.seed, [np.pi])
        assert_raises(TypeError, rs.bit_generator.seed, [0, np.pi])

    def test_seed_out_of_range(self):
        # GH #82
        rs = Generator(self.bit_generator(*self.data1['seed']))
        assert_raises(ValueError, rs.bit_generator.seed,
                      2 ** (2 * self.bits + 1))
        assert_raises(ValueError, rs.bit_generator.seed, -1)

    def test_seed_out_of_range_array(self):
        # GH #82
        rs = Generator(self.bit_generator(*self.data1['seed']))
        assert_raises(ValueError, rs.bit_generator.seed,
                      [2 ** (2 * self.bits + 1)])
        assert_raises(ValueError, rs.bit_generator.seed, [-1])

    def test_repr(self):
        rs = Generator(self.bit_generator(*self.data1['seed']))
        assert 'Generator' in repr(rs)
        assert '{:#x}'.format(id(rs)).upper().replace('X', 'x') in repr(rs)

    def test_str(self):
        rs = Generator(self.bit_generator(*self.data1['seed']))
        assert 'Generator' in str(rs)
        assert str(self.bit_generator.__name__) in str(rs)
        assert '{:#x}'.format(id(rs)).upper().replace('X', 'x') not in str(rs)

    def test_pickle(self):
        import pickle

        bit_generator = self.bit_generator(*self.data1['seed'])
        state = bit_generator.state
        bitgen_pkl = pickle.dumps(bit_generator)
        reloaded = pickle.loads(bitgen_pkl)
        reloaded_state = reloaded.state
        assert_array_equal(Generator(bit_generator).standard_normal(1000),
                           Generator(reloaded).standard_normal(1000))
        assert bit_generator is not reloaded
        assert_state_equal(reloaded_state, state)

    def test_invalid_state_type(self):
        bit_generator = self.bit_generator(*self.data1['seed'])
        with pytest.raises(TypeError):
            bit_generator.state = {'1'}

    def test_invalid_state_value(self):
        bit_generator = self.bit_generator(*self.data1['seed'])
        state = bit_generator.state
        state['bit_generator'] = 'otherBitGenerator'
        with pytest.raises(ValueError):
            bit_generator.state = state

    def test_invalid_seed_type(self):
        bit_generator = self.bit_generator(*self.data1['seed'])
        for st in self.invalid_seed_types:
            with pytest.raises(TypeError):
                bit_generator.seed(*st)

    def test_invalid_seed_values(self):
        bit_generator = self.bit_generator(*self.data1['seed'])
        for st in self.invalid_seed_values:
            with pytest.raises(ValueError):
                bit_generator.seed(*st)

    def test_benchmark(self):
        bit_generator = self.bit_generator(*self.data1['seed'])
        bit_generator._benchmark(1)
        bit_generator._benchmark(1, 'double')
        with pytest.raises(ValueError):
            bit_generator._benchmark(1, 'int32')

    @pytest.mark.skipif(MISSING_CFFI, reason='cffi not available')
    def test_cffi(self):
        bit_generator = self.bit_generator(*self.data1['seed'])
        cffi_interface = bit_generator.cffi
        assert isinstance(cffi_interface, interface)
        other_cffi_interface = bit_generator.cffi
        assert other_cffi_interface is cffi_interface

    @pytest.mark.skipif(MISSING_CTYPES, reason='ctypes not available')
    def test_ctypes(self):
        bit_generator = self.bit_generator(*self.data1['seed'])
        ctypes_interface = bit_generator.ctypes
        assert isinstance(ctypes_interface, interface)
        other_ctypes_interface = bit_generator.ctypes
        assert other_ctypes_interface is ctypes_interface

    def test_getstate(self):
        bit_generator = self.bit_generator(*self.data1['seed'])
        state = bit_generator.state
        alt_state = bit_generator.__getstate__()
        assert_state_equal(state, alt_state)


class TestXoshiro256(Base):
    @classmethod
    def setup_class(cls):
        cls.bit_generator = Xoshiro256
        cls.bits = 64
        cls.dtype = np.uint64
        cls.data1 = cls._read_csv(
            join(pwd, './data/xoshiro256-testset-1.csv'))
        cls.data2 = cls._read_csv(
            join(pwd, './data/xoshiro256-testset-2.csv'))
        cls.seed_error_type = TypeError
        cls.invalid_seed_types = [('apple',), (2 + 3j,), (3.1,)]
        cls.invalid_seed_values = [(-2,), (np.empty((2, 2), dtype=np.int64),)]


class TestXoshiro512(Base):
    @classmethod
    def setup_class(cls):
        cls.bit_generator = Xoshiro512
        cls.bits = 64
        cls.dtype = np.uint64
        cls.data1 = cls._read_csv(
            join(pwd, './data/xoshiro512-testset-1.csv'))
        cls.data2 = cls._read_csv(
            join(pwd, './data/xoshiro512-testset-2.csv'))
        cls.seed_error_type = TypeError
        cls.invalid_seed_types = [('apple',), (2 + 3j,), (3.1,)]
        cls.invalid_seed_values = [(-2,), (np.empty((2, 2), dtype=np.int64),)]


class TestThreeFry(Base):
    @classmethod
    def setup_class(cls):
        cls.bit_generator = ThreeFry
        cls.bits = 64
        cls.dtype = np.uint64
        cls.data1 = cls._read_csv(
            join(pwd, './data/threefry-testset-1.csv'))
        cls.data2 = cls._read_csv(
            join(pwd, './data/threefry-testset-2.csv'))
        cls.seed_error_type = TypeError
        cls.invalid_seed_types = []
        cls.invalid_seed_values = [(1, None, 1), (-1,), (2 ** 257 + 1,),
                                   (None, None, 2 ** 257 + 1)]

    def test_set_key(self):
        bit_generator = self.bit_generator(*self.data1['seed'])
        state = bit_generator.state
        keyed = self.bit_generator(counter=state['state']['counter'],
                                   key=state['state']['key'])
        assert_state_equal(bit_generator.state, keyed.state)


class TestPhilox(Base):
    @classmethod
    def setup_class(cls):
        cls.bit_generator = Philox
        cls.bits = 64
        cls.dtype = np.uint64
        cls.data1 = cls._read_csv(
            join(pwd, './data/philox-testset-1.csv'))
        cls.data2 = cls._read_csv(
            join(pwd, './data/philox-testset-2.csv'))
        cls.seed_error_type = TypeError
        cls.invalid_seed_types = []
        cls.invalid_seed_values = [(1, None, 1), (-1,), (2 ** 257 + 1,),
                                   (None, None, 2 ** 257 + 1)]

    def test_set_key(self):
        bit_generator = self.bit_generator(*self.data1['seed'])
        state = bit_generator.state
        keyed = self.bit_generator(counter=state['state']['counter'],
                                   key=state['state']['key'])
        assert_state_equal(bit_generator.state, keyed.state)


class TestPCG64(Base):
    @classmethod
    def setup_class(cls):
        cls.bit_generator = PCG64
        cls.bits = 64
        cls.dtype = np.uint64
        cls.data1 = cls._read_csv(join(pwd, './data/pcg64-testset-1.csv'))
        cls.data2 = cls._read_csv(join(pwd, './data/pcg64-testset-2.csv'))
        cls.seed_error_type = TypeError
        cls.invalid_seed_types = [(np.array([1, 2]),), (3.2,),
                                  (None, np.zeros(1))]
        cls.invalid_seed_values = [(-1,), (2 ** 129 + 1,), (None, -1),
                                   (None, 2 ** 129 + 1)]

    def test_seed_float_array(self):
        rs = Generator(self.bit_generator(*self.data1['seed']))
        assert_raises(self.seed_error_type, rs.bit_generator.seed,
                      np.array([np.pi]))
        assert_raises(self.seed_error_type, rs.bit_generator.seed,
                      np.array([-np.pi]))
        assert_raises(self.seed_error_type, rs.bit_generator.seed,
                      np.array([np.pi, -np.pi]))
        assert_raises(self.seed_error_type, rs.bit_generator.seed,
                      np.array([0, np.pi]))
        assert_raises(self.seed_error_type, rs.bit_generator.seed, [np.pi])
        assert_raises(self.seed_error_type, rs.bit_generator.seed, [0, np.pi])

    def test_seed_out_of_range_array(self):
        rs = Generator(self.bit_generator(*self.data1['seed']))
        assert_raises(self.seed_error_type, rs.bit_generator.seed,
                      [2 ** (2 * self.bits + 1)])
        assert_raises(self.seed_error_type, rs.bit_generator.seed, [-1])

    def test_advance_symmetry(self):
        rs = Generator(self.bit_generator(*self.data1['seed']))
        state = rs.bit_generator.state
        step = -0x9e3779b97f4a7c150000000000000000
        rs.bit_generator.advance(step)
        val_neg = rs.integers(10)
        rs.bit_generator.state = state
        rs.bit_generator.advance(2**128 + step)
        val_pos = rs.integers(10)
        rs.bit_generator.state = state
        rs.bit_generator.advance(10 * 2**128 + step)
        val_big = rs.integers(10)
        assert val_neg == val_pos
        assert val_big == val_pos

class TestPCG32(TestPCG64):
    @classmethod
    def setup_class(cls):
        cls.bit_generator = PCG32
        cls.bits = 32
        cls.dtype = np.uint32
        cls.data1 = cls._read_csv(join(pwd, './data/pcg32-testset-1.csv'))
        cls.data2 = cls._read_csv(join(pwd, './data/pcg32-testset-2.csv'))
        cls.seed_error_type = TypeError
        cls.invalid_seed_types = [(np.array([1, 2]),), (3.2,),
                                  (None, np.zeros(1))]
        cls.invalid_seed_values = [(-1,), (2 ** 129 + 1,), (None, -1),
                                   (None, 2 ** 129 + 1)]


class TestMT19937(Base):
    @classmethod
    def setup_class(cls):
        cls.bit_generator = MT19937
        cls.bits = 32
        cls.dtype = np.uint32
        cls.data1 = cls._read_csv(join(pwd, './data/mt19937-testset-1.csv'))
        cls.data2 = cls._read_csv(join(pwd, './data/mt19937-testset-2.csv'))
        cls.seed_error_type = ValueError
        cls.invalid_seed_types = []
        cls.invalid_seed_values = [(-1,), np.array([2 ** 33])]

    def test_seed_out_of_range(self):
        # GH #82
        rs = Generator(self.bit_generator(*self.data1['seed']))
        assert_raises(ValueError, rs.bit_generator.seed, 2 ** (self.bits + 1))
        assert_raises(ValueError, rs.bit_generator.seed, -1)
        assert_raises(ValueError, rs.bit_generator.seed,
                      2 ** (2 * self.bits + 1))

    def test_seed_out_of_range_array(self):
        # GH #82
        rs = Generator(self.bit_generator(*self.data1['seed']))
        assert_raises(ValueError, rs.bit_generator.seed,
                      [2 ** (self.bits + 1)])
        assert_raises(ValueError, rs.bit_generator.seed, [-1])
        assert_raises(TypeError, rs.bit_generator.seed,
                      [2 ** (2 * self.bits + 1)])

    def test_seed_float(self):
        # GH #82
        rs = Generator(self.bit_generator(*self.data1['seed']))
        assert_raises(TypeError, rs.bit_generator.seed, np.pi)
        assert_raises(TypeError, rs.bit_generator.seed, -np.pi)

    def test_seed_float_array(self):
        # GH #82
        rs = Generator(self.bit_generator(*self.data1['seed']))
        bit_generator = rs.bit_generator
        assert_raises(TypeError, bit_generator.seed, np.array([np.pi]))
        assert_raises(TypeError, bit_generator.seed, np.array([-np.pi]))
        assert_raises(TypeError, bit_generator.seed, np.array([np.pi, -np.pi]))
        assert_raises(TypeError, bit_generator.seed, np.array([0, np.pi]))
        assert_raises(TypeError, bit_generator.seed, [np.pi])
        assert_raises(TypeError, bit_generator.seed, [0, np.pi])

    def test_state_tuple(self):
        rs = Generator(self.bit_generator(*self.data1['seed']))
        bit_generator = rs.bit_generator
        state = bit_generator.state
        desired = rs.integers(2 ** 16)
        tup = (state['bit_generator'], state['state']['key'],
               state['state']['pos'])
        bit_generator.state = tup
        actual = rs.integers(2 ** 16)
        assert_equal(actual, desired)
        tup = tup + (0, 0.0)
        bit_generator.state = tup
        actual = rs.integers(2 ** 16)
        assert_equal(actual, desired)


class TestDSFMT(Base):
    @classmethod
    def setup_class(cls):
        cls.bit_generator = DSFMT
        cls.bits = 53
        cls.dtype = np.uint64
        cls.data1 = cls._read_csv(join(pwd, './data/dSFMT-testset-1.csv'))
        cls.data2 = cls._read_csv(join(pwd, './data/dSFMT-testset-2.csv'))
        cls.seed_error_type = TypeError
        cls.invalid_seed_types = []
        cls.invalid_seed_values = [(-1,), np.array([2 ** 33]),
                                   (np.array([2 ** 33, 2 ** 33]),)]

    def test_uniform_double(self):
        rs = Generator(self.bit_generator(*self.data1['seed']))
        assert_array_equal(uniform_from_dsfmt(self.data1['data']),
                           rs.random(1000))

        rs = Generator(self.bit_generator(*self.data2['seed']))
        assert_equal(uniform_from_dsfmt(self.data2['data']),
                     rs.random(1000))

    def test_gauss_inv(self):
        n = 25
        rs = RandomState(self.bit_generator(*self.data1['seed']))
        gauss = rs.standard_normal(n)
        assert_allclose(gauss,
                        gauss_from_uint(self.data1['data'], n, 'dsfmt'))

        rs = RandomState(self.bit_generator(*self.data2['seed']))
        gauss = rs.standard_normal(25)
        assert_allclose(gauss,
                        gauss_from_uint(self.data2['data'], n, 'dsfmt'))

    def test_seed_out_of_range_array(self):
        # GH #82
        rs = Generator(self.bit_generator(*self.data1['seed']))
        assert_raises(ValueError, rs.bit_generator.seed,
                      [2 ** (self.bits + 1)])
        assert_raises(ValueError, rs.bit_generator.seed, [-1])
        assert_raises(TypeError, rs.bit_generator.seed,
                      [2 ** (2 * self.bits + 1)])

    def test_seed_float(self):
        # GH #82
        rs = Generator(self.bit_generator(*self.data1['seed']))
        assert_raises(TypeError, rs.bit_generator.seed, np.pi)
        assert_raises(TypeError, rs.bit_generator.seed, -np.pi)

    def test_seed_float_array(self):
        # GH #82
        rs = Generator(self.bit_generator(*self.data1['seed']))
        assert_raises(TypeError, rs.bit_generator.seed, np.array([np.pi]))
        assert_raises(TypeError, rs.bit_generator.seed, np.array([-np.pi]))
        assert_raises(TypeError, rs.bit_generator.seed,
                      np.array([np.pi, -np.pi]))
        assert_raises(TypeError, rs.bit_generator.seed, np.array([0, np.pi]))
        assert_raises(TypeError, rs.bit_generator.seed, [np.pi])
        assert_raises(TypeError, rs.bit_generator.seed, [0, np.pi])

    def test_uniform_float(self):
        rs = Generator(self.bit_generator(*self.data1['seed']))
        vals = uniform32_from_uint(self.data1['data'], self.bits)
        uniforms = rs.random(len(vals), dtype=np.float32)
        assert_allclose(uniforms, vals)
        assert_equal(uniforms.dtype, np.float32)

        rs = Generator(self.bit_generator(*self.data2['seed']))
        vals = uniform32_from_uint(self.data2['data'], self.bits)
        uniforms = rs.random(len(vals), dtype=np.float32)
        assert_allclose(uniforms, vals)
        assert_equal(uniforms.dtype, np.float32)

    def test_buffer_reset(self):
        rs = Generator(self.bit_generator(*self.data1['seed']))
        rs.random(1)
        assert rs.bit_generator.state['buffer_loc'] != 382
        rs.bit_generator.seed(*self.data1['seed'])
        assert rs.bit_generator.state['buffer_loc'] == 382
