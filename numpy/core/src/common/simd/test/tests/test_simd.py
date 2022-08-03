# NOTE: Please avoid the use of numpy.testing since NPYV intrinsics
# may be involved in their functionality.
import pytest
import math
import re
import operator
import itertools

from contextlib import suppress
from pytest import mark
from numpy.core._simd import (
    targets, Array, Vec, Vec2, Vec3, Vec4, Mask,
    uint8_t, int8_t,  uint16_t, int16_t,
    uint32_t, int32_t, uint64_t, int64_t,
    float_, double, size_t, intptr_t,
    int_, floatstatus
)

def pytest_generate_tests(metafunc):
    metafunc.parametrize('simd', targets.values(), indirect=True)

@pytest.fixture
def simd(request):
    ext = request.param
    name = str(ext)
    if not hasattr(ext, 'SIMD'):
        pytest.skip(
            f"SIMD extention '{name}' isn't supported by the platform"
        )
    elif not ext.SIMD:
        pytest.skip(
            f"SIMD extention '{name}' isn't supported by "
            "the universal intrinsics"
        )
    for fname in ("TLane", "TLane0", "TLane1", "TLane2"):
        if fname not in request.fixturenames:
            continue
        tlane = request.getfixturevalue(fname)
        if not ext.kSupportLane(tlane(0)):
            pytest.skip(
                f"SIMD extention '{name}' doesn't support {tlane.__name__}"
            )
    return ext

@pytest.fixture
def NLanes(request):
    simd = request.getfixturevalue('simd')
    TLane = request.getfixturevalue('TLane')
    return int(simd.NLanes(TLane(0)))

@pytest.fixture
def TULane(request):
    simd = request.getfixturevalue('simd')
    TLane = request.getfixturevalue('TLane')
    ulane = {
        1: uint8_t,
        2: uint16_t,
        4: uint32_t,
        8: uint64_t
    }.get(TLane.element_size)
    return ulane

@pytest.fixture
def SetMask(request):
    simd = request.getfixturevalue('simd')
    TLane = request.getfixturevalue('TLane')
    if simd.kStrongMask:
        ulane = {
            1: uint8_t,
            2: uint16_t,
            4: uint32_t,
            8: uint64_t
        }.get(TLane.element_size)
    else:
        ulane = uint8_t
    diff_size = TLane.element_size // ulane.element_size
    return lambda *args, ulane=ulane, diff_size=diff_size: Mask(ulane)(
        *itertools.chain(*zip(*[args]*diff_size))
    )

@pytest.fixture
def VData(request):
    """
    Create list of consecutive numbers according to number
    of vector's lanes and casted based on lane's type..
    """
    def call(start, count, TLane, container):
        rng = range(start, start + count)
        return container(TLane)(*rng)

    simd = request.getfixturevalue('simd')
    TLane = request.getfixturevalue('TLane')
    nlanes = int(simd.NLanes(TLane(0)))
    return lambda start=1, count=nlanes, TLane=TLane, container=Vec: call(
        start, count, TLane, container
    )

@pytest.fixture
def ClipEdge(request):
    TLane = request.getfixturevalue('TLane')
    lane_size = TLane.element_size
    is_signed = TLane in (int8_t, int16_t, int32_t, int64_t)
    max_u = (1 << (lane_size * 8)) - 1
    max_TLane = max_u // 2 if is_signed else max_u
    min_TLane = -(max_TLane + 1) if is_signed else 0
    return lambda val, maxt=max_TLane, mint=min_TLane: min(
        max(val, mint), maxt
    )

@pytest.fixture
def LimitInt(request):
    TLane = request.getfixturevalue('TLane')
    lane_size = TLane.element_size
    is_signed = TLane in (int8_t, int16_t, int32_t, int64_t)
    max_u = (1 << (lane_size * 8)) - 1
    max_TLane = max_u // 2 if is_signed else max_u
    min_TLane = -(max_TLane + 1) if is_signed else 0
    return min_TLane, max_TLane

#
# `Miscellaneous`:
# | Name           |u8 |i8 |u16|i16|u32|i32|u64|i64|f32|f64|
# |:---------------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
# | Width          |   |   |   |   |   |   |   |   |   |   |
# | Nlanes         | x | x | x | x | x | x | x | x | x | x |
# | Undef          | x | x | x | x | x | x | x | x | x | x |
# | Zero           | x | x | x | x | x | x | x | x | x | x |
# | Set            | x | x | x | x | x | x | x | x | x | x |
# | Get0           | x | x | x | x | x | x | x | x | x | x |
# | SetTuple       | x | x | x | x | x | x | x | x | x | x |
# | GetTuple       | x | x | x | x | x | x | x | x | x | x |
# | Select         | x | x | x | x | x | x | x | x | x | x |
# | Reinterpret    | x | x | x | x | x | x | x | x | x | x |
# | Cleanup        |   |   |   |   |   |   |   |   |   |   |
#
@mark.parametrize('TLane', [
    uint8_t, int8_t, uint16_t, int16_t,
    uint32_t, int32_t, uint64_t, int64_t,
    float_, double
])
class TestMisc:
    def test_undef(self, simd, TLane):
        # dummy test just to make sure it coverd by the python module
        v = simd.Undef(TLane(0))
        assert v == v

    def test_nlanes_width(self, simd, TLane):
        data = simd.Width() // TLane.element_size
        test = simd.NLanes(TLane(0))
        assert test == data

    def test_zero(self, simd, TLane, NLanes):
        data = Vec(TLane)(0) * NLanes
        test = simd.Zero(TLane(0))
        assert test == data

    def test_select(self, simd, TLane, NLanes, SetMask):
        a = Vec(TLane)(0) * NLanes
        b = Vec(TLane)(1) * NLanes
        mask = SetMask(*[True]*NLanes)
        test = simd.Select(mask, a, b)
        assert test == a
        test = simd.Select(mask, b, a)
        assert test == b

    @mark.parametrize('TLane2', [
        uint8_t, int8_t, uint16_t, int16_t,
        uint32_t, int32_t, uint64_t, int64_t,
        float_, double
    ])
    def test_reinterpret(self, simd, TLane, TLane2):
        # We're testing the sanity of _simd's type-vector,
        # reinterpret* intrinsics itself are tested via compiler
        # during the build of _simd module
        v = simd.Reinterpret(simd.Zero(TLane(0)), TLane2(0))
        assert v == simd.Zero(TLane2(0))

    @mark.parametrize("nset", (1, 2))  # , 4, 8l 16, 32, 64))
    def test_set(self, simd, TLane, nset, VData, NLanes):
        for val in (
            0, 1, -1, -0x80, -0x8000, -0x80000000, -0x8000000000000000,
            0x7f, 0x7fff, 0x7fffffff, 0x7fffffffffffffff
        ):
            data = VData(val, nset) * (NLanes // nset)
            test = simd.Set(*data[:nset])
            assert test == data

    def test_get0(self, simd, TLane, VData, LimitInt):
        data = VData(LimitInt[0])
        assert simd.Get0(data) == data[0]
        data = VData(LimitInt[1])
        assert simd.Get0(data) == data[0]

    @mark.parametrize('container, ind', [
        (Vec2, 2),
        (Vec3, 3),
        (Vec4, 4),
    ])
    def test_get_set_tuple(self, simd, TLane, container, ind, NLanes, VData):
        data = [VData(NLanes * i) for i in range(ind)]
        data = container(TLane)(*data)
        test = simd.SetTuple(*data)
        assert test == data
        for i in range(ind):
            assert simd.GetTuple(test, size_t(i)) == data[i]


def test_cleanup(simd):
    # cleanup intrinsic is only used with AVX for
    # zeroing registers to avoid the AVX-SSE transition penalty,
    # so nothing to test here
    simd.Cleanup()

#
# Memory
# | Name             |u8 |i8 |u16|i16|u32|i32|u64|i64|f32|f64|
# |:-----------------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
# | Load             | x | x | x | x | x | x | x | x | x | x |
# | LoadAligned      | x | x | x | x | x | x | x | x | x | x |
# | LoadStream       | x | x | x | x | x | x | x | x | x | x |
# | LoadLow          | x | x | x | x | x | x | x | x | x | x |
# | LoadDeinter2     | x | x | x | x | x | x | x | x | x | x |
# | LoadTill         | - | - | - | - | x | x | x | x | x | x |
# | LoadPairTill     | - | - | - | - | x | x | x | x | x | x |
# | IsLoadable       | - | - | - | - | x | x | x | x | x | x |
# | Loadn            | - | - | - | - | x | x | x | x | x | x |
# | LoadnTill        | - | - | - | - | x | x | x | x | x | x |
# | LoadnPair        | - | - | - | - | x | x | x | x | x | x |
# | LoadnPairTill    | - | - | - | - | x | x | x | x | x | x |
# | Store            | x | x | x | x | x | x | x | x | x | x |
# | StoreStream      | x | x | x | x | x | x | x | x | x | x |
# | StoreAligned     | x | x | x | x | x | x | x | x | x | x |
# | StoreLow         | x | x | x | x | x | x | x | x | x | x |
# | StoreHigh        | x | x | x | x | x | x | x | x | x | x |
# | StoreInter2      | x | x | x | x | x | x | x | x | x | x |
# | StoreTill        | - | - | - | - | x | x | x | x | x | x |
# | StorePairTill    | - | - | - | - | x | x | x | x | x | x |
# | IsStorable       | - | - | - | - | x | x | x | x | x | x |
# | Storen           | - | - | - | - | x | x | x | x | x | x |
# | StorenTill       | - | - | - | - | x | x | x | x | x | x |
# | StorenPair       | - | - | - | - | x | x | x | x | x | x |
# | StorenPairTill   | - | - | - | - | x | x | x | x | x | x |
# | Lookup128        | - | - | - | - | x | x | x | x | x | x |
#
@mark.parametrize('TLane', [
    uint8_t, int8_t, uint16_t, int16_t,
    uint32_t, int32_t, uint64_t, int64_t,
    float_, double
])
class TestContiguousMemory:
    def test_load(self, simd, TLane, VData, NLanes):
        data = VData(container=Array)
        # unaligned load
        test = simd.Load(data)
        assert test == data
        # aligned load
        test = simd.LoadAligned(data)
        assert test == data
        # stream load
        test = simd.LoadStream(data)
        assert test == data
        # load lower part
        test = simd.LoadLow(data)
        assert test != data  # detect overflow
        test = test[:NLanes//2]
        data = data[:NLanes//2]
        assert test == data

    def test_store(self, simd, TLane, VData, NLanes):
        data = VData()
        # unaligned store
        test = Array(TLane)(0) * NLanes
        simd.Store(test, data)
        assert test == data
        # aligned store
        test = Array(TLane)(0) * NLanes
        simd.StoreAligned(test, data)
        assert test == data
        # stream store
        test = Array(TLane)(0) * NLanes
        simd.StoreStream(test, data)
        assert test == data
        # store lower part
        test = Array(TLane)(0xff) * NLanes
        hf_nlanes = NLanes // 2
        simd.StoreLow(test, data)
        assert test[:hf_nlanes] == data[:hf_nlanes]
        # detect overflow
        assert test[hf_nlanes:] == Array(TLane)(0xff) * hf_nlanes
        # store higher part
        test = Array(TLane)(0xff) * NLanes
        simd.StoreHigh(test, data)
        assert test[:hf_nlanes] == data[hf_nlanes:]
        # detect overflow
        assert test[hf_nlanes:] == Array(TLane)(0xff) * hf_nlanes

    def test_load_deinterleave(self, simd, TLane, VData, NLanes):
        # Two channel
        data = VData(count=NLanes*2, container=Array)
        test = simd.LoadDeinter2(data)
        assert test[0] == data[::2]
        assert test[1] == data[1::2]

    def test_store_interleave(self, simd, TLane, VData, NLanes):
        # Two channel
        data = Vec2(TLane)(VData(), VData(NLanes))
        test = Array(TLane)(0) * NLanes * 2
        simd.StoreInter2(test, data)
        assert test[::2] == data[0]
        assert test[1::2] == data[1]


@mark.parametrize('TLane', [
    uint32_t, int32_t, uint64_t, int64_t,
    float_, double
])
class TestMemory:
    @mark.parametrize("intrin, scale, fill", [
        ("simd.LoadTill", 1, [0xffff]),
        ("simd.LoadPairTill", 2, [0xffff, 0x7fff]),
    ])
    def test_partial_load(self, simd, TLane, intrin, scale, fill,
                          NLanes, VData):
        intrin = eval(intrin)
        data = VData(container=Array)
        fill = Array(TLane)(*fill)
        lanes = list(range(1, NLanes + 1))
        lanes += [NLanes**2, NLanes**4]
        for n in lanes:
            test = intrin(data, size_t(n), *fill)
            nscale = n*scale
            # fill the rest by specfied scalar
            data_till = data[:nscale] + fill * ((NLanes-nscale) // scale)
            assert test == data_till
            # fill the rest by zero
            data_till = data[:nscale] + [0] * (NLanes-nscale)
            test = intrin(data, size_t(n))
            assert test == data_till

    @mark.parametrize("intrin, scale", [
        ("simd.StoreTill", 1),
        ("simd.StorePairTill", 2),
    ])
    def test_partial_store(self, simd, TLane, intrin, scale, NLanes, VData):
        intrin = eval(intrin)
        data = VData()
        lanes = list(range(1, NLanes + 1))
        lanes += [NLanes**2, NLanes**4]
        tarray = Array(TLane)
        for n in lanes:
            test = tarray(0xff) * NLanes
            intrin(test, size_t(n), data)
            n = min(n, NLanes // scale)
            assert test[:n] == data[:n]
            # detect overflow
            assert test[n*scale:] == Array(TLane)(0xff) * (NLanes-n*scale)

    @mark.parametrize("intrin, scale", [
        ("simd.Loadn", 1),
        ("simd.LoadnPair", 2),
    ])
    def test_noncont_load(self, simd, TLane, intrin, scale, NLanes, VData):
        intrin = eval(intrin)
        tarray = Array(TLane)
        for stride in range(-10, 10):
            if stride < 0:
                data = VData(stride, -stride*NLanes, container=Array)
                data_stride = itertools.chain(
                    *zip(*[data[-i::stride] for i in range(scale, 0, -1)])
                )
            elif stride == 0:
                data = VData(container=Array)
                data_stride = data[0:scale] * (NLanes//scale)
            else:
                data = VData(count=stride*NLanes, container=Array)
                data_stride = itertools.chain(
                    *zip(*[data[i::stride] for i in range(scale)])
                )
            data_stride = tarray(*data_stride)[:NLanes]
            test = intrin(data, intptr_t(stride))
            assert test == data_stride

    @mark.parametrize("intrin, scale, fill", [
        ("simd.LoadnTill", 1, [0xffff]),
        ("simd.LoadnPairTill", 2, [0xffff, 0x7fff]),
    ])
    def test_noncont_partial_load(self, simd, TLane, intrin, scale, fill,
                                  NLanes, VData):
        intrin = eval(intrin)
        lanes = list(range(1, NLanes + 1))
        lanes += [NLanes**2, NLanes**4]
        tarray = Array(TLane)
        fill = tarray(*fill)
        for stride in range(-10, 10):
            if stride < 0:
                data = VData(stride, -stride*NLanes, container=Array)
                data_stride = itertools.chain(
                    *zip(*[data[-i::stride] for i in range(scale, 0, -1)])
                )
            elif stride == 0:
                data = VData(container=Array)
                data_stride = data[0:scale] * (NLanes//scale)
            else:
                data = VData(count=stride*NLanes, container=Array)
                data_stride = itertools.chain(
                    *zip(*[data[i::stride] for i in range(scale)])
                )
            data_stridex = list(data_stride)
            data_stride = tarray(*data_stridex)[:NLanes]
            for n in lanes:
                nscale = n * scale
                llanes = NLanes - nscale
                data_stride_till = (
                    data_stride[:nscale] + fill * (llanes//scale)
                )
                test = intrin(data, intptr_t(stride), size_t(n), *fill)
                assert test == data_stride_till
                # fill the rest lanes by zero
                data_stride_till = data_stride[:nscale] + [0] * llanes
                test = intrin(data, intptr_t(stride), size_t(n))
                assert test == data_stride_till

    @mark.parametrize("intrin, scale", [
        ("simd.Storen", 1),
        ("simd.StorenPair", 2),
    ])
    def test_noncont_store(self, simd, TLane, intrin, scale, NLanes, VData):
        intrin = eval(intrin)
        data = VData()
        hlanes = NLanes // scale
        block_set = [0x7f]*64
        tarray = Array(TLane)
        for stride in range(1, 10):
            data_storen = [0xff] * stride * NLanes
            for s in range(0, hlanes*stride, stride):
                i = (s//stride)*scale
                data_storen[s:s+scale] = data[i:i+scale]
            test = tarray(*([0xff] * stride * NLanes), *block_set)
            intrin(test, intptr_t(stride), data)
            assert test[:-64] == data_storen
            assert test[-64:] == block_set  # detect overflow

        for stride in range(-10, 0):
            data_storen = [0xff] * -stride * NLanes
            for s in range(0, hlanes*stride, stride):
                i = (s//stride)*scale
                data_storen[s-scale:s or None] = data[i:i+scale]
            test = tarray(*block_set, *([0xff] * -stride * NLanes))
            intrin(test, intptr_t(stride), data)
            assert test[64:] == data_storen
            assert test[:64] == block_set  # detect overflow

        # stride 0
        data_storen = [0x7f] * NLanes
        data_storen[0:scale] = data[-scale:]
        test = tarray(*data_storen)
        intrin(test, intptr_t(0), data)
        assert test == data_storen

    @mark.parametrize("intrin, scale", [
        ("simd.StorenTill", 1),
        ("simd.StorenPairTill", 2),
    ])
    def test_noncont_partial_store(self, simd, TLane, intrin, scale,
                                   NLanes, VData):
        intrin = eval(intrin)
        data = VData()
        hlanes = NLanes // scale
        lanes = list(range(1, NLanes + 1))
        lanes += [NLanes**2, NLanes**4]
        block_set = [0x7f]*64
        tarray = Array(TLane)
        for stride in range(1, 10):
            for n in lanes:
                data_till = [0xff] * stride * NLanes
                tdata = data[:n*scale] + [0xff] * (NLanes-n*scale)
                for s in range(0, hlanes*stride, stride)[:n]:
                    i = (s//stride)*scale
                    data_till[s:s+scale] = tdata[i:i+scale]
                test = tarray(*([0xff] * stride * NLanes), *block_set)
                intrin(test, intptr_t(stride), size_t(n), data)
                assert test[:-64] == data_till
                assert test[-64:] == block_set  # detect overflow

        for stride in range(-10, 0):
            for n in lanes:
                data_till = [0xff] * -stride * NLanes
                tdata = data[:n*scale] + [0xff] * (NLanes-n*scale)
                for s in range(0, hlanes*stride, stride)[:n]:
                    i = (s//stride)*scale
                    data_till[s-scale:s or None] = tdata[i:i+scale]
                test = tarray(*block_set, *([0xff] * -stride * NLanes))
                intrin(test, intptr_t(stride), size_t(n), data)
                assert test[64:] == data_till
                assert test[:64] == block_set  # detect overflow

        # stride 0
        for n in lanes:
            data_till = [0x7f] * NLanes
            test = tarray(*data_till)
            data_till[0:scale] = data[:n*scale][-scale:]
            intrin(test, intptr_t(0), size_t(n), data)
            assert test == data_till

    def test_lookup(self, simd, TLane, NLanes, VData, TULane):
        table = VData(start=0, count=128//TLane.element_size, container=Array)
        for i in table:
            broadi = simd.Set(TLane(i))
            idx = simd.Set(TULane(i))
            test = simd.Lookup128(table, idx)
            assert test == broadi

#
# `bitwise`:
# | name               |u8 |i8 |u16|i16|u32|i32|u64|i64|f32|f64|
# |--------------------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
# | And                | x | x | x | x | x | x | x | x | x | x |
# | Andc               | x | x | x | x | x | x | x | x | x | x |
# | Or                 | x | x | x | x | x | x | x | x | x | x |
# | Orc                | x | x | x | x | x | x | x | x | x | x |
# | Xor                | x | x | x | x | x | x | x | x | x | x |
# | Xnor               | x | x | x | x | x | x | x | x | x | x |
# | Not                | x | x | x | x | x | x | x | x | x | x |
# | Shr                | - | - | x | x | x | x | x | x | - | - |
# | Shl                | - | - | x | x | x | x | x | x | - | - |
# | Shli               | - | - | x | x | x | x | x | x | - | - |
# | Shri               | - | - | x | x | x | x | x | x | - | - |
#
@mark.parametrize('TLane', [
    uint8_t, int8_t, uint16_t, int16_t,
    uint32_t, int32_t, uint64_t, int64_t,
    float_, double
])
class TestBitwise:
    @mark.parametrize('pyop, intrin', [
        (operator.and_, "simd.And"),
        (operator.or_, "simd.Or"),
        (operator.xor, "simd.Xor"),
        (lambda a, b: a & ~b, "simd.Andc"),
        (lambda a, b: a | ~b, "simd.Orc"),
        (lambda a, b: ~(a ^ b), "simd.Xnor"),
    ])
    def test_bin(self, simd, TLane, pyop, intrin, VData, SetMask,
                 NLanes, TULane):
        intrin = eval(intrin)
        a, b = VData(0), VData(-NLanes)
        data = pyop(a, b)
        test = intrin(a, b)
        # Reinterpret to bypass NaNs
        assert simd.Reinterpret(test, TULane(0)) == \
               simd.Reinterpret(data, TULane(0))
        a, b = SetMask(*a), SetMask(*b)
        data = pyop(a, b)
        test = intrin(a, b)
        assert test == data

    @mark.parametrize('pyop, intrin', [
        (operator.invert, "simd.Not"),
    ])
    def test_un(self, simd, TLane, pyop, intrin, VData, SetMask,
                      NLanes, TULane):
        intrin = eval(intrin)
        a = VData(0)
        data = pyop(a)
        test = intrin(a)
        # Reinterpret to bypass NaNs
        assert simd.Reinterpret(test, TULane(0)) == \
               simd.Reinterpret(data, TULane(0))
        a = SetMask(*a)
        data = pyop(a)
        test = intrin(a)
        assert test == data

@mark.parametrize('TLane', [
    uint16_t, int16_t, uint32_t, int32_t, uint64_t, int64_t
])
def test_shift(simd, TLane, VData, LimitInt, NLanes):
    data = (VData(LimitInt[0]), VData(LimitInt[1] - NLanes))
    for d in data:
        for count in range(1, TLane.element_size * 8):
            data_test = Array(TLane)(*[int(x) << count for x in d])
            test = simd.Shl(d, int_(count))
            assert test == data_test
            test = simd.Shli(d, int_(count))
            assert test == data_test
            data_test = Array(TLane)(*[int(x) >> count for x in d])
            test = simd.Shr(d, int_(count))
            assert test == data_test
            test = simd.Shri(d, int_(count))
            assert test == data_test
    # test zero
    data = data[0]
    test = simd.Shl(data, int_(0))
    assert test == data
    test = simd.Shr(data, int_(0))
    assert test == data

#
# `Comparison`:
# | Name             |u8 |i8 |u16|i16|u32|i32|u64|i64|f32|f64|
# |------------------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
# | Gt               | x | x | x | x | x | x | x | x | x | x |
# | Ge               | x | x | x | x | x | x | x | x | x | x |
# | Lt               | x | x | x | x | x | x | x | x | x | x |
# | Le               | x | x | x | x | x | x | x | x | x | x |
# | Eq               | x | x | x | x | x | x | x | x | x | x |
# | Ne               | x | x | x | x | x | x | x | x | x | x |
# | NotNan           | - | - | - | - | - | - | - | - | x | x |
# | Any              | x | x | x | x | x | x | x | x | x | x |
# | All              | x | x | x | x | x | x | x | x | x | x |
#
@mark.parametrize('TLane', [float_, double])
class TestComparisonFPSpecialCases:
    pinf, ninf, nan = float("inf"), -float("inf"), float("nan")

    @mark.parametrize('pyop, intrin', [
        (operator.lt, "simd.Lt"),
        (operator.le, "simd.Le"),
        (operator.gt, "simd.Gt"),
        (operator.ge, "simd.Ge"),
        (operator.eq, "simd.Eq"),
        (operator.ne, "simd.Ne")
    ])
    def test_special_cases(self, simd, TLane, pyop, intrin, VData,
                           SetMask, NLanes):
        intrin = eval(intrin)
        cmp_cases = ((0, nan), (nan, 0), (nan, nan), (pinf, nan),
                     (ninf, nan), (-0.0, +0.0))

        for case_operand1, case_operand2 in cmp_cases:
            a = Vec(TLane)(case_operand1) * NLanes
            b = Vec(TLane)(case_operand2) * NLanes
            cmp = SetMask(*[pyop(a, b) for a, b in zip(a, b)])
            test = intrin(a, b)
            assert test == cmp

    def test_special_cases(self, simd, TLane, SetMask, NLanes):
        data = Vec(TLane)(float("nan")) * NLanes
        test = simd.NotNan(data)
        assert test == SetMask(*[False]*NLanes)
        data = Vec(TLane)(0) * NLanes
        test = simd.NotNan(data)
        assert test == SetMask(*[True]*NLanes)

    @mark.parametrize("pyop, intrin", [
        (any, "simd.Any"),
        (all, "simd.All")
    ])
    @mark.parametrize("operand", (
        [nan, 0],
        [0, nan],
        [nan, 1],
        [1, nan],
        [nan, nan],
        [0.0, -0.0],
        [-0.0, 0.0],
        [1.0, -0.0]
    ))
    def test_crosstest(self, simd, TLane, pyop, intrin, operand, NLanes):
        intrin = eval(intrin)
        data = Vec(TLane)(*operand) * NLanes
        cmp = pyop(data)
        test = intrin(data)
        assert test == cmp

@mark.parametrize('TLane', [
    uint8_t, int8_t, uint16_t, int16_t,
    uint32_t, int32_t, uint64_t, int64_t,
    float_, double
])
class TestComparison:
    @mark.parametrize('pyop, intrin', [
        (operator.lt, "simd.Lt"),
        (operator.le, "simd.Le"),
        (operator.gt, "simd.Gt"),
        (operator.ge, "simd.Ge"),
        (operator.eq, "simd.Eq"),
        (operator.ne, "simd.Ne")
    ])
    def test_bin(self, simd, TLane, pyop, intrin, VData, NLanes, SetMask):
        intrin = eval(intrin)
        a, b = VData(), VData(NLanes)
        cmp = SetMask(*[pyop(a, b) for a, b in zip(a, b)])
        test = intrin(a, b)
        assert test == cmp
        cmp = SetMask(*[pyop(b, a) for a, b in zip(a, b)])
        test = intrin(b, a)
        assert test == cmp

    @mark.parametrize("pyop, intrin", [
        (any, "simd.Any"),
        (all, "simd.All")
    ])
    def test_crosstest(self, simd, TLane, pyop, intrin, NLanes, SetMask):
        intrin = eval(intrin)
        for data in (
            [1, 2, 3, 4],
            [-1, -2, -3, -4],
            [0, 1, 2, 0, 4],
            [0x7f, 0x7fff, 0x7fffffff, 0x7fffffffffffffff],
            [0, -1, 0, -3, 4],
            [0],
            [1],
            [-1]
        ):
            data = Vec(TLane)(*data*NLanes)
            cmp = pyop(data)
            test = intrin(data)
            assert test == cmp
            # test mask
            data = SetMask(*data)
            cmp = pyop(data)
            test = intrin(data)
            assert test == cmp

#
# `Conversion`:
# | Name                |u8 |i8 |u16|i16|u32|i32|u64|i64|f32|f64|
# |:--------------------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
# | @ref ToMask         | x | x | x | x | x | x | x | x | - | - |
# | @ref ToVec          | x | x | x | x | x | x | x | x | - | - |
# | @ref Expand         | x | - | x | - | - | - | - | - | - | - |
# | @ref Pack (Mask)    | x |   | x |   | x |   | x |   |   |   |
#
@mark.parametrize('TLane', [
    uint8_t, int8_t, uint16_t, int16_t,
    uint32_t, int32_t, uint64_t, int64_t,
])
def test_conversion_to_vec_mask(simd, TLane, NLanes, SetMask):
    a = Vec(TLane)(*[-1, 0] * (NLanes // 2))
    test = simd.ToMask(a)
    mask = SetMask(*a)
    assert test == mask
    test = simd.ToVec(mask, TLane(0))
    assert test == a

@mark.parametrize('TLane', [uint8_t, uint16_t])
def test_conversion_expand(simd, TLane, VData, NLanes, LimitInt):
    a = VData(LimitInt[1] - NLanes)
    test = simd.Expand(a)
    hf_nlanes = NLanes // 2
    assert test == (a[:hf_nlanes], a[hf_nlanes:])

@mark.parametrize('TLane', [uint16_t])
def test_conversion_pack_mask(simd, TLane, NLanes, SetMask):
    hf_nlanes = NLanes // 2
    a = SetMask(*[True, False] * hf_nlanes)
    b = SetMask(*[False, True] * hf_nlanes)
    test = simd.Pack(a, b, TLane(0))
    data = type(test)(*[True, False] * hf_nlanes,
                      *[False, True] * hf_nlanes)
    assert test == data

@mark.parametrize('TLane, to_test', [
    (uint32_t, ([True, False], [False, True], [True, True], [False, False])),
    (uint64_t, ([True, False], [False, True], [False, False], [True, True],
                [False, True], [True, False], [True, True], [False, False])),
])
def test_conversion_pack_mask8(simd, TLane, NLanes, SetMask, to_test):
    hf_nlanes = NLanes // 2
    to_test = [t * hf_nlanes for t in to_test]
    test = simd.Pack(*[SetMask(*t) for t in to_test])
    data = type(test)(*itertools.chain(*to_test))
    assert test == data

#
# `Arithmetic`:
# | Name               |u8 |i8 |u16|i16|u32|i32|u64|i64|f32|f64|
# |--------------------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
# | Add                | x | x | x | x | x | x | x | x | x | x |
# | IfAdd              | x | x | x | x | x | x | x | x | x | x |
# | Adds               | x | x | x | x | - | - | - | - | - | - |
# | Sub                | x | x | x | x | x | x | x | x | x | x |
# | IfSub              | x | x | x | x | x | x | x | x | x | x |
# | Subs               | x | x | x | x | - | - | - | - | - | - |
# | Mul                | x | x | x | x | x | x | - | - | x | x |
# | Div                | x | x | x | x | x | x | x | x | x | x |
# | Divisor            | x | x | x | x | x | x | x | x | - | - |
# | MulAdd             | - | - | - | - | - | - | - | - | x | x |
# | MulSub             | - | - | - | - | - | - | - | - | x | x |
# | NegMulAdd          | - | - | - | - | - | - | - | - | x | x |
# | NegMulSub          | - | - | - | - | - | - | - | - | x | x |
# | MulAddSub          | - | - | - | - | - | - | - | - | x | x |
# | Sum                | - | - | - | - | x | - | x | - | x | x |
# | Sumup              | x | - | x | - | - | - | - | - | - | - |
#
@mark.parametrize('TLane', [
    uint8_t, int8_t, uint16_t, int16_t,
    uint32_t, int32_t, uint64_t, int64_t
])
class TestArithmeticInt:
    @mark.parametrize("intrin, op", [
        ("Add", operator.add), ("Sub", operator.sub),
        ("Mul", operator.mul)
    ])
    def test_bin(self, simd, TLane, intrin, op, VData):
        if intrin == "Mul" and TLane in (uint64_t, int64_t):
            return
        intrin = getattr(simd, intrin)
        for start_a, start_b in (
            (0, 100), (0, -128),
            (0x7f, -1), (0xffff, 0x7fff),
            (0xffff, -0x8000), (0xffffffff, -1)
        ):
            a = VData(start_a)
            b = VData(start_b)
            data = [TLane(op(x, y)) for x, y in zip(a, b)]
            test = intrin(a, b)
            assert test == data

    @mark.parametrize("intrin, op", [
        ("IfAdd", operator.add), ("IfSub", operator.sub)
    ])
    def test_condtional(self, simd, TLane, intrin, op, VData, NLanes, SetMask):
        intrin = getattr(simd, intrin)
        for start_a, start_b, mask in (
            (0, 100, [True, False, False, True]),
            (0, -128, [False, True, False, True]),
            (0x7f, -1, [True, True, False, False]),
            (0xffff, 0x7fff, [False, False, True, True]),
            (0xffff, -0x8000, [True, False, True, False]),
            (0xffffffff, -1, [False, False, True, True])
        ):
            a = VData(start_a)
            b = VData(start_b)
            c = VData()
            mask = SetMask(mask) * NLanes
            data = [TLane(op(x, y) if m else z)
                    for x, y, z, m in zip(a, b, c, mask)]
            test = intrin(mask, a, b, c)
            assert test == data

    def test_intdiv(self, simd, TLane, VData, LimitInt):
        def trunc_div(a, d):
            """
            Divide towards zero works with large integers > 2^53,
            and wrap around overflow similar to what C does.
            """
            if d == -1 and a == int_min:
                return a
            a = int(a)
            d = int(d)
            sign_a, sign_d = a < 0, d < 0
            if a == 0 or sign_a == sign_d:
                return a // d
            return (a + sign_d - sign_a) // d + 1

        int_min = LimitInt[0]
        data = [1, -int_min]  # to test overflow
        data += range(0, 2**8, 2**5)
        data += range(0, 2**8, 2**5-1)
        bsize = TLane.element_size
        if bsize > 1:
            data += range(2**8, 2**16, 2**13)
            data += range(2**8, 2**16, 2**13-1)
        if bsize > 2:
            data += range(2**16, 2**32, 2**29)
            data += range(2**16, 2**32, 2**29-1)
        if bsize > 4:
            data += range(2**32, 2**64, 2**61)
            data += range(2**32, 2**64, 2**61-1)
        # negate
        data += [-x for x in data]
        for dividend, divisor in itertools.product(data, data):
            if divisor == 0:
                continue
            divisor = TLane(divisor)  # cast
            dividend = VData(dividend)
            data_divc = [trunc_div(a, divisor) for a in dividend]
            divisor_parms = simd.Divisor(divisor)
            test = simd.Div(dividend, divisor_parms)
            assert test == data_divc

@mark.parametrize('TLane', [
    uint8_t, int8_t, uint16_t, int16_t,
])
@mark.parametrize("intrin, op", [
    ("Adds", operator.add), ("Subs", operator.sub)
])
def test_arithmetic_sat(simd, TLane, intrin, op, VData, ClipEdge):
    intrin = getattr(simd, intrin)
    for start_a, start_b in (
        (0, 100), (0, -128), (0x7f, -1), (0xff, 0xffff),
        (0, 0xffff-100)
    ):
        a = VData(start_a)
        b = VData(start_b)
        data = [ClipEdge(op(int(x), int(y))) for x, y in zip(a, b)]
        test = intrin(a, b)
        assert test == data

@mark.parametrize('TLane', [
    float_, double
])
def test_arithmetic_fused(simd, TLane, VData):
    a, b, c = [VData()]*3
    cx2 = simd.Add(c, c)
    # multiply and add, a*b + c
    data_fma = Vec(TLane)(*[
        a * b + c for a, b, c in zip(a, b, c)
    ])
    fma = simd.MulAdd(a, b, c)
    assert fma == data_fma
    # multiply and subtract, a*b - c
    fms = simd.MulSub(a, b, c)
    data_fms = simd.Sub(data_fma, cx2)
    assert fms == data_fms
    # negate multiply and add, -(a*b) + c
    nfma = simd.NegMulAdd(a, b, c)
    data_nfma = simd.Sub(cx2, data_fma)
    assert nfma == data_nfma
    # negate multiply and subtract, -(a*b) - c
    nfms = simd.NegMulSub(a, b, c)
    data_nfms = simd.Mul(data_fma, simd.Set(TLane(-1)))
    assert nfms == data_nfms
    # multiply, add for odd elements and subtract even elements.
    # (a * b) -+ c
    fmas = simd.MulAddSub(a, b, c)
    assert fmas[0::2] == data_fms[0::2]
    assert fmas[1::2] == data_fma[1::2]

@mark.parametrize('TLane', [
    uint32_t, uint64_t, float_, double
])
def test_arithmetic_reduce_sum(simd, TLane, VData):
    data = VData()
    data_sum = sum(data)
    test = simd.Sum(data)
    assert test == data_sum

@mark.parametrize('TLane', [
    uint8_t, uint16_t
])
def test_arithmetic_reduce_sumup(simd, TLane, NLanes, VData, LimitInt):
    rdata = (0, NLanes, LimitInt[0], LimitInt[1]-NLanes)
    for r in rdata:
        data = VData(r)
        data_sum = sum([int(d) for d in data])
        test = simd.Sumup(data)
        assert test == data_sum

#
# `Math`:
# | Name              |u8 |i8 |u16|i16|u32|i32|u64|i64|f32|f64|
# |:------------------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
# | Round             |   |   |   |   |   |   |   |   | x | x |
# | Roundi            |   |   |   |   |   |   |   |   | x | x |
# | Ceil              |   |   |   |   |   |   |   |   | x | x |
# | Trunc             |   |   |   |   |   |   |   |   | x | x |
# | Floor             |   |   |   |   |   |   |   |   | x | x |
# | Sqrt              | - | - | - | - | - | - | - | - | x | x |
# | Recip             | - | - | - | - | - | - | - | - | x | x |
# | Abs               | - | - | - | - | - | - | - | - | x | x |
# | Min               | x | x | x | x | x | x | x | x | x | x |
# | Max               | x | x | x | x | x | x | x | x | x | x |
# | MinProp           |   |   |   |   |   |   |   |   | x | x |
# | MaxProp           |   |   |   |   |   |   |   |   | x | x |
# | MinPropNan        |   |   |   |   |   |   |   |   | x | x |
# | MaxPropNan        |   |   |   |   |   |   |   |   | x | x |
# | ReduceMin         | x | x | x | x | x | x | x | x | x | x |
# | ReduceMax         | x | x | x | x | x | x | x | x | x | x |
# | ReduceMinProp     |   |   |   |   |   |   |   |   | x | x |
# | ReduceMaxProp     |   |   |   |   |   |   |   |   | x | x |
# | ReduceMinPropNan  |   |   |   |   |   |   |   |   | x | x |
# | ReduceMaxPropNan  |   |   |   |   |   |   |   |   | x | x |
#
@mark.parametrize('TLane', [float_, double])
class TestMathFP:
    pinf = float("inf")
    ninf = -float("inf")
    nan = float("nan")

    @mark.parametrize("intrin, pyop, cases", [
        ("simd.Abs", abs, [
            (-0, 0), (ninf, pinf), (pinf, pinf), (nan, nan)
        ]),
        ("simd.Sqrt", math.sqrt, [
            (-0.0, 0.0), (0.0, 0.0), (-1.0, nan), (ninf, nan), (pinf, pinf)
        ]),
        ("simd.Recip", lambda x: 1.0/x, [
            (nan, nan), (pinf, 0.0), (ninf, -0.0), (0.0, pinf), (-0.0, ninf)
        ])
    ])
    def test_un(self, simd, TLane, intrin, pyop, cases, VData, NLanes, TULane):
        intrin = eval(intrin)
        for case, desired in cases:
            data = simd.Set(TLane(case))
            data_desired = simd.Set(TLane(desired))
            test = intrin(data)
            assert test.to_list() == \
                   pytest.approx(data_desired.to_list(), nan_ok=True)
            #assert simd.Reinterpret(test, TULane(0)) == \
            #       simd.Reinterpret(data_desired, TULane(0))

        data = VData()
        data_test = [pyop(d) for d in data.to_list()]
        # cast fp32
        data_test = Vec(TLane)(*data_test)
        test = intrin(data)
        assert test == data_test

    @pytest.mark.parametrize("intrin, pyop", [
        ("simd.Ceil", math.ceil),
        ("simd.Trunc", math.trunc),
        ("simd.Floor", math.floor),
        ("simd.Round", round)
    ])
    def test_rounding(self, simd, TLane, intrin, pyop, NLanes, TULane):
        intrin = eval(intrin)
        # special cases
        round_cases = (
            (self.nan, self.nan),
            (self.pinf, self.pinf),
            (self.ninf, self.ninf)
        )
        for case, desired in round_cases:
            data_round = [desired] * NLanes
            test = intrin(simd.Set(TLane(case)))
            assert test.to_list() == pytest.approx(data_round, nan_ok=True)

        for x in range(0, 2**20, 256**2):
            for w in (-1.05, -1.10, -1.15, 1.05, 1.10, 1.15):
                data = Vec(TLane)(*[(x+a)*w for a in range(NLanes)])
                data_round = [pyop(x) for x in data.to_list()]
                test = intrin(data)
                assert test == data_round

        # test large numbers
        for i in (
            1.1529215045988576e+18, 4.6116860183954304e+18,
            5.902958103546122e+20, 2.3611832414184488e+21
        ):
            data = simd.Set(TLane(i))
            data_round = [pyop(n) for n in data.to_list()]
            test = intrin(data)
            assert test == data_round

        ## signed zero
        if intrin == simd.Floor:
            data_szero = (-0.0,)
        else:
            data_szero = (-0.0, -0.25, -0.30, -0.45, -0.5)

        for w in data_szero:
            data = simd.Reinterpret(simd.Set(TLane(-0.0)), TULane(0))
            test = simd.Reinterpret(intrin(simd.Set(TLane(w))), TULane(0))
            assert test == data

    @pytest.mark.parametrize("intrin", [
        "simd.Round", "simd.Trunc", "simd.Ceil", "simd.Floor"
    ])
    def test_unary_invalid_fpexception(self, simd, TLane, intrin):
        intrin = eval(intrin)
        for d in [self.nan, self.pinf, self.ninf]:
            v = simd.Set(TLane(d))
            before = floatstatus()  # clear
            intrin(v)
            assert floatstatus()["Invalid"] == False

    @pytest.mark.parametrize("intrin, reduce_intrin, pyop", [
        ("simd.Max", "simd.ReduceMax", max),
        ("simd.Min", "simd.ReduceMin", min),
        ("simd.MaxProp", "simd.ReduceMaxProp", max),
        ("simd.MinProp", "simd.ReduceMinProp", min),
        ("simd.MaxPropNan", "simd.ReduceMaxPropNan", max),
        ("simd.MinPropNan", "simd.ReduceMinPropNan", min)
    ])
    def test_maxmin(self, simd, TLane, intrin, reduce_intrin, pyop, NLanes):
        intrin = eval(intrin)
        reduce_intrin = eval(reduce_intrin)
        hf_nlanes = NLanes // 2
        cases = (
            ([0.0, -0.0], [-0.0, 0.0]),
            ([10, -10],  [10, -10]),
            ([self.pinf, 10], [10, self.ninf]),
            ([10, self.pinf], [self.ninf, 10]),
            ([10, -10], [10, -10]),
            ([-10, 10], [-10, 10])
        )
        for op1, op2 in cases:
            a = Vec(TLane)(*op1*hf_nlanes)
            b = Vec(TLane)(*op2*hf_nlanes)
            data = pyop(a, b)
            test = intrin(a, b)
            assert test == data
            data = pyop(a)
            test = reduce_intrin(a)
            assert test == data

    @pytest.mark.parametrize("intrin, reduce_intrin, pyop, propagate_nan", [
        ("simd.MaxProp", "simd.ReduceMaxProp", max, False),
        ("simd.MinProp", "simd.ReduceMinProp", min, False),
        ("simd.MaxPropNan", "simd.ReduceMaxPropNan", max, True),
        ("simd.MinPropNan", "simd.ReduceMinPropNan", min, True)
    ])
    def test_maxmin_special(self, simd, TLane, intrin, reduce_intrin,
                            pyop, NLanes, propagate_nan):
        intrin = eval(intrin)
        reduce_intrin = eval(reduce_intrin)
        hf_nlanes = NLanes // 2
        if propagate_nan:
            test_nan = lambda a, b: (
                self.nan if math.isnan(a) or math.isnan(b) else b
            )
        else:
            test_nan = lambda a, b: (
                b if math.isnan(a) else a if math.isnan(b) else b
            )
        cases = (
            (self.nan, 10),
            (10, self.nan),
            (self.nan, self.pinf),
            (self.pinf, self.nan),
            (self.nan, self.nan)
        )
        for op1, op2 in cases:
            ab = Vec(TLane)(*[op1, op2]*hf_nlanes)
            data = test_nan(op1, op2)
            test = reduce_intrin(ab)
            assert float(test) == pytest.approx(data, nan_ok=True)
            a = simd.Set(TLane(op1))
            b = simd.Set(TLane(op2))
            data = [data] * NLanes
            test = intrin(a, b)
            assert test.to_list() == pytest.approx(data, nan_ok=True)

@mark.parametrize('TLane', [
    uint8_t, int8_t, uint16_t, int16_t,
    uint32_t, int32_t, uint64_t, int64_t
])
class TestMathInt:
    @pytest.mark.parametrize("intrin, reduce_intrin, pyop", [
        ("simd.Max", "simd.ReduceMax", max),
        ("simd.Min", "simd.ReduceMin", min),
    ])
    def test_maxmin(self, simd, TLane, intrin, reduce_intrin, pyop,
                    VData, NLanes):
        intrin = eval(intrin)
        reduce_intrin = eval(reduce_intrin)
        a = VData()
        b = VData(NLanes)
        data = pyop(a, b)
        test = intrin(a, b)
        assert test == data
        data = pyop(a)
        test = reduce_intrin(a)
        assert test == data

#
# `Math`:
# | Name           |u8 |i8 |u16|i16|u32|i32|u64|i64|f32|f64|
# |:---------------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
# | CombineLow     | x | x | x | x | x | x | x | x | x | x |
# | CombineHigh    | x | x | x | x | x | x | x | x | x | x |
# | Combine        | x | x | x | x | x | x | x | x | x | x |
# | Zip            | x | x | x | x | x | x | x | x | x | x |
# | Unzip          | x | x | x | x | x | x | x | x | x | x |
# | Reverse64      | x | x | x | x | x | x |   |   | x |   |
# | Permute128     | - | - | - | - | x | x | x | x | x | x |
@mark.parametrize('TLane', [
    uint8_t, int8_t, uint16_t, int16_t,
    uint32_t, int32_t, uint64_t, int64_t,
    float_, double
])
def test_reorder_combine_zip(simd, TLane, VData, NLanes):
    a = VData()
    b = VData(NLanes)
    hf_nlanes = NLanes // 2
    # lower half part
    a_lo = a[:hf_nlanes]
    b_lo = b[:hf_nlanes]
    # higher half part
    a_hi = a[hf_nlanes:]
    b_hi = b[hf_nlanes:]
    # combine two lower parts
    test = simd.CombineLow(a, b)
    assert test == a_lo + b_lo
    # combine two higher parts
    test = simd.CombineHigh(a, b)
    assert test == a_hi + b_hi
    # combine x2
    test = simd.Combine(a, b)
    assert test == (a_lo + b_lo, a_hi + b_hi)
    # zip(interleave)
    zip_low = Vec(TLane)(*[v for p in zip(a_lo, b_lo) for v in p])
    zip_high = Vec(TLane)(*[v for p in zip(a_hi, b_hi) for v in p])
    test = simd.Zip(a, b)
    assert test == (zip_low, zip_high)
    # unzip(deinterleave)
    test = simd.Unzip(zip_low, zip_high)
    assert test == (a, b)


@mark.parametrize('TLane', [
    uint8_t, int8_t, uint16_t, int16_t,
    uint32_t, int32_t
])
def test_reorder_reverse64(simd, TLane, VData, NLanes):
    a = VData(0)
    rev64 = [
        y for x in range(0, NLanes, 8//TLane.element_size)
        for y in reversed(range(x, x + 8//TLane.element_size))
    ]
    test = simd.Reverse64(a)
    assert test == rev64


@mark.parametrize('TLane', [
    uint32_t, int32_t, uint64_t, int64_t,
    float_, double
])
def test_reorder_permute128(simd, TLane, VData, NLanes):
    permn = 16 // TLane.element_size
    permd = permn-1
    nlane128 = NLanes//permn
    shfl = [0, 1] if TLane.element_size == 8 else [0, 2, 4, 6]
    data = VData()
    for i in range(permn):
        indices = [int_((i >> shf) & permd) for shf in shfl]
        vperm = [
            data[int(j) + (e & -permn)]
            for e, j in enumerate(indices*nlane128)
        ]
        test = simd.Permute128(data, *indices)
        assert test == vperm
