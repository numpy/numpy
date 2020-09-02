import pytest
from numpy.core._simd import targets

npyv = None
npyv2 = None
for target_name, npyv_mod in targets.items():
    if npyv:
        if npyv_mod and npyv_mod.simd:
            npyv2 = npyv_mod
            break
        continue
    if npyv_mod and npyv_mod.simd:
        npyv = npyv_mod

unsigned_sfx = ["u8", "u16", "u32", "u64"]
signed_sfx = ["s8", "s16", "s32", "s64"]
fp_sfx = ["f32"]
if npyv and npyv.simd_f64:
    fp_sfx.append("f64")

int_sfx = unsigned_sfx + signed_sfx
all_sfx = unsigned_sfx + int_sfx

@pytest.mark.skipif(not npyv, reason="could not find any SIMD extension with NPYV support")
class Test_SIMD_MODULE(object):
    def test_num_lanes(self):
        for sfx in all_sfx:
            nlanes = getattr(npyv, "nlanes_" + sfx)
            vector = getattr(npyv, "setall_" + sfx)(1)
            assert len(vector) == nlanes

    def test_type_name(self):
        for sfx in all_sfx:
            vector = getattr(npyv, "setall_" + sfx)(1)
            assert vector.__name__ == "npyv_" + sfx

    def test_raises(self):
        def assert_raises(e, callback, *args):
            __tracebackhide__ = True  # Hide traceback for py.test
            try:
                callback(*args)
                raise AssertionError("expected to raise " + e.__name__)
            except e:
                pass

        a, b = [npyv.setall_u32(1)]*2
        for sfx in all_sfx:
            vcb = lambda intrin: getattr(npyv, f"{intrin}_{sfx}")
            assert_raises(TypeError, vcb("add"), a)
            assert_raises(TypeError, vcb("add"), a, b, a)
            assert_raises(TypeError, vcb("setall"))
            assert_raises(TypeError, vcb("setall"), [1])
            assert_raises(TypeError, vcb("load"), 1)
            assert_raises(ValueError, vcb("load"), [1])
            assert_raises(ValueError, vcb("store"), [1], getattr(npyv, f"reinterpret_{sfx}_u32")(a))

        # mix among submodules isn't allowed
        if not npyv2:
            return
        a2 = npyv2.setall_u32(1)
        assert_raises(TypeError, npyv.add_u32, a2, a2)
        assert_raises(TypeError, npyv2.add_u32, a, a)

    def test_unsigned_overflow(self):
        for sfx in unsigned_sfx:
            nlanes = getattr(npyv, "nlanes_" + sfx)
            hfbyte_len = int(sfx[1:])//4
            maxu = int(f"0x{'f'*hfbyte_len}", 16)
            maxu_72 = 0xfffffffffffffffff
            lane = getattr(npyv, "setall_" + sfx)(maxu_72)[0]
            assert lane == maxu
            lanes = getattr(npyv, "load_" + sfx)([maxu_72] * nlanes)
            assert lanes == [maxu] * nlanes
            lane = getattr(npyv, "setall_" + sfx)(-1)[0]
            assert lane == maxu
            lanes = getattr(npyv, "load_" + sfx)([-1] * nlanes)
            assert lanes == [maxu] * nlanes

    def test_signed_overflow(self):
        for sfx in signed_sfx:
            nlanes = getattr(npyv, "nlanes_" + sfx)
            maxs_72 = 0x7fffffffffffffffff
            lane = getattr(npyv, "setall_" + sfx)(maxs_72)[0]
            assert lane == -1
            lanes = getattr(npyv, "load_" + sfx)([maxs_72] * nlanes)
            assert lanes == [-1] * nlanes
            mins_72 = -0x80000000000000000
            lane = getattr(npyv, "setall_" + sfx)(mins_72)[0]
            assert lane == 0
            lanes = getattr(npyv, "load_" + sfx)([mins_72] * nlanes)
            assert lanes == [0] * nlanes

    def test_truncate_f32(self):
        f32 = npyv.setall_f32(0.1)[0]
        assert f32 != 0.1
        assert round(f32, 1) == 0.1

    def test_compare(self):
        data_range = range(0, npyv.nlanes_u32)
        vdata = npyv.load_u32(data_range)
        assert vdata == list(data_range)
        assert vdata == tuple(data_range)
        for i in data_range:
            assert vdata[i] == data_range[i]
