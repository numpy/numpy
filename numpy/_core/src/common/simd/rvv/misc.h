#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_RVV_MISC_H
#define _NPY_SIMD_RVV_MISC_H

// vector with zero lanes
#define npyv_zero_u8()  __riscv_vreinterpret_v_u32m1_u8m1(__riscv_vreinterpret_v_i32m1_u32m1(npyv_zero_s32()))
#define npyv_zero_s8()  __riscv_vreinterpret_v_i32m1_i8m1(npyv_zero_s32())
#define npyv_zero_u16() __riscv_vreinterpret_v_u32m1_u16m1(__riscv_vreinterpret_v_i32m1_u32m1(npyv_zero_s32()))
#define npyv_zero_s16() __riscv_vreinterpret_v_i32m1_i16m1(npyv_zero_s32())
#define npyv_zero_u32() __riscv_vmv_v_x_u32m1((uint32_t)0, 4)
#define npyv_zero_s32() __riscv_vmv_v_x_i32m1((int32_t)0, 4)
#define npyv_zero_u64() __riscv_vreinterpret_v_u32m1_u64m1(__riscv_vreinterpret_v_i32m1_u32m1(npyv_zero_s32()))
#define npyv_zero_s64() __riscv_vreinterpret_v_i32m1_i64m1(npyv_zero_s32())
#define npyv_zero_f32() __riscv_vfmv_v_f_f32m1(0.0f, 4)
#define npyv_zero_f64() __riscv_vfmv_v_f_f64m1(0.0, 2)

// vector with a specific value set to all lanes
NPY_FINLINE npyv_u8 npyv_setall_u8(uint8_t val)
{
    size_t vlen = __riscv_vsetvlmax_e8m1();
    return __riscv_vmv_v_x_u8m1(val, vlen);
}

NPY_FINLINE npyv_s8 npyv_setall_s8(int8_t val)
{
    size_t vlen = __riscv_vsetvlmax_e8m1();
    return __riscv_vmv_v_x_i8m1(val, vlen);
}

NPY_FINLINE npyv_u16 npyv_setall_u16(uint16_t val)
{
    size_t vlen = __riscv_vsetvlmax_e16m1();
    return __riscv_vmv_v_x_u16m1(val, vlen);
}

NPY_FINLINE npyv_s16 npyv_setall_s16(int16_t val)
{
    size_t vlen = __riscv_vsetvlmax_e16m1();
    return __riscv_vmv_v_x_i16m1(val, vlen);
}

NPY_FINLINE npyv_u32 npyv_setall_u32(uint32_t val)
{
    size_t vlen = __riscv_vsetvlmax_e32m1();
    return __riscv_vmv_v_x_u32m1(val, vlen);
}

NPY_FINLINE npyv_s32 npyv_setall_s32(int32_t val)
{
    size_t vlen = __riscv_vsetvlmax_e32m1();
    return __riscv_vmv_v_x_i32m1(val, vlen);
}

NPY_FINLINE npyv_u64 npyv_setall_u64(uint64_t val)
{
    size_t vlen = __riscv_vsetvlmax_e64m1();
    return __riscv_vmv_v_x_u64m1(val, vlen);
}

NPY_FINLINE npyv_s64 npyv_setall_s64(int64_t val)
{
    size_t vlen = __riscv_vsetvlmax_e64m1();
    return __riscv_vmv_v_x_i64m1(val, vlen);
}

NPY_FINLINE npyv_f32 npyv_setall_f32(float val)
{
    size_t vlen = __riscv_vsetvlmax_e32m1();
    return __riscv_vfmv_v_f_f32m1(val, vlen);
}

NPY_FINLINE npyv_f64 npyv_setall_f64(double val)
{
    size_t vlen = __riscv_vsetvlmax_e64m1();
    return __riscv_vfmv_v_f_f64m1(val, vlen);
}

// vector with specific values set to each lane and
// set a specific value to all remained lanes
NPY_FINLINE npyv_u8 npyv__set_u8(npy_uint8 i0, npy_uint8 i1, npy_uint8 i2, npy_uint8 i3,
    npy_uint8 i4, npy_uint8 i5, npy_uint8 i6, npy_uint8 i7, npy_uint8 i8, npy_uint8 i9,
    npy_uint8 i10, npy_uint8 i11, npy_uint8 i12, npy_uint8 i13, npy_uint8 i14, npy_uint8 i15)
{
    const uint8_t NPY_DECL_ALIGNED(16) data[16] = {
        i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15
    };
    return __riscv_vle8_v_u8m1(data, 16);
}

NPY_FINLINE npyv_s8 npyv__set_s8(npy_int8 i0, npy_int8 i1, npy_int8 i2, npy_int8 i3,
    npy_int8 i4, npy_int8 i5, npy_int8 i6, npy_int8 i7, npy_int8 i8, npy_int8 i9,
    npy_int8 i10, npy_int8 i11, npy_int8 i12, npy_int8 i13, npy_int8 i14, npy_int8 i15)
{
    const int8_t NPY_DECL_ALIGNED(16) data[16] = {
        i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15
    };
    return __riscv_vle8_v_i8m1(data, 16);
}

NPY_FINLINE vuint16m1_t npyv__set_u16(npy_uint16 i0, npy_uint16 i1, npy_uint16 i2, npy_uint16 i3,
    npy_uint16 i4, npy_uint16 i5, npy_uint16 i6, npy_uint16 i7)
{
    const uint16_t NPY_DECL_ALIGNED(16) data[8] = {i0, i1, i2, i3, i4, i5, i6, i7};
    return __riscv_vle16_v_u16m1(data, 8);
}

NPY_FINLINE vint16m1_t npyv__set_s16(npy_int16 i0, npy_int16 i1, npy_int16 i2, npy_int16 i3,
    npy_int16 i4, npy_int16 i5, npy_int16 i6, npy_int16 i7)
{
    const int16_t NPY_DECL_ALIGNED(16) data[8] = {i0, i1, i2, i3, i4, i5, i6, i7};
    return __riscv_vle16_v_i16m1(data, 8);
}

NPY_FINLINE npyv_u32 npyv__set_u32(npy_uint32 i0, npy_uint32 i1, npy_uint32 i2, npy_uint32 i3)
{
    const uint32_t NPY_DECL_ALIGNED(16) data[4] = {i0, i1, i2, i3};
    return __riscv_vle32_v_u32m1(data, 4);
}

NPY_FINLINE vint32m1_t npyv__set_s32(npy_int32 i0, npy_int32 i1, npy_int32 i2, npy_int32 i3)
{
    const int32_t NPY_DECL_ALIGNED(16) data[4] = {i0, i1, i2, i3};
    return __riscv_vle32_v_i32m1(data, 4);
}

NPY_FINLINE npyv_u64 npyv__set_u64(npy_uint64 i0, npy_uint64 i1)
{
    const uint64_t NPY_DECL_ALIGNED(16) data[2] = {i0, i1};
    return __riscv_vle64_v_u64m1(data, 2);
}

NPY_FINLINE vint64m1_t npyv__set_s64(npy_int64 i0, npy_int64 i1)
{
    const int64_t NPY_DECL_ALIGNED(16) data[2] = {i0, i1};
    return __riscv_vle64_v_i64m1(data, 2);
}

NPY_FINLINE vfloat32m1_t npyv__set_f32(float i0, float i1, float i2, float i3)
{
    const float NPY_DECL_ALIGNED(16) data[4] = {i0, i1, i2, i3};
    return __riscv_vle32_v_f32m1(data, 4);
}

NPY_FINLINE vfloat64m1_t npyv__set_f64(double i0, double i1)
{
    const double NPY_DECL_ALIGNED(16) data[2] = {i0, i1};
    return __riscv_vle64_v_f64m1(data, 2);
}

#define npyv_setf_u8(FILL, ...)  npyv__set_u8(NPYV__SET_FILL_16(npy_uint8, FILL, __VA_ARGS__))
#define npyv_setf_s8(FILL, ...)  npyv__set_s8(NPYV__SET_FILL_16(npy_int8, FILL, __VA_ARGS__))
#define npyv_setf_u16(FILL, ...) npyv__set_u16(NPYV__SET_FILL_8(npy_uint16, FILL, __VA_ARGS__))
#define npyv_setf_s16(FILL, ...) npyv__set_s16(NPYV__SET_FILL_8(npy_int16, FILL, __VA_ARGS__))
#define npyv_setf_u32(FILL, ...) npyv__set_u32(NPYV__SET_FILL_4(npy_uint32, FILL, __VA_ARGS__))
#define npyv_setf_s32(FILL, ...) npyv__set_s32(NPYV__SET_FILL_4(npy_int32, FILL, __VA_ARGS__))
#define npyv_setf_u64(FILL, ...) npyv__set_u64(NPYV__SET_FILL_2(npy_uint64, FILL, __VA_ARGS__))
#define npyv_setf_s64(FILL, ...) npyv__set_s64(NPYV__SET_FILL_2(npy_int64, FILL, __VA_ARGS__))
#define npyv_setf_f32(FILL, ...) npyv__set_f32(NPYV__SET_FILL_4(float, FILL, __VA_ARGS__))
#define npyv_setf_f64(FILL, ...) npyv__set_f64(NPYV__SET_FILL_2(double, FILL, __VA_ARGS__))

// vector with specific values set to each lane and
// set zero to all remained lanes
#define npyv_set_u8(...)  npyv_setf_u8(0,  __VA_ARGS__)
#define npyv_set_s8(...)  npyv_setf_s8(0,  __VA_ARGS__)
#define npyv_set_u16(...) npyv_setf_u16(0, __VA_ARGS__)
#define npyv_set_s16(...) npyv_setf_s16(0, __VA_ARGS__)
#define npyv_set_u32(...) npyv_setf_u32(0, __VA_ARGS__)
#define npyv_set_s32(...) npyv_setf_s32(0, __VA_ARGS__)
#define npyv_set_u64(...) npyv_setf_u64(0, __VA_ARGS__)
#define npyv_set_s64(...) npyv_setf_s64(0, __VA_ARGS__)
#define npyv_set_f32(...) npyv_setf_f32(0, __VA_ARGS__)
#define npyv_set_f64(...) npyv_setf_f64(0, __VA_ARGS__)

// Per lane select
NPY_FINLINE npyv_u8 npyv_select_u8(npyv_u8 a, npyv_u8 b, npyv_u8 c)
{
    return __riscv_vxor_vv_u8m1(__riscv_vand_vv_u8m1(__riscv_vxor_vv_u8m1(c, b, 16), a, 16), c, 16);
}

NPY_FINLINE npyv_s8 npyv_select_s8(npyv_u8 a, npyv_s8 b, npyv_s8 c)
{
    return __riscv_vxor_vv_i8m1(
        __riscv_vand_vv_i8m1(__riscv_vxor_vv_i8m1(c, b, 16), __riscv_vreinterpret_v_u8m1_i8m1(a), 16), c, 16);
}

NPY_FINLINE npyv_u16 npyv_select_u16(npyv_u16 a, npyv_u16 b, npyv_u16 c)
{
    return __riscv_vxor_vv_u16m1(__riscv_vand_vv_u16m1(__riscv_vxor_vv_u16m1(c, b, 8), a, 8), c, 8);
}

NPY_FINLINE npyv_s16 npyv_select_s16(npyv_u16 a, npyv_s16 b, npyv_s16 c)
{
    return __riscv_vxor_vv_i16m1(
        __riscv_vand_vv_i16m1(__riscv_vxor_vv_i16m1(c, b, 8), __riscv_vreinterpret_v_u16m1_i16m1(a), 8), c, 8);
}

NPY_FINLINE npyv_u32 npyv_select_u32(npyv_u32 a, npyv_u32 b, npyv_u32 c)
{
    return __riscv_vxor_vv_u32m1(__riscv_vand_vv_u32m1(__riscv_vxor_vv_u32m1(c, b, 4), a, 4), c, 4);
}

NPY_FINLINE npyv_s32 npyv_select_s32(npyv_u32 a, npyv_s32 b, npyv_s32 c)
{
    return __riscv_vxor_vv_i32m1(
        __riscv_vand_vv_i32m1(__riscv_vxor_vv_i32m1(c, b, 4), __riscv_vreinterpret_v_u32m1_i32m1(a), 4), c, 4);
}

NPY_FINLINE npyv_u64 npyv_select_u64(npyv_u64 a, npyv_u64 b, npyv_u64 c)
{
    return __riscv_vxor_vv_u64m1(__riscv_vand_vv_u64m1(__riscv_vxor_vv_u64m1(c, b, 2), a, 2), c, 2);
}

NPY_FINLINE npyv_s64 npyv_select_s64(npyv_u64 a, npyv_s64 b, npyv_s64 c)
{
    return __riscv_vxor_vv_i64m1(
        __riscv_vand_vv_i64m1(__riscv_vxor_vv_i64m1(c, b, 2), __riscv_vreinterpret_v_u64m1_i64m1(a), 2), c, 2);
}

NPY_FINLINE npyv_f32 npyv_select_f32(npyv_u32 a, npyv_f32 b, npyv_f32 c)
{
    npyv_u32 b_u32 = __riscv_vreinterpret_v_f32m1_u32m1(b);
    npyv_u32 c_u32 = __riscv_vreinterpret_v_f32m1_u32m1(c);
    return __riscv_vreinterpret_v_u32m1_f32m1(
        __riscv_vxor_vv_u32m1(__riscv_vand_vv_u32m1(__riscv_vxor_vv_u32m1(c_u32, b_u32, 4), a, 4), c_u32, 4));
}

NPY_FINLINE npyv_f64 npyv_select_f64(npyv_u64 a, npyv_f64 b, npyv_f64 c)
{
    npyv_u64 b_u64 = __riscv_vreinterpret_v_f64m1_u64m1(b);
    npyv_u64 c_u64 = __riscv_vreinterpret_v_f64m1_u64m1(c);
    return __riscv_vreinterpret_v_u64m1_f64m1(
        __riscv_vxor_vv_u64m1(__riscv_vand_vv_u64m1(__riscv_vxor_vv_u64m1(c_u64, b_u64, 2), a, 2), c_u64, 2));
}

// extract the first vector's lane
NPY_FINLINE npy_uint8 npyv_extract0_u8(npyv_u8 a)
{
    return __riscv_vmv_x_s_u8m1_u8(__riscv_vslidedown_vx_u8m1(a, 0, 16));
}

NPY_FINLINE npy_int8 npyv_extract0_s8(npyv_s8 a)
{
    return __riscv_vmv_x_s_i8m1_i8(__riscv_vslidedown_vx_i8m1(a, 0, 16));
}

NPY_FINLINE npy_uint16 npyv_extract0_u16(npyv_u16 a)
{
    return __riscv_vmv_x_s_u16m1_u16(__riscv_vslidedown_vx_u16m1(a, 0, 8));
}

NPY_FINLINE npy_int16 npyv_extract0_s16(npyv_s16 a)
{
    return __riscv_vmv_x_s_i16m1_i16(__riscv_vslidedown_vx_i16m1(a, 0, 8));
}

NPY_FINLINE npy_uint32 npyv_extract0_u32(npyv_u32 a)
{
    return __riscv_vmv_x_s_u32m1_u32(__riscv_vslidedown_vx_u32m1(a, 0, 4));
}

NPY_FINLINE npy_int32 npyv_extract0_s32(npyv_s32 a)
{
    return __riscv_vmv_x_s_i32m1_i32(__riscv_vslidedown_vx_i32m1(a, 0, 4));
}

NPY_FINLINE npy_uint64 npyv_extract0_u64(npyv_u64 a)
{
    return __riscv_vmv_x_s_u64m1_u64(__riscv_vslidedown_vx_u64m1(a, 0, 2));
}

NPY_FINLINE npy_int64 npyv_extract0_s64(npyv_s64 a)
{
    return __riscv_vmv_x_s_i64m1_i64(__riscv_vslidedown_vx_i64m1(a, 0, 2));
}

NPY_FINLINE float npyv_extract0_f32(npyv_f32 a)
{
    return __riscv_vfmv_f_s_f32m1_f32(__riscv_vslidedown_vx_f32m1(a, 0, 4));
}

NPY_FINLINE double npyv_extract0_f64(npyv_f64 a)
{
    return __riscv_vfmv_f_s_f64m1_f64(__riscv_vslidedown_vx_f64m1(a, 0, 2));
}

// Reinterpret
#define npyv_reinterpret_u8_u8(X) X
NPY_FINLINE npyv_u8 npyv_reinterpret_u8_s8(npyv_s8 a)
{
    return __riscv_vreinterpret_v_i8m1_u8m1(a);
}

NPY_FINLINE npyv_u8 npyv_reinterpret_u8_u16(npyv_u16 a)
{
    return __riscv_vreinterpret_v_u16m1_u8m1(a);
}

NPY_FINLINE npyv_u8 npyv_reinterpret_u8_s16(npyv_s16 a)
{
    return __riscv_vreinterpret_v_u16m1_u8m1(__riscv_vreinterpret_v_i16m1_u16m1(a));
}

NPY_FINLINE npyv_u8 npyv_reinterpret_u8_u32(npyv_u32 a)
{
    return __riscv_vreinterpret_v_u32m1_u8m1(a);
}

NPY_FINLINE npyv_u8 npyv_reinterpret_u8_s32(npyv_s32 a)
{
    return __riscv_vreinterpret_v_u32m1_u8m1(__riscv_vreinterpret_v_i32m1_u32m1(a));
}

NPY_FINLINE npyv_u8 npyv_reinterpret_u8_u64(npyv_u64 a)
{
    return __riscv_vreinterpret_v_u64m1_u8m1(a);
}

NPY_FINLINE npyv_u8 npyv_reinterpret_u8_s64(npyv_s64 a)
{
    return __riscv_vreinterpret_v_u64m1_u8m1(__riscv_vreinterpret_v_i64m1_u64m1(a));
}

NPY_FINLINE npyv_u8 npyv_reinterpret_u8_f32(npyv_f32 a)
{
    return __riscv_vreinterpret_v_u32m1_u8m1(__riscv_vreinterpret_v_f32m1_u32m1(a));
}

NPY_FINLINE npyv_u8 npyv_reinterpret_u8_f64(npyv_f64 a)
{
    return __riscv_vreinterpret_v_u64m1_u8m1(__riscv_vreinterpret_v_f64m1_u64m1(a));
}

#define npyv_reinterpret_s8_s8(X) X
NPY_FINLINE npyv_s8 npyv_reinterpret_s8_u8(npyv_u8 a)
{
    return __riscv_vreinterpret_v_u8m1_i8m1(a);
}

NPY_FINLINE npyv_s8 npyv_reinterpret_s8_u16(npyv_u16 a)
{
    return __riscv_vreinterpret_v_i16m1_i8m1(__riscv_vreinterpret_v_u16m1_i16m1(a));
}

NPY_FINLINE npyv_s8 npyv_reinterpret_s8_s16(npyv_s16 a)
{
    return __riscv_vreinterpret_v_i16m1_i8m1(a);
}

NPY_FINLINE npyv_s8 npyv_reinterpret_s8_u32(npyv_u32 a)
{
    return __riscv_vreinterpret_v_i32m1_i8m1(__riscv_vreinterpret_v_u32m1_i32m1(a));
}

NPY_FINLINE npyv_s8 npyv_reinterpret_s8_s32(npyv_s32 a)
{
    return __riscv_vreinterpret_v_i32m1_i8m1(a);
}

NPY_FINLINE npyv_s8 npyv_reinterpret_s8_u64(npyv_u64 a)
{
    return __riscv_vreinterpret_v_i64m1_i8m1(__riscv_vreinterpret_v_u64m1_i64m1(a));
}

NPY_FINLINE npyv_s8 npyv_reinterpret_s8_s64(npyv_s64 a)
{
    return __riscv_vreinterpret_v_i64m1_i8m1(a);
}

NPY_FINLINE npyv_s8 npyv_reinterpret_s8_f32(npyv_f32 a)
{
    return __riscv_vreinterpret_v_i32m1_i8m1(__riscv_vreinterpret_v_f32m1_i32m1(a));
}

NPY_FINLINE npyv_s8 npyv_reinterpret_s8_f64(npyv_f64 a)
{
    return __riscv_vreinterpret_v_i64m1_i8m1(__riscv_vreinterpret_v_f64m1_i64m1(a));
}

#define npyv_reinterpret_u16_u16(X) X
NPY_FINLINE npyv_u16 npyv_reinterpret_u16_u8(npyv_u8 a)
{
    return __riscv_vreinterpret_v_u8m1_u16m1(a);
}

NPY_FINLINE npyv_u16 npyv_reinterpret_u16_s8(npyv_s8 a)
{
    return __riscv_vreinterpret_v_u8m1_u16m1(__riscv_vreinterpret_v_i8m1_u8m1(a));
}

NPY_FINLINE npyv_u16 npyv_reinterpret_u16_s16(npyv_s16 a)
{
    return __riscv_vreinterpret_v_i16m1_u16m1(a);
}

NPY_FINLINE npyv_u16 npyv_reinterpret_u16_u32(npyv_u32 a)
{
    return __riscv_vreinterpret_v_u32m1_u16m1(a);
}

NPY_FINLINE npyv_u16 npyv_reinterpret_u16_s32(npyv_s32 a)
{
    return __riscv_vreinterpret_v_u32m1_u16m1(__riscv_vreinterpret_v_i32m1_u32m1(a));
}

NPY_FINLINE npyv_u16 npyv_reinterpret_u16_u64(npyv_u64 a)
{
    return __riscv_vreinterpret_v_u64m1_u16m1(a);
}

NPY_FINLINE npyv_u16 npyv_reinterpret_u16_s64(npyv_s64 a)
{
    return __riscv_vreinterpret_v_u64m1_u16m1(__riscv_vreinterpret_v_i64m1_u64m1(a));
}

NPY_FINLINE npyv_u16 npyv_reinterpret_u16_f32(npyv_f32 a)
{
    return __riscv_vreinterpret_v_u32m1_u16m1(__riscv_vreinterpret_v_f32m1_u32m1(a));
}

NPY_FINLINE npyv_u16 npyv_reinterpret_u16_f64(npyv_f64 a)
{
    return __riscv_vreinterpret_v_u64m1_u16m1(__riscv_vreinterpret_v_f64m1_u64m1(a));
}

#define npyv_reinterpret_s16_s16(X) X
NPY_FINLINE npyv_s16 npyv_reinterpret_s16_u8(npyv_u8 a)
{
    return __riscv_vreinterpret_v_i8m1_i16m1(__riscv_vreinterpret_v_u8m1_i8m1(a));
}

NPY_FINLINE npyv_s16 npyv_reinterpret_s16_s8(npyv_s8 a)
{
    return __riscv_vreinterpret_v_i8m1_i16m1(a);
}

NPY_FINLINE npyv_s16 npyv_reinterpret_s16_u16(npyv_u16 a)
{
    return __riscv_vreinterpret_v_u16m1_i16m1(a);
}

NPY_FINLINE npyv_s16 npyv_reinterpret_s16_u32(npyv_u32 a)
{
    return __riscv_vreinterpret_v_i32m1_i16m1(__riscv_vreinterpret_v_u32m1_i32m1(a));
}

NPY_FINLINE npyv_s16 npyv_reinterpret_s16_s32(npyv_s32 a)
{
    return __riscv_vreinterpret_v_i32m1_i16m1(a);
}

NPY_FINLINE npyv_s16 npyv_reinterpret_s16_u64(npyv_u64 a)
{
    return __riscv_vreinterpret_v_i64m1_i16m1(__riscv_vreinterpret_v_u64m1_i64m1(a));
}

NPY_FINLINE npyv_s16 npyv_reinterpret_s16_s64(npyv_s64 a)
{
    return __riscv_vreinterpret_v_i64m1_i16m1(a);
}

NPY_FINLINE npyv_s16 npyv_reinterpret_s16_f32(npyv_f32 a)
{
    return __riscv_vreinterpret_v_i32m1_i16m1(__riscv_vreinterpret_v_f32m1_i32m1(a));
}

NPY_FINLINE npyv_s16 npyv_reinterpret_s16_f64(npyv_f64 a)
{
    return __riscv_vreinterpret_v_i64m1_i16m1(__riscv_vreinterpret_v_f64m1_i64m1(a));
}

#define npyv_reinterpret_u32_u32(X) X
NPY_FINLINE npyv_u32 npyv_reinterpret_u32_u8(npyv_u8 a)
{
    return __riscv_vreinterpret_v_u8m1_u32m1(a);
}

NPY_FINLINE npyv_u32 npyv_reinterpret_u32_s8(npyv_s8 a)
{
    return __riscv_vreinterpret_v_u8m1_u32m1(__riscv_vreinterpret_v_i8m1_u8m1(a));
}

NPY_FINLINE npyv_u32 npyv_reinterpret_u32_u16(npyv_u16 a)
{
    return __riscv_vreinterpret_v_u16m1_u32m1(a);
}

NPY_FINLINE npyv_u32 npyv_reinterpret_u32_s16(npyv_s16 a)
{
    return __riscv_vreinterpret_v_u16m1_u32m1(__riscv_vreinterpret_v_i16m1_u16m1(a));
}

NPY_FINLINE npyv_u32 npyv_reinterpret_u32_s32(npyv_s32 a)
{
    return __riscv_vreinterpret_v_i32m1_u32m1(a);
}

NPY_FINLINE npyv_u32 npyv_reinterpret_u32_u64(npyv_u64 a)
{
    return __riscv_vreinterpret_v_u64m1_u32m1(a);
}

NPY_FINLINE npyv_u32 npyv_reinterpret_u32_s64(npyv_s64 a)
{
    return __riscv_vreinterpret_v_u64m1_u32m1(__riscv_vreinterpret_v_i64m1_u64m1(a));
}

NPY_FINLINE npyv_u32 npyv_reinterpret_u32_f32(npyv_f32 a)
{
    return __riscv_vreinterpret_v_f32m1_u32m1(a);
}

NPY_FINLINE npyv_u32 npyv_reinterpret_u32_f64(npyv_f64 a)
{
    return __riscv_vreinterpret_v_u64m1_u32m1(__riscv_vreinterpret_v_f64m1_u64m1(a));
}

#define npyv_reinterpret_s32_s32(X) X
NPY_FINLINE npyv_s32 npyv_reinterpret_s32_u8(npyv_u8 a)
{
    return __riscv_vreinterpret_v_i8m1_i32m1(__riscv_vreinterpret_v_u8m1_i8m1(a));
}

NPY_FINLINE npyv_s32 npyv_reinterpret_s32_s8(npyv_s8 a)
{
    return __riscv_vreinterpret_v_i8m1_i32m1(a);
}

NPY_FINLINE npyv_s32 npyv_reinterpret_s32_u16(npyv_u16 a)
{
    return __riscv_vreinterpret_v_i16m1_i32m1(__riscv_vreinterpret_v_u16m1_i16m1(a));
}

NPY_FINLINE npyv_s32 npyv_reinterpret_s32_s16(npyv_s16 a)
{
    return __riscv_vreinterpret_v_i16m1_i32m1(a);
}

NPY_FINLINE npyv_s32 npyv_reinterpret_s32_u32(npyv_u32 a)
{
    return __riscv_vreinterpret_v_u32m1_i32m1(a);
}

NPY_FINLINE npyv_s32 npyv_reinterpret_s32_u64(npyv_u64 a)
{
    return __riscv_vreinterpret_v_i64m1_i32m1(__riscv_vreinterpret_v_u64m1_i64m1(a));
}

NPY_FINLINE npyv_s32 npyv_reinterpret_s32_s64(npyv_s64 a)
{
    return __riscv_vreinterpret_v_i64m1_i32m1(a);
}

NPY_FINLINE npyv_s32 npyv_reinterpret_s32_f32(npyv_f32 a)
{
    return __riscv_vreinterpret_v_f32m1_i32m1(a);
}

NPY_FINLINE npyv_s32 npyv_reinterpret_s32_f64(npyv_f64 a)
{
    return __riscv_vreinterpret_v_i64m1_i32m1(__riscv_vreinterpret_v_f64m1_i64m1(a));
}

#define npyv_reinterpret_u64_u64(X) X
NPY_FINLINE npyv_u64 npyv_reinterpret_u64_u8(npyv_u8 a)
{
    return __riscv_vreinterpret_v_u8m1_u64m1(a);
}

NPY_FINLINE npyv_u64 npyv_reinterpret_u64_s8(npyv_s8 a)
{
    return __riscv_vreinterpret_v_u8m1_u64m1(__riscv_vreinterpret_v_i8m1_u8m1(a));
}

NPY_FINLINE npyv_u64 npyv_reinterpret_u64_u16(npyv_u16 a)
{
    return __riscv_vreinterpret_v_u16m1_u64m1(a);
}

NPY_FINLINE npyv_u64 npyv_reinterpret_u64_s16(npyv_s16 a)
{
    return __riscv_vreinterpret_v_u16m1_u64m1(__riscv_vreinterpret_v_i16m1_u16m1(a));
}

NPY_FINLINE npyv_u64 npyv_reinterpret_u64_u32(npyv_u32 a)
{
    return __riscv_vreinterpret_v_u32m1_u64m1(a);
}

NPY_FINLINE npyv_u64 npyv_reinterpret_u64_s32(npyv_s32 a)
{
    return __riscv_vreinterpret_v_u32m1_u64m1(__riscv_vreinterpret_v_i32m1_u32m1(a));
}

NPY_FINLINE npyv_u64 npyv_reinterpret_u64_s64(npyv_s64 a)
{
    return __riscv_vreinterpret_v_i64m1_u64m1(a);
}

NPY_FINLINE npyv_u64 npyv_reinterpret_u64_f32(npyv_f32 a)
{
    return __riscv_vreinterpret_v_u32m1_u64m1(__riscv_vreinterpret_v_f32m1_u32m1(a));
}

NPY_FINLINE npyv_u64 npyv_reinterpret_u64_f64(npyv_f64 a)
{
    return __riscv_vreinterpret_v_f64m1_u64m1(a);
}

#define npyv_reinterpret_s64_s64(X) X
NPY_FINLINE npyv_s64 npyv_reinterpret_s64_u8(npyv_u8 a)
{
    return __riscv_vreinterpret_v_i8m1_i64m1(__riscv_vreinterpret_v_u8m1_i8m1(a));
}

NPY_FINLINE npyv_s64 npyv_reinterpret_s64_s8(npyv_s8 a)
{
    return __riscv_vreinterpret_v_i8m1_i64m1(a);
}

NPY_FINLINE npyv_s64 npyv_reinterpret_s64_u16(npyv_u16 a)
{
    return __riscv_vreinterpret_v_i16m1_i64m1(__riscv_vreinterpret_v_u16m1_i16m1(a));
}

NPY_FINLINE npyv_s64 npyv_reinterpret_s64_s16(npyv_s16 a)
{
    return __riscv_vreinterpret_v_i16m1_i64m1(a);
}

NPY_FINLINE npyv_s64 npyv_reinterpret_s64_u32(npyv_u32 a)
{
    return __riscv_vreinterpret_v_i32m1_i64m1(__riscv_vreinterpret_v_u32m1_i32m1(a));
}

NPY_FINLINE npyv_s64 npyv_reinterpret_s64_s32(npyv_s32 a)
{
    return __riscv_vreinterpret_v_i32m1_i64m1(a);
}

NPY_FINLINE npyv_s64 npyv_reinterpret_s64_u64(npyv_u64 a)
{
    return __riscv_vreinterpret_v_u64m1_i64m1(a);
}

NPY_FINLINE npyv_s64 npyv_reinterpret_s64_f32(npyv_f32 a)
{
    return __riscv_vreinterpret_v_i32m1_i64m1(__riscv_vreinterpret_v_f32m1_i32m1(a));
}

NPY_FINLINE npyv_s64 npyv_reinterpret_s64_f64(npyv_f64 a)
{
    return __riscv_vreinterpret_v_f64m1_i64m1(a);
}

#define npyv_reinterpret_f32_f32(X) X
NPY_FINLINE npyv_f32 npyv_reinterpret_f32_u8(npyv_u8 a)
{
    return __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vreinterpret_v_u8m1_u32m1(a));
}

NPY_FINLINE npyv_f32 npyv_reinterpret_f32_s8(npyv_s8 a)
{
    return __riscv_vreinterpret_v_i32m1_f32m1(__riscv_vreinterpret_v_i8m1_i32m1(a));
}

NPY_FINLINE npyv_f32 npyv_reinterpret_f32_u16(npyv_u16 a)
{
    return __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vreinterpret_v_u16m1_u32m1(a));
}

NPY_FINLINE npyv_f32 npyv_reinterpret_f32_s16(npyv_s16 a)
{
    return __riscv_vreinterpret_v_i32m1_f32m1(__riscv_vreinterpret_v_i16m1_i32m1(a));
}

NPY_FINLINE npyv_f32 npyv_reinterpret_f32_u32(npyv_u32 a)
{
    return __riscv_vreinterpret_v_u32m1_f32m1(a);
}

NPY_FINLINE npyv_f32 npyv_reinterpret_f32_s32(npyv_s32 a)
{
    return __riscv_vreinterpret_v_i32m1_f32m1(a);
}

NPY_FINLINE npyv_f32 npyv_reinterpret_f32_u64(npyv_u64 a)
{
    return __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vreinterpret_v_u64m1_u32m1(a));
}

NPY_FINLINE npyv_f32 npyv_reinterpret_f32_s64(npyv_s64 a)
{
    return __riscv_vreinterpret_v_i32m1_f32m1(__riscv_vreinterpret_v_i64m1_i32m1(a));
}

NPY_FINLINE npyv_f32 npyv_reinterpret_f32_f64(npyv_f64 a)
{
    return __riscv_vreinterpret_v_u32m1_f32m1(__riscv_vreinterpret_v_u64m1_u32m1(__riscv_vreinterpret_v_f64m1_u64m1(a)));
}

#define npyv_reinterpret_f64_f64(X) X
NPY_FINLINE npyv_f64 npyv_reinterpret_f64_u8(npyv_u8 a)
{
    return __riscv_vreinterpret_v_u64m1_f64m1(__riscv_vreinterpret_v_u8m1_u64m1(a));
}

NPY_FINLINE npyv_f64 npyv_reinterpret_f64_s8(npyv_s8 a)
{
    return __riscv_vreinterpret_v_i64m1_f64m1(__riscv_vreinterpret_v_i8m1_i64m1(a));
}

NPY_FINLINE npyv_f64 npyv_reinterpret_f64_u16(npyv_u16 a)
{
    return __riscv_vreinterpret_v_u64m1_f64m1(__riscv_vreinterpret_v_u16m1_u64m1(a));
}

NPY_FINLINE npyv_f64 npyv_reinterpret_f64_s16(npyv_s16 a)
{
    return __riscv_vreinterpret_v_i64m1_f64m1(__riscv_vreinterpret_v_i16m1_i64m1(a));
}

NPY_FINLINE npyv_f64 npyv_reinterpret_f64_u32(npyv_u32 a)
{
    return __riscv_vreinterpret_v_u64m1_f64m1(__riscv_vreinterpret_v_u32m1_u64m1(a));
}

NPY_FINLINE npyv_f64 npyv_reinterpret_f64_s32(npyv_s32 a)
{
    return __riscv_vreinterpret_v_i64m1_f64m1(__riscv_vreinterpret_v_i32m1_i64m1(a));
}

NPY_FINLINE npyv_f64 npyv_reinterpret_f64_u64(npyv_u64 a)
{
    return __riscv_vreinterpret_v_u64m1_f64m1(a);
}

NPY_FINLINE npyv_f64 npyv_reinterpret_f64_s64(npyv_s64 a)
{
    return __riscv_vreinterpret_v_i64m1_f64m1(a);
}

NPY_FINLINE npyv_f64 npyv_reinterpret_f64_f32(npyv_f32 a)
{
    return __riscv_vreinterpret_v_u64m1_f64m1(__riscv_vreinterpret_v_u32m1_u64m1(__riscv_vreinterpret_v_f32m1_u32m1(a)));
}

// Only required by AVX2/AVX512
#define npyv_cleanup() ((void)0)

#endif // _NPY_SIMD_RVV_MISC_H
