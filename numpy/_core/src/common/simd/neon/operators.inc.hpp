#ifndef NUMPY_CORE_SRC_COMMON_SIMD_NEON_OPERATORS_INC_HPP_
#define NUMPY_CORE_SRC_COMMON_SIMD_NEON_OPERATORS_INC_HPP_

#if NPY_SIMD
namespace np::NPY_CPU_DISPATCH_CURFX(simd_ext) {
/***************************
 * Logical
 ***************************/

template<typename TVecOrMask>
NPY_FINLINE TVecOrMask Xnor(const TVecOrMask &a, const TVecOrMask &b)
{ return Not(Xor(a, b)); }
template<>
NPY_FINLINE Mask<uint8_t> Xnor(const Mask<uint8_t> &a, const Mask<uint8_t> &b)
{ return Mask<uint8_t>(vceqq_u8(a.val, b.val)); }
template<>
NPY_FINLINE Mask<uint16_t> Xnor(const Mask<uint16_t> &a, const Mask<uint16_t> &b)
{ return Mask<uint16_t>(vceqq_u16(a.val, b.val)); }
template<>
NPY_FINLINE Mask<uint32_t> Xnor(const Mask<uint32_t> &a, const Mask<uint32_t> &b)
{ return Mask<uint32_t>(vceqq_u32(a.val, b.val)); }
template<>
NPY_FINLINE Mask<uint64_t> Xnor(const Mask<uint64_t> &a, const Mask<uint64_t> &b)
{
    return Mask<uint64_t>(vreinterpretq_u64_u32(
        vceqq_u32(vreinterpretq_u32_u64(a.val), vreinterpretq_u32_u64(b.val))
    ));
}

#define NPYV_IMPL_NEON_BITWISE(TVEC, SFX)                 \
    template<>                                            \
    NPY_FINLINE TVEC Andc(const TVEC &a, const TVEC &b)   \
    { return TVEC(vbicq_##SFX(a.val, b.val)); }           \
    template<>                                            \
    NPY_FINLINE TVEC Orc(const TVEC &a, const TVEC &b)    \
    { return TVEC(vornq_##SFX(a.val, b.val)); }
NPYV_IMPL_NEON_BITWISE(Vec<uint8_t>,  u8)
NPYV_IMPL_NEON_BITWISE(Vec<int8_t>,   s8)
NPYV_IMPL_NEON_BITWISE(Vec<uint16_t>, u16)
NPYV_IMPL_NEON_BITWISE(Vec<int16_t>,  s16)
NPYV_IMPL_NEON_BITWISE(Vec<uint32_t>, u32)
NPYV_IMPL_NEON_BITWISE(Vec<int32_t>,  s32)
NPYV_IMPL_NEON_BITWISE(Vec<uint64_t>, u64)
NPYV_IMPL_NEON_BITWISE(Vec<int64_t>,  s64)
NPYV_IMPL_NEON_BITWISE(Mask<uint8_t>,  u8)
NPYV_IMPL_NEON_BITWISE(Mask<uint16_t>, u16)
NPYV_IMPL_NEON_BITWISE(Mask<uint32_t>, u32)
NPYV_IMPL_NEON_BITWISE(Mask<uint64_t>, u64)
#undef NPYV_IMPL_NEON_BITWISE

template<>
NPY_FINLINE Vec<float> Orc(const Vec<float> &a, const Vec<float> &b)
{ return Reinterpret<float>(Orc(Reinterpret<uint32_t>(a), Reinterpret<uint32_t>(b))); }
template<>
NPY_FINLINE Vec<float> Andc(const Vec<float> &a, const Vec<float> &b)
{ return Reinterpret<float>(Andc(Reinterpret<uint32_t>(a), Reinterpret<uint32_t>(b))); }
#if NPY_SIMD_F64
    template<>
    NPY_FINLINE Vec<double> Orc(const Vec<double> &a, const Vec<double> &b)
    { return Reinterpret<double>(Orc(Reinterpret<uint64_t>(a), Reinterpret<uint64_t>(b))); }
    template<>
    NPY_FINLINE Vec<double> Andc(const Vec<double> &a, const Vec<double> &b)
    { return Reinterpret<double>(Andc(Reinterpret<uint64_t>(a), Reinterpret<uint64_t>(b))); }
#endif

} // namespace np::NPY_CPU_DISPATCH_CURFX(simd_ext)
#endif // NPY_SIMD
#endif // NUMPY_CORE_SRC_COMMON_SIMD_NEON_OPERATORS_INC_HPP_

