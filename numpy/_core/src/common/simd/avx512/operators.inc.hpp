#ifndef NUMPY_CORE_SRC_COMMON_SIMD_AVX512_OPERATORS_INC_HPP_
#define NUMPY_CORE_SRC_COMMON_SIMD_AVX512_OPERATORS_INC_HPP_

#if NPY_SIMD
namespace np::NPY_CPU_DISPATCH_CURFX(simd_ext) {
/***************************
 * Logical
 ***************************/
template<typename TVecOrMask>
NPY_FINLINE TVecOrMask Orc(const TVecOrMask &a, const TVecOrMask &b)
{ return Or(a, Not(b)); }

template<typename TVecOrMask>
NPY_FINLINE TVecOrMask Andc(const TVecOrMask &a, const TVecOrMask &b)
{ return TVecOrMask(_mm512_andnot_si512(b.val, a.val)); }

template<>
NPY_FINLINE Vec<float> Andc(const Vec<float> &a, const Vec<float> &b)
{
#ifdef NPY_HAVE_AVX512DQ
    return Vec<float>(_mm512_andnot_ps(b.val, a.val));
#else
    return Reinterpret<float>(Vec<uint8_t>(
        _mm512_andnot_si512(
            Reinterpret<uint8_t>(b).val,
            Reinterpret<uint8_t>(a).val
        )
    ));
#endif
}
template<>
NPY_FINLINE Vec<double> Andc(const Vec<double> &a, const Vec<double> &b)
{
#ifdef NPY_HAVE_AVX512DQ
    return Vec<double>(_mm512_andnot_pd(b.val, a.val));
#else
    return Reinterpret<double>(Vec<uint8_t>(
        _mm512_andnot_si512(
            Reinterpret<uint8_t>(b).val,
            Reinterpret<uint8_t>(a).val
        )
    ));
#endif
}

template<>
NPY_FINLINE Mask<uint8_t> Andc(const Mask<uint8_t> &a, const Mask<uint8_t> &b)
{
#ifdef NPY_HAVE_AVX512BW_MASK
    return Mask<uint8_t>(_kandn_mask64(b.val, a.val));
#elif defined(NPY_HAVE_AVX512BW)
    return And(a, Not(b));
#else
    return Mask<uint8_t>(_mm512_andnot_si512(b.val, a.val));
#endif
}
template<>
NPY_FINLINE Mask<uint16_t> Andc(const Mask<uint16_t> &a, const Mask<uint16_t> &b)
{
#ifdef NPY_HAVE_AVX512BW_MASK
    return Mask<uint16_t>(_kandn_mask32(b.val, a.val));
#elif defined(NPY_HAVE_AVX512BW)
    return And(a, Not(b));
#else
    return Mask<uint16_t>(_mm512_andnot_si512(b.val, a.val));
#endif
}
template<>
NPY_FINLINE Mask<uint32_t> Andc(const Mask<uint32_t> &a, const Mask<uint32_t> &b)
{ return Mask<uint32_t>(_kandn_mask16(b.val, a.val)); }
template<>
NPY_FINLINE Mask<uint64_t> Andc(const Mask<uint64_t> &a, const Mask<uint64_t> &b)
{
#ifdef NPY_HAVE_AVX512DQ_MASK
    return Mask<uint64_t>(_kandn_mask8(b.val, a.val));
#else
    return And(a, Not(b));
#endif
}

template<typename TVecOrMask>
NPY_FINLINE TVecOrMask Xnor(const TVecOrMask &a, const TVecOrMask &b)
{ return Not(Xor(a, b)); }
#ifdef NPY_HAVE_AVX512BW_MASK
    template<>
    NPY_FINLINE Mask<uint8_t> Xnor(const Mask<uint8_t> &a, const Mask<uint8_t> &b)
    { return Mask<uint8_t>(_kxnor_mask64(a.val, b.val)); }
    template<>
    NPY_FINLINE Mask<uint16_t> Xnor(const Mask<uint16_t> &a, const Mask<uint16_t> &b)
    { return Mask<uint16_t>(_kxnor_mask32(a.val, b.val)); }
#endif
template<>
NPY_FINLINE Mask<uint32_t> Xnor(const Mask<uint32_t> &a, const Mask<uint32_t> &b)
{ return Mask<uint32_t>(_kxnor_mask16(a.val, b.val)); }
#ifdef NPY_HAVE_AVX512DQ_MASK
    template<>
    NPY_FINLINE Mask<uint64_t> Xnor(const Mask<uint64_t> &a, const Mask<uint64_t> &b)
    { return Mask<uint64_t>(_kxnor_mask8(a.val, b.val)); }
#endif

} // namespace np::NPY_CPU_DISPATCH_CURFX(simd_ext)
#endif // NPY_SIMD
#endif // NUMPY_CORE_SRC_COMMON_SIMD_AVX512_OPERATORS_INC_HPP_

