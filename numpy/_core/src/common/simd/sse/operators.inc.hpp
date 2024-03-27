#ifndef NUMPY_CORE_SRC_COMMON_SIMD_SSE_OPERATORS_INC_HPP_
#define NUMPY_CORE_SRC_COMMON_SIMD_SSE_OPERATORS_INC_HPP_

#if NPY_SIMD
namespace np::NPY_CPU_DISPATCH_CURFX(simd_ext) {
/***************************
 * Logical
 ***************************/
template<typename TVecOrMask>
NPY_FINLINE TVecOrMask Andc(const TVecOrMask &a, const TVecOrMask &b)
{ return TVecOrMask(_mm_andnot_si128(b.val, a.val)); }
template<>
NPY_FINLINE Vec<float> Andc(const Vec<float> &a, const Vec<float> &b)
{ return Vec<float>(_mm_andnot_ps(b.val, a.val)); }
template<>
NPY_FINLINE Vec<double> Andc(const Vec<double> &a, const Vec<double> &b)
{ return Vec<double>(_mm_andnot_pd(b.val, a.val)); }

template<typename TVecOrMask>
NPY_FINLINE TVecOrMask Orc(const TVecOrMask &a, const TVecOrMask &b)
{ return Or(a, Not(b)); }

template<typename TVecOrMask>
NPY_FINLINE TVecOrMask Xnor(const TVecOrMask &a, const TVecOrMask &b)
{ return Not(Xor(a, b)); }
template<>
NPY_FINLINE Mask<uint8_t> Xnor(const Mask<uint8_t> &a, const Mask<uint8_t> &b)
{ return Mask<uint8_t>(_mm_cmpeq_epi8(a.val, b.val)); }

} // namespace np::NPY_CPU_DISPATCH_CURFX(simd_ext)
#endif // NPY_SIMD
#endif // NUMPY_CORE_SRC_COMMON_SIMD_SSE_OPERATORS_INC_HPP_

