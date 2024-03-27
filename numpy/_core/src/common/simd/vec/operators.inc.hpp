#ifndef NUMPY_CORE_SRC_COMMON_SIMD_VEC_OPERATORS_INC_HPP_
#define NUMPY_CORE_SRC_COMMON_SIMD_VEC_OPERATORS_INC_HPP_

#if NPY_SIMD
namespace np::NPY_CPU_DISPATCH_CURFX(simd_ext) {
/***************************
 * Logical
 ***************************/

template<typename TVecOrMask>
NPY_FINLINE TVecOrMask Xnor(const TVecOrMask &a, const TVecOrMask &b)
{
#if defined(NPY_HAVE_VXE) || defined(NPY_HAVE_VSX)
    return TVecOrMask(vec_eqv(a.val, b.val));
#else
    return Not(Xor(a, b));
#endif
}

template<typename TVecOrMask>
NPY_FINLINE TVecOrMask Andc(const TVecOrMask &a, const TVecOrMask &b)
{
    return TVecOrMask(vec_andc(a.val, b.val));
}

template<typename TVecOrMask>
NPY_FINLINE TVecOrMask Orc(const TVecOrMask &a, const TVecOrMask &b)
{
#if defined(NPY_HAVE_VXE) || defined(NPY_HAVE_VSX)
    return TVecOrMask(vec_orc(a.val, b.val));
#else
    return TVecOrMask(Or(a, Not(b)));
#endif
}

#if NPY_SIMD_F32
    template<>
    NPY_FINLINE Vec<float> Xnor(const Vec<float> &a, const Vec<float> &b)
    { return Reinterpret<float>(Xnor(Reinterpret<uint32_t>(a), Reinterpret<uint32_t>(b))); }
    template<>
    NPY_FINLINE Vec<float> Orc(const Vec<float> &a, const Vec<float> &b)
    { return Reinterpret<float>(Orc(Reinterpret<uint32_t>(a), Reinterpret<uint32_t>(b))); }
    template<>
    NPY_FINLINE Vec<float> Andc(const Vec<float> &a, const Vec<float> &b)
    { return Reinterpret<float>(Andc(Reinterpret<uint32_t>(a), Reinterpret<uint32_t>(b))); }
#endif
template<>
NPY_FINLINE Vec<double> Xnor(const Vec<double> &a, const Vec<double> &b)
{ return Reinterpret<double>(Xnor(Reinterpret<uint64_t>(a), Reinterpret<uint64_t>(b))); }
template<>
NPY_FINLINE Vec<double> Orc(const Vec<double> &a, const Vec<double> &b)
{ return Reinterpret<double>(Orc(Reinterpret<uint64_t>(a), Reinterpret<uint64_t>(b))); }
template<>
NPY_FINLINE Vec<double> Andc(const Vec<double> &a, const Vec<double> &b)
{ return Reinterpret<double>(Andc(Reinterpret<uint64_t>(a), Reinterpret<uint64_t>(b))); }

} // namespace np::NPY_CPU_DISPATCH_CURFX(simd_ext)
#endif // NPY_SIMD
#endif // NUMPY_CORE_SRC_COMMON_SIMD_VEC_OPERATORS_INC_HPP_

