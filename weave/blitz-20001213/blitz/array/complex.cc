// Special functions for complex arrays

#ifndef BZ_ARRAYCOMPLEX_CC
#define BZ_ARRAYCOMPLEX_CC

#ifndef BZ_ARRAY_H
 #error <blitz/array/complex.cc> must be included via <blitz/array/array.h>
#endif

BZ_NAMESPACE(blitz)

#ifdef BZ_HAVE_COMPLEX

template<class T_numtype, int N_rank>
inline Array<T_numtype, N_rank> real(const Array<complex<T_numtype>,N_rank>& A)
{
    return A.extractComponent(T_numtype(), 0, 2);
}

template<class T_numtype, int N_rank>
inline Array<T_numtype, N_rank> imag(const Array<complex<T_numtype>,N_rank>& A)
{
    return A.extractComponent(T_numtype(), 1, 2);
}


#endif

BZ_NAMESPACE_END

#endif // BZ_ARRAYCOMPLEX_CC

