#ifndef BZ_ARRAY_CONVOLVE_H
#define BZ_ARRAY_CONVOLVE_H

#ifndef BZ_ARRAY_H
 #error <blitz/array/convolve.h> must be included after <blitz/array.h>
#endif

BZ_NAMESPACE(blitz)

template<class T>
Array<T,1> convolve(const Array<T,1>& B, const Array<T,1>& C);

BZ_NAMESPACE_END

#include <blitz/array/convolve.cc>

#endif // BZ_ARRAY_CONVOLVE_H
