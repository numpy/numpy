/*
 * Copyright (C) 1997 Todd Veldhuizen <tveldhui@oonumerics.org>
 * All rights reserved.  Please see <blitz/blitz.h> for terms and
 * conditions of use.
 *
 */

#ifndef BZ_ZERO_H
 #include <blitz/zero.h>
#endif

#ifndef BZ_ZERO_CC
#define BZ_ZERO_CC

BZ_NAMESPACE(blitz)

#ifdef BZ_HAVE_COMPLEX

complex<float>  ZeroElement<complex<float> >::zero_ = 
    complex<float>(0.0f, 0.0f);

complex<double> ZeroElement<complex<double> >::zero_ =
    complex<double>(0.,0.);

complex<long double> ZeroElement<complex<long double> >::zero_ =
    complex<long double>(0.0L, 0.0L);

#endif // BZ_HAVE_COMPLEX

BZ_NAMESPACE_END

#endif // BZ_ZERO_CC

