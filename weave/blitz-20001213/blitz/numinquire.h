/***************************************************************************
 * blitz/numinquire.h    Numeric inquiry functions
 *
 * $Id$
 *
 * Copyright (C) 1997-1999 Todd Veldhuizen <tveldhui@oonumerics.org>
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * Suggestions:          blitz-dev@oonumerics.org
 * Bugs:                 blitz-bugs@oonumerics.org
 *
 * For more information, please see the Blitz++ Home Page:
 *    http://oonumerics.org/blitz/
 *
 ***************************************************************************
 * $Log$
 * Revision 1.1  2002/01/03 19:50:34  eric
 * renaming compiler to weave
 *
 * Revision 1.1  2001/04/27 17:22:04  ej
 * first attempt to include needed pieces of blitz
 *
 * Revision 1.1.1.1  2000/06/19 12:26:12  tveldhui
 * Imported sources
 *
 * Revision 1.2  1998/03/14 00:04:47  tveldhui
 * 0.2-alpha-05
 *
 * Revision 1.1  1997/07/16 14:51:20  tveldhui
 * Update: Alpha release 0.2 (Arrays)
 *
 */

/*
 * These numeric inquiry functions are provided as an alternative
 * to the somewhat klunky numeric_limits<T>::yadda_yadda syntax.
 * Where a similar Fortran 90 function exists, the same name has
 * been used.
 *
 * The argument in all cases is a dummy of the appropriate type
 * (double, int, etc.)
 *
 * These functions assume that numeric_limits<T> has been specialized
 * for the appropriate case.  If not, the results are not useful.
 */

#ifndef BZ_NUMINQUIRE_H
#define BZ_NUMINQUIRE_H

#ifndef BZ_HAVE_NUMERIC_LIMITS
  #include <blitz/limits-hack.h>
#else
  #include <limits>
#endif

#ifndef BZ_RANGE_H
 #include <blitz/range.h>
#endif

BZ_NAMESPACE(blitz)

/*
 * This traits class provides zero and one values for numeric
 * types.  This was previously a template function with specializations,
 * but the specializations were causing multiply-defined symbols
 * at link time.  TV 980226
 */

template<class T_numtype>
struct _bz_OneZeroTraits {
    static inline T_numtype zero() { return 0; }
    static inline T_numtype one()  { return 1; }
};

#ifdef BZ_HAVE_COMPLEX

template<>
struct _bz_OneZeroTraits<complex<float> > {
    static inline complex<float> zero() { return complex<float>(0.0f,0.0f); }
    static inline complex<float> one()  { return complex<float>(1.0f,0.0f); }
};

template<>
struct _bz_OneZeroTraits<complex<double> > {
    static inline complex<double> zero() { return complex<double>(0.0,0.0); }
    static inline complex<double> one()  { return complex<double>(1.0,0.0); }
};

template<>
struct _bz_OneZeroTraits<complex<long double> > {
    static inline complex<long double> zero() 
    { return complex<long double>(0.0,0.0); }

    static inline complex<long double> one()  
    { return complex<long double>(1.0,0.0); }
};

#endif // BZ_HAVE_COMPLEX

template<class T>
inline T zero(T)
{
    return _bz_OneZeroTraits<T>::zero();
}

template<class T>
inline T one(T)
{
    return _bz_OneZeroTraits<T>::one();
}

template<class T>
inline int digits(T)
{
    return numeric_limits<T>::digits;
}

template<class T>
inline int digits10(T)
{
    return numeric_limits<T>::digits10;
}

template<class T>
inline T epsilon(T) BZ_THROW
{
    return numeric_limits<T>::epsilon();
}

// neghuge() by Theodore Papadopoulo, to fix a problem with
// max() reductions.
template<class T>
inline T neghuge(T) BZ_THROW
{
    return numeric_limits<T>::is_integer ?    numeric_limits<T>::min()
                                         : - numeric_limits<T>::max();
}

template<class T>
inline T huge(T) BZ_THROW
{
    return numeric_limits<T>::max();
}

template<class T>
inline T tiny(T) BZ_THROW
{
    return numeric_limits<T>::min();
}

template<class T>
inline int max_exponent(T)
{
    return numeric_limits<T>::max_exponent;
}

template<class T>
inline int min_exponent(T)
{
    return numeric_limits<T>::min_exponent;
}

template<class T>
inline int min_exponent10(T)
{
    return numeric_limits<T>::min_exponent10;
}

template<class T>
inline int max_exponent10(T)
{
    return numeric_limits<T>::max_exponent10;
}

template<class T>
inline int precision(T)
{
    return numeric_limits<T>::digits10;
}

template<class T>
inline int radix(T)
{
    return numeric_limits<T>::radix;
}

template<class T>
inline Range range(T)
{
    return Range(numeric_limits<T>::min_exponent10, 
        numeric_limits<T>::max_exponent10);
}

template<class T>
inline bool is_signed(T)
{
    return numeric_limits<T>::is_signed;
}

template<class T>
inline bool is_integer(T)
{
    return numeric_limits<T>::is_integer;
}

template<class T>
inline bool is_exact(T)
{
    return numeric_limits<T>::is_exact;
}

template<class T>
inline T round_error(T) BZ_THROW
{
    return numeric_limits<T>::round_error();
}

template<class T>
inline bool has_infinity(T) 
{
    return numeric_limits<T>::has_infinity;
}

template<class T>
inline bool has_quiet_NaN(T)
{
    return numeric_limits<T>::has_quiet_NaN;
}

template<class T>
inline bool has_signaling_NaN(T)
{
    return numeric_limits<T>::has_signaling_NaN;
}

// Provided for non-US english users
template<class T>
inline bool has_signalling_NaN(T)
{
    return numeric_limits<T>::has_signaling_NaN;
}

template<class T>
inline bool has_denorm(T)
{
    return numeric_limits<T>::has_denorm;
}

template<class T>
inline bool has_denorm_loss(T)
{
    return numeric_limits<T>::has_denorm_loss;
}

template<class T>
inline T infinity(T) BZ_THROW
{
    return numeric_limits<T>::infinity();
}

template<class T>
inline T quiet_NaN(T) BZ_THROW
{
    return numeric_limits<T>::quiet_NaN();
}

template<class T>
inline T signaling_NaN(T) BZ_THROW
{
    return numeric_limits<T>::signaling_NaN();
}

template<class T>
inline T signalling_NaN(T) BZ_THROW
{
    return numeric_limits<T>::signaling_NaN();
}

template<class T>
inline T denorm_min(T) BZ_THROW
{
    return numeric_limits<T>::denorm_min();
}

template<class T>
inline bool is_iec559(T)
{
    return numeric_limits<T>::is_iec559;
}

template<class T>
inline bool is_bounded(T)
{
    return numeric_limits<T>::is_bounded;
}

template<class T>
inline bool is_modulo(T)
{
    return numeric_limits<T>::is_modulo;
}

template<class T>
inline bool traps(T)
{
    return numeric_limits<T>::traps;
}

template<class T>
inline bool tinyness_before(T)
{
    return numeric_limits<T>::tinyness_before;
}

template<class T>
inline BZ_STD_SCOPE(float_round_style) round_style(T)
{
    return numeric_limits<T>::round_style;
}

BZ_NAMESPACE_END

#endif // BZ_NUMINQUIRE_H

