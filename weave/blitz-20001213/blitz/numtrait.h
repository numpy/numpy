/***************************************************************************
 * blitz/numtrait.h      Declaration of the NumericTypeTraits class
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
 * Revision 1.1.1.1  2000/06/19 12:26:09  tveldhui
 * Imported sources
 *
 * Revision 1.5  1998/03/14 00:04:47  tveldhui
 * 0.2-alpha-05
 *
 * Revision 1.4  1997/07/16 14:51:20  tveldhui
 * Update: Alpha release 0.2 (Arrays)
 *
 * Revision 1.3  1997/01/24 14:42:00  tveldhui
 * Periodic RCS update
 *
 * Revision 1.2  1997/01/23 03:28:28  tveldhui
 * Periodic RCS update
 *
 * Revision 1.1  1997/01/13 22:19:58  tveldhui
 * Periodic RCS update
 *
 *
 */

#ifndef BZ_NUMTRAIT_H
#define BZ_NUMTRAIT_H

#ifndef BZ_BLITZ_H
 #include <blitz/blitz.h>
#endif

BZ_NAMESPACE(blitz)

#ifndef BZ_USE_NUMTRAIT
  #define BZ_SUMTYPE(X)    X
  #define BZ_DIFFTYPE(X)   X
  #define BZ_FLOATTYPE(X)  X
  #define BZ_SIGNEDTYPE(X) X
#else

#define BZ_SUMTYPE(X)   _bz_typename NumericTypeTraits<X>::T_sumtype
#define BZ_DIFFTYPE(X)  _bz_typename NumericTypeTraits<X>::T_difftype
#define BZ_FLOATTYPE(X) _bz_typename NumericTypeTraits<X>::T_floattype
#define BZ_SIGNEDTYPE(X) _bz_typename NumericTypeTraits<X>::T_signedtype

template<class P_numtype>
class NumericTypeTraits {
public:
    typedef P_numtype T_sumtype;    // Type to be used for summing
    typedef P_numtype T_difftype;   // Type to be used for difference
    typedef P_numtype T_floattype;  // Type to be used for floating-point
                                    // calculations
    typedef P_numtype T_signedtype; // Type to be used for signed calculations
    enum { hasTrivialCtor = 0 };    // Assume the worst
};

#define BZDECLNUMTRAIT(X,Y,Z,W,U)                                   \
    template<>                                                      \
    class NumericTypeTraits<X> {                                    \
    public:                                                         \
        typedef Y T_sumtype;                                        \
        typedef Z T_difftype;                                       \
        typedef W T_floattype;                                      \
        typedef U T_signedtype;                                     \
        enum { hasTrivialCtor = 1 };                                \
    }                                                               

#ifdef BZ_BOOL
    BZDECLNUMTRAIT(bool,unsigned,int,float,int);
#endif

BZDECLNUMTRAIT(char,int,int,float,char);
BZDECLNUMTRAIT(unsigned char, unsigned, int, float,int);
BZDECLNUMTRAIT(short int, int, int, float, short int);
BZDECLNUMTRAIT(short unsigned int, unsigned int, int, float, int);
BZDECLNUMTRAIT(int, long, int, float, int);
BZDECLNUMTRAIT(unsigned int, unsigned long, int, float, long);
BZDECLNUMTRAIT(long, long, long, double, long);
BZDECLNUMTRAIT(unsigned long, unsigned long, long, double, long);
BZDECLNUMTRAIT(float, double, float, float, float);
BZDECLNUMTRAIT(double, double, double, double, double);

#ifdef BZ_HAVE_COMPLEX
// BZDECLNUMTRAIT(complex<float>, complex<double>, complex<float>, complex<float>);
// BZDECLNUMTRAIT(complex<double>, complex<long double>, complex<double>, complex<double>);
#endif // BZ_HAVE_COMPLEX

#endif // BZ_USE_NUMTRAIT

BZ_NAMESPACE_END

#endif // BZ_NUMTRAIT_H
