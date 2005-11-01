// -*- C++ -*-
/***************************************************************************
 * blitz/meta/sum.h      TinyVector sum metaprogram
 *
 * $Id$
 *
 * Copyright (C) 1997-2001 Todd Veldhuizen <tveldhui@oonumerics.org>
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
 ***************************************************************************/

#ifndef BZ_META_SUM_H
#define BZ_META_SUM_H

#ifndef BZ_METAPROG_H
 #include <blitz/meta/metaprog.h>
#endif

BZ_NAMESPACE(blitz)

template<int N, int I>
class _bz_meta_vectorSum {
public:
    static const int loopFlag = (I < N-1) ? 1 : 0;

    template<typename T_expr1>
    static inline _bz_typename T_expr1::T_numtype
    f(const T_expr1& a)
    {
        return a[I] +
            _bz_meta_vectorSum<loopFlag * N, loopFlag * (I+1)>::f(a);
    }
};

template<>
class _bz_meta_vectorSum<0,0> {
public:
    template<typename T_expr1>
    static inline _bz_meta_nullOperand f(const T_expr1&)
    { return _bz_meta_nullOperand(); }

};

BZ_NAMESPACE_END

#endif // BZ_META_SUM_H
