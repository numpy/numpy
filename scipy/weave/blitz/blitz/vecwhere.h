// -*- C++ -*-
/***************************************************************************
 * blitz/vecwhere.h      where(X,Y,Z) function for vectors
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
 ****************************************************************************/

#ifndef BZ_VECWHERE_H 
#define BZ_VECWHERE_H

#ifndef BZ_VECEXPR_H
 #error <blitz/vecwhere.h> must be included via <blitz/vector.h>
#endif

BZ_NAMESPACE(blitz)

template<typename P_expr1, typename P_expr2, typename P_expr3>
class _bz_VecWhere {

public:
    typedef P_expr1 T_expr1;
    typedef P_expr2 T_expr2;
    typedef P_expr3 T_expr3;
    typedef _bz_typename T_expr2::T_numtype T_numtype2;
    typedef _bz_typename T_expr3::T_numtype T_numtype3;
    typedef BZ_PROMOTE(T_numtype2, T_numtype3) T_numtype;

#ifdef BZ_PASS_EXPR_BY_VALUE
    _bz_VecWhere(T_expr1 a, T_expr2 b, T_expr3 c)
        : iter1_(a), iter2_(b), iter3_(c)
    { }
#else
    _bz_VecWhere(const T_expr1& a, const T_expr2& b, const T_expr3& c)
        : iter1_(a), iter2_(b), iter3_(c)
    { }
#endif

#ifdef BZ_MANUAL_VECEXPR_COPY_CONSTRUCTOR
    _bz_VecWhere(const _bz_VecWhere<T_expr1, T_expr2, T_expr3>& x)
        : iter1_(x.iter1_), iter2_(x.iter2_), iter3_(x.iter3_)
    { }
#endif

    T_numtype operator[](int i) const
    { 
        return iter1_[i] ? iter2_[i] : iter3_[i];
    }

    static const int
        _bz_staticLengthCount = P_expr1::_bz_staticLengthCount
                              + P_expr2::_bz_staticLengthCount
                              + P_expr3::_bz_staticLengthCount,
        _bz_dynamicLengthCount = P_expr1::_bz_dynamicLengthCount
                               + P_expr2::_bz_dynamicLengthCount
                               + P_expr3::_bz_dynamicLengthCount,
        _bz_staticLength =
            _bz_meta_max<_bz_meta_max<P_expr1::_bz_staticLength, 
            P_expr2::_bz_staticLength>::max, P_expr3::_bz_staticLength>::max;

    T_numtype _bz_fastAccess(int i) const
    { 
        return iter1_._bz_fastAccess(i) 
            ? iter2_._bz_fastAccess(i) 
            : iter3_._bz_fastAccess(i); 
    }

    bool _bz_hasFastAccess() const
    {
        return iter1_._bz_hasFastAccess() &&
            iter2_._bz_hasFastAccess() &&
            iter3_._bz_hasFastAccess();
    }

    int length(int recommendedLength) const
    {
        return iter1_.length(recommendedLength);
    }

    int _bz_suggestLength() const
    {
        BZPRECONDITION(
            (iter1_._bz_suggestLength() == iter2_._bz_suggestLength())
         && (iter2_._bz_suggestLength() == iter3_._bz_suggestLength()));

        return iter1_._bz_suggestLength();
    }

private:
    _bz_VecWhere() { }

    T_expr1 iter1_;
    T_expr2 iter2_;
    T_expr3 iter3_;
};

BZ_NAMESPACE_END

#include <blitz/vecwhere.cc>        // Expression templates

#endif // BZ_VECWHERE_H

