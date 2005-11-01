// -*- C++ -*-
/***************************************************************************
 * blitz/tinyveciter.h   Declaration of TinyVectorIter<T,N,stride>
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


#ifndef BZ_TINYVECITER_H
#define BZ_TINYVECITER_H

#ifndef BZ_TINYVEC_H
 #include <blitz/tinyvec.h>
#endif

#ifndef BZ_MANUAL_VECEXPR_COPY_CONSTRUCTOR
 #error "Debug in tinyveciter.h (this line shouldn't be here)"
#endif

BZ_NAMESPACE(blitz)

// N_stride has default 1, in forward declaration in <blitz/tinyvec.h>
template<typename P_numtype, int N_length, int N_stride>
class TinyVectorIter {
public:
    typedef P_numtype T_numtype;

    explicit TinyVectorIter(TinyVector<T_numtype, N_length>& x)
        : data_(x.data())
    { }

#ifdef BZ_MANUAL_VECEXPR_COPY_CONSTRUCTOR
    TinyVectorIter(const TinyVectorIter<T_numtype, N_length, N_stride>& iter)
        : data_(iter.data_)
    {
    }
#endif

    T_numtype operator[](int i) const
    {
        BZPRECONDITION(i >= 0 && i < N_length);
        return data_[i * N_stride];
    }

    T_numtype& restrict operator[](int i)
    {
        BZPRECONDITION(i >= 0 && i < N_length);
        return data_[i * N_stride];
    }

    T_numtype operator()(int i) const
    {
        BZPRECONDITION(i >= 0 && i < N_length);
        return data_[i * N_stride];
    }

    T_numtype& restrict operator()(int i)
    {
        BZPRECONDITION(i >= 0 && i < N_length);
        return data_[i * N_stride];
    }

    int length(int) const
    { return N_length; }

    static const int _bz_staticLengthCount = 1,
                     _bz_dynamicLengthCount = 0,
                     _bz_staticLength = 0;

    bool _bz_hasFastAccess() const
    { return true; }

    T_numtype _bz_fastAccess(int i) const
    { return data_[i * N_stride]; }

    T_numtype& _bz_fastAccess(int i)
    { return data_[i * N_stride]; }

    int _bz_suggestLength() const
    { return N_length; }

private:
    T_numtype * restrict data_;
};

// N_stride has default 1, in forward declaration in <blitz/tinyvec.h>
template<typename P_numtype, int N_length, int N_stride>
class TinyVectorIterConst {
public:
    typedef P_numtype T_numtype;

    explicit TinyVectorIterConst(const TinyVector<T_numtype, N_length>& x)
        : data_(x.data())
    { }

#ifdef BZ_MANUAL_VECEXPR_COPY_CONSTRUCTOR
    TinyVectorIterConst(const TinyVectorIterConst<T_numtype, N_length, 
        N_stride>& iter)
        : data_(iter.data_)
    {
    }

    void operator=(const TinyVectorIterConst<T_numtype, N_length, N_stride>& 
        iter)
    {
        data_ = iter.data_;
    }
#endif

    T_numtype operator[](int i) const
    {
        BZPRECONDITION(i >= 0 && i < N_length);
        return data_[i * N_stride];
    }

    T_numtype operator()(int i) const
    {
        BZPRECONDITION(i >= 0 && i < N_length);
        return data_[i * N_stride];
    }

    int length(int) const
    { return N_length; }

    static const int _bz_staticLengthCount = 1,
                     _bz_dynamicLengthCount = 0,
                     _bz_staticLength = 0;

    bool _bz_hasFastAccess() const
    { return true; }

    T_numtype _bz_fastAccess(int i) const
    { return data_[i * N_stride]; }

    int _bz_suggestLength() const
    { return N_length; }

private:
    const T_numtype * restrict data_;
};

BZ_NAMESPACE_END

#endif // BZ_TINYVECITER_H
