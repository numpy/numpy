/***************************************************************************
 * blitz/tinyveciter.h   Declaration of TinyVectorIter<T,N,stride>
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
 * Revision 1.1.1.1  2000/06/19 12:26:11  tveldhui
 * Imported sources
 *
 * Revision 1.2  1998/03/14 00:04:47  tveldhui
 * 0.2-alpha-05
 *
 * Revision 1.1  1997/07/16 14:51:20  tveldhui
 * Update: Alpha release 0.2 (Arrays)
 *
 */


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
template<class P_numtype, int N_length, int N_stride>
class TinyVectorIter {
public:
    typedef P_numtype T_numtype;

    _bz_explicit TinyVectorIter(TinyVector<T_numtype, N_length>& x)
        : data_(x.data())
    { }

#ifdef BZ_MANUAL_VECEXPR_COPY_CONSTRUCTOR
    TinyVectorIter(const TinyVectorIter<T_numtype, N_length, N_stride>& iter)
        : data_(iter.data_)
    {
    }
#endif

    T_numtype operator[](unsigned i) const
    {
        BZPRECONDITION(i < N_length);
        return data_[i * N_stride];
    }

    T_numtype& _bz_restrict operator[](unsigned i)
    {
        BZPRECONDITION(i < N_length);
        return data_[i * N_stride];
    }

    T_numtype operator()(unsigned i) const
    {
        BZPRECONDITION(i < N_length);
        return data_[i * N_stride];
    }

    T_numtype& _bz_restrict operator()(unsigned i)
    {
        BZPRECONDITION(i < N_length);
        return data_[i * N_stride];
    }

    unsigned length(unsigned) const
    { return N_length; }

    enum { _bz_staticLengthCount = 1,
           _bz_dynamicLengthCount = 0,
           _bz_staticLength = 0 };

    _bz_bool _bz_hasFastAccess() const
    { return _bz_true; }

    T_numtype _bz_fastAccess(unsigned i) const
    { return data_[i * N_stride]; }

    T_numtype& _bz_fastAccess(unsigned i)
    { return data_[i * N_stride]; }

    unsigned _bz_suggestLength() const
    { return N_length; }

private:
    T_numtype * _bz_restrict data_;
};

// N_stride has default 1, in forward declaration in <blitz/tinyvec.h>
template<class P_numtype, int N_length, int N_stride>
class TinyVectorIterConst {
public:
    typedef P_numtype T_numtype;

    _bz_explicit TinyVectorIterConst(const TinyVector<T_numtype, N_length>& x)
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

    T_numtype operator[](unsigned i) const
    {
        BZPRECONDITION(i < N_length);
        return data_[i * N_stride];
    }

    T_numtype operator()(unsigned i) const
    {
        BZPRECONDITION(i < N_length);
        return data_[i * N_stride];
    }

    unsigned length(unsigned) const
    { return N_length; }

    enum { _bz_staticLengthCount = 1,
           _bz_dynamicLengthCount = 0,
           _bz_staticLength = 0 };

    _bz_bool _bz_hasFastAccess() const
    { return _bz_true; }

    T_numtype _bz_fastAccess(unsigned i) const
    { return data_[i * N_stride]; }

    unsigned _bz_suggestLength() const
    { return N_length; }

private:
    const T_numtype * _bz_restrict data_;
};

BZ_NAMESPACE_END

#endif // BZ_TINYVECITER_H
