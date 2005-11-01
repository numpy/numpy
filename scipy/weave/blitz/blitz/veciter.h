// -*- C++ -*-
/***************************************************************************
 * blitz/veciter.h      Iterator classes for Vector<P_numtype>
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


#ifndef BZ_VECITER_H
#define BZ_VECITER_H

#ifndef BZ_VECTOR_H
 #error <blitz/veciter.h> should be included via <blitz/vector.h>
#endif

BZ_NAMESPACE(blitz)

// Declaration of class VectorIter
template<typename P_numtype>
class VectorIter {
public:
    typedef P_numtype T_numtype;

    explicit VectorIter(Vector<P_numtype>& x)
        : data_(x.data())
    {
        stride_ = x.stride();
        length_ = x.length();
    }

    VectorIter(P_numtype* restrict data, int stride, int length)
        : data_(data), stride_(stride), length_(length)
    { }

#ifdef BZ_MANUAL_VECEXPR_COPY_CONSTRUCTOR
    VectorIter(const VectorIter<P_numtype>& x)
    {
        data_ = x.data_;
        stride_ = x.stride_;
        length_ = x.length_;
    }
#endif

    P_numtype operator[](int i) const
    { 
        BZPRECONDITION(i < length_);
        return data_[i*stride_]; 
    }

    P_numtype& restrict operator[](int i)
    { 
        BZPRECONDITION(i < length_);
        return data_[i*stride_]; 
    }

    P_numtype operator()(int i) const
    {
        BZPRECONDITION(i < length_);
        return data_[i*stride_];
    }

    P_numtype& restrict operator()(int i) 
    {
        BZPRECONDITION(i < length_);
        return data_[i*stride_];
    }

    P_numtype operator*() const
    { return *data_; }

    P_numtype& operator*()
    { return *data_; }

    VectorIter<P_numtype> operator+(int i)
    {
        // NEEDS_WORK -- precondition checking?
        return VectorIter<P_numtype>(data_+i*stride_, stride_, length_-i);
    }

    int length(int) const
    { return length_; }

    bool isUnitStride() const
    { return (stride_ == 1); }

    /////////////////////////////////////////////
    // Library-internal member functions
    // These are undocumented and may change or
    // disappear in future releases.
    /////////////////////////////////////////////

    static const int 
        _bz_staticLengthCount = 0,
        _bz_dynamicLengthCount = 1,
        _bz_staticLength = 0;

    bool _bz_hasFastAccess() const
    { return isUnitStride(); }

    P_numtype _bz_fastAccess(int i) const
    { return data_[i]; }

    P_numtype& restrict _bz_fastAccess(int i)
    { return data_[i]; }

    int _bz_suggestLength() const
    { return length_; }

private:
    VectorIter() { }
    P_numtype * restrict data_;
    int stride_;
    int length_;
};


template<typename P_numtype>
class VectorIterConst {
public:
    typedef P_numtype T_numtype;

    explicit VectorIterConst(const Vector<P_numtype>& x)
        : data_(x.data())
    {
        stride_ = x.stride();
        length_ = x.length();
    }

#ifdef BZ_MANUAL_VECEXPR_COPY_CONSTRUCTOR
    VectorIterConst(const VectorIterConst<P_numtype>& x)
    {
        data_ = x.data_;
        stride_ = x.stride_;
        length_ = x.length_;
    }
#endif

    P_numtype operator[](int i) const
    { 
        BZPRECONDITION(i < length_);
        return data_[i*stride_]; 
    }

    P_numtype operator()(int i) const
    {
        BZPRECONDITION(i < length_);
        return data_[i*stride_];
    }

    int length(int) const
    { return length_; }

    bool isUnitStride() const
    { return (stride_ == 1); }

    /////////////////////////////////////////////
    // Library-internal member functions
    // These are undocumented and may change or
    // disappear in future releases.
    /////////////////////////////////////////////

    static const int 
        _bz_staticLengthCount = 0,
        _bz_dynamicLengthCount = 1,
        _bz_staticLength = 0;

    bool  _bz_hasFastAccess() const
    { return isUnitStride(); }

    P_numtype _bz_fastAccess(int i) const
    {
        return data_[i];
    }

    int _bz_suggestLength() const
    { return length_; }

private:
    const P_numtype * restrict data_;
    int stride_;
    int length_;
};

BZ_NAMESPACE_END

#endif // BZ_VECITER_H
