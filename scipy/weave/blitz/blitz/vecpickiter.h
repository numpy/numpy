// -*- C++ -*-
/***************************************************************************
 * blitz/vecpickiter.h      Declaration of VectorPickIter<T_numtype> and
 *                          VectorPickIterConst<T_numtype> classes
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

#ifndef BZ_VECPICKITER_H
#define BZ_VECPICKITER_H

#ifndef BZ_VECPICK_H
 #include <blitz/vecpick.h>
#endif

BZ_NAMESPACE(blitz)

template<typename P_numtype>
class VectorPickIter {

public:
    typedef P_numtype  T_numtype;

    explicit VectorPickIter(VectorPick<T_numtype>& x)
        : data_(x.vector().data()), index_(x.indexSet().data())
    {
        dataStride_  = x.vector().stride();
        indexStride_ = x.indexSet().stride();
        length_ = x.indexSet().length();
    }

#ifdef BZ_MANUAL_VECEXPR_COPY_CONSTRUCTOR
    VectorPickIter(const VectorPickIter<T_numtype>& x)
    {
        data_ = x.data_;
        index_ = x.index_;
        dataStride_ = x.dataStride_;
        indexStride_ = x.indexStride_;
        length_ = x.length_;
    }
#endif

    T_numtype operator[](int i) const
    {
        BZPRECONDITION(i < length_);
        return data_[dataStride_ * index_[i * indexStride_]];
    }

    T_numtype& operator[](int i)
    {
        BZPRECONDITION(i < length_);
        return data_[dataStride_ * index_[i * indexStride_]];
    }

    int length(int) const
    { return length_; }

    int _bz_suggestLength() const
    { return length_; }

    bool isUnitStride() const
    { return (dataStride_  == 1) && (indexStride_ == 1); }

    bool _bz_hasFastAccess() const
    { return isUnitStride(); }

    T_numtype _bz_fastAccess(int i) const
    {    
         return data_[index_[i]];
    }

    T_numtype&  _bz_fastAccess(int i)
    {
         return data_[index_[i]];
    }

    static const int
        _bz_staticLengthCount = 0,
        _bz_dynamicLengthCount = 1,
        _bz_staticLength = 0;

private:
    T_numtype * restrict data_;
    int dataStride_;
    const int * restrict index_;
    int indexStride_;
    int length_;
};

template<typename P_numtype>
class VectorPickIterConst {

public:
    typedef P_numtype  T_numtype;

    explicit VectorPickIterConst(const VectorPick<T_numtype>& x)
        : data_(x.vector().data()), index_(x.indexSet().data())
    {
        dataStride_  = x.vector().stride();
        indexStride_ = x.indexSet().stride();
        length_ = x.indexSet().length();
    }

#ifdef BZ_MANUAL_VECEXPR_COPY_CONSTRUCTOR
    VectorPickIterConst(const VectorPickIterConst<T_numtype>& x)
    {
        data_ = x.data_;
        index_ = x.index_;
        dataStride_ = x.dataStride_;
        indexStride_ = x.indexStride_;
        length_ = x.length_;
    }
#endif

    T_numtype operator[](int i) const
    {
        BZPRECONDITION(i < length_);
        return data_[dataStride_ * index_[i * indexStride_]];
    }

    int length(int) const
    { return length_; }

    int _bz_suggestLength() const
    { return length_; }

    bool isUnitStride() const
    { return (dataStride_  == 1) && (indexStride_ == 1); }

    bool _bz_hasFastAccess() const
    { return isUnitStride(); }

    T_numtype _bz_fastAccess(int i) const
    {
         return data_[index_[i]];
    }

    static const int 
        _bz_staticLengthCount = 0,
        _bz_dynamicLengthCount = 1,
        _bz_staticLength = 0;

private:
    const T_numtype * restrict data_;
    int dataStride_;
    const int * restrict index_;
    int indexStride_;
    int length_;
};

BZ_NAMESPACE_END

#endif // BZ_VECPICKITER_H

