/***************************************************************************
 * blitz/vector.h      Declaration of the Vector<P_numtype> class
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
 * Revision 1.1.1.1  2000/06/19 12:26:10  tveldhui
 * Imported sources
 *
 * Revision 1.8  1998/03/14 00:04:47  tveldhui
 * 0.2-alpha-05
 *
 * Revision 1.7  1997/07/16 14:51:20  tveldhui
 * Update: Alpha release 0.2 (Arrays)
 *
 * Revision 1.6  1997/01/24 14:42:00  tveldhui
 * Periodic RCS update
 *
 * Revision 1.5  1997/01/23 03:28:28  tveldhui
 * Periodic RCS update
 *
 * Revision 1.4  1997/01/13 22:19:58  tveldhui
 * Periodic RCS update
 *
 * Revision 1.3  1996/11/11 17:29:13  tveldhui
 * Periodic RCS update
 *
 * Revision 1.2  1996/10/31 21:06:54  tveldhui
 * Did away with multiple template parameters.
 */

/*
 * KNOWN BUGS
 *
 * 1. operator[](Vector<int>) won't match; compiler complains of no
 *       suitable copy constructor for VectorPick<T>
 * 2. Vector<T>(_bz_VecExpr<E>) constructor generates warning
 * 3. operator+=,-=,..etc.(Random<D>) won't match; compiler complains of
 *       no suitable copy constructor for _bz_VecExpr(...).
 */

#ifndef BZ_VECTOR_H
#define BZ_VECTOR_H

#ifndef BZ_BLITZ_H
 #include <blitz/blitz.h>
#endif

#ifndef BZ_MEMBLOCK_H
 #include <blitz/memblock.h>
#endif

#ifndef BZ_RANGE_H
 #include <blitz/range.h>
#endif

#ifndef BZ_LISTINIT_H
 #include <blitz/listinit.h>
#endif

BZ_NAMESPACE(blitz)

// Forward declarations
template<class P_numtype> class VectorIter;
template<class P_numtype> class VectorIterConst;
template<class P_expr>    class _bz_VecExpr;       
template<class P_numtype> class VectorPick;
template<class P_numtype> class Random;

// Declaration of class Vector<P_numtype>

template<class P_numtype>
class Vector : protected MemoryBlockReference<P_numtype> {

public:
    //////////////////////////////////////////////
    // Public Types
    //////////////////////////////////////////////

    typedef MemoryBlockReference<P_numtype> T_Base;
    typedef P_numtype                  T_numtype;
    typedef Vector<T_numtype>          T_vector;
    typedef VectorIter<T_numtype>      T_iterator;
    typedef VectorIterConst<T_numtype> T_constIterator;
    typedef VectorPick<T_numtype>      T_pick;
    typedef Vector<int>                T_indexVector;

    //////////////////////////////////////////////
    // Constructors                             //
    //////////////////////////////////////////////
    // Most of the constructors are inlined so that
    // the setting of the stride_ data member will
    // be visible to the optimizer.

    Vector()
    { 
        length_ = 0;
        stride_ = 0;
    }

    // This constructor is provided inline because it involves
    // no memory allocation.
    Vector(Vector<T_numtype>& vec)
        : MemoryBlockReference<T_numtype>(vec)
    {
        length_ = vec.length_;
        stride_ = vec.stride_;
    }

    // Slightly unsafe cast-away-const version
    Vector(const Vector<T_numtype>& vec)
        : MemoryBlockReference<T_numtype>
           (const_cast<Vector<T_numtype>& >(vec))
    {
        length_ = vec.length_;
        stride_ = vec.stride_;
    }

    _bz_explicit Vector(int length)
        : MemoryBlockReference<T_numtype>(length)
    {
        length_ = length;
        stride_ = 1;
    }

    Vector(Vector<T_numtype>& vec, Range r)
        : MemoryBlockReference<T_numtype>(vec, 
            r.first() * vec.stride())
    {
        BZPRECONDITION((r.first() >= 0) && (r.first() < vec.length()));
        BZPRECONDITION((r.last(vec.length()-1) >= 0) 
            && (r.last(vec.length()-1) < vec.length()));
        length_ = (r.last(vec.length()-1) - r.first()) / r.stride() + 1;
        stride_ = vec.stride_ * r.stride();
    }

    Vector(int length, T_numtype initValue)
        : MemoryBlockReference<T_numtype>(length)
    {
        length_ = length;
        stride_ = 1;
        (*this) = initValue;
    }

    Vector(int length, T_numtype firstValue, T_numtype delta)
        : MemoryBlockReference<T_numtype>(length)
    {
        length_ = length;
        stride_ = 1;
        for (int i=0; i < length; ++i)
            data_[i] = firstValue + i * delta;
    }

    template<class P_distribution>
    Vector(int length, Random<P_distribution>& random)
        : MemoryBlockReference<T_numtype>(length)
    {
        length_ = length;
        stride_ = 1;
        (*this) = random;
    }

    template<class P_expr>
    Vector(_bz_VecExpr<P_expr> expr)
        : MemoryBlockReference<T_numtype>(expr._bz_suggestLength())
    {
        length_ = expr._bz_suggestLength();
        stride_ = 1;
        (*this) = expr;
    }

    // Create a vector view of an already allocated block of memory.
    // Note that the memory will not be freed when this vector is
    // destroyed.
    Vector(int length, T_numtype* _bz_restrict data, int stride = 1)
        : MemoryBlockReference<T_numtype>(length, data, neverDeleteData)
    {
        length_ = length;
        stride_ = stride;
    }

    // Create a vector containing a range of numbers
    Vector(Range r)
        : MemoryBlockReference<T_numtype>(r._bz_suggestLength())
    {
        length_ = r._bz_suggestLength();
        stride_ = 1;
        (*this) = _bz_VecExpr<Range>(r);
    }
    
    //////////////////////////////////////////////
    // Member functions
    //////////////////////////////////////////////

    // assertUnitStride() is provided as an optimizing trick.  When
    // vectors are constructed outside the function scope, the optimizer
    // is unaware that they have unit stride.  This function sets the
    // stride to 1 in the local scope so the optimizer can do copy
    // propagation & dead code elimination.  Obviously, you don't
    // want to use this routine unless you are certain that the
    // vectors have unit stride.
    void            assertUnitStride()
    {
        BZPRECONDITION(stride_ == 1);
        stride_ = 1;
    }

    T_iterator      begin()
    { return T_iterator(*this); }

    T_constIterator begin()  const
    { return T_constIterator(*this); }

    T_vector        copy()   const;

    // T_iterator      end();
    // T_constIterator end()    const;

    T_numtype * _bz_restrict data()  
    { return data_; }

    const T_numtype * _bz_restrict data() const
    { return data_; }

    _bz_bool        isUnitStride() const
    { return stride_ == 1; }

    int        length() const
    { return length_; }

    void            makeUnique();

    // int        storageSize() const;

    // void            storeToBuffer(void* buffer, int bufferLength) const;

    void            reference(T_vector&);

    void            resize(int length);

    void            resizeAndPreserve(int newLength);

    // int             restoreFromBuffer(void* buffer, int bufferLength);

    T_vector        reverse()
    { return T_vector(*this,Range(length()-1,0,-1)); }

    int             stride() const
    { return stride_; }

    operator _bz_VecExpr<VectorIterConst<T_numtype> > () const
    { return _bz_VecExpr<VectorIterConst<T_numtype> >(begin()); }

    /////////////////////////////////////////////
    // Library-internal member functions
    // These are undocumented and may change or
    // disappear in future releases.
    /////////////////////////////////////////////

    int        _bz_suggestLength() const
    { return length_; }

    _bz_bool        _bz_hasFastAccess() const
    { return stride_ == 1; }

    T_numtype&      _bz_fastAccess(int i)
    { return data_[i]; }

    T_numtype       _bz_fastAccess(int i) const
    { return data_[i]; }

    template<class P_expr, class P_updater>
    void            _bz_assign(P_expr, P_updater);

    _bz_VecExpr<T_constIterator> _bz_asVecExpr() const
    { return _bz_VecExpr<T_constIterator>(begin()); }

    //////////////////////////////////////////////
    // Subscripting operators
    //////////////////////////////////////////////

    // operator()(int) may be used only when the vector has unit
    // stride.  Otherwise, use operator[].
    T_numtype        operator()(int i) const
    {
        BZPRECONDITION(i < length_);
        BZPRECONDITION(stride_ == 1);
        return data_[i];
    }

    // operator()(int) may be used only when the vector has unit
    // stride.  Otherwise, use operator[].
    T_numtype& _bz_restrict operator()(int i) 
    {
        BZPRECONDITION(i < length_);
        BZPRECONDITION(stride_ == 1);
        return data_[i];
    }

    T_numtype        operator[](int i) const
    {
        BZPRECONDITION(i < length_);
        return data_[i * stride_];
    }

    T_numtype& _bz_restrict operator[](int i)
    {
        BZPRECONDITION(i < length_);
        return data_[i * stride_];
    }

    T_vector      operator()(Range r)
    {
        return T_vector(*this, r);
    }

    T_vector      operator[](Range r)
    {
        return T_vector(*this, r);
    }

    T_pick        operator()(T_indexVector i)
    {
        return T_pick(*this, i);
    }

    T_pick        operator[](T_indexVector i)
    {
        return T_pick(*this, i);
    }

    // T_vector      operator()(difference-equation-expression)

    //////////////////////////////////////////////
    // Assignment operators
    //////////////////////////////////////////////

    // Scalar operand
    ListInitializationSwitch<T_vector,T_iterator> operator=(T_numtype x)
    {
        return ListInitializationSwitch<T_vector,T_iterator>(*this, x);
    }

    T_iterator getInitializationIterator()
    { return begin(); }

    T_vector& initialize(T_numtype);
    T_vector& operator+=(T_numtype);
    T_vector& operator-=(T_numtype);
    T_vector& operator*=(T_numtype);
    T_vector& operator/=(T_numtype);
    T_vector& operator%=(T_numtype);
    T_vector& operator^=(T_numtype);
    T_vector& operator&=(T_numtype);
    T_vector& operator|=(T_numtype);
    T_vector& operator>>=(int);
    T_vector& operator<<=(int); 

    // Vector operand
   
    template<class P_numtype2> T_vector& operator=(const Vector<P_numtype2> &);

    // Specialization uses memcpy instead of element-by-element cast and
    // copy
    // NEEDS_WORK -- KCC won't accept this syntax; standard??
    // template<> T_vector& operator=(const T_vector&);

    template<class P_numtype2> T_vector& operator+=(const Vector<P_numtype2> &);
    template<class P_numtype2> T_vector& operator-=(const Vector<P_numtype2> &);
    template<class P_numtype2> T_vector& operator*=(const Vector<P_numtype2> &);
    template<class P_numtype2> T_vector& operator/=(const Vector<P_numtype2> &);
    template<class P_numtype2> T_vector& operator%=(const Vector<P_numtype2> &);
    template<class P_numtype2> T_vector& operator^=(const Vector<P_numtype2> &);
    template<class P_numtype2> T_vector& operator&=(const Vector<P_numtype2> &);
    template<class P_numtype2> T_vector& operator|=(const Vector<P_numtype2> &);
    template<class P_numtype2> T_vector& operator>>=(const Vector<P_numtype2> &);
    template<class P_numtype2> T_vector& operator<<=(const Vector<P_numtype2> &);

    // Vector expression operand
    template<class P_expr> T_vector& operator=(_bz_VecExpr<P_expr>);
    template<class P_expr> T_vector& operator+=(_bz_VecExpr<P_expr>); 
    template<class P_expr> T_vector& operator-=(_bz_VecExpr<P_expr>);
    template<class P_expr> T_vector& operator*=(_bz_VecExpr<P_expr>);
    template<class P_expr> T_vector& operator/=(_bz_VecExpr<P_expr>);
    template<class P_expr> T_vector& operator%=(_bz_VecExpr<P_expr>);
    template<class P_expr> T_vector& operator^=(_bz_VecExpr<P_expr>);
    template<class P_expr> T_vector& operator&=(_bz_VecExpr<P_expr>);
    template<class P_expr> T_vector& operator|=(_bz_VecExpr<P_expr>);
    template<class P_expr> T_vector& operator>>=(_bz_VecExpr<P_expr>);
    template<class P_expr> T_vector& operator<<=(_bz_VecExpr<P_expr>);
    
    // VectorPick operand
    template<class P_numtype2> 
    T_vector& operator=(const VectorPick<P_numtype2> &);
    template<class P_numtype2> 
    T_vector& operator+=(const VectorPick<P_numtype2> &);
    template<class P_numtype2> 
    T_vector& operator-=(const VectorPick<P_numtype2> &);
    template<class P_numtype2> 
    T_vector& operator*=(const VectorPick<P_numtype2> &);
    template<class P_numtype2> 
    T_vector& operator/=(const VectorPick<P_numtype2> &);
    template<class P_numtype2>
    T_vector& operator%=(const VectorPick<P_numtype2> &);
    template<class P_numtype2>
    T_vector& operator^=(const VectorPick<P_numtype2> &);
    template<class P_numtype2>
    T_vector& operator&=(const VectorPick<P_numtype2> &);
    template<class P_numtype2>
    T_vector& operator|=(const VectorPick<P_numtype2> &);
    template<class P_numtype2>
    T_vector& operator>>=(const VectorPick<P_numtype2> &);
    template<class P_numtype2>
    T_vector& operator<<=(const VectorPick<P_numtype2> &);

    // Range operand
    T_vector& operator=(Range);
    T_vector& operator+=(Range);
    T_vector& operator-=(Range);
    T_vector& operator*=(Range);
    T_vector& operator/=(Range);
    T_vector& operator%=(Range);
    T_vector& operator^=(Range);
    T_vector& operator&=(Range);
    T_vector& operator|=(Range);
    T_vector& operator>>=(Range);
    T_vector& operator<<=(Range);

    // Random operand
    template<class P_distribution>
    T_vector& operator=(Random<P_distribution>& random);
    template<class P_distribution>
    T_vector& operator+=(Random<P_distribution>& random);
    template<class P_distribution>
    T_vector& operator-=(Random<P_distribution>& random);
    template<class P_distribution>
    T_vector& operator*=(Random<P_distribution>& random);
    template<class P_distribution>
    T_vector& operator/=(Random<P_distribution>& random);
    template<class P_distribution>
    T_vector& operator%=(Random<P_distribution>& random);
    template<class P_distribution>
    T_vector& operator^=(Random<P_distribution>& random);
    template<class P_distribution>
    T_vector& operator&=(Random<P_distribution>& random);
    template<class P_distribution>
    T_vector& operator|=(Random<P_distribution>& random);

    //////////////////////////////////////////////
    // Unary operators
    //////////////////////////////////////////////

//    T_vector& operator++();
//    void operator++(int);
//    T_vector& operator--();
//    void operator--(int);
    
    using    T_Base::data_;

private:
    int      length_;
    int      stride_;
};

// Global I/O functions

template<class P_numtype>
ostream& operator<<(ostream& os, const Vector<P_numtype>& x);

template<class P_expr>
ostream& operator<<(ostream& os, _bz_VecExpr<P_expr> expr);

BZ_NAMESPACE_END

#include <blitz/veciter.h>          // Iterators
#include <blitz/vecpick.h>          // VectorPick
#include <blitz/vecexpr.h>          // Expression templates
#include <blitz/vecglobs.h>         // Global functions
#include <blitz/vector.cc>          // Member functions
#include <blitz/vecio.cc>           // IO functions

#endif // BZ_VECTOR_H
