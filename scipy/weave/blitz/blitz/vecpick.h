/***************************************************************************
 * blitz/vecpick.h      Declaration of the VectorPick<T_numtype> class
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

#ifndef BZ_VECPICK_H
#define BZ_VECPICK_H

#include <blitz/vector.h>

BZ_NAMESPACE(blitz)

// Forward declarations

template<typename P_numtype> class VectorPickIter;
template<typename P_numtype> class VectorPickIterConst;

// Declaration of class VectorPick<P_numtype>

template<typename P_numtype>
class VectorPick {

public:
    //////////////////////////////////////////////
    // Public Types
    //////////////////////////////////////////////

    typedef P_numtype                      T_numtype;
    typedef Vector<T_numtype>              T_vector;
    typedef Vector<int>                    T_indexVector;
    typedef VectorPick<T_numtype>          T_pick;
    typedef VectorPickIter<T_numtype>      T_iterator;
    typedef VectorPickIterConst<T_numtype> T_constIterator;

    //////////////////////////////////////////////
    // Constructors                             //
    //////////////////////////////////////////////

    VectorPick(T_vector& vector, T_indexVector& indexarg)
        : vector_(vector), index_(indexarg)
    { }

    VectorPick(const T_pick& vecpick)
        : vector_(const_cast<T_vector&>(vecpick.vector_)), 
          index_(const_cast<T_indexVector&>(vecpick.index_))
    { }

    VectorPick(T_pick& vecpick, Range r)
        : vector_(vecpick.vector_), index_(vecpick.index_[r])
    { }
 
    //////////////////////////////////////////////
    // Member functions
    //////////////////////////////////////////////

    T_iterator         beginFast()
    { return VectorPickIter<T_numtype>(*this); }

    T_constIterator    beginFast()      const
    { return VectorPickIterConst<T_numtype>(*this); }

    // T_vector           copy()       const;

    // T_iterator         end();

    // T_constIterator    end()        const;

    T_indexVector&     indexSet()
    { return index_; }
 
    const T_indexVector& indexSet()      const
    { return index_; }

    int           length()     const
    { return index_.length(); }

    void               setVector(Vector<T_numtype>& x)
    { vector_.reference(x); }

    void               setIndex(Vector<int>& index)
    { index_.reference(index); }

    T_vector&          vector()
    { return vector_; }

    const T_vector&    vector()     const
    { return vector_; }

    /////////////////////////////////////////////
    // Library-internal member functions
    // These are undocumented and may change or
    // disappear in future releases.
    /////////////////////////////////////////////

    int        _bz_suggestLength() const
    { return index_.length(); }

    bool        _bz_hasFastAccess() const
    { return vector_._bz_hasFastAccess() && index_._bz_hasFastAccess(); }

    T_numtype&      _bz_fastAccess(int i)
    { return vector_._bz_fastAccess(index_._bz_fastAccess(i)); }

    T_numtype       _bz_fastAccess(int i) const
    { return vector_._bz_fastAccess(index_._bz_fastAccess(i)); }

    _bz_VecExpr<T_constIterator> _bz_asVecExpr() const
    { return _bz_VecExpr<T_constIterator>(beginFast()); }

    //////////////////////////////////////////////
    // Subscripting operators
    //////////////////////////////////////////////

    T_numtype       operator()(int i) const
    { 
        BZPRECONDITION(index_.stride() == 1);
        BZPRECONDITION(vector_.stride() == 1);
        BZPRECONDITION(i < index_.length());
        BZPRECONDITION(index_[i] < vector_.length());
        return vector_(index_(i));
    }

    T_numtype&      operator()(int i)
    {
        BZPRECONDITION(index_.stride() == 1);
        BZPRECONDITION(vector_.stride() == 1);
        BZPRECONDITION(i < index_.length());
        BZPRECONDITION(index_[i] < vector_.length());
        return vector_(index_(i));
    }

    T_numtype       operator[](int i) const
    {
        BZPRECONDITION(index_.stride() == 1);
        BZPRECONDITION(vector_.stride() == 1);
        BZPRECONDITION(i < index_.length());
        BZPRECONDITION(index_[i] < vector_.length());
        return vector_[index_[i]];
    }

    T_numtype&      operator[](int i)
    {
        BZPRECONDITION(index_.stride() == 1);
        BZPRECONDITION(vector_.stride() == 1);
        BZPRECONDITION(i < index_.length());
        BZPRECONDITION(index_[i] < vector_.length());
        return vector_[index_[i]];
    }

    T_pick          operator()(Range r)
    {
        return T_pick(*this, index_[r]);
    }

    T_pick          operator[](Range r)
    {
        return T_pick(*this, index_[r]);
    }

    //////////////////////////////////////////////
    // Assignment operators
    //////////////////////////////////////////////

    // Scalar operand
    T_pick& operator=(T_numtype);
    T_pick& operator+=(T_numtype);
    T_pick& operator-=(T_numtype);
    T_pick& operator*=(T_numtype);
    T_pick& operator/=(T_numtype);
    T_pick& operator%=(T_numtype);
    T_pick& operator^=(T_numtype);
    T_pick& operator&=(T_numtype);
    T_pick& operator|=(T_numtype);
    T_pick& operator>>=(int);
    T_pick& operator<<=(int);

    // Vector operand
    template<typename P_numtype2> T_pick& operator=(const Vector<P_numtype2> &);
    template<typename P_numtype2> T_pick& operator+=(const Vector<P_numtype2> &);
    template<typename P_numtype2> T_pick& operator-=(const Vector<P_numtype2> &);
    template<typename P_numtype2> T_pick& operator*=(const Vector<P_numtype2> &);
    template<typename P_numtype2> T_pick& operator/=(const Vector<P_numtype2> &);
    template<typename P_numtype2> T_pick& operator%=(const Vector<P_numtype2> &);
    template<typename P_numtype2> T_pick& operator^=(const Vector<P_numtype2> &);
    template<typename P_numtype2> T_pick& operator&=(const Vector<P_numtype2> &);
    template<typename P_numtype2> T_pick& operator|=(const Vector<P_numtype2> &);
    template<typename P_numtype2> T_pick& operator>>=(const Vector<P_numtype2> &);
    template<typename P_numtype2> T_pick& operator<<=(const Vector<P_numtype2> &);

    // Vector expression operand
    template<typename P_expr> T_pick& operator=(_bz_VecExpr<P_expr>);
    template<typename P_expr> T_pick& operator+=(_bz_VecExpr<P_expr>);
    template<typename P_expr> T_pick& operator-=(_bz_VecExpr<P_expr>);
    template<typename P_expr> T_pick& operator*=(_bz_VecExpr<P_expr>);
    template<typename P_expr> T_pick& operator/=(_bz_VecExpr<P_expr>);
    template<typename P_expr> T_pick& operator%=(_bz_VecExpr<P_expr>);
    template<typename P_expr> T_pick& operator^=(_bz_VecExpr<P_expr>);
    template<typename P_expr> T_pick& operator&=(_bz_VecExpr<P_expr>);
    template<typename P_expr> T_pick& operator|=(_bz_VecExpr<P_expr>);
    template<typename P_expr> T_pick& operator>>=(_bz_VecExpr<P_expr>);
    template<typename P_expr> T_pick& operator<<=(_bz_VecExpr<P_expr>);

    // Range operand
    T_pick& operator=(Range);
    T_pick& operator+=(Range);
    T_pick& operator-=(Range);
    T_pick& operator*=(Range);
    T_pick& operator/=(Range);
    T_pick& operator%=(Range);
    T_pick& operator^=(Range);
    T_pick& operator&=(Range);
    T_pick& operator|=(Range);
    T_pick& operator>>=(Range);
    T_pick& operator<<=(Range);

    // Vector pick operand
    template<typename P_numtype2> 
    T_pick& operator=(const VectorPick<P_numtype2> &);
    template<typename P_numtype2> 
    T_pick& operator+=(const VectorPick<P_numtype2> &);
    template<typename P_numtype2> 
    T_pick& operator-=(const VectorPick<P_numtype2> &);
    template<typename P_numtype2> 
    T_pick& operator*=(const VectorPick<P_numtype2> &);
    template<typename P_numtype2> 
    T_pick& operator/=(const VectorPick<P_numtype2> &);
    template<typename P_numtype2> 
    T_pick& operator%=(const VectorPick<P_numtype2> &);
    template<typename P_numtype2> 
    T_pick& operator^=(const VectorPick<P_numtype2> &);
    template<typename P_numtype2> 
    T_pick& operator&=(const VectorPick<P_numtype2> &);
    template<typename P_numtype2> 
    T_pick& operator|=(const VectorPick<P_numtype2> &);
    template<typename P_numtype2> 
    T_pick& operator>>=(const VectorPick<P_numtype2> &);
    template<typename P_numtype2> 
    T_pick& operator<<=(const VectorPick<P_numtype2> &);

    // Random operand
    template<typename P_distribution>
    T_pick& operator=(Random<P_distribution>& random);
    template<typename P_distribution>
    T_pick& operator+=(Random<P_distribution>& random);
    template<typename P_distribution>
    T_pick& operator-=(Random<P_distribution>& random);
    template<typename P_distribution>
    T_pick& operator*=(Random<P_distribution>& random);
    template<typename P_distribution>
    T_pick& operator/=(Random<P_distribution>& random);
    template<typename P_distribution>
    T_pick& operator%=(Random<P_distribution>& random);
    template<typename P_distribution>
    T_pick& operator^=(Random<P_distribution>& random);
    template<typename P_distribution>
    T_pick& operator&=(Random<P_distribution>& random);
    template<typename P_distribution>
    T_pick& operator|=(Random<P_distribution>& random);

private:
    VectorPick() { }

    template<typename P_expr, typename P_updater>
    inline void _bz_assign(P_expr, P_updater);

private:
    T_vector vector_;
    T_indexVector index_;
};

BZ_NAMESPACE_END

#include <blitz/vecpick.cc>
#include <blitz/vecpickio.cc>
#include <blitz/vecpickiter.h>

#endif // BZ_VECPICK_H
