/***************************************************************************
 * blitz/vecpick.h      Declaration of the VectorPick<T_numtype> class
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

#ifndef BZ_VECPICK_H
#define BZ_VECPICK_H

// >>>HOMMEL990316
#ifndef BZ_BLITZ_H
 #include <blitz/blitz.h>
#endif

#ifndef BZ_VECTOR_H
 #include <blitz/vector.h>
#endif

BZ_NAMESPACE(blitz)

// Forward declarations

template<class P_numtype> class VectorPickIter;
template<class P_numtype> class VectorPickIterConst;

// Declaration of class VectorPick<P_numtype>

template<class P_numtype>
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

    T_iterator         begin()
    { return VectorPickIter<T_numtype>(*this); }

    T_constIterator    begin()      const
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

    _bz_bool        _bz_hasFastAccess() const
    { return vector_._bz_hasFastAccess() && index_._bz_hasFastAccess(); }

    T_numtype&      _bz_fastAccess(int i)
    { return vector_._bz_fastAccess(index_._bz_fastAccess(i)); }

    T_numtype       _bz_fastAccess(int i) const
    { return vector_._bz_fastAccess(index_._bz_fastAccess(i)); }

    _bz_VecExpr<T_constIterator> _bz_asVecExpr() const
    { return _bz_VecExpr<T_constIterator>(begin()); }

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
    template<class P_numtype2> T_pick& operator=(const Vector<P_numtype2> &);
    template<class P_numtype2> T_pick& operator+=(const Vector<P_numtype2> &);
    template<class P_numtype2> T_pick& operator-=(const Vector<P_numtype2> &);
    template<class P_numtype2> T_pick& operator*=(const Vector<P_numtype2> &);
    template<class P_numtype2> T_pick& operator/=(const Vector<P_numtype2> &);
    template<class P_numtype2> T_pick& operator%=(const Vector<P_numtype2> &);
    template<class P_numtype2> T_pick& operator^=(const Vector<P_numtype2> &);
    template<class P_numtype2> T_pick& operator&=(const Vector<P_numtype2> &);
    template<class P_numtype2> T_pick& operator|=(const Vector<P_numtype2> &);
    template<class P_numtype2> T_pick& operator>>=(const Vector<P_numtype2> &);
    template<class P_numtype2> T_pick& operator<<=(const Vector<P_numtype2> &);

    // Vector expression operand
    template<class P_expr> T_pick& operator=(_bz_VecExpr<P_expr>);
    template<class P_expr> T_pick& operator+=(_bz_VecExpr<P_expr>);
    template<class P_expr> T_pick& operator-=(_bz_VecExpr<P_expr>);
    template<class P_expr> T_pick& operator*=(_bz_VecExpr<P_expr>);
    template<class P_expr> T_pick& operator/=(_bz_VecExpr<P_expr>);
    template<class P_expr> T_pick& operator%=(_bz_VecExpr<P_expr>);
    template<class P_expr> T_pick& operator^=(_bz_VecExpr<P_expr>);
    template<class P_expr> T_pick& operator&=(_bz_VecExpr<P_expr>);
    template<class P_expr> T_pick& operator|=(_bz_VecExpr<P_expr>);
    template<class P_expr> T_pick& operator>>=(_bz_VecExpr<P_expr>);
    template<class P_expr> T_pick& operator<<=(_bz_VecExpr<P_expr>);

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
    template<class P_numtype2> 
    T_pick& operator=(const VectorPick<P_numtype2> &);
    template<class P_numtype2> 
    T_pick& operator+=(const VectorPick<P_numtype2> &);
    template<class P_numtype2> 
    T_pick& operator-=(const VectorPick<P_numtype2> &);
    template<class P_numtype2> 
    T_pick& operator*=(const VectorPick<P_numtype2> &);
    template<class P_numtype2> 
    T_pick& operator/=(const VectorPick<P_numtype2> &);
    template<class P_numtype2> 
    T_pick& operator%=(const VectorPick<P_numtype2> &);
    template<class P_numtype2> 
    T_pick& operator^=(const VectorPick<P_numtype2> &);
    template<class P_numtype2> 
    T_pick& operator&=(const VectorPick<P_numtype2> &);
    template<class P_numtype2> 
    T_pick& operator|=(const VectorPick<P_numtype2> &);
    template<class P_numtype2> 
    T_pick& operator>>=(const VectorPick<P_numtype2> &);
    template<class P_numtype2> 
    T_pick& operator<<=(const VectorPick<P_numtype2> &);

    // Random operand
    template<class P_distribution>
    T_pick& operator=(Random<P_distribution>& random);
    template<class P_distribution>
    T_pick& operator+=(Random<P_distribution>& random);
    template<class P_distribution>
    T_pick& operator-=(Random<P_distribution>& random);
    template<class P_distribution>
    T_pick& operator*=(Random<P_distribution>& random);
    template<class P_distribution>
    T_pick& operator/=(Random<P_distribution>& random);
    template<class P_distribution>
    T_pick& operator%=(Random<P_distribution>& random);
    template<class P_distribution>
    T_pick& operator^=(Random<P_distribution>& random);
    template<class P_distribution>
    T_pick& operator&=(Random<P_distribution>& random);
    template<class P_distribution>
    T_pick& operator|=(Random<P_distribution>& random);

private:
    VectorPick() { }

    template<class P_expr, class P_updater>
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
