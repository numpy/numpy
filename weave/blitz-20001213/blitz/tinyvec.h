/***************************************************************************
 * blitz/tinyvec.h      Declaration of the TinyVector<T, N> class
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

#ifndef BZ_TINYVEC_H
#define BZ_TINYVEC_H

#ifndef BZ_BLITZ_H
 #include <blitz/blitz.h>
#endif

#ifndef BZ_RANGE_H
 #include <blitz/range.h>
#endif

#ifndef BZ_LISTINIT_H
 #include <blitz/listinit.h>
#endif

#include <blitz/tiny.h>

BZ_NAMESPACE(blitz)

/*****************************************************************************
 * Forward declarations
 */

template<class P_numtype, int N_length, int N_stride BZ_TEMPLATE_DEFAULT(1) >
class TinyVectorIter;

template<class P_numtype, int N_length, int N_stride BZ_TEMPLATE_DEFAULT(1) >
class TinyVectorIterConst;

template<class P_numtype>
class Vector;

template<class P_expr>
class _bz_VecExpr;

template<class P_distribution>
class Random;

template<class P_numtype>
class VectorPick;

template<class T_numtype1, class T_numtype2, int N_rows, int N_columns,
    int N_vecStride>
class _bz_matrixVectorProduct;



/*****************************************************************************
 * Declaration of class TinyVector
 */

template<class P_numtype, int N_length>
class TinyVector {

public:
    //////////////////////////////////////////////
    // Public Types
    //////////////////////////////////////////////

    typedef P_numtype                                    T_numtype;
    typedef TinyVector<T_numtype,N_length>               T_vector;
    typedef TinyVectorIter<T_numtype,N_length,1>         T_iterator;
    typedef TinyVectorIterConst<T_numtype,N_length,1>    T_constIterator;
    typedef T_iterator iterator;
    typedef T_constIterator const_iterator;
    enum { numElements = N_length };

    TinyVector()
    { }

    ~TinyVector() 
    { }

    inline TinyVector(const TinyVector<P_numtype,N_length>& x);

    inline TinyVector(T_numtype initValue);

    TinyVector(T_numtype x0, T_numtype x1)
    {
        data_[0] = x0;
        data_[1] = x1;
    }

    TinyVector(T_numtype x0, T_numtype x1, T_numtype x2)
    {
        data_[0] = x0;
        data_[1] = x1;
        data_[2] = x2;
    }

    TinyVector(T_numtype x0, T_numtype x1, T_numtype x2,
        T_numtype x3)
    {
        data_[0] = x0;
        data_[1] = x1;
        data_[2] = x2;
        data_[3] = x3;
    }

    TinyVector(T_numtype x0, T_numtype x1, T_numtype x2,
        T_numtype x3, T_numtype x4)
    {
        data_[0] = x0;
        data_[1] = x1;
        data_[2] = x2;
        data_[3] = x3;
        data_[4] = x4;
    }

    TinyVector(T_numtype x0, T_numtype x1, T_numtype x2,
        T_numtype x3, T_numtype x4, T_numtype x5)
    {
        data_[0] = x0;
        data_[1] = x1;
        data_[2] = x2;
        data_[3] = x3;
        data_[4] = x4;
        data_[5] = x5;
    }

    TinyVector(T_numtype x0, T_numtype x1, T_numtype x2,
        T_numtype x3, T_numtype x4, T_numtype x5, T_numtype x6)
    {
        data_[0] = x0;
        data_[1] = x1;
        data_[2] = x2;
        data_[3] = x3;
        data_[4] = x4;
        data_[5] = x5;
        data_[6] = x6;
    }

    TinyVector(T_numtype x0, T_numtype x1, T_numtype x2,
        T_numtype x3, T_numtype x4, T_numtype x5, T_numtype x6,
        T_numtype x7)
    {
        data_[0] = x0;
        data_[1] = x1;
        data_[2] = x2;
        data_[3] = x3;
        data_[4] = x4;
        data_[5] = x5;
        data_[6] = x6;
        data_[7] = x7;
    }

    TinyVector(T_numtype x0, T_numtype x1, T_numtype x2,
        T_numtype x3, T_numtype x4, T_numtype x5, T_numtype x6,
        T_numtype x7, T_numtype x8)
    {
        data_[0] = x0;
        data_[1] = x1;
        data_[2] = x2;
        data_[3] = x3;
        data_[4] = x4;
        data_[5] = x5;
        data_[6] = x6;
        data_[7] = x7;
        data_[8] = x8;
    }

    TinyVector(T_numtype x0, T_numtype x1, T_numtype x2,
        T_numtype x3, T_numtype x4, T_numtype x5, T_numtype x6,
        T_numtype x7, T_numtype x8, T_numtype x9)
    {
        data_[0] = x0;
        data_[1] = x1;
        data_[2] = x2;
        data_[3] = x3;
        data_[4] = x4;
        data_[5] = x5;
        data_[6] = x6;
        data_[7] = x7;
        data_[8] = x8;
        data_[9] = x9;
    }

    TinyVector(T_numtype x0, T_numtype x1, T_numtype x2,
        T_numtype x3, T_numtype x4, T_numtype x5, T_numtype x6,
        T_numtype x7, T_numtype x8, T_numtype x9, T_numtype x10)
    {
        data_[0] = x0;
        data_[1] = x1;
        data_[2] = x2;
        data_[3] = x3;
        data_[4] = x4;
        data_[5] = x5;
        data_[6] = x6;
        data_[7] = x7;
        data_[8] = x8;
        data_[9] = x9;
        data_[10] = x10;
    }

    // Constructor added by Peter Nordlund
    template<class P_expr>
    inline TinyVector(_bz_VecExpr<P_expr> expr);

    T_iterator begin()
    { return T_iterator(*this); }

    T_constIterator begin() const
    { return T_constIterator(*this); }

    // T_iterator end();
    // T_constIterator end() const;

    T_numtype * _bz_restrict data()
    { return data_; }

    const T_numtype * _bz_restrict data() const
    { return data_; }

    T_numtype * _bz_restrict dataFirst()
    { return data_; }

    const T_numtype * _bz_restrict dataFirst() const
    { return data_; }

    unsigned length() const
    { return N_length; }

    /////////////////////////////////////////////
    // Library-internal member functions
    // These are undocumented and may change or
    // disappear in future releases.
    /////////////////////////////////////////////

    unsigned        _bz_suggestLength() const
    { return N_length; }

    _bz_bool        _bz_hasFastAccess() const
    { return _bz_true; }

    T_numtype& _bz_restrict     _bz_fastAccess(unsigned i)
    { return data_[i]; }

    T_numtype       _bz_fastAccess(unsigned i) const
    { return data_[i]; }

    template<class P_expr, class P_updater>
    void _bz_assign(P_expr, P_updater);

    _bz_VecExpr<T_constIterator> _bz_asVecExpr() const
    { return _bz_VecExpr<T_constIterator>(begin()); }
   
    //////////////////////////////////////////////
    // Subscripting operators
    //////////////////////////////////////////////

    int lengthCheck(unsigned i) const
    {
        BZPRECHECK(i < N_length, 
            "TinyVector<" << BZ_DEBUG_TEMPLATE_AS_STRING_LITERAL(T_numtype) 
            << "," << N_length << "> index out of bounds: " << i);
        return 1;
    }

    T_numtype operator()(unsigned i) const
    {
        BZPRECONDITION(lengthCheck(i));
        return data_[i];
    }

    T_numtype& _bz_restrict operator()(unsigned i)
    { 
        BZPRECONDITION(lengthCheck(i));
        return data_[i];
    }

    T_numtype operator[](unsigned i) const
    {
        BZPRECONDITION(lengthCheck(i));
        return data_[i];
    }

    T_numtype& _bz_restrict operator[](unsigned i)
    {
        BZPRECONDITION(lengthCheck(i));
        return data_[i];
    }

    //////////////////////////////////////////////
    // Assignment operators
    //////////////////////////////////////////////

    // Scalar operand
    ListInitializationSwitch<T_vector> operator=(T_numtype x)
    {
        return ListInitializationSwitch<T_vector>(*this, x);
    }

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

    template<class P_numtype2> 
    T_vector& operator=(const TinyVector<P_numtype2, N_length> &);
    template<class P_numtype2>
    T_vector& operator+=(const TinyVector<P_numtype2, N_length> &);
    template<class P_numtype2>
    T_vector& operator-=(const TinyVector<P_numtype2, N_length> &);
    template<class P_numtype2>
    T_vector& operator*=(const TinyVector<P_numtype2, N_length> &);
    template<class P_numtype2>
    T_vector& operator/=(const TinyVector<P_numtype2, N_length> &);
    template<class P_numtype2>
    T_vector& operator%=(const TinyVector<P_numtype2, N_length> &);
    template<class P_numtype2>
    T_vector& operator^=(const TinyVector<P_numtype2, N_length> &);
    template<class P_numtype2>
    T_vector& operator&=(const TinyVector<P_numtype2, N_length> &);
    template<class P_numtype2>
    T_vector& operator|=(const TinyVector<P_numtype2, N_length> &);
    template<class P_numtype2>
    T_vector& operator>>=(const TinyVector<P_numtype2, N_length> &);
    template<class P_numtype2>
    T_vector& operator<<=(const TinyVector<P_numtype2, N_length> &);

    template<class P_numtype2> T_vector& operator=(const Vector<P_numtype2> &);
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

    T_numtype* _bz_restrict getInitializationIterator()
    { return dataFirst(); }

private:
    T_numtype data_[N_length];
};


// Specialization for N = 0: KCC is giving some
// peculiar errors, perhaps this will fix.

template<class T>
class TinyVector<T,0> {
};

BZ_NAMESPACE_END

#include <blitz/tinyveciter.h>  // Iterators
#include <blitz/tvecglobs.h>    // Global functions
#include <blitz/vector.h>       // Expression templates
#include <blitz/tinyvec.cc>     // Member functions
#include <blitz/tinyvecio.cc>   // I/O functions

#endif // BZ_TINYVEC_H

