/*
 * $Id$
 *
 * Copyright (C) 1997 Todd Veldhuizen <tveldhui@oonumerics.org>
 * All rights reserved.  Please see <blitz/blitz.h> for terms and
 * conditions of use.
 *
 * $Log$
 * Revision 1.1  2002/01/03 19:50:34  eric
 * renaming compiler to weave
 *
 * Revision 1.1  2001/04/27 17:25:28  ej
 * Looks like I need all the .cc files for blitz also
 *
 * Revision 1.1.1.1  2000/06/19 12:26:09  tveldhui
 * Imported sources
 *
 * Revision 1.6  1998/03/14 00:04:47  tveldhui
 * 0.2-alpha-05
 *
 * Revision 1.5  1997/07/16 14:51:20  tveldhui
 * Update: Alpha release 0.2 (Arrays)
 *
 * Revision 1.4  1997/01/24 14:42:00  tveldhui
 * Periodic RCS update
 *
 */

#ifndef BZ_VECTOR_CC
#define BZ_VECTOR_CC

#ifndef BZ_VECTOR_H
 #include <blitz/vector.h>
#endif

#ifndef BZ_UPDATE_H
 #include <blitz/update.h>
#endif

BZ_NAMESPACE(blitz)

template<class P_numtype>
Vector<P_numtype> Vector<P_numtype>::copy() const
{
    Vector<P_numtype> newCopy(length_);

    if (stride_ == 1)
    {
        memcpy(newCopy.data(), data(), length_ * sizeof(P_numtype));
    }
    else {
        for (int i=0; i < length_; ++i)
        {
            // Can assume that newCopy has unit stride, and hence use
            // the operator(), which assumes unit stride.  Since this
            // vector doesn't have unit stride, use [].
            newCopy(i) = (*this)[i];
        }
    }

    return newCopy;
}

template<class P_numtype>
void Vector<P_numtype>::makeUnique()
{
    if ((stride_ != 1) || (numReferences() > 1))
    {
        Vector<P_numtype> tmp = copy();
        reference(tmp);
    }
}

template<class P_numtype>
void Vector<P_numtype>::reference(Vector<P_numtype>& x)
{
    MemoryBlockReference<P_numtype>::changeBlock(x, 0);
    length_ = x.length_;
    stride_ = x.stride_;
}

template<class P_numtype>
void Vector<P_numtype>::resize(int length)
{
    if (length != length_)
    {
        MemoryBlockReference<P_numtype>::newBlock(length);
        length_ = length;
        stride_ = 1;
    }
}

template<class P_numtype>
void Vector<P_numtype>::resizeAndPreserve(int newLength)
{
    Vector<P_numtype> newVector(newLength);

    if (newLength > length_)
        newVector(Range(0,length_-1)) = (*this);
    else 
        newVector(Range(0,newLength-1)) = (*this);

    reference(newVector);
}

/*****************************************************************************
 * Assignment operators with vector expression operand
 */

template<class P_numtype> template<class P_expr, class P_updater>
inline
void Vector<P_numtype>::_bz_assign(P_expr expr, P_updater)
{
    BZPRECONDITION(expr.length(length_) == length_);

    // If all vectors in the expression, plus the vector to which the
    // result is being assigned have unit stride, then avoid stride
    // calculations.
    if ((stride_ == 1) && (expr._bz_hasFastAccess()))
    {
#ifndef BZ_PARTIAL_LOOP_UNROLL
        for (int i=0; i < length_; ++i)
            P_updater::update(data_[i], expr._bz_fastAccess(i));
#else
        // Unwind the inner loop, four elements at a time.
        int leftover = length_ & 0x03;

        int i=0;
        for (; i < leftover; ++i)
            P_updater::update(data_[i], expr._bz_fastAccess(i));

        for (; i < length_; i += 4)
        {
            // Common subexpression elimination: avoid doing i+1, i+2, i+3
            // multiple times (compilers *won't* do this cse automatically)
            int t1 = i+1;
            int t2 = i+2;
            int t3 = i+3;
    
            _bz_typename P_expr::T_numtype tmp1, tmp2, tmp3, tmp4;

            // Reading all the results first avoids aliasing
            // ambiguities, and provides more flexibility in
            // register allocation & instruction scheduling
            tmp1 = expr._bz_fastAccess(i);
            tmp2 = expr._bz_fastAccess(BZ_NO_PROPAGATE(t1));
            tmp3 = expr._bz_fastAccess(BZ_NO_PROPAGATE(t2));
            tmp4 = expr._bz_fastAccess(BZ_NO_PROPAGATE(t3));

            P_updater::update(data_[i], BZ_NO_PROPAGATE(tmp1));
            P_updater::update(data_[t1], tmp2);
            P_updater::update(data_[t2], tmp3);
            P_updater::update(data_[t3], tmp4);
        }
#endif
    }
    else {
        // Not all unit strides -- have to access all the vector elements
        // as data_[i*stride_], which is slower.
        for (int i=0; i < length_; ++i)
            P_updater::update((*this)[i], expr[i]);
    }
}

template<class P_numtype> template<class P_expr>
inline Vector<P_numtype>& Vector<P_numtype>::operator=(_bz_VecExpr<P_expr> expr)
{
    BZPRECONDITION(expr.length(length_) == length_);

    // If all vectors in the expression, plus the vector to which the
    // result is being assigned have unit stride, then avoid stride
    // calculations.
    if ((stride_ == 1) && (expr._bz_hasFastAccess()))
    {
#ifndef BZ_PARTIAL_LOOP_UNROLL
        for (int i=0; i < length_; ++i)
            data_[i] = (P_numtype)expr._bz_fastAccess(i);
#else
        // Unwind the inner loop, four elements at a time.
        int leftover = length_ & 3;

        int i=0;
        for (; i < leftover; ++i)
            data_[i] = (P_numtype)expr._bz_fastAccess(i);

        for (; i < length_; i += 4)
        {
            // Common subexpression elimination: avoid doing i+1, i+2, i+3
            // multiple times (compilers *won't* do this cse automatically)
            int t1 = i+1;
            int t2 = i+2;
            int t3 = i+3;

            _bz_typename P_expr::T_numtype tmp1, tmp2, tmp3, tmp4;

            // Reading all the results first avoids aliasing
            // ambiguities, and provides more flexibility in
            // register allocation & instruction scheduling
            tmp1 = expr._bz_fastAccess(i);
            tmp2 = expr._bz_fastAccess(BZ_NO_PROPAGATE(t1));
            tmp3 = expr._bz_fastAccess(BZ_NO_PROPAGATE(t2));
            tmp4 = expr._bz_fastAccess(BZ_NO_PROPAGATE(t3));

            data_[i] = (P_numtype)BZ_NO_PROPAGATE(tmp1);
            data_[t1] = (P_numtype)tmp2;
            data_[t2] = (P_numtype)tmp3;
            data_[t3] = (P_numtype)tmp4;
        }
#endif
    }
    else {
        // Not all unit strides -- have to access all the vector elements
        // as data_[i*stride_], which is slower.
        for (int i=0; i < length_; ++i)
            (*this)[i] = (P_numtype)expr[i];
    }
    return *this;
}

#ifdef BZ_PARTIAL_LOOP_UNROLL
 
#define BZ_VECTOR_ASSIGN(op)                                            \
template<class P_numtype> template<class P_expr>                        \
inline Vector<P_numtype>& Vector<P_numtype>::                           \
    operator op (_bz_VecExpr<P_expr> expr)                              \
{                                                                       \
    BZPRECONDITION(expr.length(length_) == length_);                    \
    if ((stride_ == 1) && (expr._bz_hasFastAccess()))                   \
    {                                                                   \
        int leftover = length_ & 0x03;                                  \
                                                                        \
        int i=0;                                                        \
        for (; i < leftover; ++i)                                       \
            data_[i] op expr._bz_fastAccess(i);                         \
                                                                        \
        for (; i < length_; i += 4)                                     \
        {                                                               \
            int t1 = i+1;                                               \
            int t2 = i+2;                                               \
            int t3 = i+3;                                               \
                                                                        \
            _bz_typename P_expr::T_numtype tmp1, tmp2, tmp3, tmp4;      \
                                                                        \
            tmp1 = expr._bz_fastAccess(i);                              \
            tmp2 = expr._bz_fastAccess(t1);                             \
            tmp3 = expr._bz_fastAccess(t2);                             \
            tmp4 = expr._bz_fastAccess(t3);                             \
                                                                        \
            data_[i] op tmp1;                                           \
            data_[t1] op tmp2;                                          \
            data_[t2] op tmp3;                                          \
            data_[t3] op tmp4;                                          \
        }                                                               \
    }                                                                   \
    else {                                                              \
        for (int i=0; i < length_; ++i)                               \
            (*this)[i] op expr[i];                                      \
    }                                                                   \
    return *this;                                                       \
}

#else   // not BZ_PARTIAL_LOOP_UNROLL

#ifdef BZ_ALTERNATE_FORWARD_BACKWARD_TRAVERSALS

/*
 * NEEDS_WORK: need to incorporate BZ_NO_PROPAGATE here.  This
 * will require doing away with the macro BZ_VECTOR_ASSIGN, and
 * adopting an approach similar to that used in arrayeval.cc.
 */

#define BZ_VECTOR_ASSIGN(op)                                            \
template<class P_numtype> template<class P_expr>                        \
inline Vector<P_numtype>& Vector<P_numtype>::                           \
    operator op (_bz_VecExpr<P_expr> expr)                              \
{                                                                       \
    BZPRECONDITION(expr.length(length_) == length_);                    \
    static int traversalOrder = 0;                                      \
    if ((stride_ == 1) && (expr._bz_hasFastAccess()))                   \
    {                                                                   \
        if (traversalOrder & 0x01)                                      \
            for (int i=length_-1; i >= 0; --i)                        \
                data_[i] op expr._bz_fastAccess(i);                     \
        else                                                            \
            for (int i=0; i < length_; ++i)                           \
                data_[i] op expr._bz_fastAccess(i);                     \
    }                                                                   \
    else {                                                              \
        for (int i=0; i < length_; ++i)                               \
            (*this)[i] op expr[i];                                      \
    }                                                                   \
    traversalOrder ^= 0x01;                                             \
    return *this;                                                       \
}

#else   // not BZ_ALTERNATE_FORWARD_BACKWARD_TRAVERSALS

#define BZ_VECTOR_ASSIGN(op)                                            \
template<class P_numtype> template<class P_expr>                        \
inline Vector<P_numtype>& Vector<P_numtype>::                           \
    operator op (_bz_VecExpr<P_expr> expr)                              \
{                                                                       \
    BZPRECONDITION(expr.length(length_) == length_);                    \
    if ((stride_ == 1) && (expr._bz_hasFastAccess()))                   \
    {                                                                   \
        for (int i=0; i < length_; ++i)                               \
            data_[i] op expr._bz_fastAccess(i);                         \
    }                                                                   \
    else {                                                              \
        for (int i=0; i < length_; ++i)                               \
            (*this)[i] op expr[i];                                      \
    }                                                                   \
    return *this;                                                       \
}                 
#endif // BZ_ALTERNATE_FORWARD_BACKWARD_TRAVERSALS
#endif // BZ_PARTIAL_LOOP_UNROLL

BZ_VECTOR_ASSIGN(+=)
BZ_VECTOR_ASSIGN(-=)
BZ_VECTOR_ASSIGN(*=)
BZ_VECTOR_ASSIGN(/=)
BZ_VECTOR_ASSIGN(%=)
BZ_VECTOR_ASSIGN(^=)
BZ_VECTOR_ASSIGN(&=)
BZ_VECTOR_ASSIGN(|=)
BZ_VECTOR_ASSIGN(>>=)
BZ_VECTOR_ASSIGN(<<=)

#if NOT_DEFINED

template<class P_numtype> template<class P_expr>
inline Vector<P_numtype>& Vector<P_numtype>::operator+=(_bz_VecExpr<P_expr> expr)
{
    _bz_assign(expr, _bz_plus_update<P_numtype,
        _bz_typename P_expr::T_numtype>());
    return *this;
}

template<class P_numtype> template<class P_expr>
inline Vector<P_numtype>& Vector<P_numtype>::operator-=(_bz_VecExpr<P_expr> expr)
{
    _bz_assign(expr, _bz_minus_update<P_numtype,
        _bz_typename P_expr::T_numtype>());
    return *this;
}

template<class P_numtype> template<class P_expr>
inline Vector<P_numtype>& Vector<P_numtype>::operator*=(_bz_VecExpr<P_expr> expr)
{
    _bz_assign(expr, _bz_multiply_update<P_numtype,
        _bz_typename P_expr::T_numtype>());
    return *this;
}

template<class P_numtype> template<class P_expr>
inline Vector<P_numtype>& Vector<P_numtype>::operator/=(_bz_VecExpr<P_expr> expr)
{
    _bz_assign(expr, _bz_divide_update<P_numtype,
        _bz_typename P_expr::T_numtype>());
    return *this;
}

template<class P_numtype> template<class P_expr>
inline Vector<P_numtype>& Vector<P_numtype>::operator%=(_bz_VecExpr<P_expr> expr)
{
    _bz_assign(expr, _bz_mod_update<P_numtype,
        _bz_typename P_expr::T_numtype>());
    return *this;
}

template<class P_numtype> template<class P_expr>
inline Vector<P_numtype>& Vector<P_numtype>::operator^=(_bz_VecExpr<P_expr> expr)
{
    _bz_assign(expr, _bz_xor_update<P_numtype,
        _bz_typename P_expr::T_numtype>());
    return *this;
}

template<class P_numtype> template<class P_expr>
inline Vector<P_numtype>& Vector<P_numtype>::operator&=(_bz_VecExpr<P_expr> expr)
{
    _bz_assign(expr, _bz_bitand_update<P_numtype,
        _bz_typename P_expr::T_numtype>());
    return *this;
}

template<class P_numtype> template<class P_expr>
inline Vector<P_numtype>& Vector<P_numtype>::operator|=(_bz_VecExpr<P_expr> expr)
{
    _bz_assign(expr, _bz_bitor_update<P_numtype,
        _bz_typename P_expr::T_numtype>());
    return *this;
}

template<class P_numtype> template<class P_expr>
inline Vector<P_numtype>& Vector<P_numtype>::operator>>=(_bz_VecExpr<P_expr> expr)
{
    _bz_assign(expr, _bz_shiftr_update<P_numtype,
        _bz_typename P_expr::T_numtype>());
    return *this;
}

template<class P_numtype> template<class P_expr>
inline Vector<P_numtype>& Vector<P_numtype>::operator<<=(_bz_VecExpr<P_expr> expr)
{
    _bz_assign(expr, _bz_shiftl_update<P_numtype,
        _bz_typename P_expr::T_numtype>());
    return *this;
}
#endif   // NOT_DEFINED

/*****************************************************************************
 * Assignment operators with scalar operand
 */

template<class P_numtype>
inline Vector<P_numtype>& Vector<P_numtype>::initialize(P_numtype x)
{
    typedef _bz_VecExprConstant<P_numtype> T_expr;
    (*this) = _bz_VecExpr<T_expr>(T_expr(x));
    return *this;
}

template<class P_numtype>
inline Vector<P_numtype>& Vector<P_numtype>::operator+=(P_numtype x)
{
    typedef _bz_VecExprConstant<P_numtype> T_expr;
    (*this) += _bz_VecExpr<T_expr>(T_expr(x));
    return *this;
}

template<class P_numtype>
inline Vector<P_numtype>& Vector<P_numtype>::operator-=(P_numtype x)
{
    typedef _bz_VecExprConstant<P_numtype> T_expr;
    (*this) -= _bz_VecExpr<T_expr>(T_expr(x));
    return *this;
}

template<class P_numtype>
inline Vector<P_numtype>& Vector<P_numtype>::operator*=(P_numtype x)
{
    typedef _bz_VecExprConstant<P_numtype> T_expr;
    (*this) *= _bz_VecExpr<T_expr>(T_expr(x));
    return *this;
}

template<class P_numtype>
inline Vector<P_numtype>& Vector<P_numtype>::operator/=(P_numtype x)
{
    typedef _bz_VecExprConstant<P_numtype> T_expr;
    (*this) /= _bz_VecExpr<T_expr>(T_expr(x));
    return *this;
}

template<class P_numtype>
inline Vector<P_numtype>& Vector<P_numtype>::operator%=(P_numtype x)
{
    typedef _bz_VecExprConstant<P_numtype> T_expr;
    (*this) %= _bz_VecExpr<T_expr>(T_expr(x));
    return *this;
}

template<class P_numtype>
inline Vector<P_numtype>& Vector<P_numtype>::operator^=(P_numtype x)
{
    typedef _bz_VecExprConstant<P_numtype> T_expr;
    (*this) ^= _bz_VecExpr<T_expr>(T_expr(x));
    return *this;
}

template<class P_numtype>
inline Vector<P_numtype>& Vector<P_numtype>::operator&=(P_numtype x)
{
    typedef _bz_VecExprConstant<P_numtype> T_expr;
    (*this) &= _bz_VecExpr<T_expr>(T_expr(x));
    return *this;
}

template<class P_numtype>
inline Vector<P_numtype>& Vector<P_numtype>::operator|=(P_numtype x)
{
    typedef _bz_VecExprConstant<P_numtype> T_expr;
    (*this) |= _bz_VecExpr<T_expr>(T_expr(x));
    return *this;
}

template<class P_numtype>
inline Vector<P_numtype>& Vector<P_numtype>::operator>>=(int x)
{
    typedef _bz_VecExprConstant<int> T_expr;
    (*this) >>= _bz_VecExpr<T_expr>(T_expr(x));
    return *this;
}

template<class P_numtype>
inline Vector<P_numtype>& Vector<P_numtype>::operator<<=(int x)
{
    typedef _bz_VecExprConstant<P_numtype> T_expr;
    (*this) <<= _bz_VecExpr<T_expr>(T_expr(x));
    return *this;
}

/*****************************************************************************
 * Assignment operators with vector operand
 */

// This version is for two vectors with the same template parameter.
// Does not appear to be supported by the current C++ standard; or
// is it?
#if 0
template<class P_numtype> template<>
inline Vector<P_numtype>&
Vector<P_numtype>::operator=(const Vector<P_numtype>& x)
{
    // NEEDS_WORK: if unit strides, use memcpy or something similar.

    typedef VectorIterConst<P_numtype> T_expr;
    (*this) = _bz_VecExpr<T_expr>(T_expr(*this));
    return *this;
}
#endif

// This version is for two vectors with *different* template
// parameters.
template<class P_numtype> template<class P_numtype2>
inline Vector<P_numtype>& 
Vector<P_numtype>::operator=(const Vector<P_numtype2>& x)
{
    (*this) = _bz_VecExpr<VectorIterConst<P_numtype2> >(x.begin());
    return *this;
}

template<class P_numtype> template<class P_numtype2>
inline Vector<P_numtype>&
Vector<P_numtype>::operator+=(const Vector<P_numtype2>& x)
{
    (*this) += _bz_VecExpr<VectorIterConst<P_numtype2> >(x.begin());
    return *this;
}

template<class P_numtype> template<class P_numtype2>
inline Vector<P_numtype>&
Vector<P_numtype>::operator-=(const Vector<P_numtype2>& x)
{
    (*this) -= _bz_VecExpr<VectorIterConst<P_numtype2> >(x.begin());
    return *this;
}

template<class P_numtype> template<class P_numtype2>
inline Vector<P_numtype>&
Vector<P_numtype>::operator*=(const Vector<P_numtype2>& x)
{
    (*this) *= _bz_VecExpr<VectorIterConst<P_numtype2> >(x.begin());
    return *this;
}

template<class P_numtype> template<class P_numtype2>
inline Vector<P_numtype>&
Vector<P_numtype>::operator/=(const Vector<P_numtype2>& x)
{
    (*this) /= _bz_VecExpr<VectorIterConst<P_numtype2> >(x.begin());
    return *this;
}

template<class P_numtype> template<class P_numtype2>
inline Vector<P_numtype>&
Vector<P_numtype>::operator%=(const Vector<P_numtype2>& x)
{
    (*this) %= _bz_VecExpr<VectorIterConst<P_numtype2> >(x.begin());
    return *this;
}

template<class P_numtype> template<class P_numtype2>
inline Vector<P_numtype>&
Vector<P_numtype>::operator^=(const Vector<P_numtype2>& x)
{
    (*this) ^= _bz_VecExpr<VectorIterConst<P_numtype2> >(x.begin());
    return *this;
}

template<class P_numtype> template<class P_numtype2>
inline Vector<P_numtype>&
Vector<P_numtype>::operator&=(const Vector<P_numtype2>& x)
{
    (*this) &= _bz_VecExpr<VectorIterConst<P_numtype2> >(x.begin());
    return *this;
}

template<class P_numtype> template<class P_numtype2>
inline Vector<P_numtype>&
Vector<P_numtype>::operator|=(const Vector<P_numtype2>& x)
{
    (*this) |= _bz_VecExpr<VectorIterConst<P_numtype2> >(x.begin());
    return *this;
}

template<class P_numtype> template<class P_numtype2>
inline Vector<P_numtype>&
Vector<P_numtype>::operator<<=(const Vector<P_numtype2>& x)
{
    (*this) <<= _bz_VecExpr<VectorIterConst<P_numtype2> >(x.begin());
    return *this;
}

template<class P_numtype> template<class P_numtype2>
inline Vector<P_numtype>&
Vector<P_numtype>::operator>>=(const Vector<P_numtype2>& x)
{
    (*this) >>= _bz_VecExpr<VectorIterConst<P_numtype2> >(x.begin());
    return *this;
}

/*****************************************************************************
 * Assignment operators with Range operand
 */

template<class P_numtype>
inline Vector<P_numtype>& Vector<P_numtype>::operator=(Range r)
{
    (*this) = _bz_VecExpr<Range>(r);
    return *this;
}

template<class P_numtype>
inline Vector<P_numtype>& Vector<P_numtype>::operator+=(Range r)
{
    (*this) += _bz_VecExpr<Range>(r);
    return *this;
}

template<class P_numtype>
inline Vector<P_numtype>& Vector<P_numtype>::operator-=(Range r)
{
    (*this) -= _bz_VecExpr<Range>(r);
    return *this;
}

template<class P_numtype>
inline Vector<P_numtype>& Vector<P_numtype>::operator*=(Range r)
{
    (*this) *= _bz_VecExpr<Range>(r);
    return *this;
}

template<class P_numtype>
inline Vector<P_numtype>& Vector<P_numtype>::operator/=(Range r)
{
    (*this) /= _bz_VecExpr<Range>(r);
    return *this;
}

template<class P_numtype>
inline Vector<P_numtype>& Vector<P_numtype>::operator%=(Range r)
{
    (*this) %= _bz_VecExpr<Range>(r);
    return *this;
}

template<class P_numtype>
inline Vector<P_numtype>& Vector<P_numtype>::operator^=(Range r)
{
    (*this) ^= _bz_VecExpr<Range>(r);
    return *this;
}

template<class P_numtype>
inline Vector<P_numtype>& Vector<P_numtype>::operator&=(Range r)
{
    (*this) &= _bz_VecExpr<Range>(r);
    return *this;
}

template<class P_numtype>
inline Vector<P_numtype>& Vector<P_numtype>::operator|=(Range r)
{
    (*this) |= _bz_VecExpr<Range>(r);
    return *this;
}

template<class P_numtype>
inline Vector<P_numtype>& Vector<P_numtype>::operator>>=(Range r)
{
    (*this) >>= _bz_VecExpr<Range>(r);
    return *this;
}

template<class P_numtype>
inline Vector<P_numtype>& Vector<P_numtype>::operator<<=(Range r)
{
    (*this) <<= _bz_VecExpr<Range>(r);
    return *this;
}

/*****************************************************************************
 * Assignment operators with VectorPick operand
 */

template<class P_numtype> template<class P_numtype2>
inline Vector<P_numtype>& Vector<P_numtype>::operator=(const 
    VectorPick<P_numtype2>& x)
{
    typedef VectorPickIterConst<P_numtype2> T_expr;
    (*this) = _bz_VecExpr<T_expr>(x.begin());
    return *this;
}

template<class P_numtype> template<class P_numtype2>
inline Vector<P_numtype>& Vector<P_numtype>::operator+=(const
    VectorPick<P_numtype2>& x)
{
    typedef VectorPickIterConst<P_numtype2> T_expr;
    (*this) += _bz_VecExpr<T_expr>(x.begin());
    return *this;
}


template<class P_numtype> template<class P_numtype2>
inline Vector<P_numtype>& Vector<P_numtype>::operator-=(const
    VectorPick<P_numtype2>& x)
{
    typedef VectorPickIterConst<P_numtype2> T_expr;
    (*this) -= _bz_VecExpr<T_expr>(x.begin());
    return *this;
}

template<class P_numtype> template<class P_numtype2>
inline Vector<P_numtype>& Vector<P_numtype>::operator*=(const
    VectorPick<P_numtype2>& x)
{
    typedef VectorPickIterConst<P_numtype2> T_expr;
    (*this) *= _bz_VecExpr<T_expr>(x.begin());
    return *this;
}

template<class P_numtype> template<class P_numtype2>
inline Vector<P_numtype>& Vector<P_numtype>::operator/=(const
    VectorPick<P_numtype2>& x)
{
    typedef VectorPickIterConst<P_numtype2> T_expr;
    (*this) /= _bz_VecExpr<T_expr>(x.begin());
    return *this;
}

template<class P_numtype> template<class P_numtype2>
inline Vector<P_numtype>& Vector<P_numtype>::operator%=(const
    VectorPick<P_numtype2>& x)
{
    typedef VectorPickIterConst<P_numtype2> T_expr;
    (*this) %= _bz_VecExpr<T_expr>(x.begin());
    return *this;
}

template<class P_numtype> template<class P_numtype2>
inline Vector<P_numtype>& Vector<P_numtype>::operator^=(const
    VectorPick<P_numtype2>& x)
{
    typedef VectorPickIterConst<P_numtype2> T_expr;
    (*this) ^= _bz_VecExpr<T_expr>(x.begin());
    return *this;
}

template<class P_numtype> template<class P_numtype2>
inline Vector<P_numtype>& Vector<P_numtype>::operator&=(const
    VectorPick<P_numtype2>& x)
{
    typedef VectorPickIterConst<P_numtype2> T_expr;
    (*this) &= _bz_VecExpr<T_expr>(x.begin());
    return *this;
}

template<class P_numtype> template<class P_numtype2>
inline Vector<P_numtype>& Vector<P_numtype>::operator|=(const
    VectorPick<P_numtype2>& x)
{
    typedef VectorPickIterConst<P_numtype2> T_expr;
    (*this) |= _bz_VecExpr<T_expr>(x.begin());
    return *this;
}

/*****************************************************************************
 * Assignment operators with Random operand
 */

template<class P_numtype> template<class P_distribution>
Vector<P_numtype>& Vector<P_numtype>::operator=(Random<P_distribution>& rand)
{
    for (int i=0; i < length_; ++i) 
        (*this)[i] = rand.random();
    return *this;
}

#ifdef BZ_PECULIAR_RANDOM_VECTOR_ASSIGN_BUG

template<class P_numtype> template<class P_distribution>
Vector<P_numtype>& Vector<P_numtype>::operator=(Random<P_distribution>& rand)
{
    (*this) = _bz_VecExpr<_bz_VecExprRandom<P_distribution> > 
        (_bz_VecExprRandom<P_distribution>(rand));
    return *this;
}

template<class P_numtype> template<class P_distribution>
Vector<P_numtype>& Vector<P_numtype>::operator+=(Random<P_distribution>& rand)
{
    (*this) += _bz_VecExpr<_bz_VecExprRandom<P_distribution> >
        (_bz_VecExprRandom<P_distribution>(rand));
    return *this;
}

template<class P_numtype> template<class P_distribution>
Vector<P_numtype>& Vector<P_numtype>::operator-=(Random<P_distribution>& rand)
{
    (*this) -= _bz_VecExpr<_bz_VecExprRandom<P_distribution> >
        (_bz_VecExprRandom<P_distribution>(rand));
    return *this;
}

template<class P_numtype> template<class P_distribution>
Vector<P_numtype>& Vector<P_numtype>::operator*=(Random<P_distribution>& rand)
{
    (*this) *= _bz_VecExpr<_bz_VecExprRandom<P_distribution> >
        (_bz_VecExprRandom<P_distribution>(rand));
    return *this;
}

template<class P_numtype> template<class P_distribution>
Vector<P_numtype>& Vector<P_numtype>::operator/=(Random<P_distribution>& rand)
{
    (*this) /= _bz_VecExpr<_bz_VecExprRandom<P_distribution> >
        (_bz_VecExprRandom<P_distribution>(rand));
    return *this;
}

template<class P_numtype> template<class P_distribution>
Vector<P_numtype>& Vector<P_numtype>::operator%=(Random<P_distribution>& rand)
{
    (*this) %= _bz_VecExpr<_bz_VecExprRandom<P_distribution> >
        (_bz_VecExprRandom<P_distribution>(rand));
    return *this;
}

template<class P_numtype> template<class P_distribution>
Vector<P_numtype>& Vector<P_numtype>::operator^=(Random<P_distribution>& rand)
{
    (*this) ^= _bz_VecExpr<_bz_VecExprRandom<P_distribution> >
        (_bz_VecExprRandom<P_distribution>(rand));
    return *this;
}

template<class P_numtype> template<class P_distribution>
Vector<P_numtype>& Vector<P_numtype>::operator&=(Random<P_distribution>& rand)
{
    (*this) &= _bz_VecExpr<_bz_VecExprRandom<P_distribution> >
        (_bz_VecExprRandom<P_distribution>(rand));
    return *this;
}

template<class P_numtype> template<class P_distribution>
Vector<P_numtype>& Vector<P_numtype>::operator|=(Random<P_distribution>& rand)
{
    (*this) |= _bz_VecExpr<_bz_VecExprRandom<P_distribution> >
        (_bz_VecExprRandom<P_distribution>(rand));
    return *this;
}

#endif // BZ_PECULIAR_RANDOM_VECTOR_ASSIGN_BUG

BZ_NAMESPACE_END

#endif // BZ_VECTOR_CC
