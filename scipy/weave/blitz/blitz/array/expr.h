// -*- C++ -*-
/***************************************************************************
 * blitz/array/expr.h     Array<T,N> expression templates
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
#ifndef BZ_ARRAYEXPR_H
#define BZ_ARRAYEXPR_H

#ifndef BZ_ARRAY_H
 #error <blitz/array/expr.h> must be included via <blitz/array.h>
#endif

#include <blitz/ops.h>
#include <blitz/prettyprint.h>
#include <blitz/shapecheck.h>
#include <blitz/numinquire.h>

/*
 * The array expression templates iterator interface is followed by
 * these classes:
 *
 * FastArrayIterator          <blitz/array/fastiter.h>
 * _bz_ArrayExpr              <blitz/array/expr.h>
 * _bz_ArrayExprUnaryOp               "
 * _bz_ArrayExprBinaryOp              "
 * _bz_ArrayExprTernaryOp             "
 * _bz_ArrayExprConstant              "
 * _bz_ArrayMap               <blitz/array/map.h>
 * _bz_ArrayExprReduce        <blitz/array/reduce.h>
 * IndexPlaceholder           <blitz/indexexpr.h>
 */

BZ_NAMESPACE(blitz)

template<typename T1, typename T2>
class _bz_ExprPair {
public:
    _bz_ExprPair(const T1& a, const T2& b)
        : first_(a), second_(b)
    { }

    const T1& first() const
    { return first_; }

    const T2& second() const
    { return second_; }

protected:
    T1 first_;
    T2 second_;
};

template<typename T1, typename T2>
inline _bz_ExprPair<T1,T2> makeExprPair(const T1& a, const T2& b)
{
    return _bz_ExprPair<T1,T2>(a,b);
}

template<typename P_expr>
class _bz_ArrayExpr 
#ifdef BZ_NEW_EXPRESSION_TEMPLATES
    : public ETBase<_bz_ArrayExpr<P_expr> >
#endif
{

public:
    typedef P_expr T_expr;
    typedef _bz_typename T_expr::T_numtype T_numtype;
    typedef T_expr T_ctorArg1;
    typedef int    T_ctorArg2;    // dummy

    static const int 
        numArrayOperands = T_expr::numArrayOperands,
        numIndexPlaceholders = T_expr::numIndexPlaceholders,
        rank = T_expr::rank;

    _bz_ArrayExpr(const _bz_ArrayExpr<T_expr>& a)
#ifdef BZ_NEW_EXPRESSION_TEMPLATES
        : ETBase< _bz_ArrayExpr<T_expr> >(a), iter_(a.iter_)
#else
        : iter_(a.iter_)
#endif
    { }

#if defined(BZ_NEW_EXPRESSION_TEMPLATES) && ! defined(__MWERKS__)
    template<typename T>
    _bz_ArrayExpr(const T& a)
        : iter_(a)
    { }
#else

    _bz_ArrayExpr(BZ_ETPARM(T_expr) a)
        : iter_(a)
    { }
#if !defined(__MWERKS__)
    _bz_ArrayExpr(BZ_ETPARM(_bz_typename T_expr::T_ctorArg1) a)
        : iter_(a)
    { }
#endif
#endif

    template<typename T1, typename T2>
    _bz_ArrayExpr(BZ_ETPARM(T1) a, BZ_ETPARM(T2) b)
        : iter_(a, b)
    { }

    template<typename T1, typename T2, typename T3>
    _bz_ArrayExpr(BZ_ETPARM(T1) a, BZ_ETPARM(T2) b, BZ_ETPARM(T3) c)
        : iter_(a, b, c)
    { }

    template<typename T1, typename T2, typename T3, typename T4>
    _bz_ArrayExpr(BZ_ETPARM(T1) a, BZ_ETPARM(T2) b, BZ_ETPARM(T3) c,
        BZ_ETPARM(T4) d) : iter_(a, b, c, d)
    { }

    template<typename T1, typename T2>
    _bz_ArrayExpr(const _bz_ExprPair<T1,T2>& pair)
        : iter_(pair.first(), pair.second())
    { }

    T_numtype operator*()
    { return *iter_; }

#ifdef BZ_ARRAY_EXPR_PASS_INDEX_BY_VALUE
    template<int N_rank>
    T_numtype operator()(TinyVector<int, N_rank> i)
    { return iter_(i); }
#else
    template<int N_rank>
    T_numtype operator()(const TinyVector<int, N_rank>& i)
    { return iter_(i); }
#endif

    int ascending(int rank)
    { return iter_.ascending(rank); }

    int ordering(int rank)
    { return iter_.ordering(rank); }

    int lbound(int rank)
    { return iter_.lbound(rank); }

    int ubound(int rank)
    { return iter_.ubound(rank); }

    void push(int position)
    { iter_.push(position); }

    void pop(int position)
    { iter_.pop(position); }

    void advance()
    { iter_.advance(); }

    void advance(int n)
    { iter_.advance(n); }

    void loadStride(int rank)
    { iter_.loadStride(rank); }

    bool isUnitStride(int rank) const
    { return iter_.isUnitStride(rank); }

    void advanceUnitStride()
    { iter_.advanceUnitStride(); }

    bool canCollapse(int outerLoopRank, int innerLoopRank) const
    { 
        // BZ_DEBUG_MESSAGE("_bz_ArrayExpr<>::canCollapse()");
        return iter_.canCollapse(outerLoopRank, innerLoopRank); 
    }

    T_numtype operator[](int i)
    { return iter_[i]; }

    T_numtype fastRead(int i)
    { return iter_.fastRead(i); }

    int suggestStride(int rank) const
    { return iter_.suggestStride(rank); }

    bool isStride(int rank, int stride) const
    { return iter_.isStride(rank,stride); }

    void prettyPrint(BZ_STD_SCOPE(string) &str) const
    {
        prettyPrintFormat format(true);  // Terse formatting by default
        iter_.prettyPrint(str, format);
    }

    void prettyPrint(BZ_STD_SCOPE(string) &str, 
        prettyPrintFormat& format) const
    { iter_.prettyPrint(str, format); }

    template<typename T_shape>
    bool shapeCheck(const T_shape& shape)
    { return iter_.shapeCheck(shape); }

    template<int N_rank>
    void moveTo(const TinyVector<int,N_rank>& i)
    {
        iter_.moveTo(i);
    }

protected:
    _bz_ArrayExpr() { }

    T_expr iter_;
};

struct bounds {
    static int compute_ascending(int BZ_DEBUG_PARAM(rank),
                                 int ascending1, int ascending2)
    {
        // The value INT_MIN indicates that there are no arrays
        // in a subtree of the expression.  This logic returns
        // whichever ascending is available.  If there are two
        // conflicting ascending values, this is an error.

        if (ascending1 == ascending2)
            return ascending1;
        else if (ascending1 == INT_MIN)
            return ascending2;
        else if (ascending2 == INT_MIN)
            return ascending1;

        BZ_DEBUG_MESSAGE("Two array operands have different"
            << endl << "ascending flags: for rank " << rank 
            << ", the flags are " << ascending1 << " and " 
            << ascending2 << endl);
        BZ_PRE_FAIL;
        return 0;
    }

    static int compute_ordering(int BZ_DEBUG_PARAM(rank),
                                int order1, int order2)
    {
        // The value INT_MIN indicates that there are no arrays
        // in a subtree of the expression.  This logic returns
        // whichever ordering is available.  If there are two
        // conflicting ordering values, this is an error.

        if (order1 == order2)
            return order1;
        else if (order1 == INT_MIN)
            return order2;
        else if (order2 == INT_MIN)
            return order1;

        BZ_DEBUG_MESSAGE("Two array operands have different"
            << endl << "orders: for rank " << rank << ", the orders are "
            << order1 << " and " << order2 << endl);
        BZ_PRE_FAIL;
        return 0;
    }

    static int compute_lbound(int BZ_DEBUG_PARAM(rank),
                              int lbound1, int lbound2)
    {
        // The value INT_MIN indicates that there are no arrays
        // in a subtree of the expression.  This logic returns
        // whichever lbound is available.  If there are two
        // conflicting lbound values, this is an error.

        if (lbound1 == lbound2)
            return lbound1;
        else if (lbound1 == INT_MIN)
            return lbound2;
        else if (lbound2 == INT_MIN)
            return lbound1;

        BZ_DEBUG_MESSAGE("Two array operands have different"
            << endl << "lower bounds: in rank " << rank << ", the bounds are "
            << lbound1 << " and " << lbound2 << endl);
        BZ_PRE_FAIL;
        return 0;
    }

    static int compute_ubound(int BZ_DEBUG_PARAM(rank),
                              int ubound1, int ubound2)
    {
        // The value INT_MAX indicates that there are no arrays
        // in a subtree of the expression.  This logic returns
        // whichever ubound is available.  If there are two
        // conflicting ubound values, this is an error.

        if (ubound1 == ubound2)
            return ubound1;
        else if (ubound1 == INT_MAX)
            return ubound2;
        else if (ubound2 == INT_MAX)
            return ubound1;

        BZ_DEBUG_MESSAGE("Two array operands have different"
            << endl << "upper bounds: in rank " << rank << ", the bounds are "
            << ubound1 << " and " << ubound2 << endl);
        BZ_PRE_FAIL;
        return 0;
    }
};

template<typename P_expr, typename P_op>
class _bz_ArrayExprUnaryOp {
public:
    typedef P_expr T_expr;
    typedef P_op T_op;
    typedef _bz_typename T_expr::T_numtype T_numtype1;
    typedef _bz_typename T_op::T_numtype T_numtype;
    typedef T_expr T_ctorArg1;
    typedef int    T_ctorArg2;    // dummy

    static const int 
        numArrayOperands = T_expr::numArrayOperands,
        numIndexPlaceholders = T_expr::numIndexPlaceholders,
        rank = T_expr::rank;

    _bz_ArrayExprUnaryOp(const _bz_ArrayExprUnaryOp<T_expr, T_op>& a)
        : iter_(a.iter_)
    { }

    _bz_ArrayExprUnaryOp(BZ_ETPARM(T_expr) a)
        : iter_(a)
    { }

    _bz_ArrayExprUnaryOp(_bz_typename T_expr::T_ctorArg1 a)
        : iter_(a)
    { }

#if BZ_TEMPLATE_CTOR_DOESNT_CAUSE_HAVOC
    template<typename T1>
    explicit _bz_ArrayExprUnaryOp(BZ_ETPARM(T1) a)
        : iter_(a)
    { }
#endif

    int ascending(int rank)
    { return iter_.ascending(rank); }

    int ordering(int rank)
    { return iter_.ordering(rank); }

    int lbound(int rank)
    { return iter_.lbound(rank); }

    int ubound(int rank)
    { return iter_.ubound(rank); }

    T_numtype operator*()
    { return T_op::apply(*iter_); }

#ifdef BZ_ARRAY_EXPR_PASS_INDEX_BY_VALUE
    template<int N_rank>
    T_numtype operator()(TinyVector<int, N_rank> i)
    { return T_op::apply(iter_(i)); }
#else
    template<int N_rank>
    T_numtype operator()(const TinyVector<int, N_rank>& i)
    { return T_op::apply(iter_(i)); }
#endif

    void push(int position)
    {
        iter_.push(position);
    }

    void pop(int position)
    {
        iter_.pop(position);
    }

    void advance()
    {
        iter_.advance();
    }

    void advance(int n)
    {
        iter_.advance(n);
    }

    void loadStride(int rank)
    {
        iter_.loadStride(rank);
    }

    bool isUnitStride(int rank) const
    { return iter_.isUnitStride(rank); }

    void advanceUnitStride()
    {
        iter_.advanceUnitStride();
    }

    template<int N_rank>
    void moveTo(const TinyVector<int,N_rank>& i)
    {
        iter_.moveTo(i);
    }

    bool canCollapse(int outerLoopRank, int innerLoopRank) const
    { 
        // BZ_DEBUG_MESSAGE("_bz_ArrayExprUnaryOp<>::canCollapse");
        return iter_.canCollapse(outerLoopRank, innerLoopRank); 
    }

    T_numtype operator[](int i)
    { return T_op::apply(iter_[i]); }

    T_numtype fastRead(int i)
    { return T_op::apply(iter_.fastRead(i)); }

    int suggestStride(int rank) const
    { return iter_.suggestStride(rank); }

    bool isStride(int rank, int stride) const
    { return iter_.isStride(rank,stride); }

    void prettyPrint(BZ_STD_SCOPE(string) &str, 
        prettyPrintFormat& format) const
    { T_op::prettyPrint(str, format, iter_); }

    template<typename T_shape>
    bool shapeCheck(const T_shape& shape)
    { return iter_.shapeCheck(shape); }

protected:
    _bz_ArrayExprUnaryOp() { }

    T_expr iter_;
};


template<typename P_expr1, typename P_expr2, typename P_op>
class _bz_ArrayExprBinaryOp {
public:
    typedef P_expr1 T_expr1;
    typedef P_expr2 T_expr2;
    typedef P_op T_op;
    typedef _bz_typename T_expr1::T_numtype T_numtype1;
    typedef _bz_typename T_expr2::T_numtype T_numtype2;
    typedef _bz_typename T_op::T_numtype T_numtype;
    typedef T_expr1 T_ctorArg1;
    typedef T_expr2 T_ctorArg2;

    static const int 
        numArrayOperands = T_expr1::numArrayOperands
                         + T_expr2::numArrayOperands,
        numIndexPlaceholders = T_expr1::numIndexPlaceholders
                             + T_expr2::numIndexPlaceholders,
        rank = (T_expr1::rank > T_expr2::rank) 
             ? T_expr1::rank : T_expr2::rank;

    _bz_ArrayExprBinaryOp(
        const _bz_ArrayExprBinaryOp<T_expr1, T_expr2, T_op>& a)
        : iter1_(a.iter1_), iter2_(a.iter2_)
    { }

    template<typename T1, typename T2>
    _bz_ArrayExprBinaryOp(BZ_ETPARM(T1) a, BZ_ETPARM(T2) b)
        : iter1_(a), iter2_(b)
    { }

    T_numtype operator*()
    { return T_op::apply(*iter1_, *iter2_); }

#ifdef BZ_ARRAY_EXPR_PASS_INDEX_BY_VALUE
    template<int N_rank>
    T_numtype operator()(TinyVector<int, N_rank> i)
    { return T_op::apply(iter1_(i), iter2_(i)); }
#else
    template<int N_rank>
    T_numtype operator()(const TinyVector<int, N_rank>& i)
    { return T_op::apply(iter1_(i), iter2_(i)); }
#endif

    int ascending(int rank)
    {
        return bounds::compute_ascending(rank, iter1_.ascending(rank),
            iter2_.ascending(rank));
    }

    int ordering(int rank)
    {
        return bounds::compute_ordering(rank, iter1_.ordering(rank),
            iter2_.ordering(rank));
    }

    int lbound(int rank)
    { 
        return bounds::compute_lbound(rank, iter1_.lbound(rank),
            iter2_.lbound(rank));
    }

    int ubound(int rank)
    {
        return bounds::compute_ubound(rank, iter1_.ubound(rank),
            iter2_.ubound(rank));
    }

    void push(int position)
    { 
        iter1_.push(position); 
        iter2_.push(position);
    }

    void pop(int position)
    { 
        iter1_.pop(position); 
        iter2_.pop(position);
    }

    void advance()
    { 
        iter1_.advance(); 
        iter2_.advance();
    }

    void advance(int n)
    {
        iter1_.advance(n);
        iter2_.advance(n);
    }

    void loadStride(int rank)
    { 
        iter1_.loadStride(rank); 
        iter2_.loadStride(rank);
    }
    
    bool isUnitStride(int rank) const
    { return iter1_.isUnitStride(rank) && iter2_.isUnitStride(rank); }

    void advanceUnitStride()
    { 
        iter1_.advanceUnitStride(); 
        iter2_.advanceUnitStride();
    }

    bool canCollapse(int outerLoopRank, int innerLoopRank) const
    { 
        // BZ_DEBUG_MESSAGE("_bz_ArrayExprBinaryOp<>::canCollapse");
        return iter1_.canCollapse(outerLoopRank, innerLoopRank)
            && iter2_.canCollapse(outerLoopRank, innerLoopRank);
    } 

    T_numtype operator[](int i)
    { return T_op::apply(iter1_[i], iter2_[i]); }

    T_numtype fastRead(int i)
    { return T_op::apply(iter1_.fastRead(i), iter2_.fastRead(i)); }

    int suggestStride(int rank) const
    {
        int stride1 = iter1_.suggestStride(rank);
        int stride2 = iter2_.suggestStride(rank);
        return (stride1 > stride2) ? stride1 : stride2;
    }

    bool isStride(int rank, int stride) const
    {
        return iter1_.isStride(rank,stride) && iter2_.isStride(rank,stride);
    }

    template<int N_rank>
    void moveTo(const TinyVector<int,N_rank>& i)
    {
        iter1_.moveTo(i);
        iter2_.moveTo(i);
    }

    void prettyPrint(BZ_STD_SCOPE(string) &str, 
        prettyPrintFormat& format) const
    {
        T_op::prettyPrint(str, format, iter1_, iter2_);
    }

    template<typename T_shape>
    bool shapeCheck(const T_shape& shape)
    { return iter1_.shapeCheck(shape) && iter2_.shapeCheck(shape); }

protected:
    _bz_ArrayExprBinaryOp() { }

    T_expr1 iter1_;
    T_expr2 iter2_; 
};

template<typename P_expr1, typename P_expr2, typename P_expr3, typename P_op>
class _bz_ArrayExprTernaryOp {
public:
    typedef P_expr1 T_expr1;
    typedef P_expr2 T_expr2;
    typedef P_expr3 T_expr3;
    typedef P_op T_op;
    typedef _bz_typename T_expr1::T_numtype T_numtype1;
    typedef _bz_typename T_expr2::T_numtype T_numtype2;
    typedef _bz_typename T_expr3::T_numtype T_numtype3;
    typedef _bz_typename T_op::T_numtype T_numtype;
    typedef T_expr1 T_ctorArg1;
    typedef T_expr2 T_ctorArg2;
    typedef T_expr3 T_ctorArg3;

    static const int 
        numArrayOperands = T_expr1::numArrayOperands
                         + T_expr2::numArrayOperands
                         + T_expr3::numArrayOperands,
        numIndexPlaceholders = T_expr1::numIndexPlaceholders
                             + T_expr2::numIndexPlaceholders
                             + T_expr3::numIndexPlaceholders,
        rank = (T_expr1::rank > T_expr2::rank) 
             ? ((T_expr1::rank > T_expr3::rank)
                ? T_expr1::rank : T_expr3::rank)
             : ((T_expr2::rank > T_expr3::rank) 
                ? T_expr2::rank : T_expr3::rank);

    _bz_ArrayExprTernaryOp(
        const _bz_ArrayExprTernaryOp<T_expr1, T_expr2, T_expr3, T_op>& a)
        : iter1_(a.iter1_), iter2_(a.iter2_), iter3_(a.iter3_)
    { }

    template<typename T1, typename T2, typename T3>
    _bz_ArrayExprTernaryOp(BZ_ETPARM(T1) a, BZ_ETPARM(T2) b, BZ_ETPARM(T3) c)
        : iter1_(a), iter2_(b), iter3_(c)
    { }

    T_numtype operator*()
    { return T_op::apply(*iter1_, *iter2_, *iter3_); }

#ifdef BZ_ARRAY_EXPR_PASS_INDEX_BY_VALUE
    template<int N_rank>
    T_numtype operator()(TinyVector<int, N_rank> i)
    { return T_op::apply(iter1_(i), iter2_(i), iter3_(i)); }
#else
    template<int N_rank>
    T_numtype operator()(const TinyVector<int, N_rank>& i)
    { return T_op::apply(iter1_(i), iter2_(i), iter3_(i)); }
#endif

    int ascending(int rank)
    {
        return bounds::compute_ascending(rank, bounds::compute_ascending(
            rank, iter1_.ascending(rank), iter2_.ascending(rank)),
            iter3_.ascending(rank));
    }

    int ordering(int rank)
    {
        return bounds::compute_ordering(rank, bounds::compute_ordering(
            rank, iter1_.ordering(rank), iter2_.ordering(rank)),
            iter3_.ordering(rank));
    }

    int lbound(int rank)
    { 
        return bounds::compute_lbound(rank, bounds::compute_lbound(
            rank, iter1_.lbound(rank), iter2_.lbound(rank)), 
            iter3_.lbound(rank));
    }

    int ubound(int rank)
    {
        return bounds::compute_ubound(rank, bounds::compute_ubound(
            rank, iter1_.ubound(rank), iter2_.ubound(rank)), 
            iter3_.ubound(rank));
    }

    void push(int position)
    { 
        iter1_.push(position); 
        iter2_.push(position);
        iter3_.push(position);
    }

    void pop(int position)
    { 
        iter1_.pop(position); 
        iter2_.pop(position);
        iter3_.pop(position);
    }

    void advance()
    { 
        iter1_.advance(); 
        iter2_.advance();
        iter3_.advance();
    }

    void advance(int n)
    {
        iter1_.advance(n);
        iter2_.advance(n);
        iter3_.advance(n);
    }

    void loadStride(int rank)
    { 
        iter1_.loadStride(rank); 
        iter2_.loadStride(rank);
        iter3_.loadStride(rank);
    }
    
    bool isUnitStride(int rank) const
    {
        return iter1_.isUnitStride(rank)
            && iter2_.isUnitStride(rank)
            && iter3_.isUnitStride(rank);
    }

    void advanceUnitStride()
    { 
        iter1_.advanceUnitStride(); 
        iter2_.advanceUnitStride();
        iter3_.advanceUnitStride();
    }

    bool canCollapse(int outerLoopRank, int innerLoopRank) const
    { 
        // BZ_DEBUG_MESSAGE("_bz_ArrayExprTernaryOp<>::canCollapse");
        return iter1_.canCollapse(outerLoopRank, innerLoopRank)
            && iter2_.canCollapse(outerLoopRank, innerLoopRank)
            && iter3_.canCollapse(outerLoopRank, innerLoopRank);
    } 

    T_numtype operator[](int i)
    { return T_op::apply(iter1_[i], iter2_[i], iter3_[i]); }

    T_numtype fastRead(int i)
    {
        return T_op::apply(iter1_.fastRead(i),
                           iter2_.fastRead(i),
                           iter3_.fastRead(i));
    }

    int suggestStride(int rank) const
    {
        int stride1 = iter1_.suggestStride(rank);
        int stride2 = iter2_.suggestStride(rank);
        int stride3 = iter3_.suggestStride(rank);
        return stride1 > ( stride2 = (stride2>stride3 ? stride2 : stride3) ) ?
            stride1 : stride2;
    }

    bool isStride(int rank, int stride) const
    {
        return iter1_.isStride(rank,stride)
            && iter2_.isStride(rank,stride)
            && iter3_.isStride(rank,stride);
    }

    template<int N_rank>
    void moveTo(const TinyVector<int,N_rank>& i)
    {
        iter1_.moveTo(i);
        iter2_.moveTo(i);
        iter3_.moveTo(i);
    }

    void prettyPrint(BZ_STD_SCOPE(string) &str, 
        prettyPrintFormat& format) const
    {
        T_op::prettyPrint(str, format, iter1_, iter2_, iter3_);
    }

    template<typename T_shape>
    bool shapeCheck(const T_shape& shape)
    {
        return iter1_.shapeCheck(shape)
            && iter2_.shapeCheck(shape)
            && iter3_.shapeCheck(shape);
    }

protected:
    _bz_ArrayExprTernaryOp() { }

    T_expr1 iter1_;
    T_expr2 iter2_; 
    T_expr3 iter3_; 
};


template<typename P_numtype>
class _bz_ArrayExprConstant {
public:
    typedef P_numtype T_numtype;
    typedef T_numtype T_ctorArg1;
    typedef int       T_ctorArg2;    // dummy

    static const int 
        numArrayOperands = 0, 
        numIndexPlaceholders = 0, 
        rank = 0;

    _bz_ArrayExprConstant(const _bz_ArrayExprConstant<T_numtype>& a)
        : value_(a.value_)
    { }

    _bz_ArrayExprConstant(T_numtype value)
        : value_(BZ_NO_PROPAGATE(value))
    { 
    }

    // tiny() and huge() return the smallest and largest representable
    // integer values.  See <blitz/numinquire.h>
    // NEEDS_WORK: use tiny(int()) once numeric_limits<T> available on
    // all platforms

    int ascending(int)
    { return INT_MIN; }

    int ordering(int)
    { return INT_MIN; }

    int lbound(int)
    { return INT_MIN; }

    int ubound(int)
    { return INT_MAX; }
    // NEEDS_WORK: use huge(int()) once numeric_limits<T> available on
    // all platforms

    T_numtype operator*()
    { return value_; }

#ifdef BZ_ARRAY_EXPR_PASS_INDEX_BY_VALUE
    template<int N_rank>
    T_numtype operator()(TinyVector<int,N_rank>)
    { return value_; }
#else
    template<int N_rank>
    T_numtype operator()(const TinyVector<int,N_rank>&)
    { return value_; }
#endif

    void push(int) { }
    void pop(int) { }
    void advance() { }
    void advance(int) { }
    void loadStride(int) { }

    bool isUnitStride(int) const
    { return true; }

    void advanceUnitStride()
    { }

    bool canCollapse(int,int) const 
    { return true; }

    T_numtype operator[](int)
    { return value_; }

    T_numtype fastRead(int)
    { return value_; }

    int suggestStride(int) const
    { return 1; }

    bool isStride(int,int) const
    { return true; }

    template<int N_rank>
    void moveTo(const TinyVector<int,N_rank>&)
    {
    }

    void prettyPrint(BZ_STD_SCOPE(string) &str, 
        prettyPrintFormat& format) const
    {
        if (format.tersePrintingSelected())
            str += format.nextScalarOperandSymbol();
        else
            str += BZ_DEBUG_TEMPLATE_AS_STRING_LITERAL(T_numtype);
    }

    template<typename T_shape>
    bool shapeCheck(const T_shape&)
    { return true; }

protected:
    _bz_ArrayExprConstant() { }

    T_numtype value_;
};

BZ_NAMESPACE_END

#include <blitz/array/asexpr.h>

#endif // BZ_ARRAYEXPR_H

