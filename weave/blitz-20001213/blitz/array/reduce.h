/***************************************************************************
 * blitz/array/reduce.h   Reductions of an array (or array expression) in a 
 *                        single rank: sum, mean, min, minIndex, max, maxIndex, 
 *                        product, count, any, all
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
 * Revision 1.1  2002/01/03 19:50:36  eric
 * renaming compiler to weave
 *
 * Revision 1.1  2001/04/27 17:22:04  ej
 * first attempt to include needed pieces of blitz
 *
 * Revision 1.1.1.1  2000/06/19 12:26:14  tveldhui
 * Imported sources
 *
 * Revision 1.2  1998/03/14 00:04:47  tveldhui
 * 0.2-alpha-05
 *
 * Revision 1.1  1997/07/16 14:51:20  tveldhui
 * Update: Alpha release 0.2 (Arrays)
 *
 */

#ifndef BZ_ARRAYREDUCE_H
#define BZ_ARRAYREDUCE_H

#ifndef BZ_ARRAYEXPR_H
 #error <blitz/array/reduce.h> must be included after <blitz/array/expr.h>
#endif

#ifndef BZ_REDUCE_H
 #include <blitz/reduce.h>
#endif

BZ_NAMESPACE(blitz)

template<class T_expr, int N_index, class T_reduction>
class _bz_ArrayExprReduce {

public:   
    typedef _bz_typename T_reduction::T_numtype T_numtype;
    typedef T_expr      T_ctorArg1;
    typedef T_reduction T_ctorArg2;

    enum { numArrayOperands = BZ_ENUM_CAST(T_expr::numArrayOperands),
        numIndexPlaceholders = BZ_ENUM_CAST(T_expr::numIndexPlaceholders) + 1,
        rank = BZ_ENUM_CAST(T_expr::rank) - 1 };

    _bz_ArrayExprReduce(const _bz_ArrayExprReduce<T_expr,N_index,T_reduction>&
        reduce)
        : reduce_(reduce.reduce_), iter_(reduce.iter_)
    {
    }

    _bz_ArrayExprReduce(T_expr expr)
        : iter_(expr)
    { }

    _bz_ArrayExprReduce(T_expr expr, T_reduction reduce)
        : iter_(expr), reduce_(reduce)
    { }

    int ascending(int rank)
    { return iter_.ascending(rank); }

    int ordering(int rank)
    { return iter_.ordering(rank); }

    int lbound(int rank)
    { return iter_.lbound(rank); }

    int ubound(int rank)
    { return iter_.ubound(rank); }

    template<int N_destRank>
    T_numtype operator()(const TinyVector<int, N_destRank>& destIndex)
    {
        BZPRECHECK(N_destRank == N_index,  
            "Array reduction performed over rank " << N_index 
            << " to produce a rank " << N_destRank << " expression." << endl
            << "You must reduce over rank " << N_destRank << " instead.");

        TinyVector<int, N_destRank + 1> index;

        // This metaprogram copies elements 0..N-1 of destIndex into index
        _bz_meta_vecAssign<N_index, 0>::assign(index, destIndex, 
            _bz_update<int,int>());

        int lbound = iter_.lbound(N_index);
        int ubound = iter_.ubound(N_index);

        // NEEDS_WORK: replace with tiny(int()) and huge(int()) once
        // <limits> widely available
        BZPRECHECK((lbound != INT_MIN) && (ubound != INT_MAX),
           "Array reduction performed over rank " << N_index
           << " is unbounded." << endl 
           << "There must be an array object in the expression being reduced"
           << endl << "which provides a bound in rank " << N_index << ".");

        reduce_.reset();

        for (index[N_index] = iter_.lbound(N_index);
            index[N_index] <= ubound; ++index[N_index])
        {
            if (!reduce_(iter_(index), index[N_index]))
                break;
        }

        return reduce_.result(ubound-lbound+1);
    }

    // If you have a precondition failure on this routine, it means
    // you are trying to use stack iteration mode on an expression
    // which contains an index placeholder.  You must use index
    // iteration mode instead.
    int operator*()
    {
        BZPRECONDITION(0);
        return 0;
    }

    // See operator*() note
    void push(int)
    {
        BZPRECONDITION(0);
    }

    // See operator*() note
    void pop(int)
    {
        BZPRECONDITION(0);
    }

    // See operator*() note
    void advance()
    {
        BZPRECONDITION(0);
    }

    // See operator*() note
    void advance(int)
    {
        BZPRECONDITION(0);
    }

    // See operator*() note
    void loadStride(int)
    {
        BZPRECONDITION(0);
    }

    _bz_bool isUnitStride(int rank) const
    {
        BZPRECONDITION(0);
        return false;
    }

    void advanceUnitStride()
    {
        BZPRECONDITION(0);
    }

    _bz_bool canCollapse(int,int) const
    {   BZPRECONDITION(0); return _bz_false; }

    T_numtype operator[](int)
    {
        BZPRECONDITION(0);
        return T_numtype();
    }

    T_numtype fastRead(int)
    {
        BZPRECONDITION(0);
        return T_numtype();
    }

    int suggestStride(int) const
    {
        BZPRECONDITION(0);
        return 0;
    }

    _bz_bool isStride(int,int) const
    {
        BZPRECONDITION(0);
        return _bz_true;
    }

    template<int N_rank>
    void moveTo(const TinyVector<int,N_rank>& i)
    {
        BZPRECONDITION(0);
        return;
    }

    void prettyPrint(string& str, prettyPrintFormat& format) const
    {
        // NEEDS_WORK-- do real formatting for reductions
        str += "reduce[NEEDS_WORK](";
        iter_.prettyPrint(str,format);
        str += ")";
    }

    template<class T_shape>
    _bz_bool shapeCheck(const T_shape& shape) const
    { 
        // NEEDS_WORK-- do a real shape check (tricky)
        return _bz_true; 
    }

private: 
    _bz_ArrayExprReduce() { }

    T_reduction reduce_;
    T_expr iter_;
};

#define BZ_DECL_ARRAY_PARTIAL_REDUCE(fn,reduction)                      \
template<class T_expr, int N_index>                                     \
inline                                                                  \
_bz_ArrayExpr<_bz_ArrayExprReduce<_bz_ArrayExpr<T_expr>, N_index,       \
    reduction<_bz_typename T_expr::T_numtype> > >                       \
fn(_bz_ArrayExpr<T_expr> expr, const IndexPlaceholder<N_index>&)        \
{                                                                       \
    return _bz_ArrayExprReduce<_bz_ArrayExpr<T_expr>, N_index,          \
        reduction<_bz_typename T_expr::T_numtype> >(expr);              \
}                                                                       \
                                                                        \
template<class T_numtype, int N_rank, int N_index>                      \
inline                                                                  \
_bz_ArrayExpr<_bz_ArrayExprReduce<FastArrayIterator<T_numtype,N_rank>,  \
    N_index, reduction<T_numtype> > >                                   \
fn(const Array<T_numtype, N_rank>& array,                               \
    const IndexPlaceholder<N_index>&)                                   \
{                                                                       \
    return _bz_ArrayExprReduce<FastArrayIterator<T_numtype,N_rank>,     \
        N_index, reduction<T_numtype> > (array.beginFast());            \
}                        

BZ_DECL_ARRAY_PARTIAL_REDUCE(sum,      ReduceSum)
BZ_DECL_ARRAY_PARTIAL_REDUCE(mean,     ReduceMean)
BZ_DECL_ARRAY_PARTIAL_REDUCE(min,      ReduceMin)
BZ_DECL_ARRAY_PARTIAL_REDUCE(minIndex, ReduceMinIndex)
BZ_DECL_ARRAY_PARTIAL_REDUCE(max,      ReduceMax)
BZ_DECL_ARRAY_PARTIAL_REDUCE(maxIndex, ReduceMaxIndex)
BZ_DECL_ARRAY_PARTIAL_REDUCE(product,  ReduceProduct)
BZ_DECL_ARRAY_PARTIAL_REDUCE(count,    ReduceCount)
BZ_DECL_ARRAY_PARTIAL_REDUCE(any,      ReduceAny)
BZ_DECL_ARRAY_PARTIAL_REDUCE(all,      ReduceAll)
BZ_DECL_ARRAY_PARTIAL_REDUCE(first,    ReduceFirst)
BZ_DECL_ARRAY_PARTIAL_REDUCE(last,     ReduceLast)

/*
 * Complete reductions
 */

// Prototype of reduction function
template<class T_expr, class T_reduction>
_bz_typename T_reduction::T_resulttype
_bz_ArrayExprFullReduce(T_expr expr, T_reduction reduction);

#define BZ_DECL_ARRAY_FULL_REDUCE(fn,reduction)                         \
template<class T_expr>                                                  \
inline                                                                  \
_bz_typename reduction<_bz_typename T_expr::T_numtype>::T_resulttype    \
fn(_bz_ArrayExpr<T_expr> expr)                                          \
{                                                                       \
    return _bz_ArrayExprFullReduce(expr,                                \
        reduction<_bz_typename T_expr::T_numtype>());                   \
}                                                                       \
                                                                        \
template<class T_numtype, int N_rank>                                   \
inline                                                                  \
_bz_typename reduction<T_numtype>::T_resulttype                         \
fn(const Array<T_numtype, N_rank>& array)                               \
{                                                                       \
    return _bz_ArrayExprFullReduce(array.beginFast(),                   \
        reduction<T_numtype>());                                        \
}                                                                     

BZ_DECL_ARRAY_FULL_REDUCE(sum,      ReduceSum)
BZ_DECL_ARRAY_FULL_REDUCE(mean,     ReduceMean)
BZ_DECL_ARRAY_FULL_REDUCE(min,      ReduceMin)
BZ_DECL_ARRAY_FULL_REDUCE(max,      ReduceMax)
BZ_DECL_ARRAY_FULL_REDUCE(product,  ReduceProduct)
BZ_DECL_ARRAY_FULL_REDUCE(count,    ReduceCount)
BZ_DECL_ARRAY_FULL_REDUCE(any,      ReduceAny)
BZ_DECL_ARRAY_FULL_REDUCE(all,      ReduceAll)
BZ_DECL_ARRAY_FULL_REDUCE(first,    ReduceFirst)
BZ_DECL_ARRAY_FULL_REDUCE(last,     ReduceLast)

// Special versions of complete reductions: minIndex and
// maxIndex

#define BZ_DECL_ARRAY_FULL_REDUCE_INDEXVECTOR(fn,reduction)             \
template<class T_expr>                                                  \
inline                                                                  \
_bz_typename reduction<_bz_typename T_expr::T_numtype,                  \
    T_expr::rank>::T_resulttype                                         \
fn(_bz_ArrayExpr<T_expr> expr)                                          \
{                                                                       \
    return _bz_reduceWithIndexVectorTraversal(expr,                     \
        reduction<_bz_typename T_expr::T_numtype, T_expr::rank>());     \
}                                                                       \
                                                                        \
template<class T_numtype, int N_rank>                                   \
inline                                                                  \
_bz_typename reduction<T_numtype,N_rank>::T_resulttype                  \
fn(const Array<T_numtype, N_rank>& array)                               \
{                                                                       \
    return _bz_reduceWithIndexVectorTraversal( array.beginFast(),       \
        reduction<T_numtype,N_rank>());                                 \
}

BZ_DECL_ARRAY_FULL_REDUCE_INDEXVECTOR(minIndex, ReduceMinIndexVector)
BZ_DECL_ARRAY_FULL_REDUCE_INDEXVECTOR(maxIndex, ReduceMaxIndexVector)

BZ_NAMESPACE_END

#include <blitz/array/reduce.cc>

#endif // BZ_ARRAYREDUCE_H
