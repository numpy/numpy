/***************************************************************************
 * blitz/array/reduce.cc  Array reductions.
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
#ifndef BZ_ARRAYREDUCE_H
 #error <blitz/array/reduce.cc> must be included via <blitz/array/reduce.h>
#endif

BZ_NAMESPACE(blitz)

template<typename T_expr, typename T_reduction>
_bz_typename T_reduction::T_resulttype
_bz_reduceWithIndexTraversal(T_expr expr, T_reduction reduction);

template<typename T_expr, typename T_reduction>
_bz_typename T_reduction::T_resulttype
_bz_reduceWithStackTraversal(T_expr expr, T_reduction reduction);

template<typename T_expr, typename T_reduction>
_bz_typename T_reduction::T_resulttype
_bz_ArrayExprFullReduce(T_expr expr, T_reduction reduction)
{
#ifdef BZ_TAU_PROFILING
    // Tau profiling code.  Provide Tau with a pretty-printed version of
    // the expression.
    static BZ_STD_SCOPE(string) exprDescription;
    if (!exprDescription.length())      // faked static initializer
    {
        exprDescription = T_reduction::name();
        exprDescription += "(";
        prettyPrintFormat format(true);   // Terse mode on
        expr.prettyPrint(exprDescription, format);
        exprDescription += ")";
    }
    TAU_PROFILE(" ", exprDescription, TAU_BLITZ);
#endif // BZ_TAU_PROFILING

    return _bz_reduceWithIndexTraversal(expr, reduction);

#ifdef BZ_NOT_IMPLEMENTED_FLAG
    if ((T_expr::numIndexPlaceholders > 0) || (T_reduction::needIndex))
    {
        // The expression involves index placeholders, so have to
        // use index traversal rather than stack traversal.
        return reduceWithIndexTraversal(expr, reduction);
    }
    else {
        // Use a stack traversal
        return reduceWithStackTraversal(expr, reduction);
    }
#endif
}

template<typename T_expr, typename T_reduction>
_bz_typename T_reduction::T_resulttype
_bz_reduceWithIndexTraversal(T_expr expr, T_reduction reduction)
{
    // This is optimized assuming C-style arrays.

    reduction.reset();

    const int rank = T_expr::rank;

    TinyVector<int,T_expr::rank> index, first, last;

    unsigned long count = 1;

    for (int i=0; i < rank; ++i)
    {
        index(i) = expr.lbound(i);
        first(i) = index(i);
        last(i) = expr.ubound(i) + 1;
        count *= last(i) - first(i);
    }

    const int maxRank = rank - 1;
    int lastlbound = expr.lbound(maxRank);
    int lastubound = expr.ubound(maxRank);

    int lastIndex = lastubound + 1;

    bool loopFlag = true;

    while(loopFlag) {
        for (index[maxRank]=lastlbound;index[maxRank]<lastIndex;++index[maxRank])
            if (!reduction(expr(index), index[maxRank])) {
                loopFlag = false;
                break;
            }

        int j = rank-2;
        for (; j >= 0; --j) {
            index(j+1) = first(j+1);
            ++index(j);
            if (index(j) != last(j))
                break;
        }

        if (j < 0)
            break;
    }

    return reduction.result(count);
}


template<typename T_expr, typename T_reduction>
_bz_typename T_reduction::T_resulttype
_bz_reduceWithIndexVectorTraversal(T_expr expr, T_reduction reduction)
{
    // This version is for reductions that require a vector
    // of index positions.

    reduction.reset();

    const int rank = T_expr::rank;

    TinyVector<int,T_expr::rank> index, first, last;

    unsigned long count = 1;

    for (int i=0; i < rank; ++i)
    {
        index(i) = expr.lbound(i);
        first(i) = index(i);
        last(i) = expr.ubound(i) + 1;
        count *= last(i) - first(i);
    }

    const int maxRank = rank - 1;
    int lastlbound = expr.lbound(maxRank);
    int lastubound = expr.ubound(maxRank);

    int lastIndex = lastubound + 1;

    bool loopFlag = true;

    while(loopFlag) {
        for (index[maxRank]=lastlbound;index[maxRank]<lastIndex;++index[maxRank])
            if (!reduction(expr(index),index)) {
                loopFlag = false;
                break;
            }

        int j = rank-2;
        for (; j >= 0; --j) {
            index(j+1) = first(j+1);
            ++index(j);
            if (index(j) != last(j))
                break;
        }

        if (j < 0)
            break;
    }

    return reduction.result(count);
}

BZ_NAMESPACE_END

