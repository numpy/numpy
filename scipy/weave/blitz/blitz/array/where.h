// -*- C++ -*-
/***************************************************************************
 * blitz/array/where.h  where(X,Y,Z) operator for array expressions
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
#ifndef BZ_ARRAYWHERE_H
#define BZ_ARRAYWHERE_H

#ifndef BZ_ARRAYEXPR_H
 #error <blitz/array/where.h> must be included via <blitz/array/expr.h>
#endif

BZ_NAMESPACE(blitz)

template<typename P_expr1, typename P_expr2, typename P_expr3>
class _bz_ArrayWhere {

public:
    typedef P_expr1 T_expr1;
    typedef P_expr2 T_expr2;
    typedef P_expr3 T_expr3;
    typedef _bz_typename T_expr2::T_numtype T_numtype2;
    typedef _bz_typename T_expr3::T_numtype T_numtype3;
    typedef BZ_PROMOTE(T_numtype2, T_numtype3) T_numtype;
    typedef T_expr1 T_ctorArg1;
    typedef T_expr2 T_ctorArg2;
    typedef T_expr3 T_ctorArg3;

    static const int 
        numArrayOperands = P_expr1::numArrayOperands
                         + P_expr2::numArrayOperands
                         + P_expr3::numArrayOperands,
        numIndexPlaceholders = P_expr1::numIndexPlaceholders
                             + P_expr2::numIndexPlaceholders
                             + P_expr3::numIndexPlaceholders,
        rank = _bz_meta_max<_bz_meta_max<P_expr1::rank,P_expr2::rank>::max,
                            P_expr3::rank>::max;

    _bz_ArrayWhere(const _bz_ArrayWhere<T_expr1,T_expr2,T_expr3>& a)
      : iter1_(a.iter1_), iter2_(a.iter2_), iter3_(a.iter3_)
    { }

    template<typename T1, typename T2, typename T3>
    _bz_ArrayWhere(BZ_ETPARM(T1) a, BZ_ETPARM(T2) b, BZ_ETPARM(T3) c)
      : iter1_(a), iter2_(b), iter3_(c)
    { }

    T_numtype operator*()
    { return (*iter1_) ? (*iter2_) : (*iter3_); }

    template<int N_rank>
    T_numtype operator()(const TinyVector<int, N_rank>& i)
    { return iter1_(i) ? iter2_(i) : iter3_(i); }

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
        return iter1_.canCollapse(outerLoopRank, innerLoopRank)
            && iter2_.canCollapse(outerLoopRank, innerLoopRank)
            && iter3_.canCollapse(outerLoopRank, innerLoopRank);
    }

    template<int N_rank>
    void moveTo(const TinyVector<int,N_rank>& i)
    {
        iter1_.moveTo(i);
        iter2_.moveTo(i);
        iter3_.moveTo(i);
    }

    T_numtype operator[](int i)
    { return iter1_[i] ? iter2_[i] : iter3_[i]; }

    T_numtype fastRead(int i)
    { return iter1_.fastRead(i) ? iter2_.fastRead(i) : iter3_.fastRead(i); }

    int suggestStride(int rank) const
    {
        int stride1 = iter1_.suggestStride(rank);
        int stride2 = iter2_.suggestStride(rank);
        int stride3 = iter3_.suggestStride(rank);
        return stride1>(stride2=(stride2>stride3?stride2:stride3))?stride1:stride2;
        //return minmax::max(minmax::max(stride1,stride2),stride3);
    }

    bool isStride(int rank, int stride) const
    {
        return iter1_.isStride(rank,stride) 
            && iter2_.isStride(rank,stride)
            && iter3_.isStride(rank,stride);
    }

    void prettyPrint(BZ_STD_SCOPE(string) &str, 
        prettyPrintFormat& format) const
    {
        str += "where(";
        iter1_.prettyPrint(str,format);
        str += ",";
        iter2_.prettyPrint(str,format);
        str += ",";
        iter3_.prettyPrint(str,format);
        str += ")";
    }

    template<typename T_shape>
    bool shapeCheck(const T_shape& shape)
    { 
        int t1 = iter1_.shapeCheck(shape);
        int t2 = iter2_.shapeCheck(shape);
        int t3 = iter3_.shapeCheck(shape);

        return t1 && t2 && t3;
    }

private:
    _bz_ArrayWhere() { }

    T_expr1 iter1_;
    T_expr2 iter2_;
    T_expr3 iter3_;
};

template<typename T1, typename T2, typename T3>
inline
_bz_ArrayExpr<_bz_ArrayWhere<_bz_typename asExpr<T1>::T_expr,
    _bz_typename asExpr<T2>::T_expr, _bz_typename asExpr<T3>::T_expr> >
where(const T1& a, const T2& b, const T3& c)
{
    return _bz_ArrayExpr<_bz_ArrayWhere<_bz_typename asExpr<T1>::T_expr,
       _bz_typename asExpr<T2>::T_expr, 
       _bz_typename asExpr<T3>::T_expr> >(a,b,c);
}

BZ_NAMESPACE_END

#endif // BZ_ARRAYWHERE_H

