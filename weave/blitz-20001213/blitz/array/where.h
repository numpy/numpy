/***************************************************************************
 * blitz/array/where.h  where(X,Y,Z) operator for array expressions
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
 ***************************************************************************
 * $Log$
 * Revision 1.2  2002/09/12 07:02:06  eric
 * major rewrite of weave.
 *
 * 0.
 * The underlying library code is significantly re-factored and simpler. There used to be a xxx_spec.py and xxx_info.py file for every group of type conversion classes.  The spec file held the python code that handled the conversion and the info file had most of the C code templates that were generated.  This proved pretty confusing in practice, so the two files have mostly been merged into the spec file.
 *
 * Also, there was quite a bit of code duplication running around.  The re-factoring was able to trim the standard conversion code base (excluding blitz and accelerate stuff) by about 40%.  This should be a huge maintainability and extensibility win.
 *
 * 1.
 * With multiple months of using Numeric arrays, I've found some of weave's "magic variable" names unwieldy and want to change them.  The following are the old declarations for an array x of Float32 type:
 *
 *         PyArrayObject* x = convert_to_numpy(...);
 *         float* x_data = (float*) x->data;
 *         int*   _Nx = x->dimensions;
 *         int*   _Sx = x->strides;
 *         int    _Dx = x->nd;
 *
 * The new declaration looks like this:
 *
 *         PyArrayObject* x_array = convert_to_numpy(...);
 *         float* x = (float*) x->data;
 *         int*   Nx = x->dimensions;
 *         int*   Sx = x->strides;
 *         int    Dx = x->nd;
 *
 * This is obviously not backward compatible, and will break some code (including a lot of mine).  It also makes inline() code more readable and natural to write.
 *
 * 2.
 * I've switched from CXX to Gordon McMillan's SCXX for list, tuples, and dictionaries.  I like CXX pretty well, but its use of advanced C++ (templates, etc.) caused some portability problems.  The SCXX library is similar to CXX but doesn't use templates at all.  This, like (1) is not an
 * API compatible change and requires repairing existing code.
 *
 * I have also thought about boost python, but it also makes heavy use of templates.  Moving to SCXX gets rid of almost all template usage for the standard type converters which should help portability.  std::complex and std::string from the STL are the only templates left.  Of course blitz still uses templates in a major way so weave.blitz will continue to be hard on compilers.
 *
 * I've actually considered scrapping the C++ classes for list, tuples, and
 * dictionaries, and just fall back to the standard Python C API because the classes are waaay slower than the raw API in many cases.  They are also more convenient and less error prone in many cases, so I've decided to stick with them.  The PyObject variable will always be made available for variable "x" under the name "py_x" for more speedy operations.  You'll definitely want to use these for anything that needs to be speedy.
 *
 * 3.
 * strings are converted to std::string now.  I found this to be the most useful type in for strings in my code.  Py::String was used previously.
 *
 * 4.
 * There are a number of reference count "errors" in some of the less tested conversion codes such as instance, module, etc.  I've cleaned most of these up.  I put errors in quotes here because I'm actually not positive that objects passed into "inline" really need reference counting applied to them.  The dictionaries passed in by inline() hold references to these objects so it doesn't seem that they could ever be garbage collected inadvertently.  Variables used by ext_tools, though, definitely need the reference counting done.  I don't think this is a major cost in speed, so it probably isn't worth getting rid of the ref count code.
 *
 * 5.
 * Unicode objects are now supported.  This was necessary to support rendering Unicode strings in the freetype wrappers for Chaco.
 *
 * 6.
 * blitz++ was upgraded to the latest CVS.  It compiles about twice as fast as the old blitz and looks like it supports a large number of compilers (though only gcc 2.95.3 is tested).  Compile times now take about 9 seconds on my 850 MHz PIII laptop.
 *
 * Revision 1.3  2002/03/06 17:12:26  patricg
 *
 * minmax::max(minmax::max(stride1,stride2),stride3)
 * replaced by
 * stride1>(stride2=(stride2>stride3?stride2:stride3))?stride1:stride2
 *
 * Revision 1.2  2001/01/25 00:25:56  tveldhui
 * Ensured that source files have cvs logs.
 *
 */

#ifndef BZ_ARRAYWHERE_H
#define BZ_ARRAYWHERE_H

#ifndef BZ_ARRAYEXPR_H
 #error <blitz/array/where.h> must be included via <blitz/array/expr.h>
#endif

BZ_NAMESPACE(blitz)

template<class P_expr1, class P_expr2, class P_expr3>
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

    enum { numArrayOperands = BZ_ENUM_CAST(P_expr1::numArrayOperands)
                            + BZ_ENUM_CAST(P_expr2::numArrayOperands)
                            + BZ_ENUM_CAST(P_expr3::numArrayOperands),
           numIndexPlaceholders = BZ_ENUM_CAST(P_expr1::numIndexPlaceholders)
                            + BZ_ENUM_CAST(P_expr2::numIndexPlaceholders)
                            + BZ_ENUM_CAST(P_expr3::numIndexPlaceholders),
           rank = _bz_meta_max<_bz_meta_max<P_expr1::rank,P_expr2::rank>::max,
                            P_expr3::rank>::max
    };

    _bz_ArrayWhere(const _bz_ArrayWhere<T_expr1,T_expr2,T_expr3>& a)
      : iter1_(a.iter1_), iter2_(a.iter2_), iter3_(a.iter3_)
    { }

    template<class T1, class T2, class T3>
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

    _bz_bool isUnitStride(int rank) const
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

    _bz_bool canCollapse(int outerLoopRank, int innerLoopRank) const
    {
        // BZ_DEBUG_MESSAGE("_bz_ArrayExprOp<>::canCollapse");
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

    _bz_bool isStride(int rank, int stride) const
    {
        return iter1_.isStride(rank,stride) 
            && iter2_.isStride(rank,stride)
            && iter3_.isStride(rank,stride);
    }

    void prettyPrint(string& str, prettyPrintFormat& format) const
    {
        str += "where(";
        iter1_.prettyPrint(str,format);
        str += ",";
        iter2_.prettyPrint(str,format);
        str += ",";
        iter3_.prettyPrint(str,format);
        str += ")";
    }

    template<class T_shape>
    _bz_bool shapeCheck(const T_shape& shape)
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

template<class T1, class T2, class T3>
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

