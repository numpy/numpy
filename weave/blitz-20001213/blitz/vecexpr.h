/***************************************************************************
 * blitz/vecexpr.h      Vector<P_numtype> expression templates
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
 * Revision 1.2  2002/09/12 07:04:04  eric
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
 * Revision 1.5  2002/07/02 19:36:43  jcumming
 * Undid the previous change to this file.  Vector ET support is now gotten
 * by including blitz/vector-et.h explicitly.
 *
 * Revision 1.4  2002/03/06 17:53:57  patricg
 *
 * (re)inserted includes for vecbops, vecuops and vecbfn
 * in order to compile testsuite/tinyvec.cpp
 *
 * Revision 1.3  2001/01/24 22:51:50  tveldhui
 * Reorganized #include orders to avoid including the huge Vector e.t.
 * implementation when using Array.
 *
 * Revision 1.2  2001/01/24 20:22:50  tveldhui
 * Updated copyright date in headers.
 *
 * Revision 1.1.1.1  2000/06/19 12:26:10  tveldhui
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
 * Revision 1.3  1997/01/23 03:28:28  tveldhui
 * Periodic RCS update
 *
 * Revision 1.2  1997/01/13 22:19:58  tveldhui
 * Periodic RCS update
 *
 *
 */


#ifndef BZ_VECEXPR_H
#define BZ_VECEXPR_H

#ifndef BZ_VECTOR_H
 #include <blitz/vector.h>
#endif

#ifndef BZ_APPLICS_H
 #include <blitz/applics.h>
#endif

#ifndef BZ_META_METAPROG_H
 #include <blitz/meta/metaprog.h>
#endif

#ifndef BZ_VECEXPRWRAP_H
 #include <blitz/vecexprwrap.h>           // _bz_VecExpr wrapper class
#endif

BZ_NAMESPACE(blitz)

template<class P_expr1, class P_expr2, class P_op>
class _bz_VecExprOp {

public:
    typedef P_expr1 T_expr1;
    typedef P_expr2 T_expr2;
    typedef _bz_typename T_expr1::T_numtype T_numtype1;
    typedef _bz_typename T_expr2::T_numtype T_numtype2;
    typedef BZ_PROMOTE(T_numtype1, T_numtype2) T_numtype;
    typedef P_op    T_op;

#ifdef BZ_PASS_EXPR_BY_VALUE
    _bz_VecExprOp(T_expr1 a, T_expr2 b)
        : iter1_(a), iter2_(b)
    { }
#else
    _bz_VecExprOp(const T_expr1& a, const T_expr2& b)
        : iter1_(a), iter2_(b)
    { }
#endif

#ifdef BZ_MANUAL_VECEXPR_COPY_CONSTRUCTOR
    _bz_VecExprOp(const _bz_VecExprOp<P_expr1, P_expr2, P_op>& x)
        : iter1_(x.iter1_), iter2_(x.iter2_)
    { }
#endif

    T_numtype operator[](int i) const
    { return T_op::apply(iter1_[i], iter2_[i]); }

    T_numtype operator()(int i) const
    { return T_op::apply(iter1_(i), iter2_(i)); }

    int length(int recommendedLength) const
    { 
        BZPRECONDITION(iter2_.length(recommendedLength) == 
            iter1_.length(recommendedLength));
        return iter1_.length(recommendedLength); 
    }

    enum { 
           _bz_staticLengthCount = 
      BZ_ENUM_CAST(P_expr1::_bz_staticLengthCount) 
         + BZ_ENUM_CAST(P_expr2::_bz_staticLengthCount),

           _bz_dynamicLengthCount = 
      BZ_ENUM_CAST(P_expr1::_bz_dynamicLengthCount) 
        + BZ_ENUM_CAST(P_expr2::_bz_dynamicLengthCount),

           _bz_staticLength = (BZ_ENUM_CAST(P_expr1::_bz_staticLength) > BZ_ENUM_CAST(P_expr2::_bz_staticLength)) ? BZ_ENUM_CAST(P_expr1::_bz_staticLength) : BZ_ENUM_CAST(P_expr2::_bz_staticLength)

//      _bz_meta_max<P_expr1::_bz_staticLength, P_expr2::_bz_staticLength>::max 
    };

    int _bz_suggestLength() const
    {
        int length1 = iter1_._bz_suggestLength();
        if (length1 != 0)
            return length1;
        return iter2_._bz_suggestLength();
    }

    _bz_bool  _bz_hasFastAccess() const
    { return iter1_._bz_hasFastAccess() && iter2_._bz_hasFastAccess(); }

    T_numtype _bz_fastAccess(int i) const
    { 
        return T_op::apply(iter1_._bz_fastAccess(i),
            iter2_._bz_fastAccess(i)); 
    }
    
private:
    _bz_VecExprOp();

    T_expr1 iter1_;
    T_expr2 iter2_;
};

template<class P_expr, class P_unaryOp>
class _bz_VecExprUnaryOp {

public:
    typedef P_expr T_expr;
    typedef P_unaryOp T_unaryOp;
    typedef _bz_typename T_unaryOp::T_numtype T_numtype;

#ifdef BZ_PASS_EXPR_BY_VALUE
    _bz_VecExprUnaryOp(T_expr iter)
        : iter_(iter)
    { }
#else
    _bz_VecExprUnaryOp(const T_expr& iter)
        : iter_(iter)
    { }
#endif

#ifdef BZ_MANUAL_VECEXPR_COPY_CONSTRUCTOR
    _bz_VecExprUnaryOp(const _bz_VecExprUnaryOp<P_expr, P_unaryOp>& x)
        : iter_(x.iter_)
    { }
#endif

    T_numtype operator[](int i) const
    { return T_unaryOp::apply(iter_[i]); }

    T_numtype operator()(int i) const
    { return T_unaryOp::apply(iter_(i)); }

    int length(int recommendedLength) const
    { return iter_.length(recommendedLength); }

    enum { _bz_staticLengthCount = BZ_ENUM_CAST(P_expr::_bz_staticLengthCount),
           _bz_dynamicLengthCount =BZ_ENUM_CAST(P_expr::_bz_dynamicLengthCount),
           _bz_staticLength = BZ_ENUM_CAST(P_expr::_bz_staticLength) };

    int _bz_suggestLength() const
    { return iter_._bz_suggestLength(); }

    _bz_bool _bz_hasFastAccess() const
    { return iter_._bz_hasFastAccess(); }

    T_numtype _bz_fastAccess(int i) const
    { return T_unaryOp::apply(iter_._bz_fastAccess(i)); }

private:
    _bz_VecExprUnaryOp() { }

    T_expr iter_;    
};

template<class P_numtype>
class _bz_VecExprConstant {
public:
    typedef P_numtype T_numtype;

    _bz_VecExprConstant(P_numtype value)
        : value_(BZ_NO_PROPAGATE(value))
    { 
    }

#ifdef BZ_MANUAL_VECEXPR_COPY_CONSTRUCTOR
    _bz_VecExprConstant(const _bz_VecExprConstant<P_numtype>& x)
        : value_(x.value_)
    { }
#endif

    T_numtype operator[](int) const
    { return value_; }

    T_numtype operator()(int) const
    { return value_; }

    int length(int recommendedLength) const
    { return recommendedLength; }

    enum { _bz_staticLengthCount = 0,
           _bz_dynamicLengthCount = 0,
           _bz_staticLength = 0
    };

    int _bz_suggestLength() const
    { return 0; }

    _bz_bool _bz_hasFastAccess() const
    { return 1; }

    T_numtype _bz_fastAccess(int) const
    { return value_; }

private:

    _bz_VecExprConstant() { }

    T_numtype value_;
};

// Some miscellaneous operators that don't seem to belong anywhere else.

template<class P_expr>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<_bz_VecExpr<P_expr>, 
    _bz_negate<_bz_typename P_expr::T_numtype> > >
operator-(_bz_VecExpr<P_expr> a)
{
    typedef _bz_VecExprUnaryOp<_bz_VecExpr<P_expr>,
        _bz_negate<_bz_typename P_expr::T_numtype> > T_expr;
    return _bz_VecExpr<T_expr>(T_expr(a));
}

template<class P_numtype>
inline
_bz_VecExpr<_bz_VecExprUnaryOp<VectorIterConst<P_numtype>,
    _bz_negate<P_numtype> > >
operator-(const Vector<P_numtype>& a)
{
    typedef _bz_VecExprUnaryOp<VectorIterConst<P_numtype>,
        _bz_negate<P_numtype> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(a.begin()));
}

inline
_bz_VecExpr<_bz_VecExprUnaryOp<Range, _bz_negate<Range::T_numtype> > >
operator-(Range r)
{
    typedef _bz_VecExprUnaryOp<Range, _bz_negate<Range::T_numtype> > T_expr;
    
    return _bz_VecExpr<T_expr>(T_expr(r));
}


// NEEDS_WORK: implement operator- for Range, VectorPick, TinyVector

BZ_NAMESPACE_END

#ifndef BZ_TINYVEC_H
 #include <blitz/tinyvec.h>
#endif

#endif // BZ_VECEXPR_H
