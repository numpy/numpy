/*
 * $Id$
 *
 * Copyright (C) 1997 Todd Veldhuizen <tveldhui@oonumerics.org>
 * All rights reserved.  Please see <blitz/blitz.h> for terms and
 * conditions of use.
 *
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
 * Revision 1.1.1.1  2000/06/19 12:26:08  tveldhui
 * Imported sources
 *
 * Revision 1.4  1998/03/14 00:04:47  tveldhui
 * 0.2-alpha-05
 *
 * Revision 1.3  1997/07/16 14:51:20  tveldhui
 * Update: Alpha release 0.2 (Arrays)
 *
 * Revision 1.2  1997/01/24 14:42:00  tveldhui
 * Periodic RCS update
 *
 */

#ifndef BZ_VECDOT_CC
#define BZ_VECDOT_CC

#ifndef BZ_VECGLOBS_H
 #error <blitz/vecdot.cc> must be included via <blitz/vecglobs.h>
#endif

BZ_NAMESPACE(blitz)

template<class P1, class P2>
inline
BZ_SUMTYPE(BZ_PROMOTE(_bz_typename P1::T_numtype, _bz_typename P2::T_numtype))
_bz_dot(P1 vector1, P2 vector2)
{
    BZPRECONDITION(vector1._bz_suggestLength() == vector2._bz_suggestLength());

    typedef BZ_SUMTYPE(BZ_PROMOTE(_bz_typename P1::T_numtype,
        _bz_typename P2::T_numtype))  T_sumtype;

    T_sumtype sum = 0;
    int length = vector1._bz_suggestLength();

    if (vector1._bz_hasFastAccess() && vector2._bz_hasFastAccess())
    {
        for (int i=0; i < length; ++i)
            sum += vector1._bz_fastAccess(i) 
                * vector2._bz_fastAccess(i);
    }
    else {
        for (int i=0; i < length; ++i)
            sum += vector1[i] * vector2[i];
    }

    return sum;
}


// dot()
template<class P_numtype1, class P_numtype2>
inline
BZ_SUMTYPE(BZ_PROMOTE(P_numtype1,P_numtype2))
dot(const Vector<P_numtype1>& a, const Vector<P_numtype2>& b)
{
    return _bz_dot(a, b);
}

// dot(expr,expr)
template<class P_expr1, class P_expr2>
inline
BZ_SUMTYPE(BZ_PROMOTE(_bz_typename P_expr1::T_numtype,
    _bz_typename P_expr2::T_numtype))
dot(_bz_VecExpr<P_expr1> expr1, _bz_VecExpr<P_expr2> expr2)
{
    return _bz_dot(expr1, expr2);
}

// dot(expr,vec)
template<class P_expr1, class P_numtype2>
inline
BZ_SUMTYPE(BZ_PROMOTE(_bz_typename P_expr1::T_numtype, P_numtype2))
dot(_bz_VecExpr<P_expr1> expr1, const Vector<P_numtype2>& vector2)
{
    return _bz_dot(vector2, expr1);
}

// dot(vec,expr)
template<class P_numtype1, class P_expr2>
inline
BZ_SUMTYPE(BZ_PROMOTE(P_numtype1, _bz_typename P_expr2::T_numtype))
dot(const Vector<P_numtype1>& vector1, _bz_VecExpr<P_expr2> expr2)
{
    return _bz_dot(vector1, expr2);
}

// dot(vec,vecpick)
template<class P_numtype1, class P_numtype2>
inline
BZ_SUMTYPE(BZ_PROMOTE(P_numtype1, P_numtype2))
dot(const Vector<P_numtype1>& vector1, const VectorPick<P_numtype2>& vector2)
{
    return _bz_dot(vector1, vector2);
}

// dot(vecpick,vec)
template<class P_numtype1, class P_numtype2>
inline
BZ_SUMTYPE(BZ_PROMOTE(P_numtype1, P_numtype2))
dot(const VectorPick<P_numtype1>& vector1, const Vector<P_numtype2>& vector2)
{
    return _bz_dot(vector1, vector2);
}

// dot(vecpick,vecpick)
template<class P_numtype1, class P_numtype2>
inline
BZ_SUMTYPE(BZ_PROMOTE(P_numtype1, P_numtype2))
dot(const VectorPick<P_numtype1>& vector1, const VectorPick<P_numtype2>& vector2)
{
    return _bz_dot(vector1, vector2);
}

// dot(expr, vecpick)
template<class P_expr1, class P_numtype2>
inline
BZ_SUMTYPE(BZ_PROMOTE(_bz_typename P_expr1::T_numtype, P_numtype2))
dot(_bz_VecExpr<P_expr1> expr1, const VectorPick<P_numtype2>& vector2)
{
    return _bz_dot(expr1, vector2);
}

// dot(vecpick, expr)
template<class P_numtype1, class P_expr2>
inline
BZ_SUMTYPE(BZ_PROMOTE(P_numtype1, _bz_typename P_expr2::T_numtype))
dot(const VectorPick<P_numtype1>& vector1, _bz_VecExpr<P_expr2> expr2)
{
    return _bz_dot(vector1, expr2);
}

BZ_NAMESPACE_END

#endif // BZ_VECDOT_CC

