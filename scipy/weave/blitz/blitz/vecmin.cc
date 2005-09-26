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

#ifndef BZ_VECMIN_CC
#define BZ_VECMIN_CC

#ifndef BZ_VECGLOBS_H
 #error <blitz/vecmin.cc> must be included via <blitz/vecglobs.h>
#endif

BZ_NAMESPACE(blitz)

template<class P_expr>
inline
Extremum<_bz_typename P_expr::T_numtype, int> _bz_vec_min(P_expr vector)
{
    typedef _bz_typename P_expr::T_numtype T_numtype;

    T_numtype minValue = vector(0);
    int minIndex = 0;
    int length = vector._bz_suggestLength();

    if (vector._bz_hasFastAccess())
    {
        for (int i=1; i < length; ++i)
        {
            T_numtype value = vector._bz_fastAccess(i);
            if (value < minValue)
            {
                minValue = value;
                minIndex = i;
            }
        }
    }
    else {
        for (int i=1; i < length; ++i)
        {
            T_numtype value = vector(i);
            if (value < minValue)
            {
                minValue = value;
                minIndex = i;
            }
        }
    }

    return Extremum<T_numtype, int>(minValue, minIndex);
}

// min(vector)
template<class P_numtype>
inline
Extremum<P_numtype,int> min(const Vector<P_numtype>& x)
{
    return _bz_vec_min(x._bz_asVecExpr());
}

// min(expr)
template<class P_expr>
inline
Extremum<_bz_typename P_expr::T_numtype,int> min(_bz_VecExpr<P_expr> x)
{
    return _bz_vec_min(x);
}

// min(vecpick)
template<class P_numtype>
inline
Extremum<P_numtype, int> min(const VectorPick<P_numtype>& x)
{
    return _bz_vec_min(x._bz_asVecExpr());
}

// min(TinyVector)
template<class P_numtype, int N_length>
inline
Extremum<P_numtype, int> min(const TinyVector<P_numtype, N_length>& x)
{
    return _bz_vec_min(x._bz_asVecExpr());
}

// minIndex(vector)
template<class P_numtype>
inline
int  minIndex(const Vector<P_numtype>& x)
{
    return _bz_vec_min(x._bz_asVecExpr()).index();
}

// maxIndex(expr)
template<class P_expr>
inline
int  minIndex(_bz_VecExpr<P_expr> x)
{
    return _bz_vec_min(x).index();
}

// minIndex(vecpick)
template<class P_numtype>
int  minIndex(const VectorPick<P_numtype>& x)
{
    return _bz_vec_min(x._bz_asVecExpr()).index();
}

// minIndex(TinyVector)
template<class P_numtype, int N_length>
int minIndex(const TinyVector<P_numtype, N_length>& x)
{
    return _bz_vec_min(x._bz_asVecExpr()).index();
}

// minValue(vector)
template<class P_numtype>
inline
int  minValue(const Vector<P_numtype>& x)
{
    return _bz_vec_min(x._bz_asVecExpr()).value();
}

// minValue(expr)
template<class P_expr>
inline
int  minValue(_bz_VecExpr<P_expr> x)
{
    return _bz_vec_min(x).value();
}

// minValue(vecpick)
template<class P_numtype>
int  minValue(const VectorPick<P_numtype>& x)
{
    return _bz_vec_min(x._bz_asVecExpr()).value();
}

// minValue(TinyVector)
template<class P_numtype, int N_length>
int  minValue(const TinyVector<P_numtype, N_length>& x)
{
    return _bz_vec_min(x._bz_asVecExpr()).value();
}

BZ_NAMESPACE_END

#endif // BZ_VECMIN_CC

