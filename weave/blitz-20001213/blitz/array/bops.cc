/***************************************************************************
 * blitz/arraybops.cc	Array expression templates (2 operands)
 *
 * $Id$
 *
 * Copyright (C) 1997 Todd Veldhuizen <tveldhui@oonumerics.org>
 * All rights reserved.   Please see <blitz/blitz.h> for terms and
 * conditions of use.
 *
 * Suggestions:          blitz-dev@oonumerics.org
 * Bugs:                 blitz-bugs@oonumerics.org
 *
 * For more information, please see the Blitz++ Home Page:
 *    http://oonumerics.org/blitz/
 *
 ***************************************************************************
 * $Log$
 * Revision 1.1  2002/09/12 07:02:06  eric
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
 * Revision 1.1.1.1  2000/06/19 12:26:14  tveldhui
 * Imported sources
 *
 * Revision 1.2  1998/03/14 00:04:47  tveldhui
 * 0.2-alpha-05
 *
 */ 

// Generated source file.  Do not edit. 
// genarrbops.cpp Aug  7 1997 15:04:14

#ifndef BZ_ARRAYBOPS_CC
#define BZ_ARRAYBOPS_CC

#ifndef BZ_ARRAYEXPR_H
 #error <blitz/arraybops.cc> must be included after <blitz/arrayexpr.h>
#endif

BZ_NAMESPACE(blitz)

/****************************************************************************
 * Addition Operators
 ****************************************************************************/

// Array<T_numtype1, N_rank1> + Array<T_numtype2, N_rank2>
template<class T_numtype1, int N_rank1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Add<T_numtype1, T_numtype2 > > >
operator+(const Array<T_numtype1, N_rank1>& d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Add<T_numtype1, T_numtype2> >
      (d1.begin(), 
      d2.begin());
}

// Array<T_numtype1, N_rank1> + _bz_ArrayExpr<P_expr2>
template<class T_numtype1, int N_rank1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>,
      Add<T_numtype1, _bz_typename P_expr2::T_numtype > > >
operator+(const Array<T_numtype1, N_rank1>& d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>, 
      Add<T_numtype1, _bz_typename P_expr2::T_numtype> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> + IndexPlaceholder<N_index2>
template<class T_numtype1, int N_rank1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>,
      Add<T_numtype1, int > > >
operator+(const Array<T_numtype1, N_rank1>& d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>, 
      Add<T_numtype1, int> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> + int
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>,
      Add<T_numtype1, int > > >
operator+(const Array<T_numtype1, N_rank1>& d1, 
      int d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>, 
      Add<T_numtype1, int> >
      (d1.begin(), 
      _bz_ArrayExprConstant<int>(d2));
}

// Array<T_numtype1, N_rank1> + float
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<float>,
      Add<T_numtype1, float > > >
operator+(const Array<T_numtype1, N_rank1>& d1, 
      float d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<float>, 
      Add<T_numtype1, float> >
      (d1.begin(), 
      _bz_ArrayExprConstant<float>(d2));
}

// Array<T_numtype1, N_rank1> + double
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<double>,
      Add<T_numtype1, double > > >
operator+(const Array<T_numtype1, N_rank1>& d1, 
      double d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<double>, 
      Add<T_numtype1, double> >
      (d1.begin(), 
      _bz_ArrayExprConstant<double>(d2));
}

// Array<T_numtype1, N_rank1> + long double
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<long double>,
      Add<T_numtype1, long double > > >
operator+(const Array<T_numtype1, N_rank1>& d1, 
      long double d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<long double>, 
      Add<T_numtype1, long double> >
      (d1.begin(), 
      _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
// Array<T_numtype1, N_rank1> + complex<T2>
template<class T_numtype1, int N_rank1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<complex<T2> > ,
      Add<T_numtype1, complex<T2>  > > >
operator+(const Array<T_numtype1, N_rank1>& d1, 
      complex<T2> d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<complex<T2> > , 
      Add<T_numtype1, complex<T2> > >
      (d1.begin(), 
      _bz_ArrayExprConstant<complex<T2> > (d2));
}
#endif // BZ_HAVE_COMPLEX

// _bz_ArrayExpr<P_expr1> + Array<T_numtype2, N_rank2>
template<class P_expr1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Add<_bz_typename P_expr1::T_numtype, T_numtype2 > > >
operator+(_bz_ArrayExpr<P_expr1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Add<_bz_typename P_expr1::T_numtype, T_numtype2> >
      (d1, 
      d2.begin());
}

// _bz_ArrayExpr<P_expr1> + _bz_ArrayExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>,
      Add<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype > > >
operator+(_bz_ArrayExpr<P_expr1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>, 
      Add<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> + IndexPlaceholder<N_index2>
template<class P_expr1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>,
      Add<_bz_typename P_expr1::T_numtype, int > > >
operator+(_bz_ArrayExpr<P_expr1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>, 
      Add<_bz_typename P_expr1::T_numtype, int> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> + int
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>,
      Add<_bz_typename P_expr1::T_numtype, int > > >
operator+(_bz_ArrayExpr<P_expr1> d1, 
      int d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>, 
      Add<_bz_typename P_expr1::T_numtype, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// _bz_ArrayExpr<P_expr1> + float
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<float>,
      Add<_bz_typename P_expr1::T_numtype, float > > >
operator+(_bz_ArrayExpr<P_expr1> d1, 
      float d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<float>, 
      Add<_bz_typename P_expr1::T_numtype, float> >
      (d1, 
      _bz_ArrayExprConstant<float>(d2));
}

// _bz_ArrayExpr<P_expr1> + double
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<double>,
      Add<_bz_typename P_expr1::T_numtype, double > > >
operator+(_bz_ArrayExpr<P_expr1> d1, 
      double d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<double>, 
      Add<_bz_typename P_expr1::T_numtype, double> >
      (d1, 
      _bz_ArrayExprConstant<double>(d2));
}

// _bz_ArrayExpr<P_expr1> + long double
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<long double>,
      Add<_bz_typename P_expr1::T_numtype, long double > > >
operator+(_bz_ArrayExpr<P_expr1> d1, 
      long double d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<long double>, 
      Add<_bz_typename P_expr1::T_numtype, long double> >
      (d1, 
      _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
// _bz_ArrayExpr<P_expr1> + complex<T2>
template<class P_expr1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<complex<T2> > ,
      Add<_bz_typename P_expr1::T_numtype, complex<T2>  > > >
operator+(_bz_ArrayExpr<P_expr1> d1, 
      complex<T2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<complex<T2> > , 
      Add<_bz_typename P_expr1::T_numtype, complex<T2> > >
      (d1, 
      _bz_ArrayExprConstant<complex<T2> > (d2));
}
#endif // BZ_HAVE_COMPLEX

// IndexPlaceholder<N_index1> + Array<T_numtype2, N_rank2>
template<int N_index1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Add<int, T_numtype2 > > >
operator+(IndexPlaceholder<N_index1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Add<int, T_numtype2> >
      (d1, 
      d2.begin());
}

// IndexPlaceholder<N_index1> + _bz_ArrayExpr<P_expr2>
template<int N_index1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>,
      Add<int, _bz_typename P_expr2::T_numtype > > >
operator+(IndexPlaceholder<N_index1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>, 
      Add<int, _bz_typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> + IndexPlaceholder<N_index2>
template<int N_index1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>,
      Add<int, int > > >
operator+(IndexPlaceholder<N_index1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>, 
      Add<int, int> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> + int
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>,
      Add<int, int > > >
operator+(IndexPlaceholder<N_index1> d1, 
      int d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>, 
      Add<int, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// IndexPlaceholder<N_index1> + float
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<float>,
      Add<int, float > > >
operator+(IndexPlaceholder<N_index1> d1, 
      float d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<float>, 
      Add<int, float> >
      (d1, 
      _bz_ArrayExprConstant<float>(d2));
}

// IndexPlaceholder<N_index1> + double
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<double>,
      Add<int, double > > >
operator+(IndexPlaceholder<N_index1> d1, 
      double d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<double>, 
      Add<int, double> >
      (d1, 
      _bz_ArrayExprConstant<double>(d2));
}

// IndexPlaceholder<N_index1> + long double
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<long double>,
      Add<int, long double > > >
operator+(IndexPlaceholder<N_index1> d1, 
      long double d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<long double>, 
      Add<int, long double> >
      (d1, 
      _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
// IndexPlaceholder<N_index1> + complex<T2>
template<int N_index1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<complex<T2> > ,
      Add<int, complex<T2>  > > >
operator+(IndexPlaceholder<N_index1> d1, 
      complex<T2> d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<complex<T2> > , 
      Add<int, complex<T2> > >
      (d1, 
      _bz_ArrayExprConstant<complex<T2> > (d2));
}
#endif // BZ_HAVE_COMPLEX

// int + Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Add<int, T_numtype2 > > >
operator+(int d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Add<int, T_numtype2> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2.begin());
}

// int + _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>,
      Add<int, _bz_typename P_expr2::T_numtype > > >
operator+(int d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>, 
      Add<int, _bz_typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2);
}

// int + IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      IndexPlaceholder<N_index2>,
      Add<int, int > > >
operator+(int d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      IndexPlaceholder<N_index2>, 
      Add<int, int> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2);
}

// float + Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<float>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Add<float, T_numtype2 > > >
operator+(float d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<float>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Add<float, T_numtype2> >
      (_bz_ArrayExprConstant<float>(d1), 
      d2.begin());
}

// float + _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<float>, 
      _bz_ArrayExpr<P_expr2>,
      Add<float, _bz_typename P_expr2::T_numtype > > >
operator+(float d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<float>, 
      _bz_ArrayExpr<P_expr2>, 
      Add<float, _bz_typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<float>(d1), 
      d2);
}

// float + IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<float>, 
      IndexPlaceholder<N_index2>,
      Add<float, int > > >
operator+(float d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<float>, 
      IndexPlaceholder<N_index2>, 
      Add<float, int> >
      (_bz_ArrayExprConstant<float>(d1), 
      d2);
}

// double + Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<double>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Add<double, T_numtype2 > > >
operator+(double d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<double>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Add<double, T_numtype2> >
      (_bz_ArrayExprConstant<double>(d1), 
      d2.begin());
}

// double + _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<double>, 
      _bz_ArrayExpr<P_expr2>,
      Add<double, _bz_typename P_expr2::T_numtype > > >
operator+(double d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<double>, 
      _bz_ArrayExpr<P_expr2>, 
      Add<double, _bz_typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<double>(d1), 
      d2);
}

// double + IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<double>, 
      IndexPlaceholder<N_index2>,
      Add<double, int > > >
operator+(double d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<double>, 
      IndexPlaceholder<N_index2>, 
      Add<double, int> >
      (_bz_ArrayExprConstant<double>(d1), 
      d2);
}

// long double + Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<long double>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Add<long double, T_numtype2 > > >
operator+(long double d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<long double>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Add<long double, T_numtype2> >
      (_bz_ArrayExprConstant<long double>(d1), 
      d2.begin());
}

// long double + _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<long double>, 
      _bz_ArrayExpr<P_expr2>,
      Add<long double, _bz_typename P_expr2::T_numtype > > >
operator+(long double d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<long double>, 
      _bz_ArrayExpr<P_expr2>, 
      Add<long double, _bz_typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<long double>(d1), 
      d2);
}

// long double + IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<long double>, 
      IndexPlaceholder<N_index2>,
      Add<long double, int > > >
operator+(long double d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<long double>, 
      IndexPlaceholder<N_index2>, 
      Add<long double, int> >
      (_bz_ArrayExprConstant<long double>(d1), 
      d2);
}

#ifdef BZ_HAVE_COMPLEX
// complex<T1> + Array<T_numtype2, N_rank2>
template<class T1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<complex<T1> > , 
      ArrayIterator<T_numtype2, N_rank2>,
      Add<complex<T1> , T_numtype2 > > >
operator+(complex<T1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<complex<T1> > , 
      ArrayIterator<T_numtype2, N_rank2>, 
      Add<complex<T1> , T_numtype2> >
      (_bz_ArrayExprConstant<complex<T1> > (d1), 
      d2.begin());
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX
// complex<T1> + _bz_ArrayExpr<P_expr2>
template<class T1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<complex<T1> > , 
      _bz_ArrayExpr<P_expr2>,
      Add<complex<T1> , _bz_typename P_expr2::T_numtype > > >
operator+(complex<T1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<complex<T1> > , 
      _bz_ArrayExpr<P_expr2>, 
      Add<complex<T1> , _bz_typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<complex<T1> > (d1), 
      d2);
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX
// complex<T1> + IndexPlaceholder<N_index2>
template<class T1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<complex<T1> > , 
      IndexPlaceholder<N_index2>,
      Add<complex<T1> , int > > >
operator+(complex<T1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<complex<T1> > , 
      IndexPlaceholder<N_index2>, 
      Add<complex<T1> , int> >
      (_bz_ArrayExprConstant<complex<T1> > (d1), 
      d2);
}
#endif // BZ_HAVE_COMPLEX
/****************************************************************************
 * Subtraction Operators
 ****************************************************************************/

// Array<T_numtype1, N_rank1> - Array<T_numtype2, N_rank2>
template<class T_numtype1, int N_rank1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Subtract<T_numtype1, T_numtype2 > > >
operator-(const Array<T_numtype1, N_rank1>& d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Subtract<T_numtype1, T_numtype2> >
      (d1.begin(), 
      d2.begin());
}

// Array<T_numtype1, N_rank1> - _bz_ArrayExpr<P_expr2>
template<class T_numtype1, int N_rank1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>,
      Subtract<T_numtype1, _bz_typename P_expr2::T_numtype > > >
operator-(const Array<T_numtype1, N_rank1>& d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>, 
      Subtract<T_numtype1, _bz_typename P_expr2::T_numtype> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> - IndexPlaceholder<N_index2>
template<class T_numtype1, int N_rank1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>,
      Subtract<T_numtype1, int > > >
operator-(const Array<T_numtype1, N_rank1>& d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>, 
      Subtract<T_numtype1, int> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> - int
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>,
      Subtract<T_numtype1, int > > >
operator-(const Array<T_numtype1, N_rank1>& d1, 
      int d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>, 
      Subtract<T_numtype1, int> >
      (d1.begin(), 
      _bz_ArrayExprConstant<int>(d2));
}

// Array<T_numtype1, N_rank1> - float
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<float>,
      Subtract<T_numtype1, float > > >
operator-(const Array<T_numtype1, N_rank1>& d1, 
      float d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<float>, 
      Subtract<T_numtype1, float> >
      (d1.begin(), 
      _bz_ArrayExprConstant<float>(d2));
}

// Array<T_numtype1, N_rank1> - double
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<double>,
      Subtract<T_numtype1, double > > >
operator-(const Array<T_numtype1, N_rank1>& d1, 
      double d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<double>, 
      Subtract<T_numtype1, double> >
      (d1.begin(), 
      _bz_ArrayExprConstant<double>(d2));
}

// Array<T_numtype1, N_rank1> - long double
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<long double>,
      Subtract<T_numtype1, long double > > >
operator-(const Array<T_numtype1, N_rank1>& d1, 
      long double d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<long double>, 
      Subtract<T_numtype1, long double> >
      (d1.begin(), 
      _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
// Array<T_numtype1, N_rank1> - complex<T2>
template<class T_numtype1, int N_rank1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<complex<T2> > ,
      Subtract<T_numtype1, complex<T2>  > > >
operator-(const Array<T_numtype1, N_rank1>& d1, 
      complex<T2> d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<complex<T2> > , 
      Subtract<T_numtype1, complex<T2> > >
      (d1.begin(), 
      _bz_ArrayExprConstant<complex<T2> > (d2));
}
#endif // BZ_HAVE_COMPLEX

// _bz_ArrayExpr<P_expr1> - Array<T_numtype2, N_rank2>
template<class P_expr1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Subtract<_bz_typename P_expr1::T_numtype, T_numtype2 > > >
operator-(_bz_ArrayExpr<P_expr1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Subtract<_bz_typename P_expr1::T_numtype, T_numtype2> >
      (d1, 
      d2.begin());
}

// _bz_ArrayExpr<P_expr1> - _bz_ArrayExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>,
      Subtract<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype > > >
operator-(_bz_ArrayExpr<P_expr1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>, 
      Subtract<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> - IndexPlaceholder<N_index2>
template<class P_expr1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>,
      Subtract<_bz_typename P_expr1::T_numtype, int > > >
operator-(_bz_ArrayExpr<P_expr1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>, 
      Subtract<_bz_typename P_expr1::T_numtype, int> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> - int
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>,
      Subtract<_bz_typename P_expr1::T_numtype, int > > >
operator-(_bz_ArrayExpr<P_expr1> d1, 
      int d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>, 
      Subtract<_bz_typename P_expr1::T_numtype, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// _bz_ArrayExpr<P_expr1> - float
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<float>,
      Subtract<_bz_typename P_expr1::T_numtype, float > > >
operator-(_bz_ArrayExpr<P_expr1> d1, 
      float d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<float>, 
      Subtract<_bz_typename P_expr1::T_numtype, float> >
      (d1, 
      _bz_ArrayExprConstant<float>(d2));
}

// _bz_ArrayExpr<P_expr1> - double
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<double>,
      Subtract<_bz_typename P_expr1::T_numtype, double > > >
operator-(_bz_ArrayExpr<P_expr1> d1, 
      double d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<double>, 
      Subtract<_bz_typename P_expr1::T_numtype, double> >
      (d1, 
      _bz_ArrayExprConstant<double>(d2));
}

// _bz_ArrayExpr<P_expr1> - long double
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<long double>,
      Subtract<_bz_typename P_expr1::T_numtype, long double > > >
operator-(_bz_ArrayExpr<P_expr1> d1, 
      long double d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<long double>, 
      Subtract<_bz_typename P_expr1::T_numtype, long double> >
      (d1, 
      _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
// _bz_ArrayExpr<P_expr1> - complex<T2>
template<class P_expr1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<complex<T2> > ,
      Subtract<_bz_typename P_expr1::T_numtype, complex<T2>  > > >
operator-(_bz_ArrayExpr<P_expr1> d1, 
      complex<T2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<complex<T2> > , 
      Subtract<_bz_typename P_expr1::T_numtype, complex<T2> > >
      (d1, 
      _bz_ArrayExprConstant<complex<T2> > (d2));
}
#endif // BZ_HAVE_COMPLEX

// IndexPlaceholder<N_index1> - Array<T_numtype2, N_rank2>
template<int N_index1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Subtract<int, T_numtype2 > > >
operator-(IndexPlaceholder<N_index1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Subtract<int, T_numtype2> >
      (d1, 
      d2.begin());
}

// IndexPlaceholder<N_index1> - _bz_ArrayExpr<P_expr2>
template<int N_index1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>,
      Subtract<int, _bz_typename P_expr2::T_numtype > > >
operator-(IndexPlaceholder<N_index1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>, 
      Subtract<int, _bz_typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> - IndexPlaceholder<N_index2>
template<int N_index1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>,
      Subtract<int, int > > >
operator-(IndexPlaceholder<N_index1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>, 
      Subtract<int, int> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> - int
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>,
      Subtract<int, int > > >
operator-(IndexPlaceholder<N_index1> d1, 
      int d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>, 
      Subtract<int, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// IndexPlaceholder<N_index1> - float
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<float>,
      Subtract<int, float > > >
operator-(IndexPlaceholder<N_index1> d1, 
      float d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<float>, 
      Subtract<int, float> >
      (d1, 
      _bz_ArrayExprConstant<float>(d2));
}

// IndexPlaceholder<N_index1> - double
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<double>,
      Subtract<int, double > > >
operator-(IndexPlaceholder<N_index1> d1, 
      double d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<double>, 
      Subtract<int, double> >
      (d1, 
      _bz_ArrayExprConstant<double>(d2));
}

// IndexPlaceholder<N_index1> - long double
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<long double>,
      Subtract<int, long double > > >
operator-(IndexPlaceholder<N_index1> d1, 
      long double d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<long double>, 
      Subtract<int, long double> >
      (d1, 
      _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
// IndexPlaceholder<N_index1> - complex<T2>
template<int N_index1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<complex<T2> > ,
      Subtract<int, complex<T2>  > > >
operator-(IndexPlaceholder<N_index1> d1, 
      complex<T2> d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<complex<T2> > , 
      Subtract<int, complex<T2> > >
      (d1, 
      _bz_ArrayExprConstant<complex<T2> > (d2));
}
#endif // BZ_HAVE_COMPLEX

// int - Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Subtract<int, T_numtype2 > > >
operator-(int d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Subtract<int, T_numtype2> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2.begin());
}

// int - _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>,
      Subtract<int, _bz_typename P_expr2::T_numtype > > >
operator-(int d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>, 
      Subtract<int, _bz_typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2);
}

// int - IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      IndexPlaceholder<N_index2>,
      Subtract<int, int > > >
operator-(int d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      IndexPlaceholder<N_index2>, 
      Subtract<int, int> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2);
}

// float - Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<float>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Subtract<float, T_numtype2 > > >
operator-(float d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<float>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Subtract<float, T_numtype2> >
      (_bz_ArrayExprConstant<float>(d1), 
      d2.begin());
}

// float - _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<float>, 
      _bz_ArrayExpr<P_expr2>,
      Subtract<float, _bz_typename P_expr2::T_numtype > > >
operator-(float d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<float>, 
      _bz_ArrayExpr<P_expr2>, 
      Subtract<float, _bz_typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<float>(d1), 
      d2);
}

// float - IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<float>, 
      IndexPlaceholder<N_index2>,
      Subtract<float, int > > >
operator-(float d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<float>, 
      IndexPlaceholder<N_index2>, 
      Subtract<float, int> >
      (_bz_ArrayExprConstant<float>(d1), 
      d2);
}

// double - Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<double>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Subtract<double, T_numtype2 > > >
operator-(double d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<double>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Subtract<double, T_numtype2> >
      (_bz_ArrayExprConstant<double>(d1), 
      d2.begin());
}

// double - _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<double>, 
      _bz_ArrayExpr<P_expr2>,
      Subtract<double, _bz_typename P_expr2::T_numtype > > >
operator-(double d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<double>, 
      _bz_ArrayExpr<P_expr2>, 
      Subtract<double, _bz_typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<double>(d1), 
      d2);
}

// double - IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<double>, 
      IndexPlaceholder<N_index2>,
      Subtract<double, int > > >
operator-(double d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<double>, 
      IndexPlaceholder<N_index2>, 
      Subtract<double, int> >
      (_bz_ArrayExprConstant<double>(d1), 
      d2);
}

// long double - Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<long double>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Subtract<long double, T_numtype2 > > >
operator-(long double d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<long double>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Subtract<long double, T_numtype2> >
      (_bz_ArrayExprConstant<long double>(d1), 
      d2.begin());
}

// long double - _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<long double>, 
      _bz_ArrayExpr<P_expr2>,
      Subtract<long double, _bz_typename P_expr2::T_numtype > > >
operator-(long double d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<long double>, 
      _bz_ArrayExpr<P_expr2>, 
      Subtract<long double, _bz_typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<long double>(d1), 
      d2);
}

// long double - IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<long double>, 
      IndexPlaceholder<N_index2>,
      Subtract<long double, int > > >
operator-(long double d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<long double>, 
      IndexPlaceholder<N_index2>, 
      Subtract<long double, int> >
      (_bz_ArrayExprConstant<long double>(d1), 
      d2);
}

#ifdef BZ_HAVE_COMPLEX
// complex<T1> - Array<T_numtype2, N_rank2>
template<class T1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<complex<T1> > , 
      ArrayIterator<T_numtype2, N_rank2>,
      Subtract<complex<T1> , T_numtype2 > > >
operator-(complex<T1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<complex<T1> > , 
      ArrayIterator<T_numtype2, N_rank2>, 
      Subtract<complex<T1> , T_numtype2> >
      (_bz_ArrayExprConstant<complex<T1> > (d1), 
      d2.begin());
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX
// complex<T1> - _bz_ArrayExpr<P_expr2>
template<class T1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<complex<T1> > , 
      _bz_ArrayExpr<P_expr2>,
      Subtract<complex<T1> , _bz_typename P_expr2::T_numtype > > >
operator-(complex<T1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<complex<T1> > , 
      _bz_ArrayExpr<P_expr2>, 
      Subtract<complex<T1> , _bz_typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<complex<T1> > (d1), 
      d2);
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX
// complex<T1> - IndexPlaceholder<N_index2>
template<class T1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<complex<T1> > , 
      IndexPlaceholder<N_index2>,
      Subtract<complex<T1> , int > > >
operator-(complex<T1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<complex<T1> > , 
      IndexPlaceholder<N_index2>, 
      Subtract<complex<T1> , int> >
      (_bz_ArrayExprConstant<complex<T1> > (d1), 
      d2);
}
#endif // BZ_HAVE_COMPLEX
/****************************************************************************
 * Multiplication Operators
 ****************************************************************************/

// Array<T_numtype1, N_rank1> * Array<T_numtype2, N_rank2>
template<class T_numtype1, int N_rank1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Multiply<T_numtype1, T_numtype2 > > >
operator*(const Array<T_numtype1, N_rank1>& d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Multiply<T_numtype1, T_numtype2> >
      (d1.begin(), 
      d2.begin());
}

// Array<T_numtype1, N_rank1> * _bz_ArrayExpr<P_expr2>
template<class T_numtype1, int N_rank1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>,
      Multiply<T_numtype1, _bz_typename P_expr2::T_numtype > > >
operator*(const Array<T_numtype1, N_rank1>& d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>, 
      Multiply<T_numtype1, _bz_typename P_expr2::T_numtype> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> * IndexPlaceholder<N_index2>
template<class T_numtype1, int N_rank1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>,
      Multiply<T_numtype1, int > > >
operator*(const Array<T_numtype1, N_rank1>& d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>, 
      Multiply<T_numtype1, int> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> * int
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>,
      Multiply<T_numtype1, int > > >
operator*(const Array<T_numtype1, N_rank1>& d1, 
      int d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>, 
      Multiply<T_numtype1, int> >
      (d1.begin(), 
      _bz_ArrayExprConstant<int>(d2));
}

// Array<T_numtype1, N_rank1> * float
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<float>,
      Multiply<T_numtype1, float > > >
operator*(const Array<T_numtype1, N_rank1>& d1, 
      float d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<float>, 
      Multiply<T_numtype1, float> >
      (d1.begin(), 
      _bz_ArrayExprConstant<float>(d2));
}

// Array<T_numtype1, N_rank1> * double
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<double>,
      Multiply<T_numtype1, double > > >
operator*(const Array<T_numtype1, N_rank1>& d1, 
      double d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<double>, 
      Multiply<T_numtype1, double> >
      (d1.begin(), 
      _bz_ArrayExprConstant<double>(d2));
}

// Array<T_numtype1, N_rank1> * long double
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<long double>,
      Multiply<T_numtype1, long double > > >
operator*(const Array<T_numtype1, N_rank1>& d1, 
      long double d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<long double>, 
      Multiply<T_numtype1, long double> >
      (d1.begin(), 
      _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
// Array<T_numtype1, N_rank1> * complex<T2>
template<class T_numtype1, int N_rank1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<complex<T2> > ,
      Multiply<T_numtype1, complex<T2>  > > >
operator*(const Array<T_numtype1, N_rank1>& d1, 
      complex<T2> d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<complex<T2> > , 
      Multiply<T_numtype1, complex<T2> > >
      (d1.begin(), 
      _bz_ArrayExprConstant<complex<T2> > (d2));
}
#endif // BZ_HAVE_COMPLEX

// _bz_ArrayExpr<P_expr1> * Array<T_numtype2, N_rank2>
template<class P_expr1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Multiply<_bz_typename P_expr1::T_numtype, T_numtype2 > > >
operator*(_bz_ArrayExpr<P_expr1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Multiply<_bz_typename P_expr1::T_numtype, T_numtype2> >
      (d1, 
      d2.begin());
}

// _bz_ArrayExpr<P_expr1> * _bz_ArrayExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>,
      Multiply<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype > > >
operator*(_bz_ArrayExpr<P_expr1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>, 
      Multiply<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> * IndexPlaceholder<N_index2>
template<class P_expr1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>,
      Multiply<_bz_typename P_expr1::T_numtype, int > > >
operator*(_bz_ArrayExpr<P_expr1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>, 
      Multiply<_bz_typename P_expr1::T_numtype, int> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> * int
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>,
      Multiply<_bz_typename P_expr1::T_numtype, int > > >
operator*(_bz_ArrayExpr<P_expr1> d1, 
      int d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>, 
      Multiply<_bz_typename P_expr1::T_numtype, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// _bz_ArrayExpr<P_expr1> * float
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<float>,
      Multiply<_bz_typename P_expr1::T_numtype, float > > >
operator*(_bz_ArrayExpr<P_expr1> d1, 
      float d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<float>, 
      Multiply<_bz_typename P_expr1::T_numtype, float> >
      (d1, 
      _bz_ArrayExprConstant<float>(d2));
}

// _bz_ArrayExpr<P_expr1> * double
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<double>,
      Multiply<_bz_typename P_expr1::T_numtype, double > > >
operator*(_bz_ArrayExpr<P_expr1> d1, 
      double d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<double>, 
      Multiply<_bz_typename P_expr1::T_numtype, double> >
      (d1, 
      _bz_ArrayExprConstant<double>(d2));
}

// _bz_ArrayExpr<P_expr1> * long double
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<long double>,
      Multiply<_bz_typename P_expr1::T_numtype, long double > > >
operator*(_bz_ArrayExpr<P_expr1> d1, 
      long double d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<long double>, 
      Multiply<_bz_typename P_expr1::T_numtype, long double> >
      (d1, 
      _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
// _bz_ArrayExpr<P_expr1> * complex<T2>
template<class P_expr1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<complex<T2> > ,
      Multiply<_bz_typename P_expr1::T_numtype, complex<T2>  > > >
operator*(_bz_ArrayExpr<P_expr1> d1, 
      complex<T2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<complex<T2> > , 
      Multiply<_bz_typename P_expr1::T_numtype, complex<T2> > >
      (d1, 
      _bz_ArrayExprConstant<complex<T2> > (d2));
}
#endif // BZ_HAVE_COMPLEX

// IndexPlaceholder<N_index1> * Array<T_numtype2, N_rank2>
template<int N_index1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Multiply<int, T_numtype2 > > >
operator*(IndexPlaceholder<N_index1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Multiply<int, T_numtype2> >
      (d1, 
      d2.begin());
}

// IndexPlaceholder<N_index1> * _bz_ArrayExpr<P_expr2>
template<int N_index1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>,
      Multiply<int, _bz_typename P_expr2::T_numtype > > >
operator*(IndexPlaceholder<N_index1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>, 
      Multiply<int, _bz_typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> * IndexPlaceholder<N_index2>
template<int N_index1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>,
      Multiply<int, int > > >
operator*(IndexPlaceholder<N_index1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>, 
      Multiply<int, int> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> * int
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>,
      Multiply<int, int > > >
operator*(IndexPlaceholder<N_index1> d1, 
      int d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>, 
      Multiply<int, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// IndexPlaceholder<N_index1> * float
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<float>,
      Multiply<int, float > > >
operator*(IndexPlaceholder<N_index1> d1, 
      float d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<float>, 
      Multiply<int, float> >
      (d1, 
      _bz_ArrayExprConstant<float>(d2));
}

// IndexPlaceholder<N_index1> * double
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<double>,
      Multiply<int, double > > >
operator*(IndexPlaceholder<N_index1> d1, 
      double d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<double>, 
      Multiply<int, double> >
      (d1, 
      _bz_ArrayExprConstant<double>(d2));
}

// IndexPlaceholder<N_index1> * long double
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<long double>,
      Multiply<int, long double > > >
operator*(IndexPlaceholder<N_index1> d1, 
      long double d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<long double>, 
      Multiply<int, long double> >
      (d1, 
      _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
// IndexPlaceholder<N_index1> * complex<T2>
template<int N_index1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<complex<T2> > ,
      Multiply<int, complex<T2>  > > >
operator*(IndexPlaceholder<N_index1> d1, 
      complex<T2> d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<complex<T2> > , 
      Multiply<int, complex<T2> > >
      (d1, 
      _bz_ArrayExprConstant<complex<T2> > (d2));
}
#endif // BZ_HAVE_COMPLEX

// int * Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Multiply<int, T_numtype2 > > >
operator*(int d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Multiply<int, T_numtype2> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2.begin());
}

// int * _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>,
      Multiply<int, _bz_typename P_expr2::T_numtype > > >
operator*(int d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>, 
      Multiply<int, _bz_typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2);
}

// int * IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      IndexPlaceholder<N_index2>,
      Multiply<int, int > > >
operator*(int d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      IndexPlaceholder<N_index2>, 
      Multiply<int, int> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2);
}

// float * Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<float>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Multiply<float, T_numtype2 > > >
operator*(float d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<float>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Multiply<float, T_numtype2> >
      (_bz_ArrayExprConstant<float>(d1), 
      d2.begin());
}

// float * _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<float>, 
      _bz_ArrayExpr<P_expr2>,
      Multiply<float, _bz_typename P_expr2::T_numtype > > >
operator*(float d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<float>, 
      _bz_ArrayExpr<P_expr2>, 
      Multiply<float, _bz_typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<float>(d1), 
      d2);
}

// float * IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<float>, 
      IndexPlaceholder<N_index2>,
      Multiply<float, int > > >
operator*(float d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<float>, 
      IndexPlaceholder<N_index2>, 
      Multiply<float, int> >
      (_bz_ArrayExprConstant<float>(d1), 
      d2);
}

// double * Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<double>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Multiply<double, T_numtype2 > > >
operator*(double d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<double>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Multiply<double, T_numtype2> >
      (_bz_ArrayExprConstant<double>(d1), 
      d2.begin());
}

// double * _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<double>, 
      _bz_ArrayExpr<P_expr2>,
      Multiply<double, _bz_typename P_expr2::T_numtype > > >
operator*(double d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<double>, 
      _bz_ArrayExpr<P_expr2>, 
      Multiply<double, _bz_typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<double>(d1), 
      d2);
}

// double * IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<double>, 
      IndexPlaceholder<N_index2>,
      Multiply<double, int > > >
operator*(double d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<double>, 
      IndexPlaceholder<N_index2>, 
      Multiply<double, int> >
      (_bz_ArrayExprConstant<double>(d1), 
      d2);
}

// long double * Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<long double>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Multiply<long double, T_numtype2 > > >
operator*(long double d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<long double>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Multiply<long double, T_numtype2> >
      (_bz_ArrayExprConstant<long double>(d1), 
      d2.begin());
}

// long double * _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<long double>, 
      _bz_ArrayExpr<P_expr2>,
      Multiply<long double, _bz_typename P_expr2::T_numtype > > >
operator*(long double d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<long double>, 
      _bz_ArrayExpr<P_expr2>, 
      Multiply<long double, _bz_typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<long double>(d1), 
      d2);
}

// long double * IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<long double>, 
      IndexPlaceholder<N_index2>,
      Multiply<long double, int > > >
operator*(long double d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<long double>, 
      IndexPlaceholder<N_index2>, 
      Multiply<long double, int> >
      (_bz_ArrayExprConstant<long double>(d1), 
      d2);
}

#ifdef BZ_HAVE_COMPLEX
// complex<T1> * Array<T_numtype2, N_rank2>
template<class T1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<complex<T1> > , 
      ArrayIterator<T_numtype2, N_rank2>,
      Multiply<complex<T1> , T_numtype2 > > >
operator*(complex<T1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<complex<T1> > , 
      ArrayIterator<T_numtype2, N_rank2>, 
      Multiply<complex<T1> , T_numtype2> >
      (_bz_ArrayExprConstant<complex<T1> > (d1), 
      d2.begin());
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX
// complex<T1> * _bz_ArrayExpr<P_expr2>
template<class T1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<complex<T1> > , 
      _bz_ArrayExpr<P_expr2>,
      Multiply<complex<T1> , _bz_typename P_expr2::T_numtype > > >
operator*(complex<T1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<complex<T1> > , 
      _bz_ArrayExpr<P_expr2>, 
      Multiply<complex<T1> , _bz_typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<complex<T1> > (d1), 
      d2);
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX
// complex<T1> * IndexPlaceholder<N_index2>
template<class T1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<complex<T1> > , 
      IndexPlaceholder<N_index2>,
      Multiply<complex<T1> , int > > >
operator*(complex<T1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<complex<T1> > , 
      IndexPlaceholder<N_index2>, 
      Multiply<complex<T1> , int> >
      (_bz_ArrayExprConstant<complex<T1> > (d1), 
      d2);
}
#endif // BZ_HAVE_COMPLEX
/****************************************************************************
 * Division Operators
 ****************************************************************************/

// Array<T_numtype1, N_rank1> / Array<T_numtype2, N_rank2>
template<class T_numtype1, int N_rank1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Divide<T_numtype1, T_numtype2 > > >
operator/(const Array<T_numtype1, N_rank1>& d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Divide<T_numtype1, T_numtype2> >
      (d1.begin(), 
      d2.begin());
}

// Array<T_numtype1, N_rank1> / _bz_ArrayExpr<P_expr2>
template<class T_numtype1, int N_rank1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>,
      Divide<T_numtype1, _bz_typename P_expr2::T_numtype > > >
operator/(const Array<T_numtype1, N_rank1>& d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>, 
      Divide<T_numtype1, _bz_typename P_expr2::T_numtype> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> / IndexPlaceholder<N_index2>
template<class T_numtype1, int N_rank1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>,
      Divide<T_numtype1, int > > >
operator/(const Array<T_numtype1, N_rank1>& d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>, 
      Divide<T_numtype1, int> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> / int
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>,
      Divide<T_numtype1, int > > >
operator/(const Array<T_numtype1, N_rank1>& d1, 
      int d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>, 
      Divide<T_numtype1, int> >
      (d1.begin(), 
      _bz_ArrayExprConstant<int>(d2));
}

// Array<T_numtype1, N_rank1> / float
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<float>,
      Divide<T_numtype1, float > > >
operator/(const Array<T_numtype1, N_rank1>& d1, 
      float d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<float>, 
      Divide<T_numtype1, float> >
      (d1.begin(), 
      _bz_ArrayExprConstant<float>(d2));
}

// Array<T_numtype1, N_rank1> / double
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<double>,
      Divide<T_numtype1, double > > >
operator/(const Array<T_numtype1, N_rank1>& d1, 
      double d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<double>, 
      Divide<T_numtype1, double> >
      (d1.begin(), 
      _bz_ArrayExprConstant<double>(d2));
}

// Array<T_numtype1, N_rank1> / long double
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<long double>,
      Divide<T_numtype1, long double > > >
operator/(const Array<T_numtype1, N_rank1>& d1, 
      long double d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<long double>, 
      Divide<T_numtype1, long double> >
      (d1.begin(), 
      _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
// Array<T_numtype1, N_rank1> / complex<T2>
template<class T_numtype1, int N_rank1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<complex<T2> > ,
      Divide<T_numtype1, complex<T2>  > > >
operator/(const Array<T_numtype1, N_rank1>& d1, 
      complex<T2> d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<complex<T2> > , 
      Divide<T_numtype1, complex<T2> > >
      (d1.begin(), 
      _bz_ArrayExprConstant<complex<T2> > (d2));
}
#endif // BZ_HAVE_COMPLEX

// _bz_ArrayExpr<P_expr1> / Array<T_numtype2, N_rank2>
template<class P_expr1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Divide<_bz_typename P_expr1::T_numtype, T_numtype2 > > >
operator/(_bz_ArrayExpr<P_expr1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Divide<_bz_typename P_expr1::T_numtype, T_numtype2> >
      (d1, 
      d2.begin());
}

// _bz_ArrayExpr<P_expr1> / _bz_ArrayExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>,
      Divide<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype > > >
operator/(_bz_ArrayExpr<P_expr1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>, 
      Divide<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> / IndexPlaceholder<N_index2>
template<class P_expr1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>,
      Divide<_bz_typename P_expr1::T_numtype, int > > >
operator/(_bz_ArrayExpr<P_expr1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>, 
      Divide<_bz_typename P_expr1::T_numtype, int> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> / int
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>,
      Divide<_bz_typename P_expr1::T_numtype, int > > >
operator/(_bz_ArrayExpr<P_expr1> d1, 
      int d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>, 
      Divide<_bz_typename P_expr1::T_numtype, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// _bz_ArrayExpr<P_expr1> / float
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<float>,
      Divide<_bz_typename P_expr1::T_numtype, float > > >
operator/(_bz_ArrayExpr<P_expr1> d1, 
      float d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<float>, 
      Divide<_bz_typename P_expr1::T_numtype, float> >
      (d1, 
      _bz_ArrayExprConstant<float>(d2));
}

// _bz_ArrayExpr<P_expr1> / double
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<double>,
      Divide<_bz_typename P_expr1::T_numtype, double > > >
operator/(_bz_ArrayExpr<P_expr1> d1, 
      double d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<double>, 
      Divide<_bz_typename P_expr1::T_numtype, double> >
      (d1, 
      _bz_ArrayExprConstant<double>(d2));
}

// _bz_ArrayExpr<P_expr1> / long double
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<long double>,
      Divide<_bz_typename P_expr1::T_numtype, long double > > >
operator/(_bz_ArrayExpr<P_expr1> d1, 
      long double d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<long double>, 
      Divide<_bz_typename P_expr1::T_numtype, long double> >
      (d1, 
      _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
// _bz_ArrayExpr<P_expr1> / complex<T2>
template<class P_expr1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<complex<T2> > ,
      Divide<_bz_typename P_expr1::T_numtype, complex<T2>  > > >
operator/(_bz_ArrayExpr<P_expr1> d1, 
      complex<T2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<complex<T2> > , 
      Divide<_bz_typename P_expr1::T_numtype, complex<T2> > >
      (d1, 
      _bz_ArrayExprConstant<complex<T2> > (d2));
}
#endif // BZ_HAVE_COMPLEX

// IndexPlaceholder<N_index1> / Array<T_numtype2, N_rank2>
template<int N_index1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Divide<int, T_numtype2 > > >
operator/(IndexPlaceholder<N_index1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Divide<int, T_numtype2> >
      (d1, 
      d2.begin());
}

// IndexPlaceholder<N_index1> / _bz_ArrayExpr<P_expr2>
template<int N_index1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>,
      Divide<int, _bz_typename P_expr2::T_numtype > > >
operator/(IndexPlaceholder<N_index1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>, 
      Divide<int, _bz_typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> / IndexPlaceholder<N_index2>
template<int N_index1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>,
      Divide<int, int > > >
operator/(IndexPlaceholder<N_index1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>, 
      Divide<int, int> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> / int
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>,
      Divide<int, int > > >
operator/(IndexPlaceholder<N_index1> d1, 
      int d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>, 
      Divide<int, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// IndexPlaceholder<N_index1> / float
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<float>,
      Divide<int, float > > >
operator/(IndexPlaceholder<N_index1> d1, 
      float d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<float>, 
      Divide<int, float> >
      (d1, 
      _bz_ArrayExprConstant<float>(d2));
}

// IndexPlaceholder<N_index1> / double
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<double>,
      Divide<int, double > > >
operator/(IndexPlaceholder<N_index1> d1, 
      double d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<double>, 
      Divide<int, double> >
      (d1, 
      _bz_ArrayExprConstant<double>(d2));
}

// IndexPlaceholder<N_index1> / long double
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<long double>,
      Divide<int, long double > > >
operator/(IndexPlaceholder<N_index1> d1, 
      long double d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<long double>, 
      Divide<int, long double> >
      (d1, 
      _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
// IndexPlaceholder<N_index1> / complex<T2>
template<int N_index1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<complex<T2> > ,
      Divide<int, complex<T2>  > > >
operator/(IndexPlaceholder<N_index1> d1, 
      complex<T2> d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<complex<T2> > , 
      Divide<int, complex<T2> > >
      (d1, 
      _bz_ArrayExprConstant<complex<T2> > (d2));
}
#endif // BZ_HAVE_COMPLEX

// int / Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Divide<int, T_numtype2 > > >
operator/(int d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Divide<int, T_numtype2> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2.begin());
}

// int / _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>,
      Divide<int, _bz_typename P_expr2::T_numtype > > >
operator/(int d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>, 
      Divide<int, _bz_typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2);
}

// int / IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      IndexPlaceholder<N_index2>,
      Divide<int, int > > >
operator/(int d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      IndexPlaceholder<N_index2>, 
      Divide<int, int> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2);
}

// float / Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<float>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Divide<float, T_numtype2 > > >
operator/(float d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<float>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Divide<float, T_numtype2> >
      (_bz_ArrayExprConstant<float>(d1), 
      d2.begin());
}

// float / _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<float>, 
      _bz_ArrayExpr<P_expr2>,
      Divide<float, _bz_typename P_expr2::T_numtype > > >
operator/(float d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<float>, 
      _bz_ArrayExpr<P_expr2>, 
      Divide<float, _bz_typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<float>(d1), 
      d2);
}

// float / IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<float>, 
      IndexPlaceholder<N_index2>,
      Divide<float, int > > >
operator/(float d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<float>, 
      IndexPlaceholder<N_index2>, 
      Divide<float, int> >
      (_bz_ArrayExprConstant<float>(d1), 
      d2);
}

// double / Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<double>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Divide<double, T_numtype2 > > >
operator/(double d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<double>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Divide<double, T_numtype2> >
      (_bz_ArrayExprConstant<double>(d1), 
      d2.begin());
}

// double / _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<double>, 
      _bz_ArrayExpr<P_expr2>,
      Divide<double, _bz_typename P_expr2::T_numtype > > >
operator/(double d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<double>, 
      _bz_ArrayExpr<P_expr2>, 
      Divide<double, _bz_typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<double>(d1), 
      d2);
}

// double / IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<double>, 
      IndexPlaceholder<N_index2>,
      Divide<double, int > > >
operator/(double d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<double>, 
      IndexPlaceholder<N_index2>, 
      Divide<double, int> >
      (_bz_ArrayExprConstant<double>(d1), 
      d2);
}

// long double / Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<long double>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Divide<long double, T_numtype2 > > >
operator/(long double d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<long double>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Divide<long double, T_numtype2> >
      (_bz_ArrayExprConstant<long double>(d1), 
      d2.begin());
}

// long double / _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<long double>, 
      _bz_ArrayExpr<P_expr2>,
      Divide<long double, _bz_typename P_expr2::T_numtype > > >
operator/(long double d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<long double>, 
      _bz_ArrayExpr<P_expr2>, 
      Divide<long double, _bz_typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<long double>(d1), 
      d2);
}

// long double / IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<long double>, 
      IndexPlaceholder<N_index2>,
      Divide<long double, int > > >
operator/(long double d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<long double>, 
      IndexPlaceholder<N_index2>, 
      Divide<long double, int> >
      (_bz_ArrayExprConstant<long double>(d1), 
      d2);
}

#ifdef BZ_HAVE_COMPLEX
// complex<T1> / Array<T_numtype2, N_rank2>
template<class T1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<complex<T1> > , 
      ArrayIterator<T_numtype2, N_rank2>,
      Divide<complex<T1> , T_numtype2 > > >
operator/(complex<T1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<complex<T1> > , 
      ArrayIterator<T_numtype2, N_rank2>, 
      Divide<complex<T1> , T_numtype2> >
      (_bz_ArrayExprConstant<complex<T1> > (d1), 
      d2.begin());
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX
// complex<T1> / _bz_ArrayExpr<P_expr2>
template<class T1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<complex<T1> > , 
      _bz_ArrayExpr<P_expr2>,
      Divide<complex<T1> , _bz_typename P_expr2::T_numtype > > >
operator/(complex<T1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<complex<T1> > , 
      _bz_ArrayExpr<P_expr2>, 
      Divide<complex<T1> , _bz_typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<complex<T1> > (d1), 
      d2);
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX
// complex<T1> / IndexPlaceholder<N_index2>
template<class T1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<complex<T1> > , 
      IndexPlaceholder<N_index2>,
      Divide<complex<T1> , int > > >
operator/(complex<T1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<complex<T1> > , 
      IndexPlaceholder<N_index2>, 
      Divide<complex<T1> , int> >
      (_bz_ArrayExprConstant<complex<T1> > (d1), 
      d2);
}
#endif // BZ_HAVE_COMPLEX
/****************************************************************************
 * Modulus Operators
 ****************************************************************************/

// Array<T_numtype1, N_rank1> % Array<T_numtype2, N_rank2>
template<class T_numtype1, int N_rank1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Modulo<T_numtype1, T_numtype2 > > >
operator%(const Array<T_numtype1, N_rank1>& d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Modulo<T_numtype1, T_numtype2> >
      (d1.begin(), 
      d2.begin());
}

// Array<T_numtype1, N_rank1> % _bz_ArrayExpr<P_expr2>
template<class T_numtype1, int N_rank1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>,
      Modulo<T_numtype1, _bz_typename P_expr2::T_numtype > > >
operator%(const Array<T_numtype1, N_rank1>& d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>, 
      Modulo<T_numtype1, _bz_typename P_expr2::T_numtype> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> % IndexPlaceholder<N_index2>
template<class T_numtype1, int N_rank1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>,
      Modulo<T_numtype1, int > > >
operator%(const Array<T_numtype1, N_rank1>& d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>, 
      Modulo<T_numtype1, int> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> % int
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>,
      Modulo<T_numtype1, int > > >
operator%(const Array<T_numtype1, N_rank1>& d1, 
      int d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>, 
      Modulo<T_numtype1, int> >
      (d1.begin(), 
      _bz_ArrayExprConstant<int>(d2));
}

// _bz_ArrayExpr<P_expr1> % Array<T_numtype2, N_rank2>
template<class P_expr1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Modulo<_bz_typename P_expr1::T_numtype, T_numtype2 > > >
operator%(_bz_ArrayExpr<P_expr1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Modulo<_bz_typename P_expr1::T_numtype, T_numtype2> >
      (d1, 
      d2.begin());
}

// _bz_ArrayExpr<P_expr1> % _bz_ArrayExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>,
      Modulo<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype > > >
operator%(_bz_ArrayExpr<P_expr1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>, 
      Modulo<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> % IndexPlaceholder<N_index2>
template<class P_expr1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>,
      Modulo<_bz_typename P_expr1::T_numtype, int > > >
operator%(_bz_ArrayExpr<P_expr1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>, 
      Modulo<_bz_typename P_expr1::T_numtype, int> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> % int
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>,
      Modulo<_bz_typename P_expr1::T_numtype, int > > >
operator%(_bz_ArrayExpr<P_expr1> d1, 
      int d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>, 
      Modulo<_bz_typename P_expr1::T_numtype, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// IndexPlaceholder<N_index1> % Array<T_numtype2, N_rank2>
template<int N_index1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Modulo<int, T_numtype2 > > >
operator%(IndexPlaceholder<N_index1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Modulo<int, T_numtype2> >
      (d1, 
      d2.begin());
}

// IndexPlaceholder<N_index1> % _bz_ArrayExpr<P_expr2>
template<int N_index1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>,
      Modulo<int, _bz_typename P_expr2::T_numtype > > >
operator%(IndexPlaceholder<N_index1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>, 
      Modulo<int, _bz_typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> % IndexPlaceholder<N_index2>
template<int N_index1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>,
      Modulo<int, int > > >
operator%(IndexPlaceholder<N_index1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>, 
      Modulo<int, int> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> % int
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>,
      Modulo<int, int > > >
operator%(IndexPlaceholder<N_index1> d1, 
      int d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>, 
      Modulo<int, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// int % Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Modulo<int, T_numtype2 > > >
operator%(int d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Modulo<int, T_numtype2> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2.begin());
}

// int % _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>,
      Modulo<int, _bz_typename P_expr2::T_numtype > > >
operator%(int d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>, 
      Modulo<int, _bz_typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2);
}

// int % IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      IndexPlaceholder<N_index2>,
      Modulo<int, int > > >
operator%(int d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      IndexPlaceholder<N_index2>, 
      Modulo<int, int> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2);
}
/****************************************************************************
 * Greater-than Operators
 ****************************************************************************/

// Array<T_numtype1, N_rank1> > Array<T_numtype2, N_rank2>
template<class T_numtype1, int N_rank1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Greater<T_numtype1, T_numtype2 > > >
operator>(const Array<T_numtype1, N_rank1>& d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Greater<T_numtype1, T_numtype2> >
      (d1.begin(), 
      d2.begin());
}

// Array<T_numtype1, N_rank1> > _bz_ArrayExpr<P_expr2>
template<class T_numtype1, int N_rank1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>,
      Greater<T_numtype1, _bz_typename P_expr2::T_numtype > > >
operator>(const Array<T_numtype1, N_rank1>& d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>, 
      Greater<T_numtype1, _bz_typename P_expr2::T_numtype> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> > IndexPlaceholder<N_index2>
template<class T_numtype1, int N_rank1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>,
      Greater<T_numtype1, int > > >
operator>(const Array<T_numtype1, N_rank1>& d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>, 
      Greater<T_numtype1, int> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> > int
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>,
      Greater<T_numtype1, int > > >
operator>(const Array<T_numtype1, N_rank1>& d1, 
      int d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>, 
      Greater<T_numtype1, int> >
      (d1.begin(), 
      _bz_ArrayExprConstant<int>(d2));
}

// Array<T_numtype1, N_rank1> > float
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<float>,
      Greater<T_numtype1, float > > >
operator>(const Array<T_numtype1, N_rank1>& d1, 
      float d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<float>, 
      Greater<T_numtype1, float> >
      (d1.begin(), 
      _bz_ArrayExprConstant<float>(d2));
}

// Array<T_numtype1, N_rank1> > double
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<double>,
      Greater<T_numtype1, double > > >
operator>(const Array<T_numtype1, N_rank1>& d1, 
      double d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<double>, 
      Greater<T_numtype1, double> >
      (d1.begin(), 
      _bz_ArrayExprConstant<double>(d2));
}

// Array<T_numtype1, N_rank1> > long double
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<long double>,
      Greater<T_numtype1, long double > > >
operator>(const Array<T_numtype1, N_rank1>& d1, 
      long double d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<long double>, 
      Greater<T_numtype1, long double> >
      (d1.begin(), 
      _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
// Array<T_numtype1, N_rank1> > complex<T2>
template<class T_numtype1, int N_rank1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<complex<T2> > ,
      Greater<T_numtype1, complex<T2>  > > >
operator>(const Array<T_numtype1, N_rank1>& d1, 
      complex<T2> d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<complex<T2> > , 
      Greater<T_numtype1, complex<T2> > >
      (d1.begin(), 
      _bz_ArrayExprConstant<complex<T2> > (d2));
}
#endif // BZ_HAVE_COMPLEX

// _bz_ArrayExpr<P_expr1> > Array<T_numtype2, N_rank2>
template<class P_expr1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Greater<_bz_typename P_expr1::T_numtype, T_numtype2 > > >
operator>(_bz_ArrayExpr<P_expr1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Greater<_bz_typename P_expr1::T_numtype, T_numtype2> >
      (d1, 
      d2.begin());
}

// _bz_ArrayExpr<P_expr1> > _bz_ArrayExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>,
      Greater<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype > > >
operator>(_bz_ArrayExpr<P_expr1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>, 
      Greater<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> > IndexPlaceholder<N_index2>
template<class P_expr1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>,
      Greater<_bz_typename P_expr1::T_numtype, int > > >
operator>(_bz_ArrayExpr<P_expr1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>, 
      Greater<_bz_typename P_expr1::T_numtype, int> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> > int
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>,
      Greater<_bz_typename P_expr1::T_numtype, int > > >
operator>(_bz_ArrayExpr<P_expr1> d1, 
      int d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>, 
      Greater<_bz_typename P_expr1::T_numtype, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// _bz_ArrayExpr<P_expr1> > float
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<float>,
      Greater<_bz_typename P_expr1::T_numtype, float > > >
operator>(_bz_ArrayExpr<P_expr1> d1, 
      float d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<float>, 
      Greater<_bz_typename P_expr1::T_numtype, float> >
      (d1, 
      _bz_ArrayExprConstant<float>(d2));
}

// _bz_ArrayExpr<P_expr1> > double
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<double>,
      Greater<_bz_typename P_expr1::T_numtype, double > > >
operator>(_bz_ArrayExpr<P_expr1> d1, 
      double d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<double>, 
      Greater<_bz_typename P_expr1::T_numtype, double> >
      (d1, 
      _bz_ArrayExprConstant<double>(d2));
}

// _bz_ArrayExpr<P_expr1> > long double
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<long double>,
      Greater<_bz_typename P_expr1::T_numtype, long double > > >
operator>(_bz_ArrayExpr<P_expr1> d1, 
      long double d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<long double>, 
      Greater<_bz_typename P_expr1::T_numtype, long double> >
      (d1, 
      _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
// _bz_ArrayExpr<P_expr1> > complex<T2>
template<class P_expr1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<complex<T2> > ,
      Greater<_bz_typename P_expr1::T_numtype, complex<T2>  > > >
operator>(_bz_ArrayExpr<P_expr1> d1, 
      complex<T2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<complex<T2> > , 
      Greater<_bz_typename P_expr1::T_numtype, complex<T2> > >
      (d1, 
      _bz_ArrayExprConstant<complex<T2> > (d2));
}
#endif // BZ_HAVE_COMPLEX

// IndexPlaceholder<N_index1> > Array<T_numtype2, N_rank2>
template<int N_index1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Greater<int, T_numtype2 > > >
operator>(IndexPlaceholder<N_index1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Greater<int, T_numtype2> >
      (d1, 
      d2.begin());
}

// IndexPlaceholder<N_index1> > _bz_ArrayExpr<P_expr2>
template<int N_index1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>,
      Greater<int, _bz_typename P_expr2::T_numtype > > >
operator>(IndexPlaceholder<N_index1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>, 
      Greater<int, _bz_typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> > IndexPlaceholder<N_index2>
template<int N_index1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>,
      Greater<int, int > > >
operator>(IndexPlaceholder<N_index1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>, 
      Greater<int, int> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> > int
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>,
      Greater<int, int > > >
operator>(IndexPlaceholder<N_index1> d1, 
      int d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>, 
      Greater<int, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// IndexPlaceholder<N_index1> > float
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<float>,
      Greater<int, float > > >
operator>(IndexPlaceholder<N_index1> d1, 
      float d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<float>, 
      Greater<int, float> >
      (d1, 
      _bz_ArrayExprConstant<float>(d2));
}

// IndexPlaceholder<N_index1> > double
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<double>,
      Greater<int, double > > >
operator>(IndexPlaceholder<N_index1> d1, 
      double d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<double>, 
      Greater<int, double> >
      (d1, 
      _bz_ArrayExprConstant<double>(d2));
}

// IndexPlaceholder<N_index1> > long double
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<long double>,
      Greater<int, long double > > >
operator>(IndexPlaceholder<N_index1> d1, 
      long double d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<long double>, 
      Greater<int, long double> >
      (d1, 
      _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
// IndexPlaceholder<N_index1> > complex<T2>
template<int N_index1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<complex<T2> > ,
      Greater<int, complex<T2>  > > >
operator>(IndexPlaceholder<N_index1> d1, 
      complex<T2> d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<complex<T2> > , 
      Greater<int, complex<T2> > >
      (d1, 
      _bz_ArrayExprConstant<complex<T2> > (d2));
}
#endif // BZ_HAVE_COMPLEX

// int > Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Greater<int, T_numtype2 > > >
operator>(int d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Greater<int, T_numtype2> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2.begin());
}

// int > _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>,
      Greater<int, _bz_typename P_expr2::T_numtype > > >
operator>(int d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>, 
      Greater<int, _bz_typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2);
}

// int > IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      IndexPlaceholder<N_index2>,
      Greater<int, int > > >
operator>(int d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      IndexPlaceholder<N_index2>, 
      Greater<int, int> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2);
}

// float > Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<float>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Greater<float, T_numtype2 > > >
operator>(float d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<float>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Greater<float, T_numtype2> >
      (_bz_ArrayExprConstant<float>(d1), 
      d2.begin());
}

// float > _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<float>, 
      _bz_ArrayExpr<P_expr2>,
      Greater<float, _bz_typename P_expr2::T_numtype > > >
operator>(float d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<float>, 
      _bz_ArrayExpr<P_expr2>, 
      Greater<float, _bz_typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<float>(d1), 
      d2);
}

// float > IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<float>, 
      IndexPlaceholder<N_index2>,
      Greater<float, int > > >
operator>(float d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<float>, 
      IndexPlaceholder<N_index2>, 
      Greater<float, int> >
      (_bz_ArrayExprConstant<float>(d1), 
      d2);
}

// double > Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<double>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Greater<double, T_numtype2 > > >
operator>(double d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<double>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Greater<double, T_numtype2> >
      (_bz_ArrayExprConstant<double>(d1), 
      d2.begin());
}

// double > _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<double>, 
      _bz_ArrayExpr<P_expr2>,
      Greater<double, _bz_typename P_expr2::T_numtype > > >
operator>(double d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<double>, 
      _bz_ArrayExpr<P_expr2>, 
      Greater<double, _bz_typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<double>(d1), 
      d2);
}

// double > IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<double>, 
      IndexPlaceholder<N_index2>,
      Greater<double, int > > >
operator>(double d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<double>, 
      IndexPlaceholder<N_index2>, 
      Greater<double, int> >
      (_bz_ArrayExprConstant<double>(d1), 
      d2);
}

// long double > Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<long double>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Greater<long double, T_numtype2 > > >
operator>(long double d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<long double>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Greater<long double, T_numtype2> >
      (_bz_ArrayExprConstant<long double>(d1), 
      d2.begin());
}

// long double > _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<long double>, 
      _bz_ArrayExpr<P_expr2>,
      Greater<long double, _bz_typename P_expr2::T_numtype > > >
operator>(long double d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<long double>, 
      _bz_ArrayExpr<P_expr2>, 
      Greater<long double, _bz_typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<long double>(d1), 
      d2);
}

// long double > IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<long double>, 
      IndexPlaceholder<N_index2>,
      Greater<long double, int > > >
operator>(long double d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<long double>, 
      IndexPlaceholder<N_index2>, 
      Greater<long double, int> >
      (_bz_ArrayExprConstant<long double>(d1), 
      d2);
}

#ifdef BZ_HAVE_COMPLEX
// complex<T1> > Array<T_numtype2, N_rank2>
template<class T1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<complex<T1> > , 
      ArrayIterator<T_numtype2, N_rank2>,
      Greater<complex<T1> , T_numtype2 > > >
operator>(complex<T1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<complex<T1> > , 
      ArrayIterator<T_numtype2, N_rank2>, 
      Greater<complex<T1> , T_numtype2> >
      (_bz_ArrayExprConstant<complex<T1> > (d1), 
      d2.begin());
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX
// complex<T1> > _bz_ArrayExpr<P_expr2>
template<class T1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<complex<T1> > , 
      _bz_ArrayExpr<P_expr2>,
      Greater<complex<T1> , _bz_typename P_expr2::T_numtype > > >
operator>(complex<T1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<complex<T1> > , 
      _bz_ArrayExpr<P_expr2>, 
      Greater<complex<T1> , _bz_typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<complex<T1> > (d1), 
      d2);
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX
// complex<T1> > IndexPlaceholder<N_index2>
template<class T1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<complex<T1> > , 
      IndexPlaceholder<N_index2>,
      Greater<complex<T1> , int > > >
operator>(complex<T1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<complex<T1> > , 
      IndexPlaceholder<N_index2>, 
      Greater<complex<T1> , int> >
      (_bz_ArrayExprConstant<complex<T1> > (d1), 
      d2);
}
#endif // BZ_HAVE_COMPLEX
/****************************************************************************
 * Less-than Operators
 ****************************************************************************/

// Array<T_numtype1, N_rank1> < Array<T_numtype2, N_rank2>
template<class T_numtype1, int N_rank1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Less<T_numtype1, T_numtype2 > > >
operator<(const Array<T_numtype1, N_rank1>& d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Less<T_numtype1, T_numtype2> >
      (d1.begin(), 
      d2.begin());
}

// Array<T_numtype1, N_rank1> < _bz_ArrayExpr<P_expr2>
template<class T_numtype1, int N_rank1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>,
      Less<T_numtype1, _bz_typename P_expr2::T_numtype > > >
operator<(const Array<T_numtype1, N_rank1>& d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>, 
      Less<T_numtype1, _bz_typename P_expr2::T_numtype> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> < IndexPlaceholder<N_index2>
template<class T_numtype1, int N_rank1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>,
      Less<T_numtype1, int > > >
operator<(const Array<T_numtype1, N_rank1>& d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>, 
      Less<T_numtype1, int> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> < int
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>,
      Less<T_numtype1, int > > >
operator<(const Array<T_numtype1, N_rank1>& d1, 
      int d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>, 
      Less<T_numtype1, int> >
      (d1.begin(), 
      _bz_ArrayExprConstant<int>(d2));
}

// Array<T_numtype1, N_rank1> < float
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<float>,
      Less<T_numtype1, float > > >
operator<(const Array<T_numtype1, N_rank1>& d1, 
      float d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<float>, 
      Less<T_numtype1, float> >
      (d1.begin(), 
      _bz_ArrayExprConstant<float>(d2));
}

// Array<T_numtype1, N_rank1> < double
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<double>,
      Less<T_numtype1, double > > >
operator<(const Array<T_numtype1, N_rank1>& d1, 
      double d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<double>, 
      Less<T_numtype1, double> >
      (d1.begin(), 
      _bz_ArrayExprConstant<double>(d2));
}

// Array<T_numtype1, N_rank1> < long double
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<long double>,
      Less<T_numtype1, long double > > >
operator<(const Array<T_numtype1, N_rank1>& d1, 
      long double d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<long double>, 
      Less<T_numtype1, long double> >
      (d1.begin(), 
      _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
// Array<T_numtype1, N_rank1> < complex<T2>
template<class T_numtype1, int N_rank1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<complex<T2> > ,
      Less<T_numtype1, complex<T2>  > > >
operator<(const Array<T_numtype1, N_rank1>& d1, 
      complex<T2> d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<complex<T2> > , 
      Less<T_numtype1, complex<T2> > >
      (d1.begin(), 
      _bz_ArrayExprConstant<complex<T2> > (d2));
}
#endif // BZ_HAVE_COMPLEX

// _bz_ArrayExpr<P_expr1> < Array<T_numtype2, N_rank2>
template<class P_expr1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Less<_bz_typename P_expr1::T_numtype, T_numtype2 > > >
operator<(_bz_ArrayExpr<P_expr1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Less<_bz_typename P_expr1::T_numtype, T_numtype2> >
      (d1, 
      d2.begin());
}

// _bz_ArrayExpr<P_expr1> < _bz_ArrayExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>,
      Less<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype > > >
operator<(_bz_ArrayExpr<P_expr1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>, 
      Less<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> < IndexPlaceholder<N_index2>
template<class P_expr1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>,
      Less<_bz_typename P_expr1::T_numtype, int > > >
operator<(_bz_ArrayExpr<P_expr1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>, 
      Less<_bz_typename P_expr1::T_numtype, int> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> < int
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>,
      Less<_bz_typename P_expr1::T_numtype, int > > >
operator<(_bz_ArrayExpr<P_expr1> d1, 
      int d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>, 
      Less<_bz_typename P_expr1::T_numtype, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// _bz_ArrayExpr<P_expr1> < float
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<float>,
      Less<_bz_typename P_expr1::T_numtype, float > > >
operator<(_bz_ArrayExpr<P_expr1> d1, 
      float d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<float>, 
      Less<_bz_typename P_expr1::T_numtype, float> >
      (d1, 
      _bz_ArrayExprConstant<float>(d2));
}

// _bz_ArrayExpr<P_expr1> < double
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<double>,
      Less<_bz_typename P_expr1::T_numtype, double > > >
operator<(_bz_ArrayExpr<P_expr1> d1, 
      double d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<double>, 
      Less<_bz_typename P_expr1::T_numtype, double> >
      (d1, 
      _bz_ArrayExprConstant<double>(d2));
}

// _bz_ArrayExpr<P_expr1> < long double
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<long double>,
      Less<_bz_typename P_expr1::T_numtype, long double > > >
operator<(_bz_ArrayExpr<P_expr1> d1, 
      long double d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<long double>, 
      Less<_bz_typename P_expr1::T_numtype, long double> >
      (d1, 
      _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
// _bz_ArrayExpr<P_expr1> < complex<T2>
template<class P_expr1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<complex<T2> > ,
      Less<_bz_typename P_expr1::T_numtype, complex<T2>  > > >
operator<(_bz_ArrayExpr<P_expr1> d1, 
      complex<T2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<complex<T2> > , 
      Less<_bz_typename P_expr1::T_numtype, complex<T2> > >
      (d1, 
      _bz_ArrayExprConstant<complex<T2> > (d2));
}
#endif // BZ_HAVE_COMPLEX

// IndexPlaceholder<N_index1> < Array<T_numtype2, N_rank2>
template<int N_index1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Less<int, T_numtype2 > > >
operator<(IndexPlaceholder<N_index1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Less<int, T_numtype2> >
      (d1, 
      d2.begin());
}

// IndexPlaceholder<N_index1> < _bz_ArrayExpr<P_expr2>
template<int N_index1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>,
      Less<int, _bz_typename P_expr2::T_numtype > > >
operator<(IndexPlaceholder<N_index1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>, 
      Less<int, _bz_typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> < IndexPlaceholder<N_index2>
template<int N_index1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>,
      Less<int, int > > >
operator<(IndexPlaceholder<N_index1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>, 
      Less<int, int> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> < int
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>,
      Less<int, int > > >
operator<(IndexPlaceholder<N_index1> d1, 
      int d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>, 
      Less<int, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// IndexPlaceholder<N_index1> < float
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<float>,
      Less<int, float > > >
operator<(IndexPlaceholder<N_index1> d1, 
      float d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<float>, 
      Less<int, float> >
      (d1, 
      _bz_ArrayExprConstant<float>(d2));
}

// IndexPlaceholder<N_index1> < double
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<double>,
      Less<int, double > > >
operator<(IndexPlaceholder<N_index1> d1, 
      double d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<double>, 
      Less<int, double> >
      (d1, 
      _bz_ArrayExprConstant<double>(d2));
}

// IndexPlaceholder<N_index1> < long double
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<long double>,
      Less<int, long double > > >
operator<(IndexPlaceholder<N_index1> d1, 
      long double d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<long double>, 
      Less<int, long double> >
      (d1, 
      _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
// IndexPlaceholder<N_index1> < complex<T2>
template<int N_index1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<complex<T2> > ,
      Less<int, complex<T2>  > > >
operator<(IndexPlaceholder<N_index1> d1, 
      complex<T2> d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<complex<T2> > , 
      Less<int, complex<T2> > >
      (d1, 
      _bz_ArrayExprConstant<complex<T2> > (d2));
}
#endif // BZ_HAVE_COMPLEX

// int < Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Less<int, T_numtype2 > > >
operator<(int d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Less<int, T_numtype2> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2.begin());
}

// int < _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>,
      Less<int, _bz_typename P_expr2::T_numtype > > >
operator<(int d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>, 
      Less<int, _bz_typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2);
}

// int < IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      IndexPlaceholder<N_index2>,
      Less<int, int > > >
operator<(int d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      IndexPlaceholder<N_index2>, 
      Less<int, int> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2);
}

// float < Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<float>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Less<float, T_numtype2 > > >
operator<(float d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<float>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Less<float, T_numtype2> >
      (_bz_ArrayExprConstant<float>(d1), 
      d2.begin());
}

// float < _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<float>, 
      _bz_ArrayExpr<P_expr2>,
      Less<float, _bz_typename P_expr2::T_numtype > > >
operator<(float d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<float>, 
      _bz_ArrayExpr<P_expr2>, 
      Less<float, _bz_typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<float>(d1), 
      d2);
}

// float < IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<float>, 
      IndexPlaceholder<N_index2>,
      Less<float, int > > >
operator<(float d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<float>, 
      IndexPlaceholder<N_index2>, 
      Less<float, int> >
      (_bz_ArrayExprConstant<float>(d1), 
      d2);
}

// double < Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<double>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Less<double, T_numtype2 > > >
operator<(double d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<double>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Less<double, T_numtype2> >
      (_bz_ArrayExprConstant<double>(d1), 
      d2.begin());
}

// double < _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<double>, 
      _bz_ArrayExpr<P_expr2>,
      Less<double, _bz_typename P_expr2::T_numtype > > >
operator<(double d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<double>, 
      _bz_ArrayExpr<P_expr2>, 
      Less<double, _bz_typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<double>(d1), 
      d2);
}

// double < IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<double>, 
      IndexPlaceholder<N_index2>,
      Less<double, int > > >
operator<(double d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<double>, 
      IndexPlaceholder<N_index2>, 
      Less<double, int> >
      (_bz_ArrayExprConstant<double>(d1), 
      d2);
}

// long double < Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<long double>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Less<long double, T_numtype2 > > >
operator<(long double d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<long double>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Less<long double, T_numtype2> >
      (_bz_ArrayExprConstant<long double>(d1), 
      d2.begin());
}

// long double < _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<long double>, 
      _bz_ArrayExpr<P_expr2>,
      Less<long double, _bz_typename P_expr2::T_numtype > > >
operator<(long double d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<long double>, 
      _bz_ArrayExpr<P_expr2>, 
      Less<long double, _bz_typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<long double>(d1), 
      d2);
}

// long double < IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<long double>, 
      IndexPlaceholder<N_index2>,
      Less<long double, int > > >
operator<(long double d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<long double>, 
      IndexPlaceholder<N_index2>, 
      Less<long double, int> >
      (_bz_ArrayExprConstant<long double>(d1), 
      d2);
}

#ifdef BZ_HAVE_COMPLEX
// complex<T1> < Array<T_numtype2, N_rank2>
template<class T1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<complex<T1> > , 
      ArrayIterator<T_numtype2, N_rank2>,
      Less<complex<T1> , T_numtype2 > > >
operator<(complex<T1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<complex<T1> > , 
      ArrayIterator<T_numtype2, N_rank2>, 
      Less<complex<T1> , T_numtype2> >
      (_bz_ArrayExprConstant<complex<T1> > (d1), 
      d2.begin());
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX
// complex<T1> < _bz_ArrayExpr<P_expr2>
template<class T1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<complex<T1> > , 
      _bz_ArrayExpr<P_expr2>,
      Less<complex<T1> , _bz_typename P_expr2::T_numtype > > >
operator<(complex<T1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<complex<T1> > , 
      _bz_ArrayExpr<P_expr2>, 
      Less<complex<T1> , _bz_typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<complex<T1> > (d1), 
      d2);
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX
// complex<T1> < IndexPlaceholder<N_index2>
template<class T1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<complex<T1> > , 
      IndexPlaceholder<N_index2>,
      Less<complex<T1> , int > > >
operator<(complex<T1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<complex<T1> > , 
      IndexPlaceholder<N_index2>, 
      Less<complex<T1> , int> >
      (_bz_ArrayExprConstant<complex<T1> > (d1), 
      d2);
}
#endif // BZ_HAVE_COMPLEX
/****************************************************************************
 * Greater or equal (>=) operators
 ****************************************************************************/

// Array<T_numtype1, N_rank1> >= Array<T_numtype2, N_rank2>
template<class T_numtype1, int N_rank1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      GreaterOrEqual<T_numtype1, T_numtype2 > > >
operator>=(const Array<T_numtype1, N_rank1>& d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      GreaterOrEqual<T_numtype1, T_numtype2> >
      (d1.begin(), 
      d2.begin());
}

// Array<T_numtype1, N_rank1> >= _bz_ArrayExpr<P_expr2>
template<class T_numtype1, int N_rank1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>,
      GreaterOrEqual<T_numtype1, _bz_typename P_expr2::T_numtype > > >
operator>=(const Array<T_numtype1, N_rank1>& d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>, 
      GreaterOrEqual<T_numtype1, _bz_typename P_expr2::T_numtype> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> >= IndexPlaceholder<N_index2>
template<class T_numtype1, int N_rank1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>,
      GreaterOrEqual<T_numtype1, int > > >
operator>=(const Array<T_numtype1, N_rank1>& d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>, 
      GreaterOrEqual<T_numtype1, int> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> >= int
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>,
      GreaterOrEqual<T_numtype1, int > > >
operator>=(const Array<T_numtype1, N_rank1>& d1, 
      int d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>, 
      GreaterOrEqual<T_numtype1, int> >
      (d1.begin(), 
      _bz_ArrayExprConstant<int>(d2));
}

// Array<T_numtype1, N_rank1> >= float
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<float>,
      GreaterOrEqual<T_numtype1, float > > >
operator>=(const Array<T_numtype1, N_rank1>& d1, 
      float d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<float>, 
      GreaterOrEqual<T_numtype1, float> >
      (d1.begin(), 
      _bz_ArrayExprConstant<float>(d2));
}

// Array<T_numtype1, N_rank1> >= double
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<double>,
      GreaterOrEqual<T_numtype1, double > > >
operator>=(const Array<T_numtype1, N_rank1>& d1, 
      double d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<double>, 
      GreaterOrEqual<T_numtype1, double> >
      (d1.begin(), 
      _bz_ArrayExprConstant<double>(d2));
}

// Array<T_numtype1, N_rank1> >= long double
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<long double>,
      GreaterOrEqual<T_numtype1, long double > > >
operator>=(const Array<T_numtype1, N_rank1>& d1, 
      long double d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<long double>, 
      GreaterOrEqual<T_numtype1, long double> >
      (d1.begin(), 
      _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
// Array<T_numtype1, N_rank1> >= complex<T2>
template<class T_numtype1, int N_rank1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<complex<T2> > ,
      GreaterOrEqual<T_numtype1, complex<T2>  > > >
operator>=(const Array<T_numtype1, N_rank1>& d1, 
      complex<T2> d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<complex<T2> > , 
      GreaterOrEqual<T_numtype1, complex<T2> > >
      (d1.begin(), 
      _bz_ArrayExprConstant<complex<T2> > (d2));
}
#endif // BZ_HAVE_COMPLEX

// _bz_ArrayExpr<P_expr1> >= Array<T_numtype2, N_rank2>
template<class P_expr1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      GreaterOrEqual<_bz_typename P_expr1::T_numtype, T_numtype2 > > >
operator>=(_bz_ArrayExpr<P_expr1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      GreaterOrEqual<_bz_typename P_expr1::T_numtype, T_numtype2> >
      (d1, 
      d2.begin());
}

// _bz_ArrayExpr<P_expr1> >= _bz_ArrayExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>,
      GreaterOrEqual<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype > > >
operator>=(_bz_ArrayExpr<P_expr1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>, 
      GreaterOrEqual<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> >= IndexPlaceholder<N_index2>
template<class P_expr1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>,
      GreaterOrEqual<_bz_typename P_expr1::T_numtype, int > > >
operator>=(_bz_ArrayExpr<P_expr1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>, 
      GreaterOrEqual<_bz_typename P_expr1::T_numtype, int> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> >= int
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>,
      GreaterOrEqual<_bz_typename P_expr1::T_numtype, int > > >
operator>=(_bz_ArrayExpr<P_expr1> d1, 
      int d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>, 
      GreaterOrEqual<_bz_typename P_expr1::T_numtype, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// _bz_ArrayExpr<P_expr1> >= float
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<float>,
      GreaterOrEqual<_bz_typename P_expr1::T_numtype, float > > >
operator>=(_bz_ArrayExpr<P_expr1> d1, 
      float d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<float>, 
      GreaterOrEqual<_bz_typename P_expr1::T_numtype, float> >
      (d1, 
      _bz_ArrayExprConstant<float>(d2));
}

// _bz_ArrayExpr<P_expr1> >= double
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<double>,
      GreaterOrEqual<_bz_typename P_expr1::T_numtype, double > > >
operator>=(_bz_ArrayExpr<P_expr1> d1, 
      double d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<double>, 
      GreaterOrEqual<_bz_typename P_expr1::T_numtype, double> >
      (d1, 
      _bz_ArrayExprConstant<double>(d2));
}

// _bz_ArrayExpr<P_expr1> >= long double
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<long double>,
      GreaterOrEqual<_bz_typename P_expr1::T_numtype, long double > > >
operator>=(_bz_ArrayExpr<P_expr1> d1, 
      long double d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<long double>, 
      GreaterOrEqual<_bz_typename P_expr1::T_numtype, long double> >
      (d1, 
      _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
// _bz_ArrayExpr<P_expr1> >= complex<T2>
template<class P_expr1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<complex<T2> > ,
      GreaterOrEqual<_bz_typename P_expr1::T_numtype, complex<T2>  > > >
operator>=(_bz_ArrayExpr<P_expr1> d1, 
      complex<T2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<complex<T2> > , 
      GreaterOrEqual<_bz_typename P_expr1::T_numtype, complex<T2> > >
      (d1, 
      _bz_ArrayExprConstant<complex<T2> > (d2));
}
#endif // BZ_HAVE_COMPLEX

// IndexPlaceholder<N_index1> >= Array<T_numtype2, N_rank2>
template<int N_index1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      GreaterOrEqual<int, T_numtype2 > > >
operator>=(IndexPlaceholder<N_index1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      GreaterOrEqual<int, T_numtype2> >
      (d1, 
      d2.begin());
}

// IndexPlaceholder<N_index1> >= _bz_ArrayExpr<P_expr2>
template<int N_index1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>,
      GreaterOrEqual<int, _bz_typename P_expr2::T_numtype > > >
operator>=(IndexPlaceholder<N_index1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>, 
      GreaterOrEqual<int, _bz_typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> >= IndexPlaceholder<N_index2>
template<int N_index1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>,
      GreaterOrEqual<int, int > > >
operator>=(IndexPlaceholder<N_index1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>, 
      GreaterOrEqual<int, int> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> >= int
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>,
      GreaterOrEqual<int, int > > >
operator>=(IndexPlaceholder<N_index1> d1, 
      int d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>, 
      GreaterOrEqual<int, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// IndexPlaceholder<N_index1> >= float
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<float>,
      GreaterOrEqual<int, float > > >
operator>=(IndexPlaceholder<N_index1> d1, 
      float d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<float>, 
      GreaterOrEqual<int, float> >
      (d1, 
      _bz_ArrayExprConstant<float>(d2));
}

// IndexPlaceholder<N_index1> >= double
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<double>,
      GreaterOrEqual<int, double > > >
operator>=(IndexPlaceholder<N_index1> d1, 
      double d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<double>, 
      GreaterOrEqual<int, double> >
      (d1, 
      _bz_ArrayExprConstant<double>(d2));
}

// IndexPlaceholder<N_index1> >= long double
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<long double>,
      GreaterOrEqual<int, long double > > >
operator>=(IndexPlaceholder<N_index1> d1, 
      long double d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<long double>, 
      GreaterOrEqual<int, long double> >
      (d1, 
      _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
// IndexPlaceholder<N_index1> >= complex<T2>
template<int N_index1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<complex<T2> > ,
      GreaterOrEqual<int, complex<T2>  > > >
operator>=(IndexPlaceholder<N_index1> d1, 
      complex<T2> d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<complex<T2> > , 
      GreaterOrEqual<int, complex<T2> > >
      (d1, 
      _bz_ArrayExprConstant<complex<T2> > (d2));
}
#endif // BZ_HAVE_COMPLEX

// int >= Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>,
      GreaterOrEqual<int, T_numtype2 > > >
operator>=(int d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      GreaterOrEqual<int, T_numtype2> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2.begin());
}

// int >= _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>,
      GreaterOrEqual<int, _bz_typename P_expr2::T_numtype > > >
operator>=(int d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>, 
      GreaterOrEqual<int, _bz_typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2);
}

// int >= IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      IndexPlaceholder<N_index2>,
      GreaterOrEqual<int, int > > >
operator>=(int d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      IndexPlaceholder<N_index2>, 
      GreaterOrEqual<int, int> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2);
}

// float >= Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<float>, 
      ArrayIterator<T_numtype2, N_rank2>,
      GreaterOrEqual<float, T_numtype2 > > >
operator>=(float d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<float>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      GreaterOrEqual<float, T_numtype2> >
      (_bz_ArrayExprConstant<float>(d1), 
      d2.begin());
}

// float >= _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<float>, 
      _bz_ArrayExpr<P_expr2>,
      GreaterOrEqual<float, _bz_typename P_expr2::T_numtype > > >
operator>=(float d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<float>, 
      _bz_ArrayExpr<P_expr2>, 
      GreaterOrEqual<float, _bz_typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<float>(d1), 
      d2);
}

// float >= IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<float>, 
      IndexPlaceholder<N_index2>,
      GreaterOrEqual<float, int > > >
operator>=(float d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<float>, 
      IndexPlaceholder<N_index2>, 
      GreaterOrEqual<float, int> >
      (_bz_ArrayExprConstant<float>(d1), 
      d2);
}

// double >= Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<double>, 
      ArrayIterator<T_numtype2, N_rank2>,
      GreaterOrEqual<double, T_numtype2 > > >
operator>=(double d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<double>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      GreaterOrEqual<double, T_numtype2> >
      (_bz_ArrayExprConstant<double>(d1), 
      d2.begin());
}

// double >= _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<double>, 
      _bz_ArrayExpr<P_expr2>,
      GreaterOrEqual<double, _bz_typename P_expr2::T_numtype > > >
operator>=(double d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<double>, 
      _bz_ArrayExpr<P_expr2>, 
      GreaterOrEqual<double, _bz_typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<double>(d1), 
      d2);
}

// double >= IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<double>, 
      IndexPlaceholder<N_index2>,
      GreaterOrEqual<double, int > > >
operator>=(double d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<double>, 
      IndexPlaceholder<N_index2>, 
      GreaterOrEqual<double, int> >
      (_bz_ArrayExprConstant<double>(d1), 
      d2);
}

// long double >= Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<long double>, 
      ArrayIterator<T_numtype2, N_rank2>,
      GreaterOrEqual<long double, T_numtype2 > > >
operator>=(long double d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<long double>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      GreaterOrEqual<long double, T_numtype2> >
      (_bz_ArrayExprConstant<long double>(d1), 
      d2.begin());
}

// long double >= _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<long double>, 
      _bz_ArrayExpr<P_expr2>,
      GreaterOrEqual<long double, _bz_typename P_expr2::T_numtype > > >
operator>=(long double d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<long double>, 
      _bz_ArrayExpr<P_expr2>, 
      GreaterOrEqual<long double, _bz_typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<long double>(d1), 
      d2);
}

// long double >= IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<long double>, 
      IndexPlaceholder<N_index2>,
      GreaterOrEqual<long double, int > > >
operator>=(long double d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<long double>, 
      IndexPlaceholder<N_index2>, 
      GreaterOrEqual<long double, int> >
      (_bz_ArrayExprConstant<long double>(d1), 
      d2);
}

#ifdef BZ_HAVE_COMPLEX
// complex<T1> >= Array<T_numtype2, N_rank2>
template<class T1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<complex<T1> > , 
      ArrayIterator<T_numtype2, N_rank2>,
      GreaterOrEqual<complex<T1> , T_numtype2 > > >
operator>=(complex<T1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<complex<T1> > , 
      ArrayIterator<T_numtype2, N_rank2>, 
      GreaterOrEqual<complex<T1> , T_numtype2> >
      (_bz_ArrayExprConstant<complex<T1> > (d1), 
      d2.begin());
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX
// complex<T1> >= _bz_ArrayExpr<P_expr2>
template<class T1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<complex<T1> > , 
      _bz_ArrayExpr<P_expr2>,
      GreaterOrEqual<complex<T1> , _bz_typename P_expr2::T_numtype > > >
operator>=(complex<T1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<complex<T1> > , 
      _bz_ArrayExpr<P_expr2>, 
      GreaterOrEqual<complex<T1> , _bz_typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<complex<T1> > (d1), 
      d2);
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX
// complex<T1> >= IndexPlaceholder<N_index2>
template<class T1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<complex<T1> > , 
      IndexPlaceholder<N_index2>,
      GreaterOrEqual<complex<T1> , int > > >
operator>=(complex<T1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<complex<T1> > , 
      IndexPlaceholder<N_index2>, 
      GreaterOrEqual<complex<T1> , int> >
      (_bz_ArrayExprConstant<complex<T1> > (d1), 
      d2);
}
#endif // BZ_HAVE_COMPLEX
/****************************************************************************
 * Less or equal (<=) operators
 ****************************************************************************/

// Array<T_numtype1, N_rank1> <= Array<T_numtype2, N_rank2>
template<class T_numtype1, int N_rank1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      LessOrEqual<T_numtype1, T_numtype2 > > >
operator<=(const Array<T_numtype1, N_rank1>& d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      LessOrEqual<T_numtype1, T_numtype2> >
      (d1.begin(), 
      d2.begin());
}

// Array<T_numtype1, N_rank1> <= _bz_ArrayExpr<P_expr2>
template<class T_numtype1, int N_rank1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>,
      LessOrEqual<T_numtype1, _bz_typename P_expr2::T_numtype > > >
operator<=(const Array<T_numtype1, N_rank1>& d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>, 
      LessOrEqual<T_numtype1, _bz_typename P_expr2::T_numtype> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> <= IndexPlaceholder<N_index2>
template<class T_numtype1, int N_rank1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>,
      LessOrEqual<T_numtype1, int > > >
operator<=(const Array<T_numtype1, N_rank1>& d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>, 
      LessOrEqual<T_numtype1, int> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> <= int
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>,
      LessOrEqual<T_numtype1, int > > >
operator<=(const Array<T_numtype1, N_rank1>& d1, 
      int d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>, 
      LessOrEqual<T_numtype1, int> >
      (d1.begin(), 
      _bz_ArrayExprConstant<int>(d2));
}

// Array<T_numtype1, N_rank1> <= float
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<float>,
      LessOrEqual<T_numtype1, float > > >
operator<=(const Array<T_numtype1, N_rank1>& d1, 
      float d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<float>, 
      LessOrEqual<T_numtype1, float> >
      (d1.begin(), 
      _bz_ArrayExprConstant<float>(d2));
}

// Array<T_numtype1, N_rank1> <= double
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<double>,
      LessOrEqual<T_numtype1, double > > >
operator<=(const Array<T_numtype1, N_rank1>& d1, 
      double d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<double>, 
      LessOrEqual<T_numtype1, double> >
      (d1.begin(), 
      _bz_ArrayExprConstant<double>(d2));
}

// Array<T_numtype1, N_rank1> <= long double
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<long double>,
      LessOrEqual<T_numtype1, long double > > >
operator<=(const Array<T_numtype1, N_rank1>& d1, 
      long double d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<long double>, 
      LessOrEqual<T_numtype1, long double> >
      (d1.begin(), 
      _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
// Array<T_numtype1, N_rank1> <= complex<T2>
template<class T_numtype1, int N_rank1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<complex<T2> > ,
      LessOrEqual<T_numtype1, complex<T2>  > > >
operator<=(const Array<T_numtype1, N_rank1>& d1, 
      complex<T2> d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<complex<T2> > , 
      LessOrEqual<T_numtype1, complex<T2> > >
      (d1.begin(), 
      _bz_ArrayExprConstant<complex<T2> > (d2));
}
#endif // BZ_HAVE_COMPLEX

// _bz_ArrayExpr<P_expr1> <= Array<T_numtype2, N_rank2>
template<class P_expr1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      LessOrEqual<_bz_typename P_expr1::T_numtype, T_numtype2 > > >
operator<=(_bz_ArrayExpr<P_expr1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      LessOrEqual<_bz_typename P_expr1::T_numtype, T_numtype2> >
      (d1, 
      d2.begin());
}

// _bz_ArrayExpr<P_expr1> <= _bz_ArrayExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>,
      LessOrEqual<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype > > >
operator<=(_bz_ArrayExpr<P_expr1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>, 
      LessOrEqual<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> <= IndexPlaceholder<N_index2>
template<class P_expr1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>,
      LessOrEqual<_bz_typename P_expr1::T_numtype, int > > >
operator<=(_bz_ArrayExpr<P_expr1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>, 
      LessOrEqual<_bz_typename P_expr1::T_numtype, int> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> <= int
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>,
      LessOrEqual<_bz_typename P_expr1::T_numtype, int > > >
operator<=(_bz_ArrayExpr<P_expr1> d1, 
      int d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>, 
      LessOrEqual<_bz_typename P_expr1::T_numtype, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// _bz_ArrayExpr<P_expr1> <= float
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<float>,
      LessOrEqual<_bz_typename P_expr1::T_numtype, float > > >
operator<=(_bz_ArrayExpr<P_expr1> d1, 
      float d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<float>, 
      LessOrEqual<_bz_typename P_expr1::T_numtype, float> >
      (d1, 
      _bz_ArrayExprConstant<float>(d2));
}

// _bz_ArrayExpr<P_expr1> <= double
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<double>,
      LessOrEqual<_bz_typename P_expr1::T_numtype, double > > >
operator<=(_bz_ArrayExpr<P_expr1> d1, 
      double d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<double>, 
      LessOrEqual<_bz_typename P_expr1::T_numtype, double> >
      (d1, 
      _bz_ArrayExprConstant<double>(d2));
}

// _bz_ArrayExpr<P_expr1> <= long double
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<long double>,
      LessOrEqual<_bz_typename P_expr1::T_numtype, long double > > >
operator<=(_bz_ArrayExpr<P_expr1> d1, 
      long double d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<long double>, 
      LessOrEqual<_bz_typename P_expr1::T_numtype, long double> >
      (d1, 
      _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
// _bz_ArrayExpr<P_expr1> <= complex<T2>
template<class P_expr1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<complex<T2> > ,
      LessOrEqual<_bz_typename P_expr1::T_numtype, complex<T2>  > > >
operator<=(_bz_ArrayExpr<P_expr1> d1, 
      complex<T2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<complex<T2> > , 
      LessOrEqual<_bz_typename P_expr1::T_numtype, complex<T2> > >
      (d1, 
      _bz_ArrayExprConstant<complex<T2> > (d2));
}
#endif // BZ_HAVE_COMPLEX

// IndexPlaceholder<N_index1> <= Array<T_numtype2, N_rank2>
template<int N_index1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      LessOrEqual<int, T_numtype2 > > >
operator<=(IndexPlaceholder<N_index1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      LessOrEqual<int, T_numtype2> >
      (d1, 
      d2.begin());
}

// IndexPlaceholder<N_index1> <= _bz_ArrayExpr<P_expr2>
template<int N_index1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>,
      LessOrEqual<int, _bz_typename P_expr2::T_numtype > > >
operator<=(IndexPlaceholder<N_index1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>, 
      LessOrEqual<int, _bz_typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> <= IndexPlaceholder<N_index2>
template<int N_index1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>,
      LessOrEqual<int, int > > >
operator<=(IndexPlaceholder<N_index1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>, 
      LessOrEqual<int, int> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> <= int
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>,
      LessOrEqual<int, int > > >
operator<=(IndexPlaceholder<N_index1> d1, 
      int d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>, 
      LessOrEqual<int, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// IndexPlaceholder<N_index1> <= float
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<float>,
      LessOrEqual<int, float > > >
operator<=(IndexPlaceholder<N_index1> d1, 
      float d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<float>, 
      LessOrEqual<int, float> >
      (d1, 
      _bz_ArrayExprConstant<float>(d2));
}

// IndexPlaceholder<N_index1> <= double
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<double>,
      LessOrEqual<int, double > > >
operator<=(IndexPlaceholder<N_index1> d1, 
      double d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<double>, 
      LessOrEqual<int, double> >
      (d1, 
      _bz_ArrayExprConstant<double>(d2));
}

// IndexPlaceholder<N_index1> <= long double
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<long double>,
      LessOrEqual<int, long double > > >
operator<=(IndexPlaceholder<N_index1> d1, 
      long double d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<long double>, 
      LessOrEqual<int, long double> >
      (d1, 
      _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
// IndexPlaceholder<N_index1> <= complex<T2>
template<int N_index1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<complex<T2> > ,
      LessOrEqual<int, complex<T2>  > > >
operator<=(IndexPlaceholder<N_index1> d1, 
      complex<T2> d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<complex<T2> > , 
      LessOrEqual<int, complex<T2> > >
      (d1, 
      _bz_ArrayExprConstant<complex<T2> > (d2));
}
#endif // BZ_HAVE_COMPLEX

// int <= Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>,
      LessOrEqual<int, T_numtype2 > > >
operator<=(int d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      LessOrEqual<int, T_numtype2> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2.begin());
}

// int <= _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>,
      LessOrEqual<int, _bz_typename P_expr2::T_numtype > > >
operator<=(int d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>, 
      LessOrEqual<int, _bz_typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2);
}

// int <= IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      IndexPlaceholder<N_index2>,
      LessOrEqual<int, int > > >
operator<=(int d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      IndexPlaceholder<N_index2>, 
      LessOrEqual<int, int> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2);
}

// float <= Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<float>, 
      ArrayIterator<T_numtype2, N_rank2>,
      LessOrEqual<float, T_numtype2 > > >
operator<=(float d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<float>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      LessOrEqual<float, T_numtype2> >
      (_bz_ArrayExprConstant<float>(d1), 
      d2.begin());
}

// float <= _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<float>, 
      _bz_ArrayExpr<P_expr2>,
      LessOrEqual<float, _bz_typename P_expr2::T_numtype > > >
operator<=(float d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<float>, 
      _bz_ArrayExpr<P_expr2>, 
      LessOrEqual<float, _bz_typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<float>(d1), 
      d2);
}

// float <= IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<float>, 
      IndexPlaceholder<N_index2>,
      LessOrEqual<float, int > > >
operator<=(float d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<float>, 
      IndexPlaceholder<N_index2>, 
      LessOrEqual<float, int> >
      (_bz_ArrayExprConstant<float>(d1), 
      d2);
}

// double <= Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<double>, 
      ArrayIterator<T_numtype2, N_rank2>,
      LessOrEqual<double, T_numtype2 > > >
operator<=(double d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<double>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      LessOrEqual<double, T_numtype2> >
      (_bz_ArrayExprConstant<double>(d1), 
      d2.begin());
}

// double <= _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<double>, 
      _bz_ArrayExpr<P_expr2>,
      LessOrEqual<double, _bz_typename P_expr2::T_numtype > > >
operator<=(double d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<double>, 
      _bz_ArrayExpr<P_expr2>, 
      LessOrEqual<double, _bz_typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<double>(d1), 
      d2);
}

// double <= IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<double>, 
      IndexPlaceholder<N_index2>,
      LessOrEqual<double, int > > >
operator<=(double d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<double>, 
      IndexPlaceholder<N_index2>, 
      LessOrEqual<double, int> >
      (_bz_ArrayExprConstant<double>(d1), 
      d2);
}

// long double <= Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<long double>, 
      ArrayIterator<T_numtype2, N_rank2>,
      LessOrEqual<long double, T_numtype2 > > >
operator<=(long double d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<long double>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      LessOrEqual<long double, T_numtype2> >
      (_bz_ArrayExprConstant<long double>(d1), 
      d2.begin());
}

// long double <= _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<long double>, 
      _bz_ArrayExpr<P_expr2>,
      LessOrEqual<long double, _bz_typename P_expr2::T_numtype > > >
operator<=(long double d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<long double>, 
      _bz_ArrayExpr<P_expr2>, 
      LessOrEqual<long double, _bz_typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<long double>(d1), 
      d2);
}

// long double <= IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<long double>, 
      IndexPlaceholder<N_index2>,
      LessOrEqual<long double, int > > >
operator<=(long double d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<long double>, 
      IndexPlaceholder<N_index2>, 
      LessOrEqual<long double, int> >
      (_bz_ArrayExprConstant<long double>(d1), 
      d2);
}

#ifdef BZ_HAVE_COMPLEX
// complex<T1> <= Array<T_numtype2, N_rank2>
template<class T1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<complex<T1> > , 
      ArrayIterator<T_numtype2, N_rank2>,
      LessOrEqual<complex<T1> , T_numtype2 > > >
operator<=(complex<T1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<complex<T1> > , 
      ArrayIterator<T_numtype2, N_rank2>, 
      LessOrEqual<complex<T1> , T_numtype2> >
      (_bz_ArrayExprConstant<complex<T1> > (d1), 
      d2.begin());
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX
// complex<T1> <= _bz_ArrayExpr<P_expr2>
template<class T1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<complex<T1> > , 
      _bz_ArrayExpr<P_expr2>,
      LessOrEqual<complex<T1> , _bz_typename P_expr2::T_numtype > > >
operator<=(complex<T1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<complex<T1> > , 
      _bz_ArrayExpr<P_expr2>, 
      LessOrEqual<complex<T1> , _bz_typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<complex<T1> > (d1), 
      d2);
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX
// complex<T1> <= IndexPlaceholder<N_index2>
template<class T1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<complex<T1> > , 
      IndexPlaceholder<N_index2>,
      LessOrEqual<complex<T1> , int > > >
operator<=(complex<T1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<complex<T1> > , 
      IndexPlaceholder<N_index2>, 
      LessOrEqual<complex<T1> , int> >
      (_bz_ArrayExprConstant<complex<T1> > (d1), 
      d2);
}
#endif // BZ_HAVE_COMPLEX
/****************************************************************************
 * Equality operators
 ****************************************************************************/

// Array<T_numtype1, N_rank1> == Array<T_numtype2, N_rank2>
template<class T_numtype1, int N_rank1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Equal<T_numtype1, T_numtype2 > > >
operator==(const Array<T_numtype1, N_rank1>& d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Equal<T_numtype1, T_numtype2> >
      (d1.begin(), 
      d2.begin());
}

// Array<T_numtype1, N_rank1> == _bz_ArrayExpr<P_expr2>
template<class T_numtype1, int N_rank1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>,
      Equal<T_numtype1, _bz_typename P_expr2::T_numtype > > >
operator==(const Array<T_numtype1, N_rank1>& d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>, 
      Equal<T_numtype1, _bz_typename P_expr2::T_numtype> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> == IndexPlaceholder<N_index2>
template<class T_numtype1, int N_rank1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>,
      Equal<T_numtype1, int > > >
operator==(const Array<T_numtype1, N_rank1>& d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>, 
      Equal<T_numtype1, int> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> == int
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>,
      Equal<T_numtype1, int > > >
operator==(const Array<T_numtype1, N_rank1>& d1, 
      int d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>, 
      Equal<T_numtype1, int> >
      (d1.begin(), 
      _bz_ArrayExprConstant<int>(d2));
}

// Array<T_numtype1, N_rank1> == float
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<float>,
      Equal<T_numtype1, float > > >
operator==(const Array<T_numtype1, N_rank1>& d1, 
      float d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<float>, 
      Equal<T_numtype1, float> >
      (d1.begin(), 
      _bz_ArrayExprConstant<float>(d2));
}

// Array<T_numtype1, N_rank1> == double
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<double>,
      Equal<T_numtype1, double > > >
operator==(const Array<T_numtype1, N_rank1>& d1, 
      double d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<double>, 
      Equal<T_numtype1, double> >
      (d1.begin(), 
      _bz_ArrayExprConstant<double>(d2));
}

// Array<T_numtype1, N_rank1> == long double
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<long double>,
      Equal<T_numtype1, long double > > >
operator==(const Array<T_numtype1, N_rank1>& d1, 
      long double d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<long double>, 
      Equal<T_numtype1, long double> >
      (d1.begin(), 
      _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
// Array<T_numtype1, N_rank1> == complex<T2>
template<class T_numtype1, int N_rank1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<complex<T2> > ,
      Equal<T_numtype1, complex<T2>  > > >
operator==(const Array<T_numtype1, N_rank1>& d1, 
      complex<T2> d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<complex<T2> > , 
      Equal<T_numtype1, complex<T2> > >
      (d1.begin(), 
      _bz_ArrayExprConstant<complex<T2> > (d2));
}
#endif // BZ_HAVE_COMPLEX

// _bz_ArrayExpr<P_expr1> == Array<T_numtype2, N_rank2>
template<class P_expr1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Equal<_bz_typename P_expr1::T_numtype, T_numtype2 > > >
operator==(_bz_ArrayExpr<P_expr1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Equal<_bz_typename P_expr1::T_numtype, T_numtype2> >
      (d1, 
      d2.begin());
}

// _bz_ArrayExpr<P_expr1> == _bz_ArrayExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>,
      Equal<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype > > >
operator==(_bz_ArrayExpr<P_expr1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>, 
      Equal<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> == IndexPlaceholder<N_index2>
template<class P_expr1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>,
      Equal<_bz_typename P_expr1::T_numtype, int > > >
operator==(_bz_ArrayExpr<P_expr1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>, 
      Equal<_bz_typename P_expr1::T_numtype, int> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> == int
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>,
      Equal<_bz_typename P_expr1::T_numtype, int > > >
operator==(_bz_ArrayExpr<P_expr1> d1, 
      int d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>, 
      Equal<_bz_typename P_expr1::T_numtype, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// _bz_ArrayExpr<P_expr1> == float
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<float>,
      Equal<_bz_typename P_expr1::T_numtype, float > > >
operator==(_bz_ArrayExpr<P_expr1> d1, 
      float d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<float>, 
      Equal<_bz_typename P_expr1::T_numtype, float> >
      (d1, 
      _bz_ArrayExprConstant<float>(d2));
}

// _bz_ArrayExpr<P_expr1> == double
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<double>,
      Equal<_bz_typename P_expr1::T_numtype, double > > >
operator==(_bz_ArrayExpr<P_expr1> d1, 
      double d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<double>, 
      Equal<_bz_typename P_expr1::T_numtype, double> >
      (d1, 
      _bz_ArrayExprConstant<double>(d2));
}

// _bz_ArrayExpr<P_expr1> == long double
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<long double>,
      Equal<_bz_typename P_expr1::T_numtype, long double > > >
operator==(_bz_ArrayExpr<P_expr1> d1, 
      long double d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<long double>, 
      Equal<_bz_typename P_expr1::T_numtype, long double> >
      (d1, 
      _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
// _bz_ArrayExpr<P_expr1> == complex<T2>
template<class P_expr1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<complex<T2> > ,
      Equal<_bz_typename P_expr1::T_numtype, complex<T2>  > > >
operator==(_bz_ArrayExpr<P_expr1> d1, 
      complex<T2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<complex<T2> > , 
      Equal<_bz_typename P_expr1::T_numtype, complex<T2> > >
      (d1, 
      _bz_ArrayExprConstant<complex<T2> > (d2));
}
#endif // BZ_HAVE_COMPLEX

// IndexPlaceholder<N_index1> == Array<T_numtype2, N_rank2>
template<int N_index1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Equal<int, T_numtype2 > > >
operator==(IndexPlaceholder<N_index1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Equal<int, T_numtype2> >
      (d1, 
      d2.begin());
}

// IndexPlaceholder<N_index1> == _bz_ArrayExpr<P_expr2>
template<int N_index1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>,
      Equal<int, _bz_typename P_expr2::T_numtype > > >
operator==(IndexPlaceholder<N_index1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>, 
      Equal<int, _bz_typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> == IndexPlaceholder<N_index2>
template<int N_index1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>,
      Equal<int, int > > >
operator==(IndexPlaceholder<N_index1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>, 
      Equal<int, int> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> == int
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>,
      Equal<int, int > > >
operator==(IndexPlaceholder<N_index1> d1, 
      int d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>, 
      Equal<int, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// IndexPlaceholder<N_index1> == float
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<float>,
      Equal<int, float > > >
operator==(IndexPlaceholder<N_index1> d1, 
      float d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<float>, 
      Equal<int, float> >
      (d1, 
      _bz_ArrayExprConstant<float>(d2));
}

// IndexPlaceholder<N_index1> == double
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<double>,
      Equal<int, double > > >
operator==(IndexPlaceholder<N_index1> d1, 
      double d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<double>, 
      Equal<int, double> >
      (d1, 
      _bz_ArrayExprConstant<double>(d2));
}

// IndexPlaceholder<N_index1> == long double
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<long double>,
      Equal<int, long double > > >
operator==(IndexPlaceholder<N_index1> d1, 
      long double d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<long double>, 
      Equal<int, long double> >
      (d1, 
      _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
// IndexPlaceholder<N_index1> == complex<T2>
template<int N_index1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<complex<T2> > ,
      Equal<int, complex<T2>  > > >
operator==(IndexPlaceholder<N_index1> d1, 
      complex<T2> d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<complex<T2> > , 
      Equal<int, complex<T2> > >
      (d1, 
      _bz_ArrayExprConstant<complex<T2> > (d2));
}
#endif // BZ_HAVE_COMPLEX

// int == Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Equal<int, T_numtype2 > > >
operator==(int d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Equal<int, T_numtype2> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2.begin());
}

// int == _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>,
      Equal<int, _bz_typename P_expr2::T_numtype > > >
operator==(int d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>, 
      Equal<int, _bz_typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2);
}

// int == IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      IndexPlaceholder<N_index2>,
      Equal<int, int > > >
operator==(int d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      IndexPlaceholder<N_index2>, 
      Equal<int, int> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2);
}

// float == Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<float>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Equal<float, T_numtype2 > > >
operator==(float d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<float>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Equal<float, T_numtype2> >
      (_bz_ArrayExprConstant<float>(d1), 
      d2.begin());
}

// float == _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<float>, 
      _bz_ArrayExpr<P_expr2>,
      Equal<float, _bz_typename P_expr2::T_numtype > > >
operator==(float d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<float>, 
      _bz_ArrayExpr<P_expr2>, 
      Equal<float, _bz_typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<float>(d1), 
      d2);
}

// float == IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<float>, 
      IndexPlaceholder<N_index2>,
      Equal<float, int > > >
operator==(float d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<float>, 
      IndexPlaceholder<N_index2>, 
      Equal<float, int> >
      (_bz_ArrayExprConstant<float>(d1), 
      d2);
}

// double == Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<double>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Equal<double, T_numtype2 > > >
operator==(double d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<double>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Equal<double, T_numtype2> >
      (_bz_ArrayExprConstant<double>(d1), 
      d2.begin());
}

// double == _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<double>, 
      _bz_ArrayExpr<P_expr2>,
      Equal<double, _bz_typename P_expr2::T_numtype > > >
operator==(double d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<double>, 
      _bz_ArrayExpr<P_expr2>, 
      Equal<double, _bz_typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<double>(d1), 
      d2);
}

// double == IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<double>, 
      IndexPlaceholder<N_index2>,
      Equal<double, int > > >
operator==(double d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<double>, 
      IndexPlaceholder<N_index2>, 
      Equal<double, int> >
      (_bz_ArrayExprConstant<double>(d1), 
      d2);
}

// long double == Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<long double>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Equal<long double, T_numtype2 > > >
operator==(long double d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<long double>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Equal<long double, T_numtype2> >
      (_bz_ArrayExprConstant<long double>(d1), 
      d2.begin());
}

// long double == _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<long double>, 
      _bz_ArrayExpr<P_expr2>,
      Equal<long double, _bz_typename P_expr2::T_numtype > > >
operator==(long double d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<long double>, 
      _bz_ArrayExpr<P_expr2>, 
      Equal<long double, _bz_typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<long double>(d1), 
      d2);
}

// long double == IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<long double>, 
      IndexPlaceholder<N_index2>,
      Equal<long double, int > > >
operator==(long double d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<long double>, 
      IndexPlaceholder<N_index2>, 
      Equal<long double, int> >
      (_bz_ArrayExprConstant<long double>(d1), 
      d2);
}

#ifdef BZ_HAVE_COMPLEX
// complex<T1> == Array<T_numtype2, N_rank2>
template<class T1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<complex<T1> > , 
      ArrayIterator<T_numtype2, N_rank2>,
      Equal<complex<T1> , T_numtype2 > > >
operator==(complex<T1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<complex<T1> > , 
      ArrayIterator<T_numtype2, N_rank2>, 
      Equal<complex<T1> , T_numtype2> >
      (_bz_ArrayExprConstant<complex<T1> > (d1), 
      d2.begin());
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX
// complex<T1> == _bz_ArrayExpr<P_expr2>
template<class T1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<complex<T1> > , 
      _bz_ArrayExpr<P_expr2>,
      Equal<complex<T1> , _bz_typename P_expr2::T_numtype > > >
operator==(complex<T1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<complex<T1> > , 
      _bz_ArrayExpr<P_expr2>, 
      Equal<complex<T1> , _bz_typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<complex<T1> > (d1), 
      d2);
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX
// complex<T1> == IndexPlaceholder<N_index2>
template<class T1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<complex<T1> > , 
      IndexPlaceholder<N_index2>,
      Equal<complex<T1> , int > > >
operator==(complex<T1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<complex<T1> > , 
      IndexPlaceholder<N_index2>, 
      Equal<complex<T1> , int> >
      (_bz_ArrayExprConstant<complex<T1> > (d1), 
      d2);
}
#endif // BZ_HAVE_COMPLEX
/****************************************************************************
 * Not-equal operators
 ****************************************************************************/

// Array<T_numtype1, N_rank1> != Array<T_numtype2, N_rank2>
template<class T_numtype1, int N_rank1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      NotEqual<T_numtype1, T_numtype2 > > >
operator!=(const Array<T_numtype1, N_rank1>& d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      NotEqual<T_numtype1, T_numtype2> >
      (d1.begin(), 
      d2.begin());
}

// Array<T_numtype1, N_rank1> != _bz_ArrayExpr<P_expr2>
template<class T_numtype1, int N_rank1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>,
      NotEqual<T_numtype1, _bz_typename P_expr2::T_numtype > > >
operator!=(const Array<T_numtype1, N_rank1>& d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>, 
      NotEqual<T_numtype1, _bz_typename P_expr2::T_numtype> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> != IndexPlaceholder<N_index2>
template<class T_numtype1, int N_rank1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>,
      NotEqual<T_numtype1, int > > >
operator!=(const Array<T_numtype1, N_rank1>& d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>, 
      NotEqual<T_numtype1, int> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> != int
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>,
      NotEqual<T_numtype1, int > > >
operator!=(const Array<T_numtype1, N_rank1>& d1, 
      int d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>, 
      NotEqual<T_numtype1, int> >
      (d1.begin(), 
      _bz_ArrayExprConstant<int>(d2));
}

// Array<T_numtype1, N_rank1> != float
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<float>,
      NotEqual<T_numtype1, float > > >
operator!=(const Array<T_numtype1, N_rank1>& d1, 
      float d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<float>, 
      NotEqual<T_numtype1, float> >
      (d1.begin(), 
      _bz_ArrayExprConstant<float>(d2));
}

// Array<T_numtype1, N_rank1> != double
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<double>,
      NotEqual<T_numtype1, double > > >
operator!=(const Array<T_numtype1, N_rank1>& d1, 
      double d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<double>, 
      NotEqual<T_numtype1, double> >
      (d1.begin(), 
      _bz_ArrayExprConstant<double>(d2));
}

// Array<T_numtype1, N_rank1> != long double
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<long double>,
      NotEqual<T_numtype1, long double > > >
operator!=(const Array<T_numtype1, N_rank1>& d1, 
      long double d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<long double>, 
      NotEqual<T_numtype1, long double> >
      (d1.begin(), 
      _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
// Array<T_numtype1, N_rank1> != complex<T2>
template<class T_numtype1, int N_rank1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<complex<T2> > ,
      NotEqual<T_numtype1, complex<T2>  > > >
operator!=(const Array<T_numtype1, N_rank1>& d1, 
      complex<T2> d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<complex<T2> > , 
      NotEqual<T_numtype1, complex<T2> > >
      (d1.begin(), 
      _bz_ArrayExprConstant<complex<T2> > (d2));
}
#endif // BZ_HAVE_COMPLEX

// _bz_ArrayExpr<P_expr1> != Array<T_numtype2, N_rank2>
template<class P_expr1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      NotEqual<_bz_typename P_expr1::T_numtype, T_numtype2 > > >
operator!=(_bz_ArrayExpr<P_expr1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      NotEqual<_bz_typename P_expr1::T_numtype, T_numtype2> >
      (d1, 
      d2.begin());
}

// _bz_ArrayExpr<P_expr1> != _bz_ArrayExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>,
      NotEqual<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype > > >
operator!=(_bz_ArrayExpr<P_expr1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>, 
      NotEqual<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> != IndexPlaceholder<N_index2>
template<class P_expr1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>,
      NotEqual<_bz_typename P_expr1::T_numtype, int > > >
operator!=(_bz_ArrayExpr<P_expr1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>, 
      NotEqual<_bz_typename P_expr1::T_numtype, int> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> != int
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>,
      NotEqual<_bz_typename P_expr1::T_numtype, int > > >
operator!=(_bz_ArrayExpr<P_expr1> d1, 
      int d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>, 
      NotEqual<_bz_typename P_expr1::T_numtype, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// _bz_ArrayExpr<P_expr1> != float
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<float>,
      NotEqual<_bz_typename P_expr1::T_numtype, float > > >
operator!=(_bz_ArrayExpr<P_expr1> d1, 
      float d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<float>, 
      NotEqual<_bz_typename P_expr1::T_numtype, float> >
      (d1, 
      _bz_ArrayExprConstant<float>(d2));
}

// _bz_ArrayExpr<P_expr1> != double
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<double>,
      NotEqual<_bz_typename P_expr1::T_numtype, double > > >
operator!=(_bz_ArrayExpr<P_expr1> d1, 
      double d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<double>, 
      NotEqual<_bz_typename P_expr1::T_numtype, double> >
      (d1, 
      _bz_ArrayExprConstant<double>(d2));
}

// _bz_ArrayExpr<P_expr1> != long double
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<long double>,
      NotEqual<_bz_typename P_expr1::T_numtype, long double > > >
operator!=(_bz_ArrayExpr<P_expr1> d1, 
      long double d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<long double>, 
      NotEqual<_bz_typename P_expr1::T_numtype, long double> >
      (d1, 
      _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
// _bz_ArrayExpr<P_expr1> != complex<T2>
template<class P_expr1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<complex<T2> > ,
      NotEqual<_bz_typename P_expr1::T_numtype, complex<T2>  > > >
operator!=(_bz_ArrayExpr<P_expr1> d1, 
      complex<T2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<complex<T2> > , 
      NotEqual<_bz_typename P_expr1::T_numtype, complex<T2> > >
      (d1, 
      _bz_ArrayExprConstant<complex<T2> > (d2));
}
#endif // BZ_HAVE_COMPLEX

// IndexPlaceholder<N_index1> != Array<T_numtype2, N_rank2>
template<int N_index1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      NotEqual<int, T_numtype2 > > >
operator!=(IndexPlaceholder<N_index1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      NotEqual<int, T_numtype2> >
      (d1, 
      d2.begin());
}

// IndexPlaceholder<N_index1> != _bz_ArrayExpr<P_expr2>
template<int N_index1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>,
      NotEqual<int, _bz_typename P_expr2::T_numtype > > >
operator!=(IndexPlaceholder<N_index1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>, 
      NotEqual<int, _bz_typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> != IndexPlaceholder<N_index2>
template<int N_index1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>,
      NotEqual<int, int > > >
operator!=(IndexPlaceholder<N_index1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>, 
      NotEqual<int, int> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> != int
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>,
      NotEqual<int, int > > >
operator!=(IndexPlaceholder<N_index1> d1, 
      int d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>, 
      NotEqual<int, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// IndexPlaceholder<N_index1> != float
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<float>,
      NotEqual<int, float > > >
operator!=(IndexPlaceholder<N_index1> d1, 
      float d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<float>, 
      NotEqual<int, float> >
      (d1, 
      _bz_ArrayExprConstant<float>(d2));
}

// IndexPlaceholder<N_index1> != double
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<double>,
      NotEqual<int, double > > >
operator!=(IndexPlaceholder<N_index1> d1, 
      double d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<double>, 
      NotEqual<int, double> >
      (d1, 
      _bz_ArrayExprConstant<double>(d2));
}

// IndexPlaceholder<N_index1> != long double
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<long double>,
      NotEqual<int, long double > > >
operator!=(IndexPlaceholder<N_index1> d1, 
      long double d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<long double>, 
      NotEqual<int, long double> >
      (d1, 
      _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
// IndexPlaceholder<N_index1> != complex<T2>
template<int N_index1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<complex<T2> > ,
      NotEqual<int, complex<T2>  > > >
operator!=(IndexPlaceholder<N_index1> d1, 
      complex<T2> d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<complex<T2> > , 
      NotEqual<int, complex<T2> > >
      (d1, 
      _bz_ArrayExprConstant<complex<T2> > (d2));
}
#endif // BZ_HAVE_COMPLEX

// int != Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>,
      NotEqual<int, T_numtype2 > > >
operator!=(int d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      NotEqual<int, T_numtype2> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2.begin());
}

// int != _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>,
      NotEqual<int, _bz_typename P_expr2::T_numtype > > >
operator!=(int d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>, 
      NotEqual<int, _bz_typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2);
}

// int != IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      IndexPlaceholder<N_index2>,
      NotEqual<int, int > > >
operator!=(int d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      IndexPlaceholder<N_index2>, 
      NotEqual<int, int> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2);
}

// float != Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<float>, 
      ArrayIterator<T_numtype2, N_rank2>,
      NotEqual<float, T_numtype2 > > >
operator!=(float d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<float>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      NotEqual<float, T_numtype2> >
      (_bz_ArrayExprConstant<float>(d1), 
      d2.begin());
}

// float != _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<float>, 
      _bz_ArrayExpr<P_expr2>,
      NotEqual<float, _bz_typename P_expr2::T_numtype > > >
operator!=(float d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<float>, 
      _bz_ArrayExpr<P_expr2>, 
      NotEqual<float, _bz_typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<float>(d1), 
      d2);
}

// float != IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<float>, 
      IndexPlaceholder<N_index2>,
      NotEqual<float, int > > >
operator!=(float d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<float>, 
      IndexPlaceholder<N_index2>, 
      NotEqual<float, int> >
      (_bz_ArrayExprConstant<float>(d1), 
      d2);
}

// double != Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<double>, 
      ArrayIterator<T_numtype2, N_rank2>,
      NotEqual<double, T_numtype2 > > >
operator!=(double d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<double>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      NotEqual<double, T_numtype2> >
      (_bz_ArrayExprConstant<double>(d1), 
      d2.begin());
}

// double != _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<double>, 
      _bz_ArrayExpr<P_expr2>,
      NotEqual<double, _bz_typename P_expr2::T_numtype > > >
operator!=(double d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<double>, 
      _bz_ArrayExpr<P_expr2>, 
      NotEqual<double, _bz_typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<double>(d1), 
      d2);
}

// double != IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<double>, 
      IndexPlaceholder<N_index2>,
      NotEqual<double, int > > >
operator!=(double d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<double>, 
      IndexPlaceholder<N_index2>, 
      NotEqual<double, int> >
      (_bz_ArrayExprConstant<double>(d1), 
      d2);
}

// long double != Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<long double>, 
      ArrayIterator<T_numtype2, N_rank2>,
      NotEqual<long double, T_numtype2 > > >
operator!=(long double d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<long double>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      NotEqual<long double, T_numtype2> >
      (_bz_ArrayExprConstant<long double>(d1), 
      d2.begin());
}

// long double != _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<long double>, 
      _bz_ArrayExpr<P_expr2>,
      NotEqual<long double, _bz_typename P_expr2::T_numtype > > >
operator!=(long double d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<long double>, 
      _bz_ArrayExpr<P_expr2>, 
      NotEqual<long double, _bz_typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<long double>(d1), 
      d2);
}

// long double != IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<long double>, 
      IndexPlaceholder<N_index2>,
      NotEqual<long double, int > > >
operator!=(long double d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<long double>, 
      IndexPlaceholder<N_index2>, 
      NotEqual<long double, int> >
      (_bz_ArrayExprConstant<long double>(d1), 
      d2);
}

#ifdef BZ_HAVE_COMPLEX
// complex<T1> != Array<T_numtype2, N_rank2>
template<class T1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<complex<T1> > , 
      ArrayIterator<T_numtype2, N_rank2>,
      NotEqual<complex<T1> , T_numtype2 > > >
operator!=(complex<T1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<complex<T1> > , 
      ArrayIterator<T_numtype2, N_rank2>, 
      NotEqual<complex<T1> , T_numtype2> >
      (_bz_ArrayExprConstant<complex<T1> > (d1), 
      d2.begin());
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX
// complex<T1> != _bz_ArrayExpr<P_expr2>
template<class T1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<complex<T1> > , 
      _bz_ArrayExpr<P_expr2>,
      NotEqual<complex<T1> , _bz_typename P_expr2::T_numtype > > >
operator!=(complex<T1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<complex<T1> > , 
      _bz_ArrayExpr<P_expr2>, 
      NotEqual<complex<T1> , _bz_typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<complex<T1> > (d1), 
      d2);
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX
// complex<T1> != IndexPlaceholder<N_index2>
template<class T1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<complex<T1> > , 
      IndexPlaceholder<N_index2>,
      NotEqual<complex<T1> , int > > >
operator!=(complex<T1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<complex<T1> > , 
      IndexPlaceholder<N_index2>, 
      NotEqual<complex<T1> , int> >
      (_bz_ArrayExprConstant<complex<T1> > (d1), 
      d2);
}
#endif // BZ_HAVE_COMPLEX
/****************************************************************************
 * Logical AND operators
 ****************************************************************************/

// Array<T_numtype1, N_rank1> && Array<T_numtype2, N_rank2>
template<class T_numtype1, int N_rank1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      LogicalAnd<T_numtype1, T_numtype2 > > >
operator&&(const Array<T_numtype1, N_rank1>& d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      LogicalAnd<T_numtype1, T_numtype2> >
      (d1.begin(), 
      d2.begin());
}

// Array<T_numtype1, N_rank1> && _bz_ArrayExpr<P_expr2>
template<class T_numtype1, int N_rank1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>,
      LogicalAnd<T_numtype1, _bz_typename P_expr2::T_numtype > > >
operator&&(const Array<T_numtype1, N_rank1>& d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>, 
      LogicalAnd<T_numtype1, _bz_typename P_expr2::T_numtype> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> && IndexPlaceholder<N_index2>
template<class T_numtype1, int N_rank1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>,
      LogicalAnd<T_numtype1, int > > >
operator&&(const Array<T_numtype1, N_rank1>& d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>, 
      LogicalAnd<T_numtype1, int> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> && int
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>,
      LogicalAnd<T_numtype1, int > > >
operator&&(const Array<T_numtype1, N_rank1>& d1, 
      int d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>, 
      LogicalAnd<T_numtype1, int> >
      (d1.begin(), 
      _bz_ArrayExprConstant<int>(d2));
}

// _bz_ArrayExpr<P_expr1> && Array<T_numtype2, N_rank2>
template<class P_expr1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      LogicalAnd<_bz_typename P_expr1::T_numtype, T_numtype2 > > >
operator&&(_bz_ArrayExpr<P_expr1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      LogicalAnd<_bz_typename P_expr1::T_numtype, T_numtype2> >
      (d1, 
      d2.begin());
}

// _bz_ArrayExpr<P_expr1> && _bz_ArrayExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>,
      LogicalAnd<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype > > >
operator&&(_bz_ArrayExpr<P_expr1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>, 
      LogicalAnd<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> && IndexPlaceholder<N_index2>
template<class P_expr1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>,
      LogicalAnd<_bz_typename P_expr1::T_numtype, int > > >
operator&&(_bz_ArrayExpr<P_expr1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>, 
      LogicalAnd<_bz_typename P_expr1::T_numtype, int> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> && int
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>,
      LogicalAnd<_bz_typename P_expr1::T_numtype, int > > >
operator&&(_bz_ArrayExpr<P_expr1> d1, 
      int d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>, 
      LogicalAnd<_bz_typename P_expr1::T_numtype, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// IndexPlaceholder<N_index1> && Array<T_numtype2, N_rank2>
template<int N_index1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      LogicalAnd<int, T_numtype2 > > >
operator&&(IndexPlaceholder<N_index1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      LogicalAnd<int, T_numtype2> >
      (d1, 
      d2.begin());
}

// IndexPlaceholder<N_index1> && _bz_ArrayExpr<P_expr2>
template<int N_index1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>,
      LogicalAnd<int, _bz_typename P_expr2::T_numtype > > >
operator&&(IndexPlaceholder<N_index1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>, 
      LogicalAnd<int, _bz_typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> && IndexPlaceholder<N_index2>
template<int N_index1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>,
      LogicalAnd<int, int > > >
operator&&(IndexPlaceholder<N_index1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>, 
      LogicalAnd<int, int> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> && int
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>,
      LogicalAnd<int, int > > >
operator&&(IndexPlaceholder<N_index1> d1, 
      int d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>, 
      LogicalAnd<int, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// int && Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>,
      LogicalAnd<int, T_numtype2 > > >
operator&&(int d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      LogicalAnd<int, T_numtype2> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2.begin());
}

// int && _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>,
      LogicalAnd<int, _bz_typename P_expr2::T_numtype > > >
operator&&(int d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>, 
      LogicalAnd<int, _bz_typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2);
}

// int && IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      IndexPlaceholder<N_index2>,
      LogicalAnd<int, int > > >
operator&&(int d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      IndexPlaceholder<N_index2>, 
      LogicalAnd<int, int> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2);
}
/****************************************************************************
 * Logical OR operators
 ****************************************************************************/

// Array<T_numtype1, N_rank1> || Array<T_numtype2, N_rank2>
template<class T_numtype1, int N_rank1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      LogicalOr<T_numtype1, T_numtype2 > > >
operator||(const Array<T_numtype1, N_rank1>& d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      LogicalOr<T_numtype1, T_numtype2> >
      (d1.begin(), 
      d2.begin());
}

// Array<T_numtype1, N_rank1> || _bz_ArrayExpr<P_expr2>
template<class T_numtype1, int N_rank1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>,
      LogicalOr<T_numtype1, _bz_typename P_expr2::T_numtype > > >
operator||(const Array<T_numtype1, N_rank1>& d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>, 
      LogicalOr<T_numtype1, _bz_typename P_expr2::T_numtype> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> || IndexPlaceholder<N_index2>
template<class T_numtype1, int N_rank1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>,
      LogicalOr<T_numtype1, int > > >
operator||(const Array<T_numtype1, N_rank1>& d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>, 
      LogicalOr<T_numtype1, int> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> || int
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>,
      LogicalOr<T_numtype1, int > > >
operator||(const Array<T_numtype1, N_rank1>& d1, 
      int d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>, 
      LogicalOr<T_numtype1, int> >
      (d1.begin(), 
      _bz_ArrayExprConstant<int>(d2));
}

// _bz_ArrayExpr<P_expr1> || Array<T_numtype2, N_rank2>
template<class P_expr1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      LogicalOr<_bz_typename P_expr1::T_numtype, T_numtype2 > > >
operator||(_bz_ArrayExpr<P_expr1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      LogicalOr<_bz_typename P_expr1::T_numtype, T_numtype2> >
      (d1, 
      d2.begin());
}

// _bz_ArrayExpr<P_expr1> || _bz_ArrayExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>,
      LogicalOr<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype > > >
operator||(_bz_ArrayExpr<P_expr1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>, 
      LogicalOr<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> || IndexPlaceholder<N_index2>
template<class P_expr1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>,
      LogicalOr<_bz_typename P_expr1::T_numtype, int > > >
operator||(_bz_ArrayExpr<P_expr1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>, 
      LogicalOr<_bz_typename P_expr1::T_numtype, int> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> || int
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>,
      LogicalOr<_bz_typename P_expr1::T_numtype, int > > >
operator||(_bz_ArrayExpr<P_expr1> d1, 
      int d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>, 
      LogicalOr<_bz_typename P_expr1::T_numtype, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// IndexPlaceholder<N_index1> || Array<T_numtype2, N_rank2>
template<int N_index1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      LogicalOr<int, T_numtype2 > > >
operator||(IndexPlaceholder<N_index1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      LogicalOr<int, T_numtype2> >
      (d1, 
      d2.begin());
}

// IndexPlaceholder<N_index1> || _bz_ArrayExpr<P_expr2>
template<int N_index1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>,
      LogicalOr<int, _bz_typename P_expr2::T_numtype > > >
operator||(IndexPlaceholder<N_index1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>, 
      LogicalOr<int, _bz_typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> || IndexPlaceholder<N_index2>
template<int N_index1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>,
      LogicalOr<int, int > > >
operator||(IndexPlaceholder<N_index1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>, 
      LogicalOr<int, int> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> || int
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>,
      LogicalOr<int, int > > >
operator||(IndexPlaceholder<N_index1> d1, 
      int d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>, 
      LogicalOr<int, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// int || Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>,
      LogicalOr<int, T_numtype2 > > >
operator||(int d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      LogicalOr<int, T_numtype2> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2.begin());
}

// int || _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>,
      LogicalOr<int, _bz_typename P_expr2::T_numtype > > >
operator||(int d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>, 
      LogicalOr<int, _bz_typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2);
}

// int || IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      IndexPlaceholder<N_index2>,
      LogicalOr<int, int > > >
operator||(int d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      IndexPlaceholder<N_index2>, 
      LogicalOr<int, int> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2);
}
/****************************************************************************
 * Bitwise XOR Operators
 ****************************************************************************/

// Array<T_numtype1, N_rank1> ^ Array<T_numtype2, N_rank2>
template<class T_numtype1, int N_rank1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      BitwiseXor<T_numtype1, T_numtype2 > > >
operator^(const Array<T_numtype1, N_rank1>& d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      BitwiseXor<T_numtype1, T_numtype2> >
      (d1.begin(), 
      d2.begin());
}

// Array<T_numtype1, N_rank1> ^ _bz_ArrayExpr<P_expr2>
template<class T_numtype1, int N_rank1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>,
      BitwiseXor<T_numtype1, _bz_typename P_expr2::T_numtype > > >
operator^(const Array<T_numtype1, N_rank1>& d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>, 
      BitwiseXor<T_numtype1, _bz_typename P_expr2::T_numtype> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> ^ IndexPlaceholder<N_index2>
template<class T_numtype1, int N_rank1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>,
      BitwiseXor<T_numtype1, int > > >
operator^(const Array<T_numtype1, N_rank1>& d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>, 
      BitwiseXor<T_numtype1, int> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> ^ int
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>,
      BitwiseXor<T_numtype1, int > > >
operator^(const Array<T_numtype1, N_rank1>& d1, 
      int d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>, 
      BitwiseXor<T_numtype1, int> >
      (d1.begin(), 
      _bz_ArrayExprConstant<int>(d2));
}

// _bz_ArrayExpr<P_expr1> ^ Array<T_numtype2, N_rank2>
template<class P_expr1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      BitwiseXor<_bz_typename P_expr1::T_numtype, T_numtype2 > > >
operator^(_bz_ArrayExpr<P_expr1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      BitwiseXor<_bz_typename P_expr1::T_numtype, T_numtype2> >
      (d1, 
      d2.begin());
}

// _bz_ArrayExpr<P_expr1> ^ _bz_ArrayExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>,
      BitwiseXor<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype > > >
operator^(_bz_ArrayExpr<P_expr1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>, 
      BitwiseXor<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> ^ IndexPlaceholder<N_index2>
template<class P_expr1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>,
      BitwiseXor<_bz_typename P_expr1::T_numtype, int > > >
operator^(_bz_ArrayExpr<P_expr1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>, 
      BitwiseXor<_bz_typename P_expr1::T_numtype, int> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> ^ int
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>,
      BitwiseXor<_bz_typename P_expr1::T_numtype, int > > >
operator^(_bz_ArrayExpr<P_expr1> d1, 
      int d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>, 
      BitwiseXor<_bz_typename P_expr1::T_numtype, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// IndexPlaceholder<N_index1> ^ Array<T_numtype2, N_rank2>
template<int N_index1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      BitwiseXor<int, T_numtype2 > > >
operator^(IndexPlaceholder<N_index1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      BitwiseXor<int, T_numtype2> >
      (d1, 
      d2.begin());
}

// IndexPlaceholder<N_index1> ^ _bz_ArrayExpr<P_expr2>
template<int N_index1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>,
      BitwiseXor<int, _bz_typename P_expr2::T_numtype > > >
operator^(IndexPlaceholder<N_index1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>, 
      BitwiseXor<int, _bz_typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> ^ IndexPlaceholder<N_index2>
template<int N_index1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>,
      BitwiseXor<int, int > > >
operator^(IndexPlaceholder<N_index1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>, 
      BitwiseXor<int, int> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> ^ int
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>,
      BitwiseXor<int, int > > >
operator^(IndexPlaceholder<N_index1> d1, 
      int d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>, 
      BitwiseXor<int, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// int ^ Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>,
      BitwiseXor<int, T_numtype2 > > >
operator^(int d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      BitwiseXor<int, T_numtype2> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2.begin());
}

// int ^ _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>,
      BitwiseXor<int, _bz_typename P_expr2::T_numtype > > >
operator^(int d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>, 
      BitwiseXor<int, _bz_typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2);
}

// int ^ IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      IndexPlaceholder<N_index2>,
      BitwiseXor<int, int > > >
operator^(int d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      IndexPlaceholder<N_index2>, 
      BitwiseXor<int, int> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2);
}
/****************************************************************************
 * Bitwise And Operators
 ****************************************************************************/

// Array<T_numtype1, N_rank1> & Array<T_numtype2, N_rank2>
template<class T_numtype1, int N_rank1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      BitwiseAnd<T_numtype1, T_numtype2 > > >
operator&(const Array<T_numtype1, N_rank1>& d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      BitwiseAnd<T_numtype1, T_numtype2> >
      (d1.begin(), 
      d2.begin());
}

// Array<T_numtype1, N_rank1> & _bz_ArrayExpr<P_expr2>
template<class T_numtype1, int N_rank1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>,
      BitwiseAnd<T_numtype1, _bz_typename P_expr2::T_numtype > > >
operator&(const Array<T_numtype1, N_rank1>& d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>, 
      BitwiseAnd<T_numtype1, _bz_typename P_expr2::T_numtype> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> & IndexPlaceholder<N_index2>
template<class T_numtype1, int N_rank1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>,
      BitwiseAnd<T_numtype1, int > > >
operator&(const Array<T_numtype1, N_rank1>& d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>, 
      BitwiseAnd<T_numtype1, int> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> & int
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>,
      BitwiseAnd<T_numtype1, int > > >
operator&(const Array<T_numtype1, N_rank1>& d1, 
      int d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>, 
      BitwiseAnd<T_numtype1, int> >
      (d1.begin(), 
      _bz_ArrayExprConstant<int>(d2));
}

// _bz_ArrayExpr<P_expr1> & Array<T_numtype2, N_rank2>
template<class P_expr1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      BitwiseAnd<_bz_typename P_expr1::T_numtype, T_numtype2 > > >
operator&(_bz_ArrayExpr<P_expr1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      BitwiseAnd<_bz_typename P_expr1::T_numtype, T_numtype2> >
      (d1, 
      d2.begin());
}

// _bz_ArrayExpr<P_expr1> & _bz_ArrayExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>,
      BitwiseAnd<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype > > >
operator&(_bz_ArrayExpr<P_expr1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>, 
      BitwiseAnd<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> & IndexPlaceholder<N_index2>
template<class P_expr1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>,
      BitwiseAnd<_bz_typename P_expr1::T_numtype, int > > >
operator&(_bz_ArrayExpr<P_expr1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>, 
      BitwiseAnd<_bz_typename P_expr1::T_numtype, int> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> & int
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>,
      BitwiseAnd<_bz_typename P_expr1::T_numtype, int > > >
operator&(_bz_ArrayExpr<P_expr1> d1, 
      int d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>, 
      BitwiseAnd<_bz_typename P_expr1::T_numtype, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// IndexPlaceholder<N_index1> & Array<T_numtype2, N_rank2>
template<int N_index1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      BitwiseAnd<int, T_numtype2 > > >
operator&(IndexPlaceholder<N_index1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      BitwiseAnd<int, T_numtype2> >
      (d1, 
      d2.begin());
}

// IndexPlaceholder<N_index1> & _bz_ArrayExpr<P_expr2>
template<int N_index1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>,
      BitwiseAnd<int, _bz_typename P_expr2::T_numtype > > >
operator&(IndexPlaceholder<N_index1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>, 
      BitwiseAnd<int, _bz_typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> & IndexPlaceholder<N_index2>
template<int N_index1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>,
      BitwiseAnd<int, int > > >
operator&(IndexPlaceholder<N_index1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>, 
      BitwiseAnd<int, int> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> & int
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>,
      BitwiseAnd<int, int > > >
operator&(IndexPlaceholder<N_index1> d1, 
      int d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>, 
      BitwiseAnd<int, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// int & Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>,
      BitwiseAnd<int, T_numtype2 > > >
operator&(int d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      BitwiseAnd<int, T_numtype2> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2.begin());
}

// int & _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>,
      BitwiseAnd<int, _bz_typename P_expr2::T_numtype > > >
operator&(int d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>, 
      BitwiseAnd<int, _bz_typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2);
}

// int & IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      IndexPlaceholder<N_index2>,
      BitwiseAnd<int, int > > >
operator&(int d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      IndexPlaceholder<N_index2>, 
      BitwiseAnd<int, int> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2);
}
/****************************************************************************
 * Bitwise Or Operators
 ****************************************************************************/

// Array<T_numtype1, N_rank1> | Array<T_numtype2, N_rank2>
template<class T_numtype1, int N_rank1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      BitwiseOr<T_numtype1, T_numtype2 > > >
operator|(const Array<T_numtype1, N_rank1>& d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      BitwiseOr<T_numtype1, T_numtype2> >
      (d1.begin(), 
      d2.begin());
}

// Array<T_numtype1, N_rank1> | _bz_ArrayExpr<P_expr2>
template<class T_numtype1, int N_rank1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>,
      BitwiseOr<T_numtype1, _bz_typename P_expr2::T_numtype > > >
operator|(const Array<T_numtype1, N_rank1>& d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>, 
      BitwiseOr<T_numtype1, _bz_typename P_expr2::T_numtype> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> | IndexPlaceholder<N_index2>
template<class T_numtype1, int N_rank1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>,
      BitwiseOr<T_numtype1, int > > >
operator|(const Array<T_numtype1, N_rank1>& d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>, 
      BitwiseOr<T_numtype1, int> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> | int
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>,
      BitwiseOr<T_numtype1, int > > >
operator|(const Array<T_numtype1, N_rank1>& d1, 
      int d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>, 
      BitwiseOr<T_numtype1, int> >
      (d1.begin(), 
      _bz_ArrayExprConstant<int>(d2));
}

// _bz_ArrayExpr<P_expr1> | Array<T_numtype2, N_rank2>
template<class P_expr1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      BitwiseOr<_bz_typename P_expr1::T_numtype, T_numtype2 > > >
operator|(_bz_ArrayExpr<P_expr1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      BitwiseOr<_bz_typename P_expr1::T_numtype, T_numtype2> >
      (d1, 
      d2.begin());
}

// _bz_ArrayExpr<P_expr1> | _bz_ArrayExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>,
      BitwiseOr<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype > > >
operator|(_bz_ArrayExpr<P_expr1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>, 
      BitwiseOr<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> | IndexPlaceholder<N_index2>
template<class P_expr1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>,
      BitwiseOr<_bz_typename P_expr1::T_numtype, int > > >
operator|(_bz_ArrayExpr<P_expr1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>, 
      BitwiseOr<_bz_typename P_expr1::T_numtype, int> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> | int
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>,
      BitwiseOr<_bz_typename P_expr1::T_numtype, int > > >
operator|(_bz_ArrayExpr<P_expr1> d1, 
      int d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>, 
      BitwiseOr<_bz_typename P_expr1::T_numtype, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// IndexPlaceholder<N_index1> | Array<T_numtype2, N_rank2>
template<int N_index1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      BitwiseOr<int, T_numtype2 > > >
operator|(IndexPlaceholder<N_index1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      BitwiseOr<int, T_numtype2> >
      (d1, 
      d2.begin());
}

// IndexPlaceholder<N_index1> | _bz_ArrayExpr<P_expr2>
template<int N_index1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>,
      BitwiseOr<int, _bz_typename P_expr2::T_numtype > > >
operator|(IndexPlaceholder<N_index1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>, 
      BitwiseOr<int, _bz_typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> | IndexPlaceholder<N_index2>
template<int N_index1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>,
      BitwiseOr<int, int > > >
operator|(IndexPlaceholder<N_index1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>, 
      BitwiseOr<int, int> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> | int
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>,
      BitwiseOr<int, int > > >
operator|(IndexPlaceholder<N_index1> d1, 
      int d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>, 
      BitwiseOr<int, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// int | Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>,
      BitwiseOr<int, T_numtype2 > > >
operator|(int d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      BitwiseOr<int, T_numtype2> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2.begin());
}

// int | _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>,
      BitwiseOr<int, _bz_typename P_expr2::T_numtype > > >
operator|(int d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>, 
      BitwiseOr<int, _bz_typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2);
}

// int | IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      IndexPlaceholder<N_index2>,
      BitwiseOr<int, int > > >
operator|(int d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      IndexPlaceholder<N_index2>, 
      BitwiseOr<int, int> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2);
}
/****************************************************************************
 * Shift right Operators
 ****************************************************************************/

// Array<T_numtype1, N_rank1> >> Array<T_numtype2, N_rank2>
template<class T_numtype1, int N_rank1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      ShiftRight<T_numtype1, T_numtype2 > > >
operator>>(const Array<T_numtype1, N_rank1>& d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      ShiftRight<T_numtype1, T_numtype2> >
      (d1.begin(), 
      d2.begin());
}

// Array<T_numtype1, N_rank1> >> _bz_ArrayExpr<P_expr2>
template<class T_numtype1, int N_rank1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>,
      ShiftRight<T_numtype1, _bz_typename P_expr2::T_numtype > > >
operator>>(const Array<T_numtype1, N_rank1>& d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>, 
      ShiftRight<T_numtype1, _bz_typename P_expr2::T_numtype> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> >> IndexPlaceholder<N_index2>
template<class T_numtype1, int N_rank1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>,
      ShiftRight<T_numtype1, int > > >
operator>>(const Array<T_numtype1, N_rank1>& d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>, 
      ShiftRight<T_numtype1, int> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> >> int
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>,
      ShiftRight<T_numtype1, int > > >
operator>>(const Array<T_numtype1, N_rank1>& d1, 
      int d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>, 
      ShiftRight<T_numtype1, int> >
      (d1.begin(), 
      _bz_ArrayExprConstant<int>(d2));
}

// _bz_ArrayExpr<P_expr1> >> Array<T_numtype2, N_rank2>
template<class P_expr1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      ShiftRight<_bz_typename P_expr1::T_numtype, T_numtype2 > > >
operator>>(_bz_ArrayExpr<P_expr1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      ShiftRight<_bz_typename P_expr1::T_numtype, T_numtype2> >
      (d1, 
      d2.begin());
}

// _bz_ArrayExpr<P_expr1> >> _bz_ArrayExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>,
      ShiftRight<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype > > >
operator>>(_bz_ArrayExpr<P_expr1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>, 
      ShiftRight<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> >> IndexPlaceholder<N_index2>
template<class P_expr1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>,
      ShiftRight<_bz_typename P_expr1::T_numtype, int > > >
operator>>(_bz_ArrayExpr<P_expr1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>, 
      ShiftRight<_bz_typename P_expr1::T_numtype, int> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> >> int
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>,
      ShiftRight<_bz_typename P_expr1::T_numtype, int > > >
operator>>(_bz_ArrayExpr<P_expr1> d1, 
      int d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>, 
      ShiftRight<_bz_typename P_expr1::T_numtype, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// IndexPlaceholder<N_index1> >> Array<T_numtype2, N_rank2>
template<int N_index1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      ShiftRight<int, T_numtype2 > > >
operator>>(IndexPlaceholder<N_index1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      ShiftRight<int, T_numtype2> >
      (d1, 
      d2.begin());
}

// IndexPlaceholder<N_index1> >> _bz_ArrayExpr<P_expr2>
template<int N_index1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>,
      ShiftRight<int, _bz_typename P_expr2::T_numtype > > >
operator>>(IndexPlaceholder<N_index1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>, 
      ShiftRight<int, _bz_typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> >> IndexPlaceholder<N_index2>
template<int N_index1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>,
      ShiftRight<int, int > > >
operator>>(IndexPlaceholder<N_index1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>, 
      ShiftRight<int, int> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> >> int
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>,
      ShiftRight<int, int > > >
operator>>(IndexPlaceholder<N_index1> d1, 
      int d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>, 
      ShiftRight<int, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// int >> Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>,
      ShiftRight<int, T_numtype2 > > >
operator>>(int d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      ShiftRight<int, T_numtype2> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2.begin());
}

// int >> _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>,
      ShiftRight<int, _bz_typename P_expr2::T_numtype > > >
operator>>(int d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>, 
      ShiftRight<int, _bz_typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2);
}

// int >> IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      IndexPlaceholder<N_index2>,
      ShiftRight<int, int > > >
operator>>(int d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      IndexPlaceholder<N_index2>, 
      ShiftRight<int, int> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2);
}
/****************************************************************************
 * Shift left Operators
 ****************************************************************************/

// Array<T_numtype1, N_rank1> << Array<T_numtype2, N_rank2>
template<class T_numtype1, int N_rank1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      ShiftLeft<T_numtype1, T_numtype2 > > >
operator<<(const Array<T_numtype1, N_rank1>& d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      ShiftLeft<T_numtype1, T_numtype2> >
      (d1.begin(), 
      d2.begin());
}

// Array<T_numtype1, N_rank1> << _bz_ArrayExpr<P_expr2>
template<class T_numtype1, int N_rank1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>,
      ShiftLeft<T_numtype1, _bz_typename P_expr2::T_numtype > > >
operator<<(const Array<T_numtype1, N_rank1>& d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>, 
      ShiftLeft<T_numtype1, _bz_typename P_expr2::T_numtype> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> << IndexPlaceholder<N_index2>
template<class T_numtype1, int N_rank1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>,
      ShiftLeft<T_numtype1, int > > >
operator<<(const Array<T_numtype1, N_rank1>& d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>, 
      ShiftLeft<T_numtype1, int> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> << int
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>,
      ShiftLeft<T_numtype1, int > > >
operator<<(const Array<T_numtype1, N_rank1>& d1, 
      int d2)
{
    return _bz_ArrayExprOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>, 
      ShiftLeft<T_numtype1, int> >
      (d1.begin(), 
      _bz_ArrayExprConstant<int>(d2));
}

// _bz_ArrayExpr<P_expr1> << Array<T_numtype2, N_rank2>
template<class P_expr1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      ShiftLeft<_bz_typename P_expr1::T_numtype, T_numtype2 > > >
operator<<(_bz_ArrayExpr<P_expr1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      ShiftLeft<_bz_typename P_expr1::T_numtype, T_numtype2> >
      (d1, 
      d2.begin());
}

// _bz_ArrayExpr<P_expr1> << _bz_ArrayExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>,
      ShiftLeft<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype > > >
operator<<(_bz_ArrayExpr<P_expr1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>, 
      ShiftLeft<_bz_typename P_expr1::T_numtype, _bz_typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> << IndexPlaceholder<N_index2>
template<class P_expr1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>,
      ShiftLeft<_bz_typename P_expr1::T_numtype, int > > >
operator<<(_bz_ArrayExpr<P_expr1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>, 
      ShiftLeft<_bz_typename P_expr1::T_numtype, int> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> << int
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>,
      ShiftLeft<_bz_typename P_expr1::T_numtype, int > > >
operator<<(_bz_ArrayExpr<P_expr1> d1, 
      int d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>, 
      ShiftLeft<_bz_typename P_expr1::T_numtype, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// IndexPlaceholder<N_index1> << Array<T_numtype2, N_rank2>
template<int N_index1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      ShiftLeft<int, T_numtype2 > > >
operator<<(IndexPlaceholder<N_index1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      ShiftLeft<int, T_numtype2> >
      (d1, 
      d2.begin());
}

// IndexPlaceholder<N_index1> << _bz_ArrayExpr<P_expr2>
template<int N_index1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>,
      ShiftLeft<int, _bz_typename P_expr2::T_numtype > > >
operator<<(IndexPlaceholder<N_index1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>, 
      ShiftLeft<int, _bz_typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> << IndexPlaceholder<N_index2>
template<int N_index1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>,
      ShiftLeft<int, int > > >
operator<<(IndexPlaceholder<N_index1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>, 
      ShiftLeft<int, int> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> << int
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>,
      ShiftLeft<int, int > > >
operator<<(IndexPlaceholder<N_index1> d1, 
      int d2)
{
    return _bz_ArrayExprOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>, 
      ShiftLeft<int, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// int << Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>,
      ShiftLeft<int, T_numtype2 > > >
operator<<(int d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      ShiftLeft<int, T_numtype2> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2.begin());
}

// int << _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>,
      ShiftLeft<int, _bz_typename P_expr2::T_numtype > > >
operator<<(int d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>, 
      ShiftLeft<int, _bz_typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2);
}

// int << IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      IndexPlaceholder<N_index2>,
      ShiftLeft<int, int > > >
operator<<(int d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprOp<_bz_ArrayExprConstant<int>, 
      IndexPlaceholder<N_index2>, 
      ShiftLeft<int, int> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2);
}
BZ_NAMESPACE_END

#endif
