/***************************************************************************
 * blitz/vecwhere.cc	where(X,Y,Z) function for vectors
 *
 * $Id$
 *
 * Copyright (C) 1997 Todd Veldhuizen <tveldhui@oonumerics.org>
 * All rights reserved.   Please see <blitz/blitz.h> for terms and
 * conditions of use.
 *
 * Suggestions:          blitz-dev@oonumerics.org
 * Bugs:                 blitz-bugs@oonumerics.org
 * Licensing inquiries:  blitz-licenses@oonumerics.org
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
 * Revision 1.1.1.1  2000/06/19 12:26:11  tveldhui
 * Imported sources
 *
 * Revision 1.1  1997/07/16 14:51:20  tveldhui
 * Update: Alpha release 0.2 (Arrays)
 *
 */ 

// Generated source file.  Do not edit. 
// genvecwhere.cpp Feb  5 1997 09:52:29

#ifndef BZ_VECWHERE_CC
#define BZ_VECWHERE_CC

BZ_NAMESPACE(blitz)

// where(Vector<P_numtype1>, Vector<P_numtype2>, Vector<P_numtype3>)
template<class P_numtype1, class P_numtype2, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      VectorIterConst<P_numtype3> > >
where(const Vector<P_numtype1>& d1, 
      const Vector<P_numtype2>& d2, 
      const Vector<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      VectorIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      d3.begin()));
}

// where(Vector<P_numtype1>, Vector<P_numtype2>, _bz_VecExpr<P_expr3>)
template<class P_numtype1, class P_numtype2, class P_expr3>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      _bz_VecExpr<P_expr3> > >
where(const Vector<P_numtype1>& d1, 
      const Vector<P_numtype2>& d2, 
      _bz_VecExpr<P_expr3> d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      _bz_VecExpr<P_expr3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      d3));
}

// where(Vector<P_numtype1>, Vector<P_numtype2>, VectorPick<P_numtype3>)
template<class P_numtype1, class P_numtype2, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      VectorPickIterConst<P_numtype3> > >
where(const Vector<P_numtype1>& d1, 
      const Vector<P_numtype2>& d2, 
      const VectorPick<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      VectorPickIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      d3.begin()));
}

// where(Vector<P_numtype1>, Vector<P_numtype2>, Range)
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      Range > >
where(const Vector<P_numtype1>& d1, 
      const Vector<P_numtype2>& d2, 
      Range d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      Range > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      d3));
}

// where(Vector<P_numtype1>, Vector<P_numtype2>, TinyVector<P_numtype3, N_length3>)
template<class P_numtype1, class P_numtype2, class P_numtype3, int N_length3>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      TinyVectorIterConst<P_numtype3, N_length3> > >
where(const Vector<P_numtype1>& d1, 
      const Vector<P_numtype2>& d2, 
      const TinyVector<P_numtype3, N_length3>& d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      TinyVectorIterConst<P_numtype3, N_length3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      d3.begin()));
}

// where(Vector<P_numtype1>, Vector<P_numtype2>, int)
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      _bz_VecExprConstant<int> > >
where(const Vector<P_numtype1>& d1, 
      const Vector<P_numtype2>& d2, 
      int d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      _bz_VecExprConstant<int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      _bz_VecExprConstant<int>(d3)));
}

// where(Vector<P_numtype1>, Vector<P_numtype2>, float)
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      _bz_VecExprConstant<float> > >
where(const Vector<P_numtype1>& d1, 
      const Vector<P_numtype2>& d2, 
      float d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      _bz_VecExprConstant<float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      _bz_VecExprConstant<float>(d3)));
}

// where(Vector<P_numtype1>, Vector<P_numtype2>, double)
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      _bz_VecExprConstant<double> > >
where(const Vector<P_numtype1>& d1, 
      const Vector<P_numtype2>& d2, 
      double d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      _bz_VecExprConstant<double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      _bz_VecExprConstant<double>(d3)));
}

// where(Vector<P_numtype1>, Vector<P_numtype2>, long double)
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      _bz_VecExprConstant<long double> > >
where(const Vector<P_numtype1>& d1, 
      const Vector<P_numtype2>& d2, 
      long double d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      _bz_VecExprConstant<long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      _bz_VecExprConstant<long double>(d3)));
}

// where(Vector<P_numtype1>, Vector<P_numtype2>, complex<T3>)
#ifdef BZ_HAVE_COMPLEX
template<class P_numtype1, class P_numtype2, class T3>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      _bz_VecExprConstant<complex<T3> >  > >
where(const Vector<P_numtype1>& d1, 
      const Vector<P_numtype2>& d2, 
      complex<T3> d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      _bz_VecExprConstant<complex<T3> >  > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      _bz_VecExprConstant<complex<T3> > (d3)));
}
#endif // BZ_HAVE_COMPLEX

// where(Vector<P_numtype1>, _bz_VecExpr<P_expr2>, Vector<P_numtype3>)
template<class P_numtype1, class P_expr2, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      VectorIterConst<P_numtype3> > >
where(const Vector<P_numtype1>& d1, 
      _bz_VecExpr<P_expr2> d2, 
      const Vector<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      VectorIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2, 
      d3.begin()));
}

// where(Vector<P_numtype1>, _bz_VecExpr<P_expr2>, _bz_VecExpr<P_expr3>)
template<class P_numtype1, class P_expr2, class P_expr3>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_VecExpr<P_expr3> > >
where(const Vector<P_numtype1>& d1, 
      _bz_VecExpr<P_expr2> d2, 
      _bz_VecExpr<P_expr3> d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_VecExpr<P_expr3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2, 
      d3));
}

// where(Vector<P_numtype1>, _bz_VecExpr<P_expr2>, VectorPick<P_numtype3>)
template<class P_numtype1, class P_expr2, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      VectorPickIterConst<P_numtype3> > >
where(const Vector<P_numtype1>& d1, 
      _bz_VecExpr<P_expr2> d2, 
      const VectorPick<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      VectorPickIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2, 
      d3.begin()));
}

// where(Vector<P_numtype1>, _bz_VecExpr<P_expr2>, Range)
template<class P_numtype1, class P_expr2>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      Range > >
where(const Vector<P_numtype1>& d1, 
      _bz_VecExpr<P_expr2> d2, 
      Range d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      Range > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2, 
      d3));
}

// where(Vector<P_numtype1>, _bz_VecExpr<P_expr2>, TinyVector<P_numtype3, N_length3>)
template<class P_numtype1, class P_expr2, class P_numtype3, int N_length3>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      TinyVectorIterConst<P_numtype3, N_length3> > >
where(const Vector<P_numtype1>& d1, 
      _bz_VecExpr<P_expr2> d2, 
      const TinyVector<P_numtype3, N_length3>& d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      TinyVectorIterConst<P_numtype3, N_length3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2, 
      d3.begin()));
}

// where(Vector<P_numtype1>, _bz_VecExpr<P_expr2>, int)
template<class P_numtype1, class P_expr2>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_VecExprConstant<int> > >
where(const Vector<P_numtype1>& d1, 
      _bz_VecExpr<P_expr2> d2, 
      int d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_VecExprConstant<int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2, 
      _bz_VecExprConstant<int>(d3)));
}

// where(Vector<P_numtype1>, _bz_VecExpr<P_expr2>, float)
template<class P_numtype1, class P_expr2>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_VecExprConstant<float> > >
where(const Vector<P_numtype1>& d1, 
      _bz_VecExpr<P_expr2> d2, 
      float d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_VecExprConstant<float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2, 
      _bz_VecExprConstant<float>(d3)));
}

// where(Vector<P_numtype1>, _bz_VecExpr<P_expr2>, double)
template<class P_numtype1, class P_expr2>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_VecExprConstant<double> > >
where(const Vector<P_numtype1>& d1, 
      _bz_VecExpr<P_expr2> d2, 
      double d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_VecExprConstant<double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2, 
      _bz_VecExprConstant<double>(d3)));
}

// where(Vector<P_numtype1>, _bz_VecExpr<P_expr2>, long double)
template<class P_numtype1, class P_expr2>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_VecExprConstant<long double> > >
where(const Vector<P_numtype1>& d1, 
      _bz_VecExpr<P_expr2> d2, 
      long double d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_VecExprConstant<long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2, 
      _bz_VecExprConstant<long double>(d3)));
}

// where(Vector<P_numtype1>, _bz_VecExpr<P_expr2>, complex<T3>)
#ifdef BZ_HAVE_COMPLEX
template<class P_numtype1, class P_expr2, class T3>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_VecExprConstant<complex<T3> >  > >
where(const Vector<P_numtype1>& d1, 
      _bz_VecExpr<P_expr2> d2, 
      complex<T3> d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_VecExprConstant<complex<T3> >  > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2, 
      _bz_VecExprConstant<complex<T3> > (d3)));
}
#endif // BZ_HAVE_COMPLEX

// where(Vector<P_numtype1>, VectorPick<P_numtype2>, Vector<P_numtype3>)
template<class P_numtype1, class P_numtype2, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      VectorIterConst<P_numtype3> > >
where(const Vector<P_numtype1>& d1, 
      const VectorPick<P_numtype2>& d2, 
      const Vector<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      VectorIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      d3.begin()));
}

// where(Vector<P_numtype1>, VectorPick<P_numtype2>, _bz_VecExpr<P_expr3>)
template<class P_numtype1, class P_numtype2, class P_expr3>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_VecExpr<P_expr3> > >
where(const Vector<P_numtype1>& d1, 
      const VectorPick<P_numtype2>& d2, 
      _bz_VecExpr<P_expr3> d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_VecExpr<P_expr3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      d3));
}

// where(Vector<P_numtype1>, VectorPick<P_numtype2>, VectorPick<P_numtype3>)
template<class P_numtype1, class P_numtype2, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      VectorPickIterConst<P_numtype3> > >
where(const Vector<P_numtype1>& d1, 
      const VectorPick<P_numtype2>& d2, 
      const VectorPick<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      VectorPickIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      d3.begin()));
}

// where(Vector<P_numtype1>, VectorPick<P_numtype2>, Range)
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      Range > >
where(const Vector<P_numtype1>& d1, 
      const VectorPick<P_numtype2>& d2, 
      Range d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      Range > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      d3));
}

// where(Vector<P_numtype1>, VectorPick<P_numtype2>, TinyVector<P_numtype3, N_length3>)
template<class P_numtype1, class P_numtype2, class P_numtype3, int N_length3>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      TinyVectorIterConst<P_numtype3, N_length3> > >
where(const Vector<P_numtype1>& d1, 
      const VectorPick<P_numtype2>& d2, 
      const TinyVector<P_numtype3, N_length3>& d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      TinyVectorIterConst<P_numtype3, N_length3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      d3.begin()));
}

// where(Vector<P_numtype1>, VectorPick<P_numtype2>, int)
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_VecExprConstant<int> > >
where(const Vector<P_numtype1>& d1, 
      const VectorPick<P_numtype2>& d2, 
      int d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_VecExprConstant<int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      _bz_VecExprConstant<int>(d3)));
}

// where(Vector<P_numtype1>, VectorPick<P_numtype2>, float)
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_VecExprConstant<float> > >
where(const Vector<P_numtype1>& d1, 
      const VectorPick<P_numtype2>& d2, 
      float d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_VecExprConstant<float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      _bz_VecExprConstant<float>(d3)));
}

// where(Vector<P_numtype1>, VectorPick<P_numtype2>, double)
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_VecExprConstant<double> > >
where(const Vector<P_numtype1>& d1, 
      const VectorPick<P_numtype2>& d2, 
      double d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_VecExprConstant<double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      _bz_VecExprConstant<double>(d3)));
}

// where(Vector<P_numtype1>, VectorPick<P_numtype2>, long double)
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_VecExprConstant<long double> > >
where(const Vector<P_numtype1>& d1, 
      const VectorPick<P_numtype2>& d2, 
      long double d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_VecExprConstant<long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      _bz_VecExprConstant<long double>(d3)));
}

// where(Vector<P_numtype1>, VectorPick<P_numtype2>, complex<T3>)
#ifdef BZ_HAVE_COMPLEX
template<class P_numtype1, class P_numtype2, class T3>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_VecExprConstant<complex<T3> >  > >
where(const Vector<P_numtype1>& d1, 
      const VectorPick<P_numtype2>& d2, 
      complex<T3> d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_VecExprConstant<complex<T3> >  > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      _bz_VecExprConstant<complex<T3> > (d3)));
}
#endif // BZ_HAVE_COMPLEX

// where(Vector<P_numtype1>, Range, Vector<P_numtype3>)
template<class P_numtype1, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      Range, 
      VectorIterConst<P_numtype3> > >
where(const Vector<P_numtype1>& d1, 
      Range d2, 
      const Vector<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      Range, 
      VectorIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2, 
      d3.begin()));
}

// where(Vector<P_numtype1>, Range, _bz_VecExpr<P_expr3>)
template<class P_numtype1, class P_expr3>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      Range, 
      _bz_VecExpr<P_expr3> > >
where(const Vector<P_numtype1>& d1, 
      Range d2, 
      _bz_VecExpr<P_expr3> d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      Range, 
      _bz_VecExpr<P_expr3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2, 
      d3));
}

// where(Vector<P_numtype1>, Range, VectorPick<P_numtype3>)
template<class P_numtype1, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      Range, 
      VectorPickIterConst<P_numtype3> > >
where(const Vector<P_numtype1>& d1, 
      Range d2, 
      const VectorPick<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      Range, 
      VectorPickIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2, 
      d3.begin()));
}

// where(Vector<P_numtype1>, Range, Range)
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      Range, 
      Range > >
where(const Vector<P_numtype1>& d1, 
      Range d2, 
      Range d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      Range, 
      Range > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2, 
      d3));
}

// where(Vector<P_numtype1>, Range, TinyVector<P_numtype3, N_length3>)
template<class P_numtype1, class P_numtype3, int N_length3>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      Range, 
      TinyVectorIterConst<P_numtype3, N_length3> > >
where(const Vector<P_numtype1>& d1, 
      Range d2, 
      const TinyVector<P_numtype3, N_length3>& d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      Range, 
      TinyVectorIterConst<P_numtype3, N_length3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2, 
      d3.begin()));
}

// where(Vector<P_numtype1>, Range, int)
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      Range, 
      _bz_VecExprConstant<int> > >
where(const Vector<P_numtype1>& d1, 
      Range d2, 
      int d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      Range, 
      _bz_VecExprConstant<int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2, 
      _bz_VecExprConstant<int>(d3)));
}

// where(Vector<P_numtype1>, Range, float)
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      Range, 
      _bz_VecExprConstant<float> > >
where(const Vector<P_numtype1>& d1, 
      Range d2, 
      float d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      Range, 
      _bz_VecExprConstant<float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2, 
      _bz_VecExprConstant<float>(d3)));
}

// where(Vector<P_numtype1>, Range, double)
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      Range, 
      _bz_VecExprConstant<double> > >
where(const Vector<P_numtype1>& d1, 
      Range d2, 
      double d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      Range, 
      _bz_VecExprConstant<double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2, 
      _bz_VecExprConstant<double>(d3)));
}

// where(Vector<P_numtype1>, Range, long double)
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      Range, 
      _bz_VecExprConstant<long double> > >
where(const Vector<P_numtype1>& d1, 
      Range d2, 
      long double d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      Range, 
      _bz_VecExprConstant<long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2, 
      _bz_VecExprConstant<long double>(d3)));
}

// where(Vector<P_numtype1>, Range, complex<T3>)
#ifdef BZ_HAVE_COMPLEX
template<class P_numtype1, class T3>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      Range, 
      _bz_VecExprConstant<complex<T3> >  > >
where(const Vector<P_numtype1>& d1, 
      Range d2, 
      complex<T3> d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      Range, 
      _bz_VecExprConstant<complex<T3> >  > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2, 
      _bz_VecExprConstant<complex<T3> > (d3)));
}
#endif // BZ_HAVE_COMPLEX

// where(Vector<P_numtype1>, TinyVector<P_numtype2, N_length2>, Vector<P_numtype3>)
template<class P_numtype1, class P_numtype2, int N_length2, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      VectorIterConst<P_numtype3> > >
where(const Vector<P_numtype1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2, 
      const Vector<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      VectorIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      d3.begin()));
}

// where(Vector<P_numtype1>, TinyVector<P_numtype2, N_length2>, _bz_VecExpr<P_expr3>)
template<class P_numtype1, class P_numtype2, int N_length2, class P_expr3>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_VecExpr<P_expr3> > >
where(const Vector<P_numtype1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2, 
      _bz_VecExpr<P_expr3> d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_VecExpr<P_expr3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      d3));
}

// where(Vector<P_numtype1>, TinyVector<P_numtype2, N_length2>, VectorPick<P_numtype3>)
template<class P_numtype1, class P_numtype2, int N_length2, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      VectorPickIterConst<P_numtype3> > >
where(const Vector<P_numtype1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2, 
      const VectorPick<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      VectorPickIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      d3.begin()));
}

// where(Vector<P_numtype1>, TinyVector<P_numtype2, N_length2>, Range)
template<class P_numtype1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      Range > >
where(const Vector<P_numtype1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2, 
      Range d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      Range > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      d3));
}

// where(Vector<P_numtype1>, TinyVector<P_numtype2, N_length2>, TinyVector<P_numtype3, N_length3>)
template<class P_numtype1, class P_numtype2, int N_length2, class P_numtype3, int N_length3>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      TinyVectorIterConst<P_numtype3, N_length3> > >
where(const Vector<P_numtype1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2, 
      const TinyVector<P_numtype3, N_length3>& d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      TinyVectorIterConst<P_numtype3, N_length3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      d3.begin()));
}

// where(Vector<P_numtype1>, TinyVector<P_numtype2, N_length2>, int)
template<class P_numtype1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_VecExprConstant<int> > >
where(const Vector<P_numtype1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2, 
      int d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_VecExprConstant<int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      _bz_VecExprConstant<int>(d3)));
}

// where(Vector<P_numtype1>, TinyVector<P_numtype2, N_length2>, float)
template<class P_numtype1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_VecExprConstant<float> > >
where(const Vector<P_numtype1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2, 
      float d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_VecExprConstant<float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      _bz_VecExprConstant<float>(d3)));
}

// where(Vector<P_numtype1>, TinyVector<P_numtype2, N_length2>, double)
template<class P_numtype1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_VecExprConstant<double> > >
where(const Vector<P_numtype1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2, 
      double d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_VecExprConstant<double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      _bz_VecExprConstant<double>(d3)));
}

// where(Vector<P_numtype1>, TinyVector<P_numtype2, N_length2>, long double)
template<class P_numtype1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_VecExprConstant<long double> > >
where(const Vector<P_numtype1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2, 
      long double d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_VecExprConstant<long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      _bz_VecExprConstant<long double>(d3)));
}

// where(Vector<P_numtype1>, TinyVector<P_numtype2, N_length2>, complex<T3>)
#ifdef BZ_HAVE_COMPLEX
template<class P_numtype1, class P_numtype2, int N_length2, class T3>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_VecExprConstant<complex<T3> >  > >
where(const Vector<P_numtype1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2, 
      complex<T3> d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_VecExprConstant<complex<T3> >  > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      _bz_VecExprConstant<complex<T3> > (d3)));
}
#endif // BZ_HAVE_COMPLEX

// where(Vector<P_numtype1>, int, Vector<P_numtype3>)
template<class P_numtype1, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>, 
      VectorIterConst<P_numtype3> > >
where(const Vector<P_numtype1>& d1, 
      int d2, 
      const Vector<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>, 
      VectorIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2), 
      d3.begin()));
}

// where(Vector<P_numtype1>, int, _bz_VecExpr<P_expr3>)
template<class P_numtype1, class P_expr3>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>, 
      _bz_VecExpr<P_expr3> > >
where(const Vector<P_numtype1>& d1, 
      int d2, 
      _bz_VecExpr<P_expr3> d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>, 
      _bz_VecExpr<P_expr3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2), 
      d3));
}

// where(Vector<P_numtype1>, int, VectorPick<P_numtype3>)
template<class P_numtype1, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>, 
      VectorPickIterConst<P_numtype3> > >
where(const Vector<P_numtype1>& d1, 
      int d2, 
      const VectorPick<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>, 
      VectorPickIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2), 
      d3.begin()));
}

// where(Vector<P_numtype1>, int, Range)
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>, 
      Range > >
where(const Vector<P_numtype1>& d1, 
      int d2, 
      Range d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>, 
      Range > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2), 
      d3));
}

// where(Vector<P_numtype1>, int, TinyVector<P_numtype3, N_length3>)
template<class P_numtype1, class P_numtype3, int N_length3>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>, 
      TinyVectorIterConst<P_numtype3, N_length3> > >
where(const Vector<P_numtype1>& d1, 
      int d2, 
      const TinyVector<P_numtype3, N_length3>& d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>, 
      TinyVectorIterConst<P_numtype3, N_length3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2), 
      d3.begin()));
}

// where(Vector<P_numtype1>, int, int)
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>, 
      _bz_VecExprConstant<int> > >
where(const Vector<P_numtype1>& d1, 
      int d2, 
      int d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>, 
      _bz_VecExprConstant<int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2), 
      _bz_VecExprConstant<int>(d3)));
}

// where(Vector<P_numtype1>, float, Vector<P_numtype3>)
template<class P_numtype1, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>, 
      VectorIterConst<P_numtype3> > >
where(const Vector<P_numtype1>& d1, 
      float d2, 
      const Vector<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>, 
      VectorIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<float>(d2), 
      d3.begin()));
}

// where(Vector<P_numtype1>, float, _bz_VecExpr<P_expr3>)
template<class P_numtype1, class P_expr3>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>, 
      _bz_VecExpr<P_expr3> > >
where(const Vector<P_numtype1>& d1, 
      float d2, 
      _bz_VecExpr<P_expr3> d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>, 
      _bz_VecExpr<P_expr3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<float>(d2), 
      d3));
}

// where(Vector<P_numtype1>, float, VectorPick<P_numtype3>)
template<class P_numtype1, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>, 
      VectorPickIterConst<P_numtype3> > >
where(const Vector<P_numtype1>& d1, 
      float d2, 
      const VectorPick<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>, 
      VectorPickIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<float>(d2), 
      d3.begin()));
}

// where(Vector<P_numtype1>, float, Range)
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>, 
      Range > >
where(const Vector<P_numtype1>& d1, 
      float d2, 
      Range d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>, 
      Range > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<float>(d2), 
      d3));
}

// where(Vector<P_numtype1>, float, TinyVector<P_numtype3, N_length3>)
template<class P_numtype1, class P_numtype3, int N_length3>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>, 
      TinyVectorIterConst<P_numtype3, N_length3> > >
where(const Vector<P_numtype1>& d1, 
      float d2, 
      const TinyVector<P_numtype3, N_length3>& d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>, 
      TinyVectorIterConst<P_numtype3, N_length3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<float>(d2), 
      d3.begin()));
}

// where(Vector<P_numtype1>, float, float)
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>, 
      _bz_VecExprConstant<float> > >
where(const Vector<P_numtype1>& d1, 
      float d2, 
      float d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>, 
      _bz_VecExprConstant<float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<float>(d2), 
      _bz_VecExprConstant<float>(d3)));
}

// where(Vector<P_numtype1>, double, Vector<P_numtype3>)
template<class P_numtype1, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>, 
      VectorIterConst<P_numtype3> > >
where(const Vector<P_numtype1>& d1, 
      double d2, 
      const Vector<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>, 
      VectorIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<double>(d2), 
      d3.begin()));
}

// where(Vector<P_numtype1>, double, _bz_VecExpr<P_expr3>)
template<class P_numtype1, class P_expr3>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>, 
      _bz_VecExpr<P_expr3> > >
where(const Vector<P_numtype1>& d1, 
      double d2, 
      _bz_VecExpr<P_expr3> d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>, 
      _bz_VecExpr<P_expr3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<double>(d2), 
      d3));
}

// where(Vector<P_numtype1>, double, VectorPick<P_numtype3>)
template<class P_numtype1, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>, 
      VectorPickIterConst<P_numtype3> > >
where(const Vector<P_numtype1>& d1, 
      double d2, 
      const VectorPick<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>, 
      VectorPickIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<double>(d2), 
      d3.begin()));
}

// where(Vector<P_numtype1>, double, Range)
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>, 
      Range > >
where(const Vector<P_numtype1>& d1, 
      double d2, 
      Range d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>, 
      Range > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<double>(d2), 
      d3));
}

// where(Vector<P_numtype1>, double, TinyVector<P_numtype3, N_length3>)
template<class P_numtype1, class P_numtype3, int N_length3>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>, 
      TinyVectorIterConst<P_numtype3, N_length3> > >
where(const Vector<P_numtype1>& d1, 
      double d2, 
      const TinyVector<P_numtype3, N_length3>& d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>, 
      TinyVectorIterConst<P_numtype3, N_length3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<double>(d2), 
      d3.begin()));
}

// where(Vector<P_numtype1>, double, double)
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>, 
      _bz_VecExprConstant<double> > >
where(const Vector<P_numtype1>& d1, 
      double d2, 
      double d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>, 
      _bz_VecExprConstant<double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<double>(d2), 
      _bz_VecExprConstant<double>(d3)));
}

// where(Vector<P_numtype1>, long double, Vector<P_numtype3>)
template<class P_numtype1, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>, 
      VectorIterConst<P_numtype3> > >
where(const Vector<P_numtype1>& d1, 
      long double d2, 
      const Vector<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>, 
      VectorIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<long double>(d2), 
      d3.begin()));
}

// where(Vector<P_numtype1>, long double, _bz_VecExpr<P_expr3>)
template<class P_numtype1, class P_expr3>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>, 
      _bz_VecExpr<P_expr3> > >
where(const Vector<P_numtype1>& d1, 
      long double d2, 
      _bz_VecExpr<P_expr3> d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>, 
      _bz_VecExpr<P_expr3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<long double>(d2), 
      d3));
}

// where(Vector<P_numtype1>, long double, VectorPick<P_numtype3>)
template<class P_numtype1, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>, 
      VectorPickIterConst<P_numtype3> > >
where(const Vector<P_numtype1>& d1, 
      long double d2, 
      const VectorPick<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>, 
      VectorPickIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<long double>(d2), 
      d3.begin()));
}

// where(Vector<P_numtype1>, long double, Range)
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>, 
      Range > >
where(const Vector<P_numtype1>& d1, 
      long double d2, 
      Range d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>, 
      Range > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<long double>(d2), 
      d3));
}

// where(Vector<P_numtype1>, long double, TinyVector<P_numtype3, N_length3>)
template<class P_numtype1, class P_numtype3, int N_length3>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>, 
      TinyVectorIterConst<P_numtype3, N_length3> > >
where(const Vector<P_numtype1>& d1, 
      long double d2, 
      const TinyVector<P_numtype3, N_length3>& d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>, 
      TinyVectorIterConst<P_numtype3, N_length3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<long double>(d2), 
      d3.begin()));
}

// where(Vector<P_numtype1>, long double, long double)
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>, 
      _bz_VecExprConstant<long double> > >
where(const Vector<P_numtype1>& d1, 
      long double d2, 
      long double d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>, 
      _bz_VecExprConstant<long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<long double>(d2), 
      _bz_VecExprConstant<long double>(d3)));
}

// where(Vector<P_numtype1>, complex<T2>, Vector<P_numtype3>)
#ifdef BZ_HAVE_COMPLEX
template<class P_numtype1, class T2, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > , 
      VectorIterConst<P_numtype3> > >
where(const Vector<P_numtype1>& d1, 
      complex<T2> d2, 
      const Vector<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > , 
      VectorIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<complex<T2> > (d2), 
      d3.begin()));
}
#endif // BZ_HAVE_COMPLEX

// where(Vector<P_numtype1>, complex<T2>, _bz_VecExpr<P_expr3>)
#ifdef BZ_HAVE_COMPLEX
template<class P_numtype1, class T2, class P_expr3>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_VecExpr<P_expr3> > >
where(const Vector<P_numtype1>& d1, 
      complex<T2> d2, 
      _bz_VecExpr<P_expr3> d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_VecExpr<P_expr3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<complex<T2> > (d2), 
      d3));
}
#endif // BZ_HAVE_COMPLEX

// where(Vector<P_numtype1>, complex<T2>, VectorPick<P_numtype3>)
#ifdef BZ_HAVE_COMPLEX
template<class P_numtype1, class T2, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > , 
      VectorPickIterConst<P_numtype3> > >
where(const Vector<P_numtype1>& d1, 
      complex<T2> d2, 
      const VectorPick<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > , 
      VectorPickIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<complex<T2> > (d2), 
      d3.begin()));
}
#endif // BZ_HAVE_COMPLEX

// where(Vector<P_numtype1>, complex<T2>, Range)
#ifdef BZ_HAVE_COMPLEX
template<class P_numtype1, class T2>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > , 
      Range > >
where(const Vector<P_numtype1>& d1, 
      complex<T2> d2, 
      Range d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > , 
      Range > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<complex<T2> > (d2), 
      d3));
}
#endif // BZ_HAVE_COMPLEX

// where(Vector<P_numtype1>, complex<T2>, TinyVector<P_numtype3, N_length3>)
#ifdef BZ_HAVE_COMPLEX
template<class P_numtype1, class T2, class P_numtype3, int N_length3>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > , 
      TinyVectorIterConst<P_numtype3, N_length3> > >
where(const Vector<P_numtype1>& d1, 
      complex<T2> d2, 
      const TinyVector<P_numtype3, N_length3>& d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > , 
      TinyVectorIterConst<P_numtype3, N_length3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<complex<T2> > (d2), 
      d3.begin()));
}
#endif // BZ_HAVE_COMPLEX

// where(Vector<P_numtype1>, complex<T2>, complex<T3>)
#ifdef BZ_HAVE_COMPLEX
template<class P_numtype1, class T2, class T3>
inline
_bz_VecExpr<_bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_VecExprConstant<complex<T3> >  > >
where(const Vector<P_numtype1>& d1, 
      complex<T2> d2, 
      complex<T3> d3)
{ 
    typedef _bz_VecWhere<VectorIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_VecExprConstant<complex<T3> >  > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<complex<T2> > (d2), 
      _bz_VecExprConstant<complex<T3> > (d3)));
}
#endif // BZ_HAVE_COMPLEX

// where(_bz_VecExpr<P_expr1>, Vector<P_numtype2>, Vector<P_numtype3>)
template<class P_expr1, class P_numtype2, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      VectorIterConst<P_numtype2>, 
      VectorIterConst<P_numtype3> > >
where(_bz_VecExpr<P_expr1> d1, 
      const Vector<P_numtype2>& d2, 
      const Vector<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      VectorIterConst<P_numtype2>, 
      VectorIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin(), 
      d3.begin()));
}

// where(_bz_VecExpr<P_expr1>, Vector<P_numtype2>, _bz_VecExpr<P_expr3>)
template<class P_expr1, class P_numtype2, class P_expr3>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      VectorIterConst<P_numtype2>, 
      _bz_VecExpr<P_expr3> > >
where(_bz_VecExpr<P_expr1> d1, 
      const Vector<P_numtype2>& d2, 
      _bz_VecExpr<P_expr3> d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      VectorIterConst<P_numtype2>, 
      _bz_VecExpr<P_expr3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin(), 
      d3));
}

// where(_bz_VecExpr<P_expr1>, Vector<P_numtype2>, VectorPick<P_numtype3>)
template<class P_expr1, class P_numtype2, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      VectorIterConst<P_numtype2>, 
      VectorPickIterConst<P_numtype3> > >
where(_bz_VecExpr<P_expr1> d1, 
      const Vector<P_numtype2>& d2, 
      const VectorPick<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      VectorIterConst<P_numtype2>, 
      VectorPickIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin(), 
      d3.begin()));
}

// where(_bz_VecExpr<P_expr1>, Vector<P_numtype2>, Range)
template<class P_expr1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      VectorIterConst<P_numtype2>, 
      Range > >
where(_bz_VecExpr<P_expr1> d1, 
      const Vector<P_numtype2>& d2, 
      Range d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      VectorIterConst<P_numtype2>, 
      Range > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin(), 
      d3));
}

// where(_bz_VecExpr<P_expr1>, Vector<P_numtype2>, TinyVector<P_numtype3, N_length3>)
template<class P_expr1, class P_numtype2, class P_numtype3, int N_length3>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      VectorIterConst<P_numtype2>, 
      TinyVectorIterConst<P_numtype3, N_length3> > >
where(_bz_VecExpr<P_expr1> d1, 
      const Vector<P_numtype2>& d2, 
      const TinyVector<P_numtype3, N_length3>& d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      VectorIterConst<P_numtype2>, 
      TinyVectorIterConst<P_numtype3, N_length3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin(), 
      d3.begin()));
}

// where(_bz_VecExpr<P_expr1>, Vector<P_numtype2>, int)
template<class P_expr1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      VectorIterConst<P_numtype2>, 
      _bz_VecExprConstant<int> > >
where(_bz_VecExpr<P_expr1> d1, 
      const Vector<P_numtype2>& d2, 
      int d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      VectorIterConst<P_numtype2>, 
      _bz_VecExprConstant<int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin(), 
      _bz_VecExprConstant<int>(d3)));
}

// where(_bz_VecExpr<P_expr1>, Vector<P_numtype2>, float)
template<class P_expr1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      VectorIterConst<P_numtype2>, 
      _bz_VecExprConstant<float> > >
where(_bz_VecExpr<P_expr1> d1, 
      const Vector<P_numtype2>& d2, 
      float d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      VectorIterConst<P_numtype2>, 
      _bz_VecExprConstant<float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin(), 
      _bz_VecExprConstant<float>(d3)));
}

// where(_bz_VecExpr<P_expr1>, Vector<P_numtype2>, double)
template<class P_expr1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      VectorIterConst<P_numtype2>, 
      _bz_VecExprConstant<double> > >
where(_bz_VecExpr<P_expr1> d1, 
      const Vector<P_numtype2>& d2, 
      double d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      VectorIterConst<P_numtype2>, 
      _bz_VecExprConstant<double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin(), 
      _bz_VecExprConstant<double>(d3)));
}

// where(_bz_VecExpr<P_expr1>, Vector<P_numtype2>, long double)
template<class P_expr1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      VectorIterConst<P_numtype2>, 
      _bz_VecExprConstant<long double> > >
where(_bz_VecExpr<P_expr1> d1, 
      const Vector<P_numtype2>& d2, 
      long double d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      VectorIterConst<P_numtype2>, 
      _bz_VecExprConstant<long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin(), 
      _bz_VecExprConstant<long double>(d3)));
}

// where(_bz_VecExpr<P_expr1>, Vector<P_numtype2>, complex<T3>)
#ifdef BZ_HAVE_COMPLEX
template<class P_expr1, class P_numtype2, class T3>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      VectorIterConst<P_numtype2>, 
      _bz_VecExprConstant<complex<T3> >  > >
where(_bz_VecExpr<P_expr1> d1, 
      const Vector<P_numtype2>& d2, 
      complex<T3> d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      VectorIterConst<P_numtype2>, 
      _bz_VecExprConstant<complex<T3> >  > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin(), 
      _bz_VecExprConstant<complex<T3> > (d3)));
}
#endif // BZ_HAVE_COMPLEX

// where(_bz_VecExpr<P_expr1>, _bz_VecExpr<P_expr2>, Vector<P_numtype3>)
template<class P_expr1, class P_expr2, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExpr<P_expr2>, 
      VectorIterConst<P_numtype3> > >
where(_bz_VecExpr<P_expr1> d1, 
      _bz_VecExpr<P_expr2> d2, 
      const Vector<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExpr<P_expr2>, 
      VectorIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2, 
      d3.begin()));
}

// where(_bz_VecExpr<P_expr1>, _bz_VecExpr<P_expr2>, _bz_VecExpr<P_expr3>)
template<class P_expr1, class P_expr2, class P_expr3>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_VecExpr<P_expr3> > >
where(_bz_VecExpr<P_expr1> d1, 
      _bz_VecExpr<P_expr2> d2, 
      _bz_VecExpr<P_expr3> d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_VecExpr<P_expr3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2, 
      d3));
}

// where(_bz_VecExpr<P_expr1>, _bz_VecExpr<P_expr2>, VectorPick<P_numtype3>)
template<class P_expr1, class P_expr2, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExpr<P_expr2>, 
      VectorPickIterConst<P_numtype3> > >
where(_bz_VecExpr<P_expr1> d1, 
      _bz_VecExpr<P_expr2> d2, 
      const VectorPick<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExpr<P_expr2>, 
      VectorPickIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2, 
      d3.begin()));
}

// where(_bz_VecExpr<P_expr1>, _bz_VecExpr<P_expr2>, Range)
template<class P_expr1, class P_expr2>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExpr<P_expr2>, 
      Range > >
where(_bz_VecExpr<P_expr1> d1, 
      _bz_VecExpr<P_expr2> d2, 
      Range d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExpr<P_expr2>, 
      Range > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2, 
      d3));
}

// where(_bz_VecExpr<P_expr1>, _bz_VecExpr<P_expr2>, TinyVector<P_numtype3, N_length3>)
template<class P_expr1, class P_expr2, class P_numtype3, int N_length3>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExpr<P_expr2>, 
      TinyVectorIterConst<P_numtype3, N_length3> > >
where(_bz_VecExpr<P_expr1> d1, 
      _bz_VecExpr<P_expr2> d2, 
      const TinyVector<P_numtype3, N_length3>& d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExpr<P_expr2>, 
      TinyVectorIterConst<P_numtype3, N_length3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2, 
      d3.begin()));
}

// where(_bz_VecExpr<P_expr1>, _bz_VecExpr<P_expr2>, int)
template<class P_expr1, class P_expr2>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_VecExprConstant<int> > >
where(_bz_VecExpr<P_expr1> d1, 
      _bz_VecExpr<P_expr2> d2, 
      int d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_VecExprConstant<int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2, 
      _bz_VecExprConstant<int>(d3)));
}

// where(_bz_VecExpr<P_expr1>, _bz_VecExpr<P_expr2>, float)
template<class P_expr1, class P_expr2>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_VecExprConstant<float> > >
where(_bz_VecExpr<P_expr1> d1, 
      _bz_VecExpr<P_expr2> d2, 
      float d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_VecExprConstant<float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2, 
      _bz_VecExprConstant<float>(d3)));
}

// where(_bz_VecExpr<P_expr1>, _bz_VecExpr<P_expr2>, double)
template<class P_expr1, class P_expr2>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_VecExprConstant<double> > >
where(_bz_VecExpr<P_expr1> d1, 
      _bz_VecExpr<P_expr2> d2, 
      double d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_VecExprConstant<double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2, 
      _bz_VecExprConstant<double>(d3)));
}

// where(_bz_VecExpr<P_expr1>, _bz_VecExpr<P_expr2>, long double)
template<class P_expr1, class P_expr2>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_VecExprConstant<long double> > >
where(_bz_VecExpr<P_expr1> d1, 
      _bz_VecExpr<P_expr2> d2, 
      long double d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_VecExprConstant<long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2, 
      _bz_VecExprConstant<long double>(d3)));
}

// where(_bz_VecExpr<P_expr1>, _bz_VecExpr<P_expr2>, complex<T3>)
#ifdef BZ_HAVE_COMPLEX
template<class P_expr1, class P_expr2, class T3>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_VecExprConstant<complex<T3> >  > >
where(_bz_VecExpr<P_expr1> d1, 
      _bz_VecExpr<P_expr2> d2, 
      complex<T3> d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_VecExprConstant<complex<T3> >  > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2, 
      _bz_VecExprConstant<complex<T3> > (d3)));
}
#endif // BZ_HAVE_COMPLEX

// where(_bz_VecExpr<P_expr1>, VectorPick<P_numtype2>, Vector<P_numtype3>)
template<class P_expr1, class P_numtype2, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      VectorPickIterConst<P_numtype2>, 
      VectorIterConst<P_numtype3> > >
where(_bz_VecExpr<P_expr1> d1, 
      const VectorPick<P_numtype2>& d2, 
      const Vector<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      VectorPickIterConst<P_numtype2>, 
      VectorIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin(), 
      d3.begin()));
}

// where(_bz_VecExpr<P_expr1>, VectorPick<P_numtype2>, _bz_VecExpr<P_expr3>)
template<class P_expr1, class P_numtype2, class P_expr3>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_VecExpr<P_expr3> > >
where(_bz_VecExpr<P_expr1> d1, 
      const VectorPick<P_numtype2>& d2, 
      _bz_VecExpr<P_expr3> d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_VecExpr<P_expr3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin(), 
      d3));
}

// where(_bz_VecExpr<P_expr1>, VectorPick<P_numtype2>, VectorPick<P_numtype3>)
template<class P_expr1, class P_numtype2, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      VectorPickIterConst<P_numtype2>, 
      VectorPickIterConst<P_numtype3> > >
where(_bz_VecExpr<P_expr1> d1, 
      const VectorPick<P_numtype2>& d2, 
      const VectorPick<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      VectorPickIterConst<P_numtype2>, 
      VectorPickIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin(), 
      d3.begin()));
}

// where(_bz_VecExpr<P_expr1>, VectorPick<P_numtype2>, Range)
template<class P_expr1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      VectorPickIterConst<P_numtype2>, 
      Range > >
where(_bz_VecExpr<P_expr1> d1, 
      const VectorPick<P_numtype2>& d2, 
      Range d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      VectorPickIterConst<P_numtype2>, 
      Range > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin(), 
      d3));
}

// where(_bz_VecExpr<P_expr1>, VectorPick<P_numtype2>, TinyVector<P_numtype3, N_length3>)
template<class P_expr1, class P_numtype2, class P_numtype3, int N_length3>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      VectorPickIterConst<P_numtype2>, 
      TinyVectorIterConst<P_numtype3, N_length3> > >
where(_bz_VecExpr<P_expr1> d1, 
      const VectorPick<P_numtype2>& d2, 
      const TinyVector<P_numtype3, N_length3>& d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      VectorPickIterConst<P_numtype2>, 
      TinyVectorIterConst<P_numtype3, N_length3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin(), 
      d3.begin()));
}

// where(_bz_VecExpr<P_expr1>, VectorPick<P_numtype2>, int)
template<class P_expr1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_VecExprConstant<int> > >
where(_bz_VecExpr<P_expr1> d1, 
      const VectorPick<P_numtype2>& d2, 
      int d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_VecExprConstant<int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin(), 
      _bz_VecExprConstant<int>(d3)));
}

// where(_bz_VecExpr<P_expr1>, VectorPick<P_numtype2>, float)
template<class P_expr1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_VecExprConstant<float> > >
where(_bz_VecExpr<P_expr1> d1, 
      const VectorPick<P_numtype2>& d2, 
      float d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_VecExprConstant<float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin(), 
      _bz_VecExprConstant<float>(d3)));
}

// where(_bz_VecExpr<P_expr1>, VectorPick<P_numtype2>, double)
template<class P_expr1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_VecExprConstant<double> > >
where(_bz_VecExpr<P_expr1> d1, 
      const VectorPick<P_numtype2>& d2, 
      double d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_VecExprConstant<double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin(), 
      _bz_VecExprConstant<double>(d3)));
}

// where(_bz_VecExpr<P_expr1>, VectorPick<P_numtype2>, long double)
template<class P_expr1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_VecExprConstant<long double> > >
where(_bz_VecExpr<P_expr1> d1, 
      const VectorPick<P_numtype2>& d2, 
      long double d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_VecExprConstant<long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin(), 
      _bz_VecExprConstant<long double>(d3)));
}

// where(_bz_VecExpr<P_expr1>, VectorPick<P_numtype2>, complex<T3>)
#ifdef BZ_HAVE_COMPLEX
template<class P_expr1, class P_numtype2, class T3>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_VecExprConstant<complex<T3> >  > >
where(_bz_VecExpr<P_expr1> d1, 
      const VectorPick<P_numtype2>& d2, 
      complex<T3> d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_VecExprConstant<complex<T3> >  > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin(), 
      _bz_VecExprConstant<complex<T3> > (d3)));
}
#endif // BZ_HAVE_COMPLEX

// where(_bz_VecExpr<P_expr1>, Range, Vector<P_numtype3>)
template<class P_expr1, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      Range, 
      VectorIterConst<P_numtype3> > >
where(_bz_VecExpr<P_expr1> d1, 
      Range d2, 
      const Vector<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      Range, 
      VectorIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2, 
      d3.begin()));
}

// where(_bz_VecExpr<P_expr1>, Range, _bz_VecExpr<P_expr3>)
template<class P_expr1, class P_expr3>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      Range, 
      _bz_VecExpr<P_expr3> > >
where(_bz_VecExpr<P_expr1> d1, 
      Range d2, 
      _bz_VecExpr<P_expr3> d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      Range, 
      _bz_VecExpr<P_expr3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2, 
      d3));
}

// where(_bz_VecExpr<P_expr1>, Range, VectorPick<P_numtype3>)
template<class P_expr1, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      Range, 
      VectorPickIterConst<P_numtype3> > >
where(_bz_VecExpr<P_expr1> d1, 
      Range d2, 
      const VectorPick<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      Range, 
      VectorPickIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2, 
      d3.begin()));
}

// where(_bz_VecExpr<P_expr1>, Range, Range)
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      Range, 
      Range > >
where(_bz_VecExpr<P_expr1> d1, 
      Range d2, 
      Range d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      Range, 
      Range > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2, 
      d3));
}

// where(_bz_VecExpr<P_expr1>, Range, TinyVector<P_numtype3, N_length3>)
template<class P_expr1, class P_numtype3, int N_length3>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      Range, 
      TinyVectorIterConst<P_numtype3, N_length3> > >
where(_bz_VecExpr<P_expr1> d1, 
      Range d2, 
      const TinyVector<P_numtype3, N_length3>& d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      Range, 
      TinyVectorIterConst<P_numtype3, N_length3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2, 
      d3.begin()));
}

// where(_bz_VecExpr<P_expr1>, Range, int)
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      Range, 
      _bz_VecExprConstant<int> > >
where(_bz_VecExpr<P_expr1> d1, 
      Range d2, 
      int d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      Range, 
      _bz_VecExprConstant<int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2, 
      _bz_VecExprConstant<int>(d3)));
}

// where(_bz_VecExpr<P_expr1>, Range, float)
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      Range, 
      _bz_VecExprConstant<float> > >
where(_bz_VecExpr<P_expr1> d1, 
      Range d2, 
      float d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      Range, 
      _bz_VecExprConstant<float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2, 
      _bz_VecExprConstant<float>(d3)));
}

// where(_bz_VecExpr<P_expr1>, Range, double)
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      Range, 
      _bz_VecExprConstant<double> > >
where(_bz_VecExpr<P_expr1> d1, 
      Range d2, 
      double d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      Range, 
      _bz_VecExprConstant<double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2, 
      _bz_VecExprConstant<double>(d3)));
}

// where(_bz_VecExpr<P_expr1>, Range, long double)
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      Range, 
      _bz_VecExprConstant<long double> > >
where(_bz_VecExpr<P_expr1> d1, 
      Range d2, 
      long double d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      Range, 
      _bz_VecExprConstant<long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2, 
      _bz_VecExprConstant<long double>(d3)));
}

// where(_bz_VecExpr<P_expr1>, Range, complex<T3>)
#ifdef BZ_HAVE_COMPLEX
template<class P_expr1, class T3>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      Range, 
      _bz_VecExprConstant<complex<T3> >  > >
where(_bz_VecExpr<P_expr1> d1, 
      Range d2, 
      complex<T3> d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      Range, 
      _bz_VecExprConstant<complex<T3> >  > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2, 
      _bz_VecExprConstant<complex<T3> > (d3)));
}
#endif // BZ_HAVE_COMPLEX

// where(_bz_VecExpr<P_expr1>, TinyVector<P_numtype2, N_length2>, Vector<P_numtype3>)
template<class P_expr1, class P_numtype2, int N_length2, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      VectorIterConst<P_numtype3> > >
where(_bz_VecExpr<P_expr1> d1, 
      const TinyVector<P_numtype2, N_length2>& d2, 
      const Vector<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      VectorIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin(), 
      d3.begin()));
}

// where(_bz_VecExpr<P_expr1>, TinyVector<P_numtype2, N_length2>, _bz_VecExpr<P_expr3>)
template<class P_expr1, class P_numtype2, int N_length2, class P_expr3>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_VecExpr<P_expr3> > >
where(_bz_VecExpr<P_expr1> d1, 
      const TinyVector<P_numtype2, N_length2>& d2, 
      _bz_VecExpr<P_expr3> d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_VecExpr<P_expr3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin(), 
      d3));
}

// where(_bz_VecExpr<P_expr1>, TinyVector<P_numtype2, N_length2>, VectorPick<P_numtype3>)
template<class P_expr1, class P_numtype2, int N_length2, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      VectorPickIterConst<P_numtype3> > >
where(_bz_VecExpr<P_expr1> d1, 
      const TinyVector<P_numtype2, N_length2>& d2, 
      const VectorPick<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      VectorPickIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin(), 
      d3.begin()));
}

// where(_bz_VecExpr<P_expr1>, TinyVector<P_numtype2, N_length2>, Range)
template<class P_expr1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      Range > >
where(_bz_VecExpr<P_expr1> d1, 
      const TinyVector<P_numtype2, N_length2>& d2, 
      Range d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      Range > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin(), 
      d3));
}

// where(_bz_VecExpr<P_expr1>, TinyVector<P_numtype2, N_length2>, TinyVector<P_numtype3, N_length3>)
template<class P_expr1, class P_numtype2, int N_length2, class P_numtype3, int N_length3>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      TinyVectorIterConst<P_numtype3, N_length3> > >
where(_bz_VecExpr<P_expr1> d1, 
      const TinyVector<P_numtype2, N_length2>& d2, 
      const TinyVector<P_numtype3, N_length3>& d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      TinyVectorIterConst<P_numtype3, N_length3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin(), 
      d3.begin()));
}

// where(_bz_VecExpr<P_expr1>, TinyVector<P_numtype2, N_length2>, int)
template<class P_expr1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_VecExprConstant<int> > >
where(_bz_VecExpr<P_expr1> d1, 
      const TinyVector<P_numtype2, N_length2>& d2, 
      int d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_VecExprConstant<int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin(), 
      _bz_VecExprConstant<int>(d3)));
}

// where(_bz_VecExpr<P_expr1>, TinyVector<P_numtype2, N_length2>, float)
template<class P_expr1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_VecExprConstant<float> > >
where(_bz_VecExpr<P_expr1> d1, 
      const TinyVector<P_numtype2, N_length2>& d2, 
      float d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_VecExprConstant<float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin(), 
      _bz_VecExprConstant<float>(d3)));
}

// where(_bz_VecExpr<P_expr1>, TinyVector<P_numtype2, N_length2>, double)
template<class P_expr1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_VecExprConstant<double> > >
where(_bz_VecExpr<P_expr1> d1, 
      const TinyVector<P_numtype2, N_length2>& d2, 
      double d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_VecExprConstant<double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin(), 
      _bz_VecExprConstant<double>(d3)));
}

// where(_bz_VecExpr<P_expr1>, TinyVector<P_numtype2, N_length2>, long double)
template<class P_expr1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_VecExprConstant<long double> > >
where(_bz_VecExpr<P_expr1> d1, 
      const TinyVector<P_numtype2, N_length2>& d2, 
      long double d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_VecExprConstant<long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin(), 
      _bz_VecExprConstant<long double>(d3)));
}

// where(_bz_VecExpr<P_expr1>, TinyVector<P_numtype2, N_length2>, complex<T3>)
#ifdef BZ_HAVE_COMPLEX
template<class P_expr1, class P_numtype2, int N_length2, class T3>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_VecExprConstant<complex<T3> >  > >
where(_bz_VecExpr<P_expr1> d1, 
      const TinyVector<P_numtype2, N_length2>& d2, 
      complex<T3> d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_VecExprConstant<complex<T3> >  > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin(), 
      _bz_VecExprConstant<complex<T3> > (d3)));
}
#endif // BZ_HAVE_COMPLEX

// where(_bz_VecExpr<P_expr1>, int, Vector<P_numtype3>)
template<class P_expr1, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<int>, 
      VectorIterConst<P_numtype3> > >
where(_bz_VecExpr<P_expr1> d1, 
      int d2, 
      const Vector<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<int>, 
      VectorIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<int>(d2), 
      d3.begin()));
}

// where(_bz_VecExpr<P_expr1>, int, _bz_VecExpr<P_expr3>)
template<class P_expr1, class P_expr3>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<int>, 
      _bz_VecExpr<P_expr3> > >
where(_bz_VecExpr<P_expr1> d1, 
      int d2, 
      _bz_VecExpr<P_expr3> d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<int>, 
      _bz_VecExpr<P_expr3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<int>(d2), 
      d3));
}

// where(_bz_VecExpr<P_expr1>, int, VectorPick<P_numtype3>)
template<class P_expr1, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<int>, 
      VectorPickIterConst<P_numtype3> > >
where(_bz_VecExpr<P_expr1> d1, 
      int d2, 
      const VectorPick<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<int>, 
      VectorPickIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<int>(d2), 
      d3.begin()));
}

// where(_bz_VecExpr<P_expr1>, int, Range)
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<int>, 
      Range > >
where(_bz_VecExpr<P_expr1> d1, 
      int d2, 
      Range d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<int>, 
      Range > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<int>(d2), 
      d3));
}

// where(_bz_VecExpr<P_expr1>, int, TinyVector<P_numtype3, N_length3>)
template<class P_expr1, class P_numtype3, int N_length3>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<int>, 
      TinyVectorIterConst<P_numtype3, N_length3> > >
where(_bz_VecExpr<P_expr1> d1, 
      int d2, 
      const TinyVector<P_numtype3, N_length3>& d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<int>, 
      TinyVectorIterConst<P_numtype3, N_length3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<int>(d2), 
      d3.begin()));
}

// where(_bz_VecExpr<P_expr1>, int, int)
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<int>, 
      _bz_VecExprConstant<int> > >
where(_bz_VecExpr<P_expr1> d1, 
      int d2, 
      int d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<int>, 
      _bz_VecExprConstant<int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<int>(d2), 
      _bz_VecExprConstant<int>(d3)));
}

// where(_bz_VecExpr<P_expr1>, float, Vector<P_numtype3>)
template<class P_expr1, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<float>, 
      VectorIterConst<P_numtype3> > >
where(_bz_VecExpr<P_expr1> d1, 
      float d2, 
      const Vector<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<float>, 
      VectorIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<float>(d2), 
      d3.begin()));
}

// where(_bz_VecExpr<P_expr1>, float, _bz_VecExpr<P_expr3>)
template<class P_expr1, class P_expr3>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<float>, 
      _bz_VecExpr<P_expr3> > >
where(_bz_VecExpr<P_expr1> d1, 
      float d2, 
      _bz_VecExpr<P_expr3> d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<float>, 
      _bz_VecExpr<P_expr3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<float>(d2), 
      d3));
}

// where(_bz_VecExpr<P_expr1>, float, VectorPick<P_numtype3>)
template<class P_expr1, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<float>, 
      VectorPickIterConst<P_numtype3> > >
where(_bz_VecExpr<P_expr1> d1, 
      float d2, 
      const VectorPick<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<float>, 
      VectorPickIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<float>(d2), 
      d3.begin()));
}

// where(_bz_VecExpr<P_expr1>, float, Range)
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<float>, 
      Range > >
where(_bz_VecExpr<P_expr1> d1, 
      float d2, 
      Range d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<float>, 
      Range > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<float>(d2), 
      d3));
}

// where(_bz_VecExpr<P_expr1>, float, TinyVector<P_numtype3, N_length3>)
template<class P_expr1, class P_numtype3, int N_length3>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<float>, 
      TinyVectorIterConst<P_numtype3, N_length3> > >
where(_bz_VecExpr<P_expr1> d1, 
      float d2, 
      const TinyVector<P_numtype3, N_length3>& d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<float>, 
      TinyVectorIterConst<P_numtype3, N_length3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<float>(d2), 
      d3.begin()));
}

// where(_bz_VecExpr<P_expr1>, float, float)
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<float>, 
      _bz_VecExprConstant<float> > >
where(_bz_VecExpr<P_expr1> d1, 
      float d2, 
      float d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<float>, 
      _bz_VecExprConstant<float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<float>(d2), 
      _bz_VecExprConstant<float>(d3)));
}

// where(_bz_VecExpr<P_expr1>, double, Vector<P_numtype3>)
template<class P_expr1, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<double>, 
      VectorIterConst<P_numtype3> > >
where(_bz_VecExpr<P_expr1> d1, 
      double d2, 
      const Vector<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<double>, 
      VectorIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<double>(d2), 
      d3.begin()));
}

// where(_bz_VecExpr<P_expr1>, double, _bz_VecExpr<P_expr3>)
template<class P_expr1, class P_expr3>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<double>, 
      _bz_VecExpr<P_expr3> > >
where(_bz_VecExpr<P_expr1> d1, 
      double d2, 
      _bz_VecExpr<P_expr3> d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<double>, 
      _bz_VecExpr<P_expr3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<double>(d2), 
      d3));
}

// where(_bz_VecExpr<P_expr1>, double, VectorPick<P_numtype3>)
template<class P_expr1, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<double>, 
      VectorPickIterConst<P_numtype3> > >
where(_bz_VecExpr<P_expr1> d1, 
      double d2, 
      const VectorPick<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<double>, 
      VectorPickIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<double>(d2), 
      d3.begin()));
}

// where(_bz_VecExpr<P_expr1>, double, Range)
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<double>, 
      Range > >
where(_bz_VecExpr<P_expr1> d1, 
      double d2, 
      Range d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<double>, 
      Range > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<double>(d2), 
      d3));
}

// where(_bz_VecExpr<P_expr1>, double, TinyVector<P_numtype3, N_length3>)
template<class P_expr1, class P_numtype3, int N_length3>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<double>, 
      TinyVectorIterConst<P_numtype3, N_length3> > >
where(_bz_VecExpr<P_expr1> d1, 
      double d2, 
      const TinyVector<P_numtype3, N_length3>& d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<double>, 
      TinyVectorIterConst<P_numtype3, N_length3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<double>(d2), 
      d3.begin()));
}

// where(_bz_VecExpr<P_expr1>, double, double)
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<double>, 
      _bz_VecExprConstant<double> > >
where(_bz_VecExpr<P_expr1> d1, 
      double d2, 
      double d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<double>, 
      _bz_VecExprConstant<double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<double>(d2), 
      _bz_VecExprConstant<double>(d3)));
}

// where(_bz_VecExpr<P_expr1>, long double, Vector<P_numtype3>)
template<class P_expr1, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<long double>, 
      VectorIterConst<P_numtype3> > >
where(_bz_VecExpr<P_expr1> d1, 
      long double d2, 
      const Vector<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<long double>, 
      VectorIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<long double>(d2), 
      d3.begin()));
}

// where(_bz_VecExpr<P_expr1>, long double, _bz_VecExpr<P_expr3>)
template<class P_expr1, class P_expr3>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<long double>, 
      _bz_VecExpr<P_expr3> > >
where(_bz_VecExpr<P_expr1> d1, 
      long double d2, 
      _bz_VecExpr<P_expr3> d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<long double>, 
      _bz_VecExpr<P_expr3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<long double>(d2), 
      d3));
}

// where(_bz_VecExpr<P_expr1>, long double, VectorPick<P_numtype3>)
template<class P_expr1, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<long double>, 
      VectorPickIterConst<P_numtype3> > >
where(_bz_VecExpr<P_expr1> d1, 
      long double d2, 
      const VectorPick<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<long double>, 
      VectorPickIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<long double>(d2), 
      d3.begin()));
}

// where(_bz_VecExpr<P_expr1>, long double, Range)
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<long double>, 
      Range > >
where(_bz_VecExpr<P_expr1> d1, 
      long double d2, 
      Range d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<long double>, 
      Range > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<long double>(d2), 
      d3));
}

// where(_bz_VecExpr<P_expr1>, long double, TinyVector<P_numtype3, N_length3>)
template<class P_expr1, class P_numtype3, int N_length3>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<long double>, 
      TinyVectorIterConst<P_numtype3, N_length3> > >
where(_bz_VecExpr<P_expr1> d1, 
      long double d2, 
      const TinyVector<P_numtype3, N_length3>& d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<long double>, 
      TinyVectorIterConst<P_numtype3, N_length3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<long double>(d2), 
      d3.begin()));
}

// where(_bz_VecExpr<P_expr1>, long double, long double)
template<class P_expr1>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<long double>, 
      _bz_VecExprConstant<long double> > >
where(_bz_VecExpr<P_expr1> d1, 
      long double d2, 
      long double d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<long double>, 
      _bz_VecExprConstant<long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<long double>(d2), 
      _bz_VecExprConstant<long double>(d3)));
}

// where(_bz_VecExpr<P_expr1>, complex<T2>, Vector<P_numtype3>)
#ifdef BZ_HAVE_COMPLEX
template<class P_expr1, class T2, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<complex<T2> > , 
      VectorIterConst<P_numtype3> > >
where(_bz_VecExpr<P_expr1> d1, 
      complex<T2> d2, 
      const Vector<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<complex<T2> > , 
      VectorIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<complex<T2> > (d2), 
      d3.begin()));
}
#endif // BZ_HAVE_COMPLEX

// where(_bz_VecExpr<P_expr1>, complex<T2>, _bz_VecExpr<P_expr3>)
#ifdef BZ_HAVE_COMPLEX
template<class P_expr1, class T2, class P_expr3>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_VecExpr<P_expr3> > >
where(_bz_VecExpr<P_expr1> d1, 
      complex<T2> d2, 
      _bz_VecExpr<P_expr3> d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_VecExpr<P_expr3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<complex<T2> > (d2), 
      d3));
}
#endif // BZ_HAVE_COMPLEX

// where(_bz_VecExpr<P_expr1>, complex<T2>, VectorPick<P_numtype3>)
#ifdef BZ_HAVE_COMPLEX
template<class P_expr1, class T2, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<complex<T2> > , 
      VectorPickIterConst<P_numtype3> > >
where(_bz_VecExpr<P_expr1> d1, 
      complex<T2> d2, 
      const VectorPick<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<complex<T2> > , 
      VectorPickIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<complex<T2> > (d2), 
      d3.begin()));
}
#endif // BZ_HAVE_COMPLEX

// where(_bz_VecExpr<P_expr1>, complex<T2>, Range)
#ifdef BZ_HAVE_COMPLEX
template<class P_expr1, class T2>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<complex<T2> > , 
      Range > >
where(_bz_VecExpr<P_expr1> d1, 
      complex<T2> d2, 
      Range d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<complex<T2> > , 
      Range > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<complex<T2> > (d2), 
      d3));
}
#endif // BZ_HAVE_COMPLEX

// where(_bz_VecExpr<P_expr1>, complex<T2>, TinyVector<P_numtype3, N_length3>)
#ifdef BZ_HAVE_COMPLEX
template<class P_expr1, class T2, class P_numtype3, int N_length3>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<complex<T2> > , 
      TinyVectorIterConst<P_numtype3, N_length3> > >
where(_bz_VecExpr<P_expr1> d1, 
      complex<T2> d2, 
      const TinyVector<P_numtype3, N_length3>& d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<complex<T2> > , 
      TinyVectorIterConst<P_numtype3, N_length3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<complex<T2> > (d2), 
      d3.begin()));
}
#endif // BZ_HAVE_COMPLEX

// where(_bz_VecExpr<P_expr1>, complex<T2>, complex<T3>)
#ifdef BZ_HAVE_COMPLEX
template<class P_expr1, class T2, class T3>
inline
_bz_VecExpr<_bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_VecExprConstant<complex<T3> >  > >
where(_bz_VecExpr<P_expr1> d1, 
      complex<T2> d2, 
      complex<T3> d3)
{ 
    typedef _bz_VecWhere<_bz_VecExpr<P_expr1>, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_VecExprConstant<complex<T3> >  > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<complex<T2> > (d2), 
      _bz_VecExprConstant<complex<T3> > (d3)));
}
#endif // BZ_HAVE_COMPLEX

// where(VectorPick<P_numtype1>, Vector<P_numtype2>, Vector<P_numtype3>)
template<class P_numtype1, class P_numtype2, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      VectorIterConst<P_numtype3> > >
where(const VectorPick<P_numtype1>& d1, 
      const Vector<P_numtype2>& d2, 
      const Vector<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      VectorIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      d3.begin()));
}

// where(VectorPick<P_numtype1>, Vector<P_numtype2>, _bz_VecExpr<P_expr3>)
template<class P_numtype1, class P_numtype2, class P_expr3>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      _bz_VecExpr<P_expr3> > >
where(const VectorPick<P_numtype1>& d1, 
      const Vector<P_numtype2>& d2, 
      _bz_VecExpr<P_expr3> d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      _bz_VecExpr<P_expr3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      d3));
}

// where(VectorPick<P_numtype1>, Vector<P_numtype2>, VectorPick<P_numtype3>)
template<class P_numtype1, class P_numtype2, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      VectorPickIterConst<P_numtype3> > >
where(const VectorPick<P_numtype1>& d1, 
      const Vector<P_numtype2>& d2, 
      const VectorPick<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      VectorPickIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      d3.begin()));
}

// where(VectorPick<P_numtype1>, Vector<P_numtype2>, Range)
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      Range > >
where(const VectorPick<P_numtype1>& d1, 
      const Vector<P_numtype2>& d2, 
      Range d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      Range > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      d3));
}

// where(VectorPick<P_numtype1>, Vector<P_numtype2>, TinyVector<P_numtype3, N_length3>)
template<class P_numtype1, class P_numtype2, class P_numtype3, int N_length3>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      TinyVectorIterConst<P_numtype3, N_length3> > >
where(const VectorPick<P_numtype1>& d1, 
      const Vector<P_numtype2>& d2, 
      const TinyVector<P_numtype3, N_length3>& d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      TinyVectorIterConst<P_numtype3, N_length3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      d3.begin()));
}

// where(VectorPick<P_numtype1>, Vector<P_numtype2>, int)
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      _bz_VecExprConstant<int> > >
where(const VectorPick<P_numtype1>& d1, 
      const Vector<P_numtype2>& d2, 
      int d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      _bz_VecExprConstant<int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      _bz_VecExprConstant<int>(d3)));
}

// where(VectorPick<P_numtype1>, Vector<P_numtype2>, float)
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      _bz_VecExprConstant<float> > >
where(const VectorPick<P_numtype1>& d1, 
      const Vector<P_numtype2>& d2, 
      float d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      _bz_VecExprConstant<float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      _bz_VecExprConstant<float>(d3)));
}

// where(VectorPick<P_numtype1>, Vector<P_numtype2>, double)
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      _bz_VecExprConstant<double> > >
where(const VectorPick<P_numtype1>& d1, 
      const Vector<P_numtype2>& d2, 
      double d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      _bz_VecExprConstant<double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      _bz_VecExprConstant<double>(d3)));
}

// where(VectorPick<P_numtype1>, Vector<P_numtype2>, long double)
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      _bz_VecExprConstant<long double> > >
where(const VectorPick<P_numtype1>& d1, 
      const Vector<P_numtype2>& d2, 
      long double d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      _bz_VecExprConstant<long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      _bz_VecExprConstant<long double>(d3)));
}

// where(VectorPick<P_numtype1>, Vector<P_numtype2>, complex<T3>)
#ifdef BZ_HAVE_COMPLEX
template<class P_numtype1, class P_numtype2, class T3>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      _bz_VecExprConstant<complex<T3> >  > >
where(const VectorPick<P_numtype1>& d1, 
      const Vector<P_numtype2>& d2, 
      complex<T3> d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      VectorIterConst<P_numtype2>, 
      _bz_VecExprConstant<complex<T3> >  > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      _bz_VecExprConstant<complex<T3> > (d3)));
}
#endif // BZ_HAVE_COMPLEX

// where(VectorPick<P_numtype1>, _bz_VecExpr<P_expr2>, Vector<P_numtype3>)
template<class P_numtype1, class P_expr2, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      VectorIterConst<P_numtype3> > >
where(const VectorPick<P_numtype1>& d1, 
      _bz_VecExpr<P_expr2> d2, 
      const Vector<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      VectorIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2, 
      d3.begin()));
}

// where(VectorPick<P_numtype1>, _bz_VecExpr<P_expr2>, _bz_VecExpr<P_expr3>)
template<class P_numtype1, class P_expr2, class P_expr3>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_VecExpr<P_expr3> > >
where(const VectorPick<P_numtype1>& d1, 
      _bz_VecExpr<P_expr2> d2, 
      _bz_VecExpr<P_expr3> d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_VecExpr<P_expr3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2, 
      d3));
}

// where(VectorPick<P_numtype1>, _bz_VecExpr<P_expr2>, VectorPick<P_numtype3>)
template<class P_numtype1, class P_expr2, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      VectorPickIterConst<P_numtype3> > >
where(const VectorPick<P_numtype1>& d1, 
      _bz_VecExpr<P_expr2> d2, 
      const VectorPick<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      VectorPickIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2, 
      d3.begin()));
}

// where(VectorPick<P_numtype1>, _bz_VecExpr<P_expr2>, Range)
template<class P_numtype1, class P_expr2>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      Range > >
where(const VectorPick<P_numtype1>& d1, 
      _bz_VecExpr<P_expr2> d2, 
      Range d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      Range > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2, 
      d3));
}

// where(VectorPick<P_numtype1>, _bz_VecExpr<P_expr2>, TinyVector<P_numtype3, N_length3>)
template<class P_numtype1, class P_expr2, class P_numtype3, int N_length3>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      TinyVectorIterConst<P_numtype3, N_length3> > >
where(const VectorPick<P_numtype1>& d1, 
      _bz_VecExpr<P_expr2> d2, 
      const TinyVector<P_numtype3, N_length3>& d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      TinyVectorIterConst<P_numtype3, N_length3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2, 
      d3.begin()));
}

// where(VectorPick<P_numtype1>, _bz_VecExpr<P_expr2>, int)
template<class P_numtype1, class P_expr2>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_VecExprConstant<int> > >
where(const VectorPick<P_numtype1>& d1, 
      _bz_VecExpr<P_expr2> d2, 
      int d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_VecExprConstant<int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2, 
      _bz_VecExprConstant<int>(d3)));
}

// where(VectorPick<P_numtype1>, _bz_VecExpr<P_expr2>, float)
template<class P_numtype1, class P_expr2>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_VecExprConstant<float> > >
where(const VectorPick<P_numtype1>& d1, 
      _bz_VecExpr<P_expr2> d2, 
      float d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_VecExprConstant<float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2, 
      _bz_VecExprConstant<float>(d3)));
}

// where(VectorPick<P_numtype1>, _bz_VecExpr<P_expr2>, double)
template<class P_numtype1, class P_expr2>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_VecExprConstant<double> > >
where(const VectorPick<P_numtype1>& d1, 
      _bz_VecExpr<P_expr2> d2, 
      double d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_VecExprConstant<double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2, 
      _bz_VecExprConstant<double>(d3)));
}

// where(VectorPick<P_numtype1>, _bz_VecExpr<P_expr2>, long double)
template<class P_numtype1, class P_expr2>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_VecExprConstant<long double> > >
where(const VectorPick<P_numtype1>& d1, 
      _bz_VecExpr<P_expr2> d2, 
      long double d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_VecExprConstant<long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2, 
      _bz_VecExprConstant<long double>(d3)));
}

// where(VectorPick<P_numtype1>, _bz_VecExpr<P_expr2>, complex<T3>)
#ifdef BZ_HAVE_COMPLEX
template<class P_numtype1, class P_expr2, class T3>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_VecExprConstant<complex<T3> >  > >
where(const VectorPick<P_numtype1>& d1, 
      _bz_VecExpr<P_expr2> d2, 
      complex<T3> d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_VecExprConstant<complex<T3> >  > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2, 
      _bz_VecExprConstant<complex<T3> > (d3)));
}
#endif // BZ_HAVE_COMPLEX

// where(VectorPick<P_numtype1>, VectorPick<P_numtype2>, Vector<P_numtype3>)
template<class P_numtype1, class P_numtype2, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      VectorIterConst<P_numtype3> > >
where(const VectorPick<P_numtype1>& d1, 
      const VectorPick<P_numtype2>& d2, 
      const Vector<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      VectorIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      d3.begin()));
}

// where(VectorPick<P_numtype1>, VectorPick<P_numtype2>, _bz_VecExpr<P_expr3>)
template<class P_numtype1, class P_numtype2, class P_expr3>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_VecExpr<P_expr3> > >
where(const VectorPick<P_numtype1>& d1, 
      const VectorPick<P_numtype2>& d2, 
      _bz_VecExpr<P_expr3> d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_VecExpr<P_expr3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      d3));
}

// where(VectorPick<P_numtype1>, VectorPick<P_numtype2>, VectorPick<P_numtype3>)
template<class P_numtype1, class P_numtype2, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      VectorPickIterConst<P_numtype3> > >
where(const VectorPick<P_numtype1>& d1, 
      const VectorPick<P_numtype2>& d2, 
      const VectorPick<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      VectorPickIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      d3.begin()));
}

// where(VectorPick<P_numtype1>, VectorPick<P_numtype2>, Range)
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      Range > >
where(const VectorPick<P_numtype1>& d1, 
      const VectorPick<P_numtype2>& d2, 
      Range d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      Range > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      d3));
}

// where(VectorPick<P_numtype1>, VectorPick<P_numtype2>, TinyVector<P_numtype3, N_length3>)
template<class P_numtype1, class P_numtype2, class P_numtype3, int N_length3>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      TinyVectorIterConst<P_numtype3, N_length3> > >
where(const VectorPick<P_numtype1>& d1, 
      const VectorPick<P_numtype2>& d2, 
      const TinyVector<P_numtype3, N_length3>& d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      TinyVectorIterConst<P_numtype3, N_length3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      d3.begin()));
}

// where(VectorPick<P_numtype1>, VectorPick<P_numtype2>, int)
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_VecExprConstant<int> > >
where(const VectorPick<P_numtype1>& d1, 
      const VectorPick<P_numtype2>& d2, 
      int d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_VecExprConstant<int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      _bz_VecExprConstant<int>(d3)));
}

// where(VectorPick<P_numtype1>, VectorPick<P_numtype2>, float)
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_VecExprConstant<float> > >
where(const VectorPick<P_numtype1>& d1, 
      const VectorPick<P_numtype2>& d2, 
      float d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_VecExprConstant<float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      _bz_VecExprConstant<float>(d3)));
}

// where(VectorPick<P_numtype1>, VectorPick<P_numtype2>, double)
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_VecExprConstant<double> > >
where(const VectorPick<P_numtype1>& d1, 
      const VectorPick<P_numtype2>& d2, 
      double d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_VecExprConstant<double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      _bz_VecExprConstant<double>(d3)));
}

// where(VectorPick<P_numtype1>, VectorPick<P_numtype2>, long double)
template<class P_numtype1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_VecExprConstant<long double> > >
where(const VectorPick<P_numtype1>& d1, 
      const VectorPick<P_numtype2>& d2, 
      long double d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_VecExprConstant<long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      _bz_VecExprConstant<long double>(d3)));
}

// where(VectorPick<P_numtype1>, VectorPick<P_numtype2>, complex<T3>)
#ifdef BZ_HAVE_COMPLEX
template<class P_numtype1, class P_numtype2, class T3>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_VecExprConstant<complex<T3> >  > >
where(const VectorPick<P_numtype1>& d1, 
      const VectorPick<P_numtype2>& d2, 
      complex<T3> d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_VecExprConstant<complex<T3> >  > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      _bz_VecExprConstant<complex<T3> > (d3)));
}
#endif // BZ_HAVE_COMPLEX

// where(VectorPick<P_numtype1>, Range, Vector<P_numtype3>)
template<class P_numtype1, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      Range, 
      VectorIterConst<P_numtype3> > >
where(const VectorPick<P_numtype1>& d1, 
      Range d2, 
      const Vector<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      Range, 
      VectorIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2, 
      d3.begin()));
}

// where(VectorPick<P_numtype1>, Range, _bz_VecExpr<P_expr3>)
template<class P_numtype1, class P_expr3>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      Range, 
      _bz_VecExpr<P_expr3> > >
where(const VectorPick<P_numtype1>& d1, 
      Range d2, 
      _bz_VecExpr<P_expr3> d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      Range, 
      _bz_VecExpr<P_expr3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2, 
      d3));
}

// where(VectorPick<P_numtype1>, Range, VectorPick<P_numtype3>)
template<class P_numtype1, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      Range, 
      VectorPickIterConst<P_numtype3> > >
where(const VectorPick<P_numtype1>& d1, 
      Range d2, 
      const VectorPick<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      Range, 
      VectorPickIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2, 
      d3.begin()));
}

// where(VectorPick<P_numtype1>, Range, Range)
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      Range, 
      Range > >
where(const VectorPick<P_numtype1>& d1, 
      Range d2, 
      Range d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      Range, 
      Range > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2, 
      d3));
}

// where(VectorPick<P_numtype1>, Range, TinyVector<P_numtype3, N_length3>)
template<class P_numtype1, class P_numtype3, int N_length3>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      Range, 
      TinyVectorIterConst<P_numtype3, N_length3> > >
where(const VectorPick<P_numtype1>& d1, 
      Range d2, 
      const TinyVector<P_numtype3, N_length3>& d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      Range, 
      TinyVectorIterConst<P_numtype3, N_length3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2, 
      d3.begin()));
}

// where(VectorPick<P_numtype1>, Range, int)
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      Range, 
      _bz_VecExprConstant<int> > >
where(const VectorPick<P_numtype1>& d1, 
      Range d2, 
      int d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      Range, 
      _bz_VecExprConstant<int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2, 
      _bz_VecExprConstant<int>(d3)));
}

// where(VectorPick<P_numtype1>, Range, float)
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      Range, 
      _bz_VecExprConstant<float> > >
where(const VectorPick<P_numtype1>& d1, 
      Range d2, 
      float d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      Range, 
      _bz_VecExprConstant<float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2, 
      _bz_VecExprConstant<float>(d3)));
}

// where(VectorPick<P_numtype1>, Range, double)
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      Range, 
      _bz_VecExprConstant<double> > >
where(const VectorPick<P_numtype1>& d1, 
      Range d2, 
      double d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      Range, 
      _bz_VecExprConstant<double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2, 
      _bz_VecExprConstant<double>(d3)));
}

// where(VectorPick<P_numtype1>, Range, long double)
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      Range, 
      _bz_VecExprConstant<long double> > >
where(const VectorPick<P_numtype1>& d1, 
      Range d2, 
      long double d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      Range, 
      _bz_VecExprConstant<long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2, 
      _bz_VecExprConstant<long double>(d3)));
}

// where(VectorPick<P_numtype1>, Range, complex<T3>)
#ifdef BZ_HAVE_COMPLEX
template<class P_numtype1, class T3>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      Range, 
      _bz_VecExprConstant<complex<T3> >  > >
where(const VectorPick<P_numtype1>& d1, 
      Range d2, 
      complex<T3> d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      Range, 
      _bz_VecExprConstant<complex<T3> >  > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2, 
      _bz_VecExprConstant<complex<T3> > (d3)));
}
#endif // BZ_HAVE_COMPLEX

// where(VectorPick<P_numtype1>, TinyVector<P_numtype2, N_length2>, Vector<P_numtype3>)
template<class P_numtype1, class P_numtype2, int N_length2, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      VectorIterConst<P_numtype3> > >
where(const VectorPick<P_numtype1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2, 
      const Vector<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      VectorIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      d3.begin()));
}

// where(VectorPick<P_numtype1>, TinyVector<P_numtype2, N_length2>, _bz_VecExpr<P_expr3>)
template<class P_numtype1, class P_numtype2, int N_length2, class P_expr3>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_VecExpr<P_expr3> > >
where(const VectorPick<P_numtype1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2, 
      _bz_VecExpr<P_expr3> d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_VecExpr<P_expr3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      d3));
}

// where(VectorPick<P_numtype1>, TinyVector<P_numtype2, N_length2>, VectorPick<P_numtype3>)
template<class P_numtype1, class P_numtype2, int N_length2, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      VectorPickIterConst<P_numtype3> > >
where(const VectorPick<P_numtype1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2, 
      const VectorPick<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      VectorPickIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      d3.begin()));
}

// where(VectorPick<P_numtype1>, TinyVector<P_numtype2, N_length2>, Range)
template<class P_numtype1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      Range > >
where(const VectorPick<P_numtype1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2, 
      Range d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      Range > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      d3));
}

// where(VectorPick<P_numtype1>, TinyVector<P_numtype2, N_length2>, TinyVector<P_numtype3, N_length3>)
template<class P_numtype1, class P_numtype2, int N_length2, class P_numtype3, int N_length3>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      TinyVectorIterConst<P_numtype3, N_length3> > >
where(const VectorPick<P_numtype1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2, 
      const TinyVector<P_numtype3, N_length3>& d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      TinyVectorIterConst<P_numtype3, N_length3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      d3.begin()));
}

// where(VectorPick<P_numtype1>, TinyVector<P_numtype2, N_length2>, int)
template<class P_numtype1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_VecExprConstant<int> > >
where(const VectorPick<P_numtype1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2, 
      int d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_VecExprConstant<int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      _bz_VecExprConstant<int>(d3)));
}

// where(VectorPick<P_numtype1>, TinyVector<P_numtype2, N_length2>, float)
template<class P_numtype1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_VecExprConstant<float> > >
where(const VectorPick<P_numtype1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2, 
      float d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_VecExprConstant<float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      _bz_VecExprConstant<float>(d3)));
}

// where(VectorPick<P_numtype1>, TinyVector<P_numtype2, N_length2>, double)
template<class P_numtype1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_VecExprConstant<double> > >
where(const VectorPick<P_numtype1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2, 
      double d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_VecExprConstant<double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      _bz_VecExprConstant<double>(d3)));
}

// where(VectorPick<P_numtype1>, TinyVector<P_numtype2, N_length2>, long double)
template<class P_numtype1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_VecExprConstant<long double> > >
where(const VectorPick<P_numtype1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2, 
      long double d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_VecExprConstant<long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      _bz_VecExprConstant<long double>(d3)));
}

// where(VectorPick<P_numtype1>, TinyVector<P_numtype2, N_length2>, complex<T3>)
#ifdef BZ_HAVE_COMPLEX
template<class P_numtype1, class P_numtype2, int N_length2, class T3>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_VecExprConstant<complex<T3> >  > >
where(const VectorPick<P_numtype1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2, 
      complex<T3> d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_VecExprConstant<complex<T3> >  > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      _bz_VecExprConstant<complex<T3> > (d3)));
}
#endif // BZ_HAVE_COMPLEX

// where(VectorPick<P_numtype1>, int, Vector<P_numtype3>)
template<class P_numtype1, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>, 
      VectorIterConst<P_numtype3> > >
where(const VectorPick<P_numtype1>& d1, 
      int d2, 
      const Vector<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>, 
      VectorIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2), 
      d3.begin()));
}

// where(VectorPick<P_numtype1>, int, _bz_VecExpr<P_expr3>)
template<class P_numtype1, class P_expr3>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>, 
      _bz_VecExpr<P_expr3> > >
where(const VectorPick<P_numtype1>& d1, 
      int d2, 
      _bz_VecExpr<P_expr3> d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>, 
      _bz_VecExpr<P_expr3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2), 
      d3));
}

// where(VectorPick<P_numtype1>, int, VectorPick<P_numtype3>)
template<class P_numtype1, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>, 
      VectorPickIterConst<P_numtype3> > >
where(const VectorPick<P_numtype1>& d1, 
      int d2, 
      const VectorPick<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>, 
      VectorPickIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2), 
      d3.begin()));
}

// where(VectorPick<P_numtype1>, int, Range)
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>, 
      Range > >
where(const VectorPick<P_numtype1>& d1, 
      int d2, 
      Range d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>, 
      Range > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2), 
      d3));
}

// where(VectorPick<P_numtype1>, int, TinyVector<P_numtype3, N_length3>)
template<class P_numtype1, class P_numtype3, int N_length3>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>, 
      TinyVectorIterConst<P_numtype3, N_length3> > >
where(const VectorPick<P_numtype1>& d1, 
      int d2, 
      const TinyVector<P_numtype3, N_length3>& d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>, 
      TinyVectorIterConst<P_numtype3, N_length3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2), 
      d3.begin()));
}

// where(VectorPick<P_numtype1>, int, int)
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>, 
      _bz_VecExprConstant<int> > >
where(const VectorPick<P_numtype1>& d1, 
      int d2, 
      int d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<int>, 
      _bz_VecExprConstant<int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2), 
      _bz_VecExprConstant<int>(d3)));
}

// where(VectorPick<P_numtype1>, float, Vector<P_numtype3>)
template<class P_numtype1, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>, 
      VectorIterConst<P_numtype3> > >
where(const VectorPick<P_numtype1>& d1, 
      float d2, 
      const Vector<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>, 
      VectorIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<float>(d2), 
      d3.begin()));
}

// where(VectorPick<P_numtype1>, float, _bz_VecExpr<P_expr3>)
template<class P_numtype1, class P_expr3>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>, 
      _bz_VecExpr<P_expr3> > >
where(const VectorPick<P_numtype1>& d1, 
      float d2, 
      _bz_VecExpr<P_expr3> d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>, 
      _bz_VecExpr<P_expr3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<float>(d2), 
      d3));
}

// where(VectorPick<P_numtype1>, float, VectorPick<P_numtype3>)
template<class P_numtype1, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>, 
      VectorPickIterConst<P_numtype3> > >
where(const VectorPick<P_numtype1>& d1, 
      float d2, 
      const VectorPick<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>, 
      VectorPickIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<float>(d2), 
      d3.begin()));
}

// where(VectorPick<P_numtype1>, float, Range)
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>, 
      Range > >
where(const VectorPick<P_numtype1>& d1, 
      float d2, 
      Range d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>, 
      Range > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<float>(d2), 
      d3));
}

// where(VectorPick<P_numtype1>, float, TinyVector<P_numtype3, N_length3>)
template<class P_numtype1, class P_numtype3, int N_length3>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>, 
      TinyVectorIterConst<P_numtype3, N_length3> > >
where(const VectorPick<P_numtype1>& d1, 
      float d2, 
      const TinyVector<P_numtype3, N_length3>& d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>, 
      TinyVectorIterConst<P_numtype3, N_length3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<float>(d2), 
      d3.begin()));
}

// where(VectorPick<P_numtype1>, float, float)
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>, 
      _bz_VecExprConstant<float> > >
where(const VectorPick<P_numtype1>& d1, 
      float d2, 
      float d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<float>, 
      _bz_VecExprConstant<float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<float>(d2), 
      _bz_VecExprConstant<float>(d3)));
}

// where(VectorPick<P_numtype1>, double, Vector<P_numtype3>)
template<class P_numtype1, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>, 
      VectorIterConst<P_numtype3> > >
where(const VectorPick<P_numtype1>& d1, 
      double d2, 
      const Vector<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>, 
      VectorIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<double>(d2), 
      d3.begin()));
}

// where(VectorPick<P_numtype1>, double, _bz_VecExpr<P_expr3>)
template<class P_numtype1, class P_expr3>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>, 
      _bz_VecExpr<P_expr3> > >
where(const VectorPick<P_numtype1>& d1, 
      double d2, 
      _bz_VecExpr<P_expr3> d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>, 
      _bz_VecExpr<P_expr3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<double>(d2), 
      d3));
}

// where(VectorPick<P_numtype1>, double, VectorPick<P_numtype3>)
template<class P_numtype1, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>, 
      VectorPickIterConst<P_numtype3> > >
where(const VectorPick<P_numtype1>& d1, 
      double d2, 
      const VectorPick<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>, 
      VectorPickIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<double>(d2), 
      d3.begin()));
}

// where(VectorPick<P_numtype1>, double, Range)
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>, 
      Range > >
where(const VectorPick<P_numtype1>& d1, 
      double d2, 
      Range d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>, 
      Range > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<double>(d2), 
      d3));
}

// where(VectorPick<P_numtype1>, double, TinyVector<P_numtype3, N_length3>)
template<class P_numtype1, class P_numtype3, int N_length3>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>, 
      TinyVectorIterConst<P_numtype3, N_length3> > >
where(const VectorPick<P_numtype1>& d1, 
      double d2, 
      const TinyVector<P_numtype3, N_length3>& d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>, 
      TinyVectorIterConst<P_numtype3, N_length3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<double>(d2), 
      d3.begin()));
}

// where(VectorPick<P_numtype1>, double, double)
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>, 
      _bz_VecExprConstant<double> > >
where(const VectorPick<P_numtype1>& d1, 
      double d2, 
      double d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<double>, 
      _bz_VecExprConstant<double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<double>(d2), 
      _bz_VecExprConstant<double>(d3)));
}

// where(VectorPick<P_numtype1>, long double, Vector<P_numtype3>)
template<class P_numtype1, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>, 
      VectorIterConst<P_numtype3> > >
where(const VectorPick<P_numtype1>& d1, 
      long double d2, 
      const Vector<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>, 
      VectorIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<long double>(d2), 
      d3.begin()));
}

// where(VectorPick<P_numtype1>, long double, _bz_VecExpr<P_expr3>)
template<class P_numtype1, class P_expr3>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>, 
      _bz_VecExpr<P_expr3> > >
where(const VectorPick<P_numtype1>& d1, 
      long double d2, 
      _bz_VecExpr<P_expr3> d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>, 
      _bz_VecExpr<P_expr3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<long double>(d2), 
      d3));
}

// where(VectorPick<P_numtype1>, long double, VectorPick<P_numtype3>)
template<class P_numtype1, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>, 
      VectorPickIterConst<P_numtype3> > >
where(const VectorPick<P_numtype1>& d1, 
      long double d2, 
      const VectorPick<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>, 
      VectorPickIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<long double>(d2), 
      d3.begin()));
}

// where(VectorPick<P_numtype1>, long double, Range)
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>, 
      Range > >
where(const VectorPick<P_numtype1>& d1, 
      long double d2, 
      Range d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>, 
      Range > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<long double>(d2), 
      d3));
}

// where(VectorPick<P_numtype1>, long double, TinyVector<P_numtype3, N_length3>)
template<class P_numtype1, class P_numtype3, int N_length3>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>, 
      TinyVectorIterConst<P_numtype3, N_length3> > >
where(const VectorPick<P_numtype1>& d1, 
      long double d2, 
      const TinyVector<P_numtype3, N_length3>& d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>, 
      TinyVectorIterConst<P_numtype3, N_length3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<long double>(d2), 
      d3.begin()));
}

// where(VectorPick<P_numtype1>, long double, long double)
template<class P_numtype1>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>, 
      _bz_VecExprConstant<long double> > >
where(const VectorPick<P_numtype1>& d1, 
      long double d2, 
      long double d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<long double>, 
      _bz_VecExprConstant<long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<long double>(d2), 
      _bz_VecExprConstant<long double>(d3)));
}

// where(VectorPick<P_numtype1>, complex<T2>, Vector<P_numtype3>)
#ifdef BZ_HAVE_COMPLEX
template<class P_numtype1, class T2, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > , 
      VectorIterConst<P_numtype3> > >
where(const VectorPick<P_numtype1>& d1, 
      complex<T2> d2, 
      const Vector<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > , 
      VectorIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<complex<T2> > (d2), 
      d3.begin()));
}
#endif // BZ_HAVE_COMPLEX

// where(VectorPick<P_numtype1>, complex<T2>, _bz_VecExpr<P_expr3>)
#ifdef BZ_HAVE_COMPLEX
template<class P_numtype1, class T2, class P_expr3>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_VecExpr<P_expr3> > >
where(const VectorPick<P_numtype1>& d1, 
      complex<T2> d2, 
      _bz_VecExpr<P_expr3> d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_VecExpr<P_expr3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<complex<T2> > (d2), 
      d3));
}
#endif // BZ_HAVE_COMPLEX

// where(VectorPick<P_numtype1>, complex<T2>, VectorPick<P_numtype3>)
#ifdef BZ_HAVE_COMPLEX
template<class P_numtype1, class T2, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > , 
      VectorPickIterConst<P_numtype3> > >
where(const VectorPick<P_numtype1>& d1, 
      complex<T2> d2, 
      const VectorPick<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > , 
      VectorPickIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<complex<T2> > (d2), 
      d3.begin()));
}
#endif // BZ_HAVE_COMPLEX

// where(VectorPick<P_numtype1>, complex<T2>, Range)
#ifdef BZ_HAVE_COMPLEX
template<class P_numtype1, class T2>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > , 
      Range > >
where(const VectorPick<P_numtype1>& d1, 
      complex<T2> d2, 
      Range d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > , 
      Range > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<complex<T2> > (d2), 
      d3));
}
#endif // BZ_HAVE_COMPLEX

// where(VectorPick<P_numtype1>, complex<T2>, TinyVector<P_numtype3, N_length3>)
#ifdef BZ_HAVE_COMPLEX
template<class P_numtype1, class T2, class P_numtype3, int N_length3>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > , 
      TinyVectorIterConst<P_numtype3, N_length3> > >
where(const VectorPick<P_numtype1>& d1, 
      complex<T2> d2, 
      const TinyVector<P_numtype3, N_length3>& d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > , 
      TinyVectorIterConst<P_numtype3, N_length3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<complex<T2> > (d2), 
      d3.begin()));
}
#endif // BZ_HAVE_COMPLEX

// where(VectorPick<P_numtype1>, complex<T2>, complex<T3>)
#ifdef BZ_HAVE_COMPLEX
template<class P_numtype1, class T2, class T3>
inline
_bz_VecExpr<_bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_VecExprConstant<complex<T3> >  > >
where(const VectorPick<P_numtype1>& d1, 
      complex<T2> d2, 
      complex<T3> d3)
{ 
    typedef _bz_VecWhere<VectorPickIterConst<P_numtype1>, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_VecExprConstant<complex<T3> >  > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<complex<T2> > (d2), 
      _bz_VecExprConstant<complex<T3> > (d3)));
}
#endif // BZ_HAVE_COMPLEX

// where(Range, Vector<P_numtype2>, Vector<P_numtype3>)
template<class P_numtype2, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<Range, 
      VectorIterConst<P_numtype2>, 
      VectorIterConst<P_numtype3> > >
where(Range d1, 
      const Vector<P_numtype2>& d2, 
      const Vector<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<Range, 
      VectorIterConst<P_numtype2>, 
      VectorIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin(), 
      d3.begin()));
}

// where(Range, Vector<P_numtype2>, _bz_VecExpr<P_expr3>)
template<class P_numtype2, class P_expr3>
inline
_bz_VecExpr<_bz_VecWhere<Range, 
      VectorIterConst<P_numtype2>, 
      _bz_VecExpr<P_expr3> > >
where(Range d1, 
      const Vector<P_numtype2>& d2, 
      _bz_VecExpr<P_expr3> d3)
{ 
    typedef _bz_VecWhere<Range, 
      VectorIterConst<P_numtype2>, 
      _bz_VecExpr<P_expr3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin(), 
      d3));
}

// where(Range, Vector<P_numtype2>, VectorPick<P_numtype3>)
template<class P_numtype2, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<Range, 
      VectorIterConst<P_numtype2>, 
      VectorPickIterConst<P_numtype3> > >
where(Range d1, 
      const Vector<P_numtype2>& d2, 
      const VectorPick<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<Range, 
      VectorIterConst<P_numtype2>, 
      VectorPickIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin(), 
      d3.begin()));
}

// where(Range, Vector<P_numtype2>, Range)
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecWhere<Range, 
      VectorIterConst<P_numtype2>, 
      Range > >
where(Range d1, 
      const Vector<P_numtype2>& d2, 
      Range d3)
{ 
    typedef _bz_VecWhere<Range, 
      VectorIterConst<P_numtype2>, 
      Range > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin(), 
      d3));
}

// where(Range, Vector<P_numtype2>, TinyVector<P_numtype3, N_length3>)
template<class P_numtype2, class P_numtype3, int N_length3>
inline
_bz_VecExpr<_bz_VecWhere<Range, 
      VectorIterConst<P_numtype2>, 
      TinyVectorIterConst<P_numtype3, N_length3> > >
where(Range d1, 
      const Vector<P_numtype2>& d2, 
      const TinyVector<P_numtype3, N_length3>& d3)
{ 
    typedef _bz_VecWhere<Range, 
      VectorIterConst<P_numtype2>, 
      TinyVectorIterConst<P_numtype3, N_length3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin(), 
      d3.begin()));
}

// where(Range, Vector<P_numtype2>, int)
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecWhere<Range, 
      VectorIterConst<P_numtype2>, 
      _bz_VecExprConstant<int> > >
where(Range d1, 
      const Vector<P_numtype2>& d2, 
      int d3)
{ 
    typedef _bz_VecWhere<Range, 
      VectorIterConst<P_numtype2>, 
      _bz_VecExprConstant<int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin(), 
      _bz_VecExprConstant<int>(d3)));
}

// where(Range, Vector<P_numtype2>, float)
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecWhere<Range, 
      VectorIterConst<P_numtype2>, 
      _bz_VecExprConstant<float> > >
where(Range d1, 
      const Vector<P_numtype2>& d2, 
      float d3)
{ 
    typedef _bz_VecWhere<Range, 
      VectorIterConst<P_numtype2>, 
      _bz_VecExprConstant<float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin(), 
      _bz_VecExprConstant<float>(d3)));
}

// where(Range, Vector<P_numtype2>, double)
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecWhere<Range, 
      VectorIterConst<P_numtype2>, 
      _bz_VecExprConstant<double> > >
where(Range d1, 
      const Vector<P_numtype2>& d2, 
      double d3)
{ 
    typedef _bz_VecWhere<Range, 
      VectorIterConst<P_numtype2>, 
      _bz_VecExprConstant<double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin(), 
      _bz_VecExprConstant<double>(d3)));
}

// where(Range, Vector<P_numtype2>, long double)
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecWhere<Range, 
      VectorIterConst<P_numtype2>, 
      _bz_VecExprConstant<long double> > >
where(Range d1, 
      const Vector<P_numtype2>& d2, 
      long double d3)
{ 
    typedef _bz_VecWhere<Range, 
      VectorIterConst<P_numtype2>, 
      _bz_VecExprConstant<long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin(), 
      _bz_VecExprConstant<long double>(d3)));
}

// where(Range, Vector<P_numtype2>, complex<T3>)
#ifdef BZ_HAVE_COMPLEX
template<class P_numtype2, class T3>
inline
_bz_VecExpr<_bz_VecWhere<Range, 
      VectorIterConst<P_numtype2>, 
      _bz_VecExprConstant<complex<T3> >  > >
where(Range d1, 
      const Vector<P_numtype2>& d2, 
      complex<T3> d3)
{ 
    typedef _bz_VecWhere<Range, 
      VectorIterConst<P_numtype2>, 
      _bz_VecExprConstant<complex<T3> >  > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin(), 
      _bz_VecExprConstant<complex<T3> > (d3)));
}
#endif // BZ_HAVE_COMPLEX

// where(Range, _bz_VecExpr<P_expr2>, Vector<P_numtype3>)
template<class P_expr2, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<Range, 
      _bz_VecExpr<P_expr2>, 
      VectorIterConst<P_numtype3> > >
where(Range d1, 
      _bz_VecExpr<P_expr2> d2, 
      const Vector<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<Range, 
      _bz_VecExpr<P_expr2>, 
      VectorIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2, 
      d3.begin()));
}

// where(Range, _bz_VecExpr<P_expr2>, _bz_VecExpr<P_expr3>)
template<class P_expr2, class P_expr3>
inline
_bz_VecExpr<_bz_VecWhere<Range, 
      _bz_VecExpr<P_expr2>, 
      _bz_VecExpr<P_expr3> > >
where(Range d1, 
      _bz_VecExpr<P_expr2> d2, 
      _bz_VecExpr<P_expr3> d3)
{ 
    typedef _bz_VecWhere<Range, 
      _bz_VecExpr<P_expr2>, 
      _bz_VecExpr<P_expr3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2, 
      d3));
}

// where(Range, _bz_VecExpr<P_expr2>, VectorPick<P_numtype3>)
template<class P_expr2, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<Range, 
      _bz_VecExpr<P_expr2>, 
      VectorPickIterConst<P_numtype3> > >
where(Range d1, 
      _bz_VecExpr<P_expr2> d2, 
      const VectorPick<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<Range, 
      _bz_VecExpr<P_expr2>, 
      VectorPickIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2, 
      d3.begin()));
}

// where(Range, _bz_VecExpr<P_expr2>, Range)
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecWhere<Range, 
      _bz_VecExpr<P_expr2>, 
      Range > >
where(Range d1, 
      _bz_VecExpr<P_expr2> d2, 
      Range d3)
{ 
    typedef _bz_VecWhere<Range, 
      _bz_VecExpr<P_expr2>, 
      Range > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2, 
      d3));
}

// where(Range, _bz_VecExpr<P_expr2>, TinyVector<P_numtype3, N_length3>)
template<class P_expr2, class P_numtype3, int N_length3>
inline
_bz_VecExpr<_bz_VecWhere<Range, 
      _bz_VecExpr<P_expr2>, 
      TinyVectorIterConst<P_numtype3, N_length3> > >
where(Range d1, 
      _bz_VecExpr<P_expr2> d2, 
      const TinyVector<P_numtype3, N_length3>& d3)
{ 
    typedef _bz_VecWhere<Range, 
      _bz_VecExpr<P_expr2>, 
      TinyVectorIterConst<P_numtype3, N_length3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2, 
      d3.begin()));
}

// where(Range, _bz_VecExpr<P_expr2>, int)
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecWhere<Range, 
      _bz_VecExpr<P_expr2>, 
      _bz_VecExprConstant<int> > >
where(Range d1, 
      _bz_VecExpr<P_expr2> d2, 
      int d3)
{ 
    typedef _bz_VecWhere<Range, 
      _bz_VecExpr<P_expr2>, 
      _bz_VecExprConstant<int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2, 
      _bz_VecExprConstant<int>(d3)));
}

// where(Range, _bz_VecExpr<P_expr2>, float)
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecWhere<Range, 
      _bz_VecExpr<P_expr2>, 
      _bz_VecExprConstant<float> > >
where(Range d1, 
      _bz_VecExpr<P_expr2> d2, 
      float d3)
{ 
    typedef _bz_VecWhere<Range, 
      _bz_VecExpr<P_expr2>, 
      _bz_VecExprConstant<float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2, 
      _bz_VecExprConstant<float>(d3)));
}

// where(Range, _bz_VecExpr<P_expr2>, double)
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecWhere<Range, 
      _bz_VecExpr<P_expr2>, 
      _bz_VecExprConstant<double> > >
where(Range d1, 
      _bz_VecExpr<P_expr2> d2, 
      double d3)
{ 
    typedef _bz_VecWhere<Range, 
      _bz_VecExpr<P_expr2>, 
      _bz_VecExprConstant<double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2, 
      _bz_VecExprConstant<double>(d3)));
}

// where(Range, _bz_VecExpr<P_expr2>, long double)
template<class P_expr2>
inline
_bz_VecExpr<_bz_VecWhere<Range, 
      _bz_VecExpr<P_expr2>, 
      _bz_VecExprConstant<long double> > >
where(Range d1, 
      _bz_VecExpr<P_expr2> d2, 
      long double d3)
{ 
    typedef _bz_VecWhere<Range, 
      _bz_VecExpr<P_expr2>, 
      _bz_VecExprConstant<long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2, 
      _bz_VecExprConstant<long double>(d3)));
}

// where(Range, _bz_VecExpr<P_expr2>, complex<T3>)
#ifdef BZ_HAVE_COMPLEX
template<class P_expr2, class T3>
inline
_bz_VecExpr<_bz_VecWhere<Range, 
      _bz_VecExpr<P_expr2>, 
      _bz_VecExprConstant<complex<T3> >  > >
where(Range d1, 
      _bz_VecExpr<P_expr2> d2, 
      complex<T3> d3)
{ 
    typedef _bz_VecWhere<Range, 
      _bz_VecExpr<P_expr2>, 
      _bz_VecExprConstant<complex<T3> >  > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2, 
      _bz_VecExprConstant<complex<T3> > (d3)));
}
#endif // BZ_HAVE_COMPLEX

// where(Range, VectorPick<P_numtype2>, Vector<P_numtype3>)
template<class P_numtype2, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<Range, 
      VectorPickIterConst<P_numtype2>, 
      VectorIterConst<P_numtype3> > >
where(Range d1, 
      const VectorPick<P_numtype2>& d2, 
      const Vector<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<Range, 
      VectorPickIterConst<P_numtype2>, 
      VectorIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin(), 
      d3.begin()));
}

// where(Range, VectorPick<P_numtype2>, _bz_VecExpr<P_expr3>)
template<class P_numtype2, class P_expr3>
inline
_bz_VecExpr<_bz_VecWhere<Range, 
      VectorPickIterConst<P_numtype2>, 
      _bz_VecExpr<P_expr3> > >
where(Range d1, 
      const VectorPick<P_numtype2>& d2, 
      _bz_VecExpr<P_expr3> d3)
{ 
    typedef _bz_VecWhere<Range, 
      VectorPickIterConst<P_numtype2>, 
      _bz_VecExpr<P_expr3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin(), 
      d3));
}

// where(Range, VectorPick<P_numtype2>, VectorPick<P_numtype3>)
template<class P_numtype2, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<Range, 
      VectorPickIterConst<P_numtype2>, 
      VectorPickIterConst<P_numtype3> > >
where(Range d1, 
      const VectorPick<P_numtype2>& d2, 
      const VectorPick<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<Range, 
      VectorPickIterConst<P_numtype2>, 
      VectorPickIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin(), 
      d3.begin()));
}

// where(Range, VectorPick<P_numtype2>, Range)
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecWhere<Range, 
      VectorPickIterConst<P_numtype2>, 
      Range > >
where(Range d1, 
      const VectorPick<P_numtype2>& d2, 
      Range d3)
{ 
    typedef _bz_VecWhere<Range, 
      VectorPickIterConst<P_numtype2>, 
      Range > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin(), 
      d3));
}

// where(Range, VectorPick<P_numtype2>, TinyVector<P_numtype3, N_length3>)
template<class P_numtype2, class P_numtype3, int N_length3>
inline
_bz_VecExpr<_bz_VecWhere<Range, 
      VectorPickIterConst<P_numtype2>, 
      TinyVectorIterConst<P_numtype3, N_length3> > >
where(Range d1, 
      const VectorPick<P_numtype2>& d2, 
      const TinyVector<P_numtype3, N_length3>& d3)
{ 
    typedef _bz_VecWhere<Range, 
      VectorPickIterConst<P_numtype2>, 
      TinyVectorIterConst<P_numtype3, N_length3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin(), 
      d3.begin()));
}

// where(Range, VectorPick<P_numtype2>, int)
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecWhere<Range, 
      VectorPickIterConst<P_numtype2>, 
      _bz_VecExprConstant<int> > >
where(Range d1, 
      const VectorPick<P_numtype2>& d2, 
      int d3)
{ 
    typedef _bz_VecWhere<Range, 
      VectorPickIterConst<P_numtype2>, 
      _bz_VecExprConstant<int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin(), 
      _bz_VecExprConstant<int>(d3)));
}

// where(Range, VectorPick<P_numtype2>, float)
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecWhere<Range, 
      VectorPickIterConst<P_numtype2>, 
      _bz_VecExprConstant<float> > >
where(Range d1, 
      const VectorPick<P_numtype2>& d2, 
      float d3)
{ 
    typedef _bz_VecWhere<Range, 
      VectorPickIterConst<P_numtype2>, 
      _bz_VecExprConstant<float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin(), 
      _bz_VecExprConstant<float>(d3)));
}

// where(Range, VectorPick<P_numtype2>, double)
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecWhere<Range, 
      VectorPickIterConst<P_numtype2>, 
      _bz_VecExprConstant<double> > >
where(Range d1, 
      const VectorPick<P_numtype2>& d2, 
      double d3)
{ 
    typedef _bz_VecWhere<Range, 
      VectorPickIterConst<P_numtype2>, 
      _bz_VecExprConstant<double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin(), 
      _bz_VecExprConstant<double>(d3)));
}

// where(Range, VectorPick<P_numtype2>, long double)
template<class P_numtype2>
inline
_bz_VecExpr<_bz_VecWhere<Range, 
      VectorPickIterConst<P_numtype2>, 
      _bz_VecExprConstant<long double> > >
where(Range d1, 
      const VectorPick<P_numtype2>& d2, 
      long double d3)
{ 
    typedef _bz_VecWhere<Range, 
      VectorPickIterConst<P_numtype2>, 
      _bz_VecExprConstant<long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin(), 
      _bz_VecExprConstant<long double>(d3)));
}

// where(Range, VectorPick<P_numtype2>, complex<T3>)
#ifdef BZ_HAVE_COMPLEX
template<class P_numtype2, class T3>
inline
_bz_VecExpr<_bz_VecWhere<Range, 
      VectorPickIterConst<P_numtype2>, 
      _bz_VecExprConstant<complex<T3> >  > >
where(Range d1, 
      const VectorPick<P_numtype2>& d2, 
      complex<T3> d3)
{ 
    typedef _bz_VecWhere<Range, 
      VectorPickIterConst<P_numtype2>, 
      _bz_VecExprConstant<complex<T3> >  > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin(), 
      _bz_VecExprConstant<complex<T3> > (d3)));
}
#endif // BZ_HAVE_COMPLEX

// where(Range, Range, Vector<P_numtype3>)
template<class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<Range, 
      Range, 
      VectorIterConst<P_numtype3> > >
where(Range d1, 
      Range d2, 
      const Vector<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<Range, 
      Range, 
      VectorIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2, 
      d3.begin()));
}

// where(Range, Range, _bz_VecExpr<P_expr3>)
template<class P_expr3>
inline
_bz_VecExpr<_bz_VecWhere<Range, 
      Range, 
      _bz_VecExpr<P_expr3> > >
where(Range d1, 
      Range d2, 
      _bz_VecExpr<P_expr3> d3)
{ 
    typedef _bz_VecWhere<Range, 
      Range, 
      _bz_VecExpr<P_expr3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2, 
      d3));
}

// where(Range, Range, VectorPick<P_numtype3>)
template<class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<Range, 
      Range, 
      VectorPickIterConst<P_numtype3> > >
where(Range d1, 
      Range d2, 
      const VectorPick<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<Range, 
      Range, 
      VectorPickIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2, 
      d3.begin()));
}

// where(Range, Range, Range)

inline
_bz_VecExpr<_bz_VecWhere<Range, 
      Range, 
      Range > >
where(Range d1, 
      Range d2, 
      Range d3)
{ 
    typedef _bz_VecWhere<Range, 
      Range, 
      Range > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2, 
      d3));
}

// where(Range, Range, TinyVector<P_numtype3, N_length3>)
template<class P_numtype3, int N_length3>
inline
_bz_VecExpr<_bz_VecWhere<Range, 
      Range, 
      TinyVectorIterConst<P_numtype3, N_length3> > >
where(Range d1, 
      Range d2, 
      const TinyVector<P_numtype3, N_length3>& d3)
{ 
    typedef _bz_VecWhere<Range, 
      Range, 
      TinyVectorIterConst<P_numtype3, N_length3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2, 
      d3.begin()));
}

// where(Range, Range, int)

inline
_bz_VecExpr<_bz_VecWhere<Range, 
      Range, 
      _bz_VecExprConstant<int> > >
where(Range d1, 
      Range d2, 
      int d3)
{ 
    typedef _bz_VecWhere<Range, 
      Range, 
      _bz_VecExprConstant<int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2, 
      _bz_VecExprConstant<int>(d3)));
}

// where(Range, Range, float)

inline
_bz_VecExpr<_bz_VecWhere<Range, 
      Range, 
      _bz_VecExprConstant<float> > >
where(Range d1, 
      Range d2, 
      float d3)
{ 
    typedef _bz_VecWhere<Range, 
      Range, 
      _bz_VecExprConstant<float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2, 
      _bz_VecExprConstant<float>(d3)));
}

// where(Range, Range, double)

inline
_bz_VecExpr<_bz_VecWhere<Range, 
      Range, 
      _bz_VecExprConstant<double> > >
where(Range d1, 
      Range d2, 
      double d3)
{ 
    typedef _bz_VecWhere<Range, 
      Range, 
      _bz_VecExprConstant<double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2, 
      _bz_VecExprConstant<double>(d3)));
}

// where(Range, Range, long double)

inline
_bz_VecExpr<_bz_VecWhere<Range, 
      Range, 
      _bz_VecExprConstant<long double> > >
where(Range d1, 
      Range d2, 
      long double d3)
{ 
    typedef _bz_VecWhere<Range, 
      Range, 
      _bz_VecExprConstant<long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2, 
      _bz_VecExprConstant<long double>(d3)));
}

// where(Range, Range, complex<T3>)
#ifdef BZ_HAVE_COMPLEX
template<class T3>
inline
_bz_VecExpr<_bz_VecWhere<Range, 
      Range, 
      _bz_VecExprConstant<complex<T3> >  > >
where(Range d1, 
      Range d2, 
      complex<T3> d3)
{ 
    typedef _bz_VecWhere<Range, 
      Range, 
      _bz_VecExprConstant<complex<T3> >  > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2, 
      _bz_VecExprConstant<complex<T3> > (d3)));
}
#endif // BZ_HAVE_COMPLEX

// where(Range, TinyVector<P_numtype2, N_length2>, Vector<P_numtype3>)
template<class P_numtype2, int N_length2, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<Range, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      VectorIterConst<P_numtype3> > >
where(Range d1, 
      const TinyVector<P_numtype2, N_length2>& d2, 
      const Vector<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<Range, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      VectorIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin(), 
      d3.begin()));
}

// where(Range, TinyVector<P_numtype2, N_length2>, _bz_VecExpr<P_expr3>)
template<class P_numtype2, int N_length2, class P_expr3>
inline
_bz_VecExpr<_bz_VecWhere<Range, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_VecExpr<P_expr3> > >
where(Range d1, 
      const TinyVector<P_numtype2, N_length2>& d2, 
      _bz_VecExpr<P_expr3> d3)
{ 
    typedef _bz_VecWhere<Range, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_VecExpr<P_expr3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin(), 
      d3));
}

// where(Range, TinyVector<P_numtype2, N_length2>, VectorPick<P_numtype3>)
template<class P_numtype2, int N_length2, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<Range, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      VectorPickIterConst<P_numtype3> > >
where(Range d1, 
      const TinyVector<P_numtype2, N_length2>& d2, 
      const VectorPick<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<Range, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      VectorPickIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin(), 
      d3.begin()));
}

// where(Range, TinyVector<P_numtype2, N_length2>, Range)
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecWhere<Range, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      Range > >
where(Range d1, 
      const TinyVector<P_numtype2, N_length2>& d2, 
      Range d3)
{ 
    typedef _bz_VecWhere<Range, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      Range > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin(), 
      d3));
}

// where(Range, TinyVector<P_numtype2, N_length2>, TinyVector<P_numtype3, N_length3>)
template<class P_numtype2, int N_length2, class P_numtype3, int N_length3>
inline
_bz_VecExpr<_bz_VecWhere<Range, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      TinyVectorIterConst<P_numtype3, N_length3> > >
where(Range d1, 
      const TinyVector<P_numtype2, N_length2>& d2, 
      const TinyVector<P_numtype3, N_length3>& d3)
{ 
    typedef _bz_VecWhere<Range, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      TinyVectorIterConst<P_numtype3, N_length3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin(), 
      d3.begin()));
}

// where(Range, TinyVector<P_numtype2, N_length2>, int)
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecWhere<Range, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_VecExprConstant<int> > >
where(Range d1, 
      const TinyVector<P_numtype2, N_length2>& d2, 
      int d3)
{ 
    typedef _bz_VecWhere<Range, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_VecExprConstant<int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin(), 
      _bz_VecExprConstant<int>(d3)));
}

// where(Range, TinyVector<P_numtype2, N_length2>, float)
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecWhere<Range, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_VecExprConstant<float> > >
where(Range d1, 
      const TinyVector<P_numtype2, N_length2>& d2, 
      float d3)
{ 
    typedef _bz_VecWhere<Range, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_VecExprConstant<float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin(), 
      _bz_VecExprConstant<float>(d3)));
}

// where(Range, TinyVector<P_numtype2, N_length2>, double)
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecWhere<Range, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_VecExprConstant<double> > >
where(Range d1, 
      const TinyVector<P_numtype2, N_length2>& d2, 
      double d3)
{ 
    typedef _bz_VecWhere<Range, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_VecExprConstant<double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin(), 
      _bz_VecExprConstant<double>(d3)));
}

// where(Range, TinyVector<P_numtype2, N_length2>, long double)
template<class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecWhere<Range, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_VecExprConstant<long double> > >
where(Range d1, 
      const TinyVector<P_numtype2, N_length2>& d2, 
      long double d3)
{ 
    typedef _bz_VecWhere<Range, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_VecExprConstant<long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin(), 
      _bz_VecExprConstant<long double>(d3)));
}

// where(Range, TinyVector<P_numtype2, N_length2>, complex<T3>)
#ifdef BZ_HAVE_COMPLEX
template<class P_numtype2, int N_length2, class T3>
inline
_bz_VecExpr<_bz_VecWhere<Range, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_VecExprConstant<complex<T3> >  > >
where(Range d1, 
      const TinyVector<P_numtype2, N_length2>& d2, 
      complex<T3> d3)
{ 
    typedef _bz_VecWhere<Range, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_VecExprConstant<complex<T3> >  > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      d2.begin(), 
      _bz_VecExprConstant<complex<T3> > (d3)));
}
#endif // BZ_HAVE_COMPLEX

// where(Range, int, Vector<P_numtype3>)
template<class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<Range, 
      _bz_VecExprConstant<int>, 
      VectorIterConst<P_numtype3> > >
where(Range d1, 
      int d2, 
      const Vector<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<Range, 
      _bz_VecExprConstant<int>, 
      VectorIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<int>(d2), 
      d3.begin()));
}

// where(Range, int, _bz_VecExpr<P_expr3>)
template<class P_expr3>
inline
_bz_VecExpr<_bz_VecWhere<Range, 
      _bz_VecExprConstant<int>, 
      _bz_VecExpr<P_expr3> > >
where(Range d1, 
      int d2, 
      _bz_VecExpr<P_expr3> d3)
{ 
    typedef _bz_VecWhere<Range, 
      _bz_VecExprConstant<int>, 
      _bz_VecExpr<P_expr3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<int>(d2), 
      d3));
}

// where(Range, int, VectorPick<P_numtype3>)
template<class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<Range, 
      _bz_VecExprConstant<int>, 
      VectorPickIterConst<P_numtype3> > >
where(Range d1, 
      int d2, 
      const VectorPick<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<Range, 
      _bz_VecExprConstant<int>, 
      VectorPickIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<int>(d2), 
      d3.begin()));
}

// where(Range, int, Range)

inline
_bz_VecExpr<_bz_VecWhere<Range, 
      _bz_VecExprConstant<int>, 
      Range > >
where(Range d1, 
      int d2, 
      Range d3)
{ 
    typedef _bz_VecWhere<Range, 
      _bz_VecExprConstant<int>, 
      Range > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<int>(d2), 
      d3));
}

// where(Range, int, TinyVector<P_numtype3, N_length3>)
template<class P_numtype3, int N_length3>
inline
_bz_VecExpr<_bz_VecWhere<Range, 
      _bz_VecExprConstant<int>, 
      TinyVectorIterConst<P_numtype3, N_length3> > >
where(Range d1, 
      int d2, 
      const TinyVector<P_numtype3, N_length3>& d3)
{ 
    typedef _bz_VecWhere<Range, 
      _bz_VecExprConstant<int>, 
      TinyVectorIterConst<P_numtype3, N_length3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<int>(d2), 
      d3.begin()));
}

// where(Range, int, int)

inline
_bz_VecExpr<_bz_VecWhere<Range, 
      _bz_VecExprConstant<int>, 
      _bz_VecExprConstant<int> > >
where(Range d1, 
      int d2, 
      int d3)
{ 
    typedef _bz_VecWhere<Range, 
      _bz_VecExprConstant<int>, 
      _bz_VecExprConstant<int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<int>(d2), 
      _bz_VecExprConstant<int>(d3)));
}

// where(Range, float, Vector<P_numtype3>)
template<class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<Range, 
      _bz_VecExprConstant<float>, 
      VectorIterConst<P_numtype3> > >
where(Range d1, 
      float d2, 
      const Vector<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<Range, 
      _bz_VecExprConstant<float>, 
      VectorIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<float>(d2), 
      d3.begin()));
}

// where(Range, float, _bz_VecExpr<P_expr3>)
template<class P_expr3>
inline
_bz_VecExpr<_bz_VecWhere<Range, 
      _bz_VecExprConstant<float>, 
      _bz_VecExpr<P_expr3> > >
where(Range d1, 
      float d2, 
      _bz_VecExpr<P_expr3> d3)
{ 
    typedef _bz_VecWhere<Range, 
      _bz_VecExprConstant<float>, 
      _bz_VecExpr<P_expr3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<float>(d2), 
      d3));
}

// where(Range, float, VectorPick<P_numtype3>)
template<class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<Range, 
      _bz_VecExprConstant<float>, 
      VectorPickIterConst<P_numtype3> > >
where(Range d1, 
      float d2, 
      const VectorPick<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<Range, 
      _bz_VecExprConstant<float>, 
      VectorPickIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<float>(d2), 
      d3.begin()));
}

// where(Range, float, Range)

inline
_bz_VecExpr<_bz_VecWhere<Range, 
      _bz_VecExprConstant<float>, 
      Range > >
where(Range d1, 
      float d2, 
      Range d3)
{ 
    typedef _bz_VecWhere<Range, 
      _bz_VecExprConstant<float>, 
      Range > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<float>(d2), 
      d3));
}

// where(Range, float, TinyVector<P_numtype3, N_length3>)
template<class P_numtype3, int N_length3>
inline
_bz_VecExpr<_bz_VecWhere<Range, 
      _bz_VecExprConstant<float>, 
      TinyVectorIterConst<P_numtype3, N_length3> > >
where(Range d1, 
      float d2, 
      const TinyVector<P_numtype3, N_length3>& d3)
{ 
    typedef _bz_VecWhere<Range, 
      _bz_VecExprConstant<float>, 
      TinyVectorIterConst<P_numtype3, N_length3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<float>(d2), 
      d3.begin()));
}

// where(Range, float, float)

inline
_bz_VecExpr<_bz_VecWhere<Range, 
      _bz_VecExprConstant<float>, 
      _bz_VecExprConstant<float> > >
where(Range d1, 
      float d2, 
      float d3)
{ 
    typedef _bz_VecWhere<Range, 
      _bz_VecExprConstant<float>, 
      _bz_VecExprConstant<float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<float>(d2), 
      _bz_VecExprConstant<float>(d3)));
}

// where(Range, double, Vector<P_numtype3>)
template<class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<Range, 
      _bz_VecExprConstant<double>, 
      VectorIterConst<P_numtype3> > >
where(Range d1, 
      double d2, 
      const Vector<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<Range, 
      _bz_VecExprConstant<double>, 
      VectorIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<double>(d2), 
      d3.begin()));
}

// where(Range, double, _bz_VecExpr<P_expr3>)
template<class P_expr3>
inline
_bz_VecExpr<_bz_VecWhere<Range, 
      _bz_VecExprConstant<double>, 
      _bz_VecExpr<P_expr3> > >
where(Range d1, 
      double d2, 
      _bz_VecExpr<P_expr3> d3)
{ 
    typedef _bz_VecWhere<Range, 
      _bz_VecExprConstant<double>, 
      _bz_VecExpr<P_expr3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<double>(d2), 
      d3));
}

// where(Range, double, VectorPick<P_numtype3>)
template<class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<Range, 
      _bz_VecExprConstant<double>, 
      VectorPickIterConst<P_numtype3> > >
where(Range d1, 
      double d2, 
      const VectorPick<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<Range, 
      _bz_VecExprConstant<double>, 
      VectorPickIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<double>(d2), 
      d3.begin()));
}

// where(Range, double, Range)

inline
_bz_VecExpr<_bz_VecWhere<Range, 
      _bz_VecExprConstant<double>, 
      Range > >
where(Range d1, 
      double d2, 
      Range d3)
{ 
    typedef _bz_VecWhere<Range, 
      _bz_VecExprConstant<double>, 
      Range > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<double>(d2), 
      d3));
}

// where(Range, double, TinyVector<P_numtype3, N_length3>)
template<class P_numtype3, int N_length3>
inline
_bz_VecExpr<_bz_VecWhere<Range, 
      _bz_VecExprConstant<double>, 
      TinyVectorIterConst<P_numtype3, N_length3> > >
where(Range d1, 
      double d2, 
      const TinyVector<P_numtype3, N_length3>& d3)
{ 
    typedef _bz_VecWhere<Range, 
      _bz_VecExprConstant<double>, 
      TinyVectorIterConst<P_numtype3, N_length3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<double>(d2), 
      d3.begin()));
}

// where(Range, double, double)

inline
_bz_VecExpr<_bz_VecWhere<Range, 
      _bz_VecExprConstant<double>, 
      _bz_VecExprConstant<double> > >
where(Range d1, 
      double d2, 
      double d3)
{ 
    typedef _bz_VecWhere<Range, 
      _bz_VecExprConstant<double>, 
      _bz_VecExprConstant<double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<double>(d2), 
      _bz_VecExprConstant<double>(d3)));
}

// where(Range, long double, Vector<P_numtype3>)
template<class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<Range, 
      _bz_VecExprConstant<long double>, 
      VectorIterConst<P_numtype3> > >
where(Range d1, 
      long double d2, 
      const Vector<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<Range, 
      _bz_VecExprConstant<long double>, 
      VectorIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<long double>(d2), 
      d3.begin()));
}

// where(Range, long double, _bz_VecExpr<P_expr3>)
template<class P_expr3>
inline
_bz_VecExpr<_bz_VecWhere<Range, 
      _bz_VecExprConstant<long double>, 
      _bz_VecExpr<P_expr3> > >
where(Range d1, 
      long double d2, 
      _bz_VecExpr<P_expr3> d3)
{ 
    typedef _bz_VecWhere<Range, 
      _bz_VecExprConstant<long double>, 
      _bz_VecExpr<P_expr3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<long double>(d2), 
      d3));
}

// where(Range, long double, VectorPick<P_numtype3>)
template<class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<Range, 
      _bz_VecExprConstant<long double>, 
      VectorPickIterConst<P_numtype3> > >
where(Range d1, 
      long double d2, 
      const VectorPick<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<Range, 
      _bz_VecExprConstant<long double>, 
      VectorPickIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<long double>(d2), 
      d3.begin()));
}

// where(Range, long double, Range)

inline
_bz_VecExpr<_bz_VecWhere<Range, 
      _bz_VecExprConstant<long double>, 
      Range > >
where(Range d1, 
      long double d2, 
      Range d3)
{ 
    typedef _bz_VecWhere<Range, 
      _bz_VecExprConstant<long double>, 
      Range > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<long double>(d2), 
      d3));
}

// where(Range, long double, TinyVector<P_numtype3, N_length3>)
template<class P_numtype3, int N_length3>
inline
_bz_VecExpr<_bz_VecWhere<Range, 
      _bz_VecExprConstant<long double>, 
      TinyVectorIterConst<P_numtype3, N_length3> > >
where(Range d1, 
      long double d2, 
      const TinyVector<P_numtype3, N_length3>& d3)
{ 
    typedef _bz_VecWhere<Range, 
      _bz_VecExprConstant<long double>, 
      TinyVectorIterConst<P_numtype3, N_length3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<long double>(d2), 
      d3.begin()));
}

// where(Range, long double, long double)

inline
_bz_VecExpr<_bz_VecWhere<Range, 
      _bz_VecExprConstant<long double>, 
      _bz_VecExprConstant<long double> > >
where(Range d1, 
      long double d2, 
      long double d3)
{ 
    typedef _bz_VecWhere<Range, 
      _bz_VecExprConstant<long double>, 
      _bz_VecExprConstant<long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<long double>(d2), 
      _bz_VecExprConstant<long double>(d3)));
}

// where(Range, complex<T2>, Vector<P_numtype3>)
#ifdef BZ_HAVE_COMPLEX
template<class T2, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<Range, 
      _bz_VecExprConstant<complex<T2> > , 
      VectorIterConst<P_numtype3> > >
where(Range d1, 
      complex<T2> d2, 
      const Vector<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<Range, 
      _bz_VecExprConstant<complex<T2> > , 
      VectorIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<complex<T2> > (d2), 
      d3.begin()));
}
#endif // BZ_HAVE_COMPLEX

// where(Range, complex<T2>, _bz_VecExpr<P_expr3>)
#ifdef BZ_HAVE_COMPLEX
template<class T2, class P_expr3>
inline
_bz_VecExpr<_bz_VecWhere<Range, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_VecExpr<P_expr3> > >
where(Range d1, 
      complex<T2> d2, 
      _bz_VecExpr<P_expr3> d3)
{ 
    typedef _bz_VecWhere<Range, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_VecExpr<P_expr3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<complex<T2> > (d2), 
      d3));
}
#endif // BZ_HAVE_COMPLEX

// where(Range, complex<T2>, VectorPick<P_numtype3>)
#ifdef BZ_HAVE_COMPLEX
template<class T2, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<Range, 
      _bz_VecExprConstant<complex<T2> > , 
      VectorPickIterConst<P_numtype3> > >
where(Range d1, 
      complex<T2> d2, 
      const VectorPick<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<Range, 
      _bz_VecExprConstant<complex<T2> > , 
      VectorPickIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<complex<T2> > (d2), 
      d3.begin()));
}
#endif // BZ_HAVE_COMPLEX

// where(Range, complex<T2>, Range)
#ifdef BZ_HAVE_COMPLEX
template<class T2>
inline
_bz_VecExpr<_bz_VecWhere<Range, 
      _bz_VecExprConstant<complex<T2> > , 
      Range > >
where(Range d1, 
      complex<T2> d2, 
      Range d3)
{ 
    typedef _bz_VecWhere<Range, 
      _bz_VecExprConstant<complex<T2> > , 
      Range > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<complex<T2> > (d2), 
      d3));
}
#endif // BZ_HAVE_COMPLEX

// where(Range, complex<T2>, TinyVector<P_numtype3, N_length3>)
#ifdef BZ_HAVE_COMPLEX
template<class T2, class P_numtype3, int N_length3>
inline
_bz_VecExpr<_bz_VecWhere<Range, 
      _bz_VecExprConstant<complex<T2> > , 
      TinyVectorIterConst<P_numtype3, N_length3> > >
where(Range d1, 
      complex<T2> d2, 
      const TinyVector<P_numtype3, N_length3>& d3)
{ 
    typedef _bz_VecWhere<Range, 
      _bz_VecExprConstant<complex<T2> > , 
      TinyVectorIterConst<P_numtype3, N_length3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<complex<T2> > (d2), 
      d3.begin()));
}
#endif // BZ_HAVE_COMPLEX

// where(Range, complex<T2>, complex<T3>)
#ifdef BZ_HAVE_COMPLEX
template<class T2, class T3>
inline
_bz_VecExpr<_bz_VecWhere<Range, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_VecExprConstant<complex<T3> >  > >
where(Range d1, 
      complex<T2> d2, 
      complex<T3> d3)
{ 
    typedef _bz_VecWhere<Range, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_VecExprConstant<complex<T3> >  > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1, 
      _bz_VecExprConstant<complex<T2> > (d2), 
      _bz_VecExprConstant<complex<T3> > (d3)));
}
#endif // BZ_HAVE_COMPLEX

// where(TinyVector<P_numtype1, N_length1>, Vector<P_numtype2>, Vector<P_numtype3>)
template<class P_numtype1, int N_length1, class P_numtype2, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorIterConst<P_numtype2>, 
      VectorIterConst<P_numtype3> > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      const Vector<P_numtype2>& d2, 
      const Vector<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorIterConst<P_numtype2>, 
      VectorIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      d3.begin()));
}

// where(TinyVector<P_numtype1, N_length1>, Vector<P_numtype2>, _bz_VecExpr<P_expr3>)
template<class P_numtype1, int N_length1, class P_numtype2, class P_expr3>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorIterConst<P_numtype2>, 
      _bz_VecExpr<P_expr3> > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      const Vector<P_numtype2>& d2, 
      _bz_VecExpr<P_expr3> d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorIterConst<P_numtype2>, 
      _bz_VecExpr<P_expr3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      d3));
}

// where(TinyVector<P_numtype1, N_length1>, Vector<P_numtype2>, VectorPick<P_numtype3>)
template<class P_numtype1, int N_length1, class P_numtype2, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorIterConst<P_numtype2>, 
      VectorPickIterConst<P_numtype3> > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      const Vector<P_numtype2>& d2, 
      const VectorPick<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorIterConst<P_numtype2>, 
      VectorPickIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      d3.begin()));
}

// where(TinyVector<P_numtype1, N_length1>, Vector<P_numtype2>, Range)
template<class P_numtype1, int N_length1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorIterConst<P_numtype2>, 
      Range > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      const Vector<P_numtype2>& d2, 
      Range d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorIterConst<P_numtype2>, 
      Range > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      d3));
}

// where(TinyVector<P_numtype1, N_length1>, Vector<P_numtype2>, TinyVector<P_numtype3, N_length3>)
template<class P_numtype1, int N_length1, class P_numtype2, class P_numtype3, int N_length3>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorIterConst<P_numtype2>, 
      TinyVectorIterConst<P_numtype3, N_length3> > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      const Vector<P_numtype2>& d2, 
      const TinyVector<P_numtype3, N_length3>& d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorIterConst<P_numtype2>, 
      TinyVectorIterConst<P_numtype3, N_length3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      d3.begin()));
}

// where(TinyVector<P_numtype1, N_length1>, Vector<P_numtype2>, int)
template<class P_numtype1, int N_length1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorIterConst<P_numtype2>, 
      _bz_VecExprConstant<int> > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      const Vector<P_numtype2>& d2, 
      int d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorIterConst<P_numtype2>, 
      _bz_VecExprConstant<int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      _bz_VecExprConstant<int>(d3)));
}

// where(TinyVector<P_numtype1, N_length1>, Vector<P_numtype2>, float)
template<class P_numtype1, int N_length1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorIterConst<P_numtype2>, 
      _bz_VecExprConstant<float> > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      const Vector<P_numtype2>& d2, 
      float d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorIterConst<P_numtype2>, 
      _bz_VecExprConstant<float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      _bz_VecExprConstant<float>(d3)));
}

// where(TinyVector<P_numtype1, N_length1>, Vector<P_numtype2>, double)
template<class P_numtype1, int N_length1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorIterConst<P_numtype2>, 
      _bz_VecExprConstant<double> > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      const Vector<P_numtype2>& d2, 
      double d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorIterConst<P_numtype2>, 
      _bz_VecExprConstant<double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      _bz_VecExprConstant<double>(d3)));
}

// where(TinyVector<P_numtype1, N_length1>, Vector<P_numtype2>, long double)
template<class P_numtype1, int N_length1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorIterConst<P_numtype2>, 
      _bz_VecExprConstant<long double> > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      const Vector<P_numtype2>& d2, 
      long double d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorIterConst<P_numtype2>, 
      _bz_VecExprConstant<long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      _bz_VecExprConstant<long double>(d3)));
}

// where(TinyVector<P_numtype1, N_length1>, Vector<P_numtype2>, complex<T3>)
#ifdef BZ_HAVE_COMPLEX
template<class P_numtype1, int N_length1, class P_numtype2, class T3>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorIterConst<P_numtype2>, 
      _bz_VecExprConstant<complex<T3> >  > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      const Vector<P_numtype2>& d2, 
      complex<T3> d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorIterConst<P_numtype2>, 
      _bz_VecExprConstant<complex<T3> >  > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      _bz_VecExprConstant<complex<T3> > (d3)));
}
#endif // BZ_HAVE_COMPLEX

// where(TinyVector<P_numtype1, N_length1>, _bz_VecExpr<P_expr2>, Vector<P_numtype3>)
template<class P_numtype1, int N_length1, class P_expr2, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExpr<P_expr2>, 
      VectorIterConst<P_numtype3> > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      _bz_VecExpr<P_expr2> d2, 
      const Vector<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExpr<P_expr2>, 
      VectorIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2, 
      d3.begin()));
}

// where(TinyVector<P_numtype1, N_length1>, _bz_VecExpr<P_expr2>, _bz_VecExpr<P_expr3>)
template<class P_numtype1, int N_length1, class P_expr2, class P_expr3>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_VecExpr<P_expr3> > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      _bz_VecExpr<P_expr2> d2, 
      _bz_VecExpr<P_expr3> d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_VecExpr<P_expr3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2, 
      d3));
}

// where(TinyVector<P_numtype1, N_length1>, _bz_VecExpr<P_expr2>, VectorPick<P_numtype3>)
template<class P_numtype1, int N_length1, class P_expr2, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExpr<P_expr2>, 
      VectorPickIterConst<P_numtype3> > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      _bz_VecExpr<P_expr2> d2, 
      const VectorPick<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExpr<P_expr2>, 
      VectorPickIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2, 
      d3.begin()));
}

// where(TinyVector<P_numtype1, N_length1>, _bz_VecExpr<P_expr2>, Range)
template<class P_numtype1, int N_length1, class P_expr2>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExpr<P_expr2>, 
      Range > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      _bz_VecExpr<P_expr2> d2, 
      Range d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExpr<P_expr2>, 
      Range > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2, 
      d3));
}

// where(TinyVector<P_numtype1, N_length1>, _bz_VecExpr<P_expr2>, TinyVector<P_numtype3, N_length3>)
template<class P_numtype1, int N_length1, class P_expr2, class P_numtype3, int N_length3>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExpr<P_expr2>, 
      TinyVectorIterConst<P_numtype3, N_length3> > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      _bz_VecExpr<P_expr2> d2, 
      const TinyVector<P_numtype3, N_length3>& d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExpr<P_expr2>, 
      TinyVectorIterConst<P_numtype3, N_length3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2, 
      d3.begin()));
}

// where(TinyVector<P_numtype1, N_length1>, _bz_VecExpr<P_expr2>, int)
template<class P_numtype1, int N_length1, class P_expr2>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_VecExprConstant<int> > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      _bz_VecExpr<P_expr2> d2, 
      int d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_VecExprConstant<int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2, 
      _bz_VecExprConstant<int>(d3)));
}

// where(TinyVector<P_numtype1, N_length1>, _bz_VecExpr<P_expr2>, float)
template<class P_numtype1, int N_length1, class P_expr2>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_VecExprConstant<float> > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      _bz_VecExpr<P_expr2> d2, 
      float d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_VecExprConstant<float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2, 
      _bz_VecExprConstant<float>(d3)));
}

// where(TinyVector<P_numtype1, N_length1>, _bz_VecExpr<P_expr2>, double)
template<class P_numtype1, int N_length1, class P_expr2>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_VecExprConstant<double> > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      _bz_VecExpr<P_expr2> d2, 
      double d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_VecExprConstant<double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2, 
      _bz_VecExprConstant<double>(d3)));
}

// where(TinyVector<P_numtype1, N_length1>, _bz_VecExpr<P_expr2>, long double)
template<class P_numtype1, int N_length1, class P_expr2>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_VecExprConstant<long double> > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      _bz_VecExpr<P_expr2> d2, 
      long double d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_VecExprConstant<long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2, 
      _bz_VecExprConstant<long double>(d3)));
}

// where(TinyVector<P_numtype1, N_length1>, _bz_VecExpr<P_expr2>, complex<T3>)
#ifdef BZ_HAVE_COMPLEX
template<class P_numtype1, int N_length1, class P_expr2, class T3>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_VecExprConstant<complex<T3> >  > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      _bz_VecExpr<P_expr2> d2, 
      complex<T3> d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExpr<P_expr2>, 
      _bz_VecExprConstant<complex<T3> >  > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2, 
      _bz_VecExprConstant<complex<T3> > (d3)));
}
#endif // BZ_HAVE_COMPLEX

// where(TinyVector<P_numtype1, N_length1>, VectorPick<P_numtype2>, Vector<P_numtype3>)
template<class P_numtype1, int N_length1, class P_numtype2, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorPickIterConst<P_numtype2>, 
      VectorIterConst<P_numtype3> > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      const VectorPick<P_numtype2>& d2, 
      const Vector<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorPickIterConst<P_numtype2>, 
      VectorIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      d3.begin()));
}

// where(TinyVector<P_numtype1, N_length1>, VectorPick<P_numtype2>, _bz_VecExpr<P_expr3>)
template<class P_numtype1, int N_length1, class P_numtype2, class P_expr3>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_VecExpr<P_expr3> > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      const VectorPick<P_numtype2>& d2, 
      _bz_VecExpr<P_expr3> d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_VecExpr<P_expr3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      d3));
}

// where(TinyVector<P_numtype1, N_length1>, VectorPick<P_numtype2>, VectorPick<P_numtype3>)
template<class P_numtype1, int N_length1, class P_numtype2, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorPickIterConst<P_numtype2>, 
      VectorPickIterConst<P_numtype3> > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      const VectorPick<P_numtype2>& d2, 
      const VectorPick<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorPickIterConst<P_numtype2>, 
      VectorPickIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      d3.begin()));
}

// where(TinyVector<P_numtype1, N_length1>, VectorPick<P_numtype2>, Range)
template<class P_numtype1, int N_length1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorPickIterConst<P_numtype2>, 
      Range > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      const VectorPick<P_numtype2>& d2, 
      Range d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorPickIterConst<P_numtype2>, 
      Range > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      d3));
}

// where(TinyVector<P_numtype1, N_length1>, VectorPick<P_numtype2>, TinyVector<P_numtype3, N_length3>)
template<class P_numtype1, int N_length1, class P_numtype2, class P_numtype3, int N_length3>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorPickIterConst<P_numtype2>, 
      TinyVectorIterConst<P_numtype3, N_length3> > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      const VectorPick<P_numtype2>& d2, 
      const TinyVector<P_numtype3, N_length3>& d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorPickIterConst<P_numtype2>, 
      TinyVectorIterConst<P_numtype3, N_length3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      d3.begin()));
}

// where(TinyVector<P_numtype1, N_length1>, VectorPick<P_numtype2>, int)
template<class P_numtype1, int N_length1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_VecExprConstant<int> > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      const VectorPick<P_numtype2>& d2, 
      int d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_VecExprConstant<int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      _bz_VecExprConstant<int>(d3)));
}

// where(TinyVector<P_numtype1, N_length1>, VectorPick<P_numtype2>, float)
template<class P_numtype1, int N_length1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_VecExprConstant<float> > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      const VectorPick<P_numtype2>& d2, 
      float d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_VecExprConstant<float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      _bz_VecExprConstant<float>(d3)));
}

// where(TinyVector<P_numtype1, N_length1>, VectorPick<P_numtype2>, double)
template<class P_numtype1, int N_length1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_VecExprConstant<double> > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      const VectorPick<P_numtype2>& d2, 
      double d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_VecExprConstant<double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      _bz_VecExprConstant<double>(d3)));
}

// where(TinyVector<P_numtype1, N_length1>, VectorPick<P_numtype2>, long double)
template<class P_numtype1, int N_length1, class P_numtype2>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_VecExprConstant<long double> > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      const VectorPick<P_numtype2>& d2, 
      long double d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_VecExprConstant<long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      _bz_VecExprConstant<long double>(d3)));
}

// where(TinyVector<P_numtype1, N_length1>, VectorPick<P_numtype2>, complex<T3>)
#ifdef BZ_HAVE_COMPLEX
template<class P_numtype1, int N_length1, class P_numtype2, class T3>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_VecExprConstant<complex<T3> >  > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      const VectorPick<P_numtype2>& d2, 
      complex<T3> d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      VectorPickIterConst<P_numtype2>, 
      _bz_VecExprConstant<complex<T3> >  > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      _bz_VecExprConstant<complex<T3> > (d3)));
}
#endif // BZ_HAVE_COMPLEX

// where(TinyVector<P_numtype1, N_length1>, Range, Vector<P_numtype3>)
template<class P_numtype1, int N_length1, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      Range, 
      VectorIterConst<P_numtype3> > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      Range d2, 
      const Vector<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      Range, 
      VectorIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2, 
      d3.begin()));
}

// where(TinyVector<P_numtype1, N_length1>, Range, _bz_VecExpr<P_expr3>)
template<class P_numtype1, int N_length1, class P_expr3>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      Range, 
      _bz_VecExpr<P_expr3> > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      Range d2, 
      _bz_VecExpr<P_expr3> d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      Range, 
      _bz_VecExpr<P_expr3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2, 
      d3));
}

// where(TinyVector<P_numtype1, N_length1>, Range, VectorPick<P_numtype3>)
template<class P_numtype1, int N_length1, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      Range, 
      VectorPickIterConst<P_numtype3> > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      Range d2, 
      const VectorPick<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      Range, 
      VectorPickIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2, 
      d3.begin()));
}

// where(TinyVector<P_numtype1, N_length1>, Range, Range)
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      Range, 
      Range > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      Range d2, 
      Range d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      Range, 
      Range > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2, 
      d3));
}

// where(TinyVector<P_numtype1, N_length1>, Range, TinyVector<P_numtype3, N_length3>)
template<class P_numtype1, int N_length1, class P_numtype3, int N_length3>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      Range, 
      TinyVectorIterConst<P_numtype3, N_length3> > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      Range d2, 
      const TinyVector<P_numtype3, N_length3>& d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      Range, 
      TinyVectorIterConst<P_numtype3, N_length3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2, 
      d3.begin()));
}

// where(TinyVector<P_numtype1, N_length1>, Range, int)
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      Range, 
      _bz_VecExprConstant<int> > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      Range d2, 
      int d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      Range, 
      _bz_VecExprConstant<int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2, 
      _bz_VecExprConstant<int>(d3)));
}

// where(TinyVector<P_numtype1, N_length1>, Range, float)
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      Range, 
      _bz_VecExprConstant<float> > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      Range d2, 
      float d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      Range, 
      _bz_VecExprConstant<float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2, 
      _bz_VecExprConstant<float>(d3)));
}

// where(TinyVector<P_numtype1, N_length1>, Range, double)
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      Range, 
      _bz_VecExprConstant<double> > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      Range d2, 
      double d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      Range, 
      _bz_VecExprConstant<double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2, 
      _bz_VecExprConstant<double>(d3)));
}

// where(TinyVector<P_numtype1, N_length1>, Range, long double)
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      Range, 
      _bz_VecExprConstant<long double> > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      Range d2, 
      long double d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      Range, 
      _bz_VecExprConstant<long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2, 
      _bz_VecExprConstant<long double>(d3)));
}

// where(TinyVector<P_numtype1, N_length1>, Range, complex<T3>)
#ifdef BZ_HAVE_COMPLEX
template<class P_numtype1, int N_length1, class T3>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      Range, 
      _bz_VecExprConstant<complex<T3> >  > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      Range d2, 
      complex<T3> d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      Range, 
      _bz_VecExprConstant<complex<T3> >  > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2, 
      _bz_VecExprConstant<complex<T3> > (d3)));
}
#endif // BZ_HAVE_COMPLEX

// where(TinyVector<P_numtype1, N_length1>, TinyVector<P_numtype2, N_length2>, Vector<P_numtype3>)
template<class P_numtype1, int N_length1, class P_numtype2, int N_length2, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      VectorIterConst<P_numtype3> > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2, 
      const Vector<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      VectorIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      d3.begin()));
}

// where(TinyVector<P_numtype1, N_length1>, TinyVector<P_numtype2, N_length2>, _bz_VecExpr<P_expr3>)
template<class P_numtype1, int N_length1, class P_numtype2, int N_length2, class P_expr3>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_VecExpr<P_expr3> > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2, 
      _bz_VecExpr<P_expr3> d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_VecExpr<P_expr3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      d3));
}

// where(TinyVector<P_numtype1, N_length1>, TinyVector<P_numtype2, N_length2>, VectorPick<P_numtype3>)
template<class P_numtype1, int N_length1, class P_numtype2, int N_length2, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      VectorPickIterConst<P_numtype3> > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2, 
      const VectorPick<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      VectorPickIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      d3.begin()));
}

// where(TinyVector<P_numtype1, N_length1>, TinyVector<P_numtype2, N_length2>, Range)
template<class P_numtype1, int N_length1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      Range > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2, 
      Range d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      Range > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      d3));
}

// where(TinyVector<P_numtype1, N_length1>, TinyVector<P_numtype2, N_length2>, TinyVector<P_numtype3, N_length3>)
template<class P_numtype1, int N_length1, class P_numtype2, int N_length2, class P_numtype3, int N_length3>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      TinyVectorIterConst<P_numtype3, N_length3> > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2, 
      const TinyVector<P_numtype3, N_length3>& d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      TinyVectorIterConst<P_numtype3, N_length3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      d3.begin()));
}

// where(TinyVector<P_numtype1, N_length1>, TinyVector<P_numtype2, N_length2>, int)
template<class P_numtype1, int N_length1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_VecExprConstant<int> > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2, 
      int d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_VecExprConstant<int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      _bz_VecExprConstant<int>(d3)));
}

// where(TinyVector<P_numtype1, N_length1>, TinyVector<P_numtype2, N_length2>, float)
template<class P_numtype1, int N_length1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_VecExprConstant<float> > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2, 
      float d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_VecExprConstant<float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      _bz_VecExprConstant<float>(d3)));
}

// where(TinyVector<P_numtype1, N_length1>, TinyVector<P_numtype2, N_length2>, double)
template<class P_numtype1, int N_length1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_VecExprConstant<double> > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2, 
      double d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_VecExprConstant<double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      _bz_VecExprConstant<double>(d3)));
}

// where(TinyVector<P_numtype1, N_length1>, TinyVector<P_numtype2, N_length2>, long double)
template<class P_numtype1, int N_length1, class P_numtype2, int N_length2>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_VecExprConstant<long double> > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2, 
      long double d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_VecExprConstant<long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      _bz_VecExprConstant<long double>(d3)));
}

// where(TinyVector<P_numtype1, N_length1>, TinyVector<P_numtype2, N_length2>, complex<T3>)
#ifdef BZ_HAVE_COMPLEX
template<class P_numtype1, int N_length1, class P_numtype2, int N_length2, class T3>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_VecExprConstant<complex<T3> >  > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      const TinyVector<P_numtype2, N_length2>& d2, 
      complex<T3> d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      TinyVectorIterConst<P_numtype2, N_length2>, 
      _bz_VecExprConstant<complex<T3> >  > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      d2.begin(), 
      _bz_VecExprConstant<complex<T3> > (d3)));
}
#endif // BZ_HAVE_COMPLEX

// where(TinyVector<P_numtype1, N_length1>, int, Vector<P_numtype3>)
template<class P_numtype1, int N_length1, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<int>, 
      VectorIterConst<P_numtype3> > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      int d2, 
      const Vector<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<int>, 
      VectorIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2), 
      d3.begin()));
}

// where(TinyVector<P_numtype1, N_length1>, int, _bz_VecExpr<P_expr3>)
template<class P_numtype1, int N_length1, class P_expr3>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<int>, 
      _bz_VecExpr<P_expr3> > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      int d2, 
      _bz_VecExpr<P_expr3> d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<int>, 
      _bz_VecExpr<P_expr3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2), 
      d3));
}

// where(TinyVector<P_numtype1, N_length1>, int, VectorPick<P_numtype3>)
template<class P_numtype1, int N_length1, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<int>, 
      VectorPickIterConst<P_numtype3> > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      int d2, 
      const VectorPick<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<int>, 
      VectorPickIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2), 
      d3.begin()));
}

// where(TinyVector<P_numtype1, N_length1>, int, Range)
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<int>, 
      Range > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      int d2, 
      Range d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<int>, 
      Range > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2), 
      d3));
}

// where(TinyVector<P_numtype1, N_length1>, int, TinyVector<P_numtype3, N_length3>)
template<class P_numtype1, int N_length1, class P_numtype3, int N_length3>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<int>, 
      TinyVectorIterConst<P_numtype3, N_length3> > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      int d2, 
      const TinyVector<P_numtype3, N_length3>& d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<int>, 
      TinyVectorIterConst<P_numtype3, N_length3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2), 
      d3.begin()));
}

// where(TinyVector<P_numtype1, N_length1>, int, int)
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<int>, 
      _bz_VecExprConstant<int> > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      int d2, 
      int d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<int>, 
      _bz_VecExprConstant<int> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<int>(d2), 
      _bz_VecExprConstant<int>(d3)));
}

// where(TinyVector<P_numtype1, N_length1>, float, Vector<P_numtype3>)
template<class P_numtype1, int N_length1, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<float>, 
      VectorIterConst<P_numtype3> > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      float d2, 
      const Vector<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<float>, 
      VectorIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<float>(d2), 
      d3.begin()));
}

// where(TinyVector<P_numtype1, N_length1>, float, _bz_VecExpr<P_expr3>)
template<class P_numtype1, int N_length1, class P_expr3>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<float>, 
      _bz_VecExpr<P_expr3> > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      float d2, 
      _bz_VecExpr<P_expr3> d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<float>, 
      _bz_VecExpr<P_expr3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<float>(d2), 
      d3));
}

// where(TinyVector<P_numtype1, N_length1>, float, VectorPick<P_numtype3>)
template<class P_numtype1, int N_length1, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<float>, 
      VectorPickIterConst<P_numtype3> > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      float d2, 
      const VectorPick<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<float>, 
      VectorPickIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<float>(d2), 
      d3.begin()));
}

// where(TinyVector<P_numtype1, N_length1>, float, Range)
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<float>, 
      Range > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      float d2, 
      Range d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<float>, 
      Range > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<float>(d2), 
      d3));
}

// where(TinyVector<P_numtype1, N_length1>, float, TinyVector<P_numtype3, N_length3>)
template<class P_numtype1, int N_length1, class P_numtype3, int N_length3>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<float>, 
      TinyVectorIterConst<P_numtype3, N_length3> > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      float d2, 
      const TinyVector<P_numtype3, N_length3>& d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<float>, 
      TinyVectorIterConst<P_numtype3, N_length3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<float>(d2), 
      d3.begin()));
}

// where(TinyVector<P_numtype1, N_length1>, float, float)
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<float>, 
      _bz_VecExprConstant<float> > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      float d2, 
      float d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<float>, 
      _bz_VecExprConstant<float> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<float>(d2), 
      _bz_VecExprConstant<float>(d3)));
}

// where(TinyVector<P_numtype1, N_length1>, double, Vector<P_numtype3>)
template<class P_numtype1, int N_length1, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<double>, 
      VectorIterConst<P_numtype3> > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      double d2, 
      const Vector<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<double>, 
      VectorIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<double>(d2), 
      d3.begin()));
}

// where(TinyVector<P_numtype1, N_length1>, double, _bz_VecExpr<P_expr3>)
template<class P_numtype1, int N_length1, class P_expr3>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<double>, 
      _bz_VecExpr<P_expr3> > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      double d2, 
      _bz_VecExpr<P_expr3> d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<double>, 
      _bz_VecExpr<P_expr3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<double>(d2), 
      d3));
}

// where(TinyVector<P_numtype1, N_length1>, double, VectorPick<P_numtype3>)
template<class P_numtype1, int N_length1, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<double>, 
      VectorPickIterConst<P_numtype3> > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      double d2, 
      const VectorPick<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<double>, 
      VectorPickIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<double>(d2), 
      d3.begin()));
}

// where(TinyVector<P_numtype1, N_length1>, double, Range)
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<double>, 
      Range > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      double d2, 
      Range d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<double>, 
      Range > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<double>(d2), 
      d3));
}

// where(TinyVector<P_numtype1, N_length1>, double, TinyVector<P_numtype3, N_length3>)
template<class P_numtype1, int N_length1, class P_numtype3, int N_length3>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<double>, 
      TinyVectorIterConst<P_numtype3, N_length3> > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      double d2, 
      const TinyVector<P_numtype3, N_length3>& d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<double>, 
      TinyVectorIterConst<P_numtype3, N_length3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<double>(d2), 
      d3.begin()));
}

// where(TinyVector<P_numtype1, N_length1>, double, double)
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<double>, 
      _bz_VecExprConstant<double> > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      double d2, 
      double d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<double>, 
      _bz_VecExprConstant<double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<double>(d2), 
      _bz_VecExprConstant<double>(d3)));
}

// where(TinyVector<P_numtype1, N_length1>, long double, Vector<P_numtype3>)
template<class P_numtype1, int N_length1, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<long double>, 
      VectorIterConst<P_numtype3> > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      long double d2, 
      const Vector<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<long double>, 
      VectorIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<long double>(d2), 
      d3.begin()));
}

// where(TinyVector<P_numtype1, N_length1>, long double, _bz_VecExpr<P_expr3>)
template<class P_numtype1, int N_length1, class P_expr3>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<long double>, 
      _bz_VecExpr<P_expr3> > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      long double d2, 
      _bz_VecExpr<P_expr3> d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<long double>, 
      _bz_VecExpr<P_expr3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<long double>(d2), 
      d3));
}

// where(TinyVector<P_numtype1, N_length1>, long double, VectorPick<P_numtype3>)
template<class P_numtype1, int N_length1, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<long double>, 
      VectorPickIterConst<P_numtype3> > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      long double d2, 
      const VectorPick<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<long double>, 
      VectorPickIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<long double>(d2), 
      d3.begin()));
}

// where(TinyVector<P_numtype1, N_length1>, long double, Range)
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<long double>, 
      Range > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      long double d2, 
      Range d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<long double>, 
      Range > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<long double>(d2), 
      d3));
}

// where(TinyVector<P_numtype1, N_length1>, long double, TinyVector<P_numtype3, N_length3>)
template<class P_numtype1, int N_length1, class P_numtype3, int N_length3>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<long double>, 
      TinyVectorIterConst<P_numtype3, N_length3> > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      long double d2, 
      const TinyVector<P_numtype3, N_length3>& d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<long double>, 
      TinyVectorIterConst<P_numtype3, N_length3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<long double>(d2), 
      d3.begin()));
}

// where(TinyVector<P_numtype1, N_length1>, long double, long double)
template<class P_numtype1, int N_length1>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<long double>, 
      _bz_VecExprConstant<long double> > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      long double d2, 
      long double d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<long double>, 
      _bz_VecExprConstant<long double> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<long double>(d2), 
      _bz_VecExprConstant<long double>(d3)));
}

// where(TinyVector<P_numtype1, N_length1>, complex<T2>, Vector<P_numtype3>)
#ifdef BZ_HAVE_COMPLEX
template<class P_numtype1, int N_length1, class T2, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<complex<T2> > , 
      VectorIterConst<P_numtype3> > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      complex<T2> d2, 
      const Vector<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<complex<T2> > , 
      VectorIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<complex<T2> > (d2), 
      d3.begin()));
}
#endif // BZ_HAVE_COMPLEX

// where(TinyVector<P_numtype1, N_length1>, complex<T2>, _bz_VecExpr<P_expr3>)
#ifdef BZ_HAVE_COMPLEX
template<class P_numtype1, int N_length1, class T2, class P_expr3>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_VecExpr<P_expr3> > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      complex<T2> d2, 
      _bz_VecExpr<P_expr3> d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_VecExpr<P_expr3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<complex<T2> > (d2), 
      d3));
}
#endif // BZ_HAVE_COMPLEX

// where(TinyVector<P_numtype1, N_length1>, complex<T2>, VectorPick<P_numtype3>)
#ifdef BZ_HAVE_COMPLEX
template<class P_numtype1, int N_length1, class T2, class P_numtype3>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<complex<T2> > , 
      VectorPickIterConst<P_numtype3> > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      complex<T2> d2, 
      const VectorPick<P_numtype3>& d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<complex<T2> > , 
      VectorPickIterConst<P_numtype3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<complex<T2> > (d2), 
      d3.begin()));
}
#endif // BZ_HAVE_COMPLEX

// where(TinyVector<P_numtype1, N_length1>, complex<T2>, Range)
#ifdef BZ_HAVE_COMPLEX
template<class P_numtype1, int N_length1, class T2>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<complex<T2> > , 
      Range > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      complex<T2> d2, 
      Range d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<complex<T2> > , 
      Range > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<complex<T2> > (d2), 
      d3));
}
#endif // BZ_HAVE_COMPLEX

// where(TinyVector<P_numtype1, N_length1>, complex<T2>, TinyVector<P_numtype3, N_length3>)
#ifdef BZ_HAVE_COMPLEX
template<class P_numtype1, int N_length1, class T2, class P_numtype3, int N_length3>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<complex<T2> > , 
      TinyVectorIterConst<P_numtype3, N_length3> > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      complex<T2> d2, 
      const TinyVector<P_numtype3, N_length3>& d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<complex<T2> > , 
      TinyVectorIterConst<P_numtype3, N_length3> > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<complex<T2> > (d2), 
      d3.begin()));
}
#endif // BZ_HAVE_COMPLEX

// where(TinyVector<P_numtype1, N_length1>, complex<T2>, complex<T3>)
#ifdef BZ_HAVE_COMPLEX
template<class P_numtype1, int N_length1, class T2, class T3>
inline
_bz_VecExpr<_bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_VecExprConstant<complex<T3> >  > >
where(const TinyVector<P_numtype1, N_length1>& d1, 
      complex<T2> d2, 
      complex<T3> d3)
{ 
    typedef _bz_VecWhere<TinyVectorIterConst<P_numtype1, N_length1>, 
      _bz_VecExprConstant<complex<T2> > , 
      _bz_VecExprConstant<complex<T3> >  > T_expr;

    return _bz_VecExpr<T_expr>(T_expr(d1.begin(), 
      _bz_VecExprConstant<complex<T2> > (d2), 
      _bz_VecExprConstant<complex<T3> > (d3)));
}
#endif // BZ_HAVE_COMPLEX

BZ_NAMESPACE_END

#endif
