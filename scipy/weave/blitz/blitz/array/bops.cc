/***************************************************************************
 * blitz/../array/bops.cc	Array expression templates (2 operands)
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
 * Suggestions:          blitz-suggest@cybervision.com
 * Bugs:                 blitz-bugs@cybervision.com
 *
 * For more information, please see the Blitz++ Home Page:
 *    http://seurat.uwaterloo.ca/blitz/
 *
 ***************************************************************************
 *
 */ 

// Generated source file.  Do not edit. 
// genarrbops.cpp Dec 30 2003 16:48:46

#ifndef BZ_ARRAYBOPS_CC
#define BZ_ARRAYBOPS_CC

#ifndef BZ_ARRAYEXPR_H
 #error <blitz/array/bops.cc> must be included after <blitz/arrayexpr.h>
#endif

BZ_NAMESPACE(blitz)

/****************************************************************************
 * Addition Operators
 ****************************************************************************/

// Array<T_numtype1, N_rank1> + Array<T_numtype2, N_rank2>
template<class T_numtype1, int N_rank1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Add<T_numtype1, T_numtype2 > > >
operator+(const Array<T_numtype1, N_rank1>& d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Add<T_numtype1, T_numtype2> >
      (d1.begin(), 
      d2.begin());
}

// Array<T_numtype1, N_rank1> + _bz_ArrayExpr<P_expr2>
template<class T_numtype1, int N_rank1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>,
      Add<T_numtype1, typename P_expr2::T_numtype > > >
operator+(const Array<T_numtype1, N_rank1>& d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>, 
      Add<T_numtype1, typename P_expr2::T_numtype> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> + IndexPlaceholder<N_index2>
template<class T_numtype1, int N_rank1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>,
      Add<T_numtype1, int > > >
operator+(const Array<T_numtype1, N_rank1>& d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>, 
      Add<T_numtype1, int> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> + int
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>,
      Add<T_numtype1, int > > >
operator+(const Array<T_numtype1, N_rank1>& d1, 
      int d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>, 
      Add<T_numtype1, int> >
      (d1.begin(), 
      _bz_ArrayExprConstant<int>(d2));
}

// Array<T_numtype1, N_rank1> + float
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<float>,
      Add<T_numtype1, float > > >
operator+(const Array<T_numtype1, N_rank1>& d1, 
      float d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<float>, 
      Add<T_numtype1, float> >
      (d1.begin(), 
      _bz_ArrayExprConstant<float>(d2));
}

// Array<T_numtype1, N_rank1> + double
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<double>,
      Add<T_numtype1, double > > >
operator+(const Array<T_numtype1, N_rank1>& d1, 
      double d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<double>, 
      Add<T_numtype1, double> >
      (d1.begin(), 
      _bz_ArrayExprConstant<double>(d2));
}

// Array<T_numtype1, N_rank1> + long double
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<long double>,
      Add<T_numtype1, long double > > >
operator+(const Array<T_numtype1, N_rank1>& d1, 
      long double d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<long double>, 
      Add<T_numtype1, long double> >
      (d1.begin(), 
      _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
// Array<T_numtype1, N_rank1> + complex<T2>
template<class T_numtype1, int N_rank1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<complex<T2> > ,
      Add<T_numtype1, complex<T2>  > > >
operator+(const Array<T_numtype1, N_rank1>& d1, 
      complex<T2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<complex<T2> > , 
      Add<T_numtype1, complex<T2> > >
      (d1.begin(), 
      _bz_ArrayExprConstant<complex<T2> > (d2));
}
#endif // BZ_HAVE_COMPLEX

// _bz_ArrayExpr<P_expr1> + Array<T_numtype2, N_rank2>
template<class P_expr1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Add<typename P_expr1::T_numtype, T_numtype2 > > >
operator+(_bz_ArrayExpr<P_expr1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Add<typename P_expr1::T_numtype, T_numtype2> >
      (d1, 
      d2.begin());
}

// _bz_ArrayExpr<P_expr1> + _bz_ArrayExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>,
      Add<typename P_expr1::T_numtype, typename P_expr2::T_numtype > > >
operator+(_bz_ArrayExpr<P_expr1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>, 
      Add<typename P_expr1::T_numtype, typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> + IndexPlaceholder<N_index2>
template<class P_expr1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>,
      Add<typename P_expr1::T_numtype, int > > >
operator+(_bz_ArrayExpr<P_expr1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>, 
      Add<typename P_expr1::T_numtype, int> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> + int
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>,
      Add<typename P_expr1::T_numtype, int > > >
operator+(_bz_ArrayExpr<P_expr1> d1, 
      int d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>, 
      Add<typename P_expr1::T_numtype, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// _bz_ArrayExpr<P_expr1> + float
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<float>,
      Add<typename P_expr1::T_numtype, float > > >
operator+(_bz_ArrayExpr<P_expr1> d1, 
      float d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<float>, 
      Add<typename P_expr1::T_numtype, float> >
      (d1, 
      _bz_ArrayExprConstant<float>(d2));
}

// _bz_ArrayExpr<P_expr1> + double
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<double>,
      Add<typename P_expr1::T_numtype, double > > >
operator+(_bz_ArrayExpr<P_expr1> d1, 
      double d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<double>, 
      Add<typename P_expr1::T_numtype, double> >
      (d1, 
      _bz_ArrayExprConstant<double>(d2));
}

// _bz_ArrayExpr<P_expr1> + long double
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<long double>,
      Add<typename P_expr1::T_numtype, long double > > >
operator+(_bz_ArrayExpr<P_expr1> d1, 
      long double d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<long double>, 
      Add<typename P_expr1::T_numtype, long double> >
      (d1, 
      _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
// _bz_ArrayExpr<P_expr1> + complex<T2>
template<class P_expr1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<complex<T2> > ,
      Add<typename P_expr1::T_numtype, complex<T2>  > > >
operator+(_bz_ArrayExpr<P_expr1> d1, 
      complex<T2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<complex<T2> > , 
      Add<typename P_expr1::T_numtype, complex<T2> > >
      (d1, 
      _bz_ArrayExprConstant<complex<T2> > (d2));
}
#endif // BZ_HAVE_COMPLEX

// IndexPlaceholder<N_index1> + Array<T_numtype2, N_rank2>
template<int N_index1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Add<int, T_numtype2 > > >
operator+(IndexPlaceholder<N_index1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Add<int, T_numtype2> >
      (d1, 
      d2.begin());
}

// IndexPlaceholder<N_index1> + _bz_ArrayExpr<P_expr2>
template<int N_index1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>,
      Add<int, typename P_expr2::T_numtype > > >
operator+(IndexPlaceholder<N_index1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>, 
      Add<int, typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> + IndexPlaceholder<N_index2>
template<int N_index1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>,
      Add<int, int > > >
operator+(IndexPlaceholder<N_index1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>, 
      Add<int, int> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> + int
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>,
      Add<int, int > > >
operator+(IndexPlaceholder<N_index1> d1, 
      int d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>, 
      Add<int, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// IndexPlaceholder<N_index1> + float
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<float>,
      Add<int, float > > >
operator+(IndexPlaceholder<N_index1> d1, 
      float d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<float>, 
      Add<int, float> >
      (d1, 
      _bz_ArrayExprConstant<float>(d2));
}

// IndexPlaceholder<N_index1> + double
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<double>,
      Add<int, double > > >
operator+(IndexPlaceholder<N_index1> d1, 
      double d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<double>, 
      Add<int, double> >
      (d1, 
      _bz_ArrayExprConstant<double>(d2));
}

// IndexPlaceholder<N_index1> + long double
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<long double>,
      Add<int, long double > > >
operator+(IndexPlaceholder<N_index1> d1, 
      long double d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<long double>, 
      Add<int, long double> >
      (d1, 
      _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
// IndexPlaceholder<N_index1> + complex<T2>
template<int N_index1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<complex<T2> > ,
      Add<int, complex<T2>  > > >
operator+(IndexPlaceholder<N_index1> d1, 
      complex<T2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<complex<T2> > , 
      Add<int, complex<T2> > >
      (d1, 
      _bz_ArrayExprConstant<complex<T2> > (d2));
}
#endif // BZ_HAVE_COMPLEX

// int + Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Add<int, T_numtype2 > > >
operator+(int d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Add<int, T_numtype2> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2.begin());
}

// int + _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>,
      Add<int, typename P_expr2::T_numtype > > >
operator+(int d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>, 
      Add<int, typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2);
}

// int + IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      IndexPlaceholder<N_index2>,
      Add<int, int > > >
operator+(int d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      IndexPlaceholder<N_index2>, 
      Add<int, int> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2);
}

// float + Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Add<float, T_numtype2 > > >
operator+(float d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Add<float, T_numtype2> >
      (_bz_ArrayExprConstant<float>(d1), 
      d2.begin());
}

// float + _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, 
      _bz_ArrayExpr<P_expr2>,
      Add<float, typename P_expr2::T_numtype > > >
operator+(float d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, 
      _bz_ArrayExpr<P_expr2>, 
      Add<float, typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<float>(d1), 
      d2);
}

// float + IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, 
      IndexPlaceholder<N_index2>,
      Add<float, int > > >
operator+(float d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, 
      IndexPlaceholder<N_index2>, 
      Add<float, int> >
      (_bz_ArrayExprConstant<float>(d1), 
      d2);
}

// double + Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Add<double, T_numtype2 > > >
operator+(double d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Add<double, T_numtype2> >
      (_bz_ArrayExprConstant<double>(d1), 
      d2.begin());
}

// double + _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, 
      _bz_ArrayExpr<P_expr2>,
      Add<double, typename P_expr2::T_numtype > > >
operator+(double d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, 
      _bz_ArrayExpr<P_expr2>, 
      Add<double, typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<double>(d1), 
      d2);
}

// double + IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, 
      IndexPlaceholder<N_index2>,
      Add<double, int > > >
operator+(double d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, 
      IndexPlaceholder<N_index2>, 
      Add<double, int> >
      (_bz_ArrayExprConstant<double>(d1), 
      d2);
}

// long double + Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Add<long double, T_numtype2 > > >
operator+(long double d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Add<long double, T_numtype2> >
      (_bz_ArrayExprConstant<long double>(d1), 
      d2.begin());
}

// long double + _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, 
      _bz_ArrayExpr<P_expr2>,
      Add<long double, typename P_expr2::T_numtype > > >
operator+(long double d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, 
      _bz_ArrayExpr<P_expr2>, 
      Add<long double, typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<long double>(d1), 
      d2);
}

// long double + IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, 
      IndexPlaceholder<N_index2>,
      Add<long double, int > > >
operator+(long double d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, 
      IndexPlaceholder<N_index2>, 
      Add<long double, int> >
      (_bz_ArrayExprConstant<long double>(d1), 
      d2);
}

#ifdef BZ_HAVE_COMPLEX
// complex<T1> + Array<T_numtype2, N_rank2>
template<class T1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , 
      ArrayIterator<T_numtype2, N_rank2>,
      Add<complex<T1> , T_numtype2 > > >
operator+(complex<T1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , 
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
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , 
      _bz_ArrayExpr<P_expr2>,
      Add<complex<T1> , typename P_expr2::T_numtype > > >
operator+(complex<T1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , 
      _bz_ArrayExpr<P_expr2>, 
      Add<complex<T1> , typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<complex<T1> > (d1), 
      d2);
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX
// complex<T1> + IndexPlaceholder<N_index2>
template<class T1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , 
      IndexPlaceholder<N_index2>,
      Add<complex<T1> , int > > >
operator+(complex<T1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , 
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
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Subtract<T_numtype1, T_numtype2 > > >
operator-(const Array<T_numtype1, N_rank1>& d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Subtract<T_numtype1, T_numtype2> >
      (d1.begin(), 
      d2.begin());
}

// Array<T_numtype1, N_rank1> - _bz_ArrayExpr<P_expr2>
template<class T_numtype1, int N_rank1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>,
      Subtract<T_numtype1, typename P_expr2::T_numtype > > >
operator-(const Array<T_numtype1, N_rank1>& d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>, 
      Subtract<T_numtype1, typename P_expr2::T_numtype> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> - IndexPlaceholder<N_index2>
template<class T_numtype1, int N_rank1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>,
      Subtract<T_numtype1, int > > >
operator-(const Array<T_numtype1, N_rank1>& d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>, 
      Subtract<T_numtype1, int> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> - int
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>,
      Subtract<T_numtype1, int > > >
operator-(const Array<T_numtype1, N_rank1>& d1, 
      int d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>, 
      Subtract<T_numtype1, int> >
      (d1.begin(), 
      _bz_ArrayExprConstant<int>(d2));
}

// Array<T_numtype1, N_rank1> - float
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<float>,
      Subtract<T_numtype1, float > > >
operator-(const Array<T_numtype1, N_rank1>& d1, 
      float d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<float>, 
      Subtract<T_numtype1, float> >
      (d1.begin(), 
      _bz_ArrayExprConstant<float>(d2));
}

// Array<T_numtype1, N_rank1> - double
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<double>,
      Subtract<T_numtype1, double > > >
operator-(const Array<T_numtype1, N_rank1>& d1, 
      double d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<double>, 
      Subtract<T_numtype1, double> >
      (d1.begin(), 
      _bz_ArrayExprConstant<double>(d2));
}

// Array<T_numtype1, N_rank1> - long double
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<long double>,
      Subtract<T_numtype1, long double > > >
operator-(const Array<T_numtype1, N_rank1>& d1, 
      long double d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<long double>, 
      Subtract<T_numtype1, long double> >
      (d1.begin(), 
      _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
// Array<T_numtype1, N_rank1> - complex<T2>
template<class T_numtype1, int N_rank1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<complex<T2> > ,
      Subtract<T_numtype1, complex<T2>  > > >
operator-(const Array<T_numtype1, N_rank1>& d1, 
      complex<T2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<complex<T2> > , 
      Subtract<T_numtype1, complex<T2> > >
      (d1.begin(), 
      _bz_ArrayExprConstant<complex<T2> > (d2));
}
#endif // BZ_HAVE_COMPLEX

// _bz_ArrayExpr<P_expr1> - Array<T_numtype2, N_rank2>
template<class P_expr1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Subtract<typename P_expr1::T_numtype, T_numtype2 > > >
operator-(_bz_ArrayExpr<P_expr1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Subtract<typename P_expr1::T_numtype, T_numtype2> >
      (d1, 
      d2.begin());
}

// _bz_ArrayExpr<P_expr1> - _bz_ArrayExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>,
      Subtract<typename P_expr1::T_numtype, typename P_expr2::T_numtype > > >
operator-(_bz_ArrayExpr<P_expr1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>, 
      Subtract<typename P_expr1::T_numtype, typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> - IndexPlaceholder<N_index2>
template<class P_expr1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>,
      Subtract<typename P_expr1::T_numtype, int > > >
operator-(_bz_ArrayExpr<P_expr1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>, 
      Subtract<typename P_expr1::T_numtype, int> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> - int
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>,
      Subtract<typename P_expr1::T_numtype, int > > >
operator-(_bz_ArrayExpr<P_expr1> d1, 
      int d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>, 
      Subtract<typename P_expr1::T_numtype, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// _bz_ArrayExpr<P_expr1> - float
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<float>,
      Subtract<typename P_expr1::T_numtype, float > > >
operator-(_bz_ArrayExpr<P_expr1> d1, 
      float d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<float>, 
      Subtract<typename P_expr1::T_numtype, float> >
      (d1, 
      _bz_ArrayExprConstant<float>(d2));
}

// _bz_ArrayExpr<P_expr1> - double
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<double>,
      Subtract<typename P_expr1::T_numtype, double > > >
operator-(_bz_ArrayExpr<P_expr1> d1, 
      double d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<double>, 
      Subtract<typename P_expr1::T_numtype, double> >
      (d1, 
      _bz_ArrayExprConstant<double>(d2));
}

// _bz_ArrayExpr<P_expr1> - long double
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<long double>,
      Subtract<typename P_expr1::T_numtype, long double > > >
operator-(_bz_ArrayExpr<P_expr1> d1, 
      long double d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<long double>, 
      Subtract<typename P_expr1::T_numtype, long double> >
      (d1, 
      _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
// _bz_ArrayExpr<P_expr1> - complex<T2>
template<class P_expr1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<complex<T2> > ,
      Subtract<typename P_expr1::T_numtype, complex<T2>  > > >
operator-(_bz_ArrayExpr<P_expr1> d1, 
      complex<T2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<complex<T2> > , 
      Subtract<typename P_expr1::T_numtype, complex<T2> > >
      (d1, 
      _bz_ArrayExprConstant<complex<T2> > (d2));
}
#endif // BZ_HAVE_COMPLEX

// IndexPlaceholder<N_index1> - Array<T_numtype2, N_rank2>
template<int N_index1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Subtract<int, T_numtype2 > > >
operator-(IndexPlaceholder<N_index1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Subtract<int, T_numtype2> >
      (d1, 
      d2.begin());
}

// IndexPlaceholder<N_index1> - _bz_ArrayExpr<P_expr2>
template<int N_index1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>,
      Subtract<int, typename P_expr2::T_numtype > > >
operator-(IndexPlaceholder<N_index1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>, 
      Subtract<int, typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> - IndexPlaceholder<N_index2>
template<int N_index1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>,
      Subtract<int, int > > >
operator-(IndexPlaceholder<N_index1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>, 
      Subtract<int, int> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> - int
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>,
      Subtract<int, int > > >
operator-(IndexPlaceholder<N_index1> d1, 
      int d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>, 
      Subtract<int, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// IndexPlaceholder<N_index1> - float
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<float>,
      Subtract<int, float > > >
operator-(IndexPlaceholder<N_index1> d1, 
      float d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<float>, 
      Subtract<int, float> >
      (d1, 
      _bz_ArrayExprConstant<float>(d2));
}

// IndexPlaceholder<N_index1> - double
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<double>,
      Subtract<int, double > > >
operator-(IndexPlaceholder<N_index1> d1, 
      double d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<double>, 
      Subtract<int, double> >
      (d1, 
      _bz_ArrayExprConstant<double>(d2));
}

// IndexPlaceholder<N_index1> - long double
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<long double>,
      Subtract<int, long double > > >
operator-(IndexPlaceholder<N_index1> d1, 
      long double d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<long double>, 
      Subtract<int, long double> >
      (d1, 
      _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
// IndexPlaceholder<N_index1> - complex<T2>
template<int N_index1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<complex<T2> > ,
      Subtract<int, complex<T2>  > > >
operator-(IndexPlaceholder<N_index1> d1, 
      complex<T2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<complex<T2> > , 
      Subtract<int, complex<T2> > >
      (d1, 
      _bz_ArrayExprConstant<complex<T2> > (d2));
}
#endif // BZ_HAVE_COMPLEX

// int - Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Subtract<int, T_numtype2 > > >
operator-(int d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Subtract<int, T_numtype2> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2.begin());
}

// int - _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>,
      Subtract<int, typename P_expr2::T_numtype > > >
operator-(int d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>, 
      Subtract<int, typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2);
}

// int - IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      IndexPlaceholder<N_index2>,
      Subtract<int, int > > >
operator-(int d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      IndexPlaceholder<N_index2>, 
      Subtract<int, int> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2);
}

// float - Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Subtract<float, T_numtype2 > > >
operator-(float d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Subtract<float, T_numtype2> >
      (_bz_ArrayExprConstant<float>(d1), 
      d2.begin());
}

// float - _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, 
      _bz_ArrayExpr<P_expr2>,
      Subtract<float, typename P_expr2::T_numtype > > >
operator-(float d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, 
      _bz_ArrayExpr<P_expr2>, 
      Subtract<float, typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<float>(d1), 
      d2);
}

// float - IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, 
      IndexPlaceholder<N_index2>,
      Subtract<float, int > > >
operator-(float d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, 
      IndexPlaceholder<N_index2>, 
      Subtract<float, int> >
      (_bz_ArrayExprConstant<float>(d1), 
      d2);
}

// double - Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Subtract<double, T_numtype2 > > >
operator-(double d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Subtract<double, T_numtype2> >
      (_bz_ArrayExprConstant<double>(d1), 
      d2.begin());
}

// double - _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, 
      _bz_ArrayExpr<P_expr2>,
      Subtract<double, typename P_expr2::T_numtype > > >
operator-(double d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, 
      _bz_ArrayExpr<P_expr2>, 
      Subtract<double, typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<double>(d1), 
      d2);
}

// double - IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, 
      IndexPlaceholder<N_index2>,
      Subtract<double, int > > >
operator-(double d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, 
      IndexPlaceholder<N_index2>, 
      Subtract<double, int> >
      (_bz_ArrayExprConstant<double>(d1), 
      d2);
}

// long double - Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Subtract<long double, T_numtype2 > > >
operator-(long double d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Subtract<long double, T_numtype2> >
      (_bz_ArrayExprConstant<long double>(d1), 
      d2.begin());
}

// long double - _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, 
      _bz_ArrayExpr<P_expr2>,
      Subtract<long double, typename P_expr2::T_numtype > > >
operator-(long double d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, 
      _bz_ArrayExpr<P_expr2>, 
      Subtract<long double, typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<long double>(d1), 
      d2);
}

// long double - IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, 
      IndexPlaceholder<N_index2>,
      Subtract<long double, int > > >
operator-(long double d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, 
      IndexPlaceholder<N_index2>, 
      Subtract<long double, int> >
      (_bz_ArrayExprConstant<long double>(d1), 
      d2);
}

#ifdef BZ_HAVE_COMPLEX
// complex<T1> - Array<T_numtype2, N_rank2>
template<class T1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , 
      ArrayIterator<T_numtype2, N_rank2>,
      Subtract<complex<T1> , T_numtype2 > > >
operator-(complex<T1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , 
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
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , 
      _bz_ArrayExpr<P_expr2>,
      Subtract<complex<T1> , typename P_expr2::T_numtype > > >
operator-(complex<T1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , 
      _bz_ArrayExpr<P_expr2>, 
      Subtract<complex<T1> , typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<complex<T1> > (d1), 
      d2);
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX
// complex<T1> - IndexPlaceholder<N_index2>
template<class T1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , 
      IndexPlaceholder<N_index2>,
      Subtract<complex<T1> , int > > >
operator-(complex<T1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , 
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
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Multiply<T_numtype1, T_numtype2 > > >
operator*(const Array<T_numtype1, N_rank1>& d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Multiply<T_numtype1, T_numtype2> >
      (d1.begin(), 
      d2.begin());
}

// Array<T_numtype1, N_rank1> * _bz_ArrayExpr<P_expr2>
template<class T_numtype1, int N_rank1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>,
      Multiply<T_numtype1, typename P_expr2::T_numtype > > >
operator*(const Array<T_numtype1, N_rank1>& d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>, 
      Multiply<T_numtype1, typename P_expr2::T_numtype> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> * IndexPlaceholder<N_index2>
template<class T_numtype1, int N_rank1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>,
      Multiply<T_numtype1, int > > >
operator*(const Array<T_numtype1, N_rank1>& d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>, 
      Multiply<T_numtype1, int> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> * int
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>,
      Multiply<T_numtype1, int > > >
operator*(const Array<T_numtype1, N_rank1>& d1, 
      int d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>, 
      Multiply<T_numtype1, int> >
      (d1.begin(), 
      _bz_ArrayExprConstant<int>(d2));
}

// Array<T_numtype1, N_rank1> * float
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<float>,
      Multiply<T_numtype1, float > > >
operator*(const Array<T_numtype1, N_rank1>& d1, 
      float d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<float>, 
      Multiply<T_numtype1, float> >
      (d1.begin(), 
      _bz_ArrayExprConstant<float>(d2));
}

// Array<T_numtype1, N_rank1> * double
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<double>,
      Multiply<T_numtype1, double > > >
operator*(const Array<T_numtype1, N_rank1>& d1, 
      double d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<double>, 
      Multiply<T_numtype1, double> >
      (d1.begin(), 
      _bz_ArrayExprConstant<double>(d2));
}

// Array<T_numtype1, N_rank1> * long double
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<long double>,
      Multiply<T_numtype1, long double > > >
operator*(const Array<T_numtype1, N_rank1>& d1, 
      long double d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<long double>, 
      Multiply<T_numtype1, long double> >
      (d1.begin(), 
      _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
// Array<T_numtype1, N_rank1> * complex<T2>
template<class T_numtype1, int N_rank1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<complex<T2> > ,
      Multiply<T_numtype1, complex<T2>  > > >
operator*(const Array<T_numtype1, N_rank1>& d1, 
      complex<T2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<complex<T2> > , 
      Multiply<T_numtype1, complex<T2> > >
      (d1.begin(), 
      _bz_ArrayExprConstant<complex<T2> > (d2));
}
#endif // BZ_HAVE_COMPLEX

// _bz_ArrayExpr<P_expr1> * Array<T_numtype2, N_rank2>
template<class P_expr1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Multiply<typename P_expr1::T_numtype, T_numtype2 > > >
operator*(_bz_ArrayExpr<P_expr1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Multiply<typename P_expr1::T_numtype, T_numtype2> >
      (d1, 
      d2.begin());
}

// _bz_ArrayExpr<P_expr1> * _bz_ArrayExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>,
      Multiply<typename P_expr1::T_numtype, typename P_expr2::T_numtype > > >
operator*(_bz_ArrayExpr<P_expr1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>, 
      Multiply<typename P_expr1::T_numtype, typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> * IndexPlaceholder<N_index2>
template<class P_expr1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>,
      Multiply<typename P_expr1::T_numtype, int > > >
operator*(_bz_ArrayExpr<P_expr1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>, 
      Multiply<typename P_expr1::T_numtype, int> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> * int
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>,
      Multiply<typename P_expr1::T_numtype, int > > >
operator*(_bz_ArrayExpr<P_expr1> d1, 
      int d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>, 
      Multiply<typename P_expr1::T_numtype, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// _bz_ArrayExpr<P_expr1> * float
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<float>,
      Multiply<typename P_expr1::T_numtype, float > > >
operator*(_bz_ArrayExpr<P_expr1> d1, 
      float d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<float>, 
      Multiply<typename P_expr1::T_numtype, float> >
      (d1, 
      _bz_ArrayExprConstant<float>(d2));
}

// _bz_ArrayExpr<P_expr1> * double
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<double>,
      Multiply<typename P_expr1::T_numtype, double > > >
operator*(_bz_ArrayExpr<P_expr1> d1, 
      double d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<double>, 
      Multiply<typename P_expr1::T_numtype, double> >
      (d1, 
      _bz_ArrayExprConstant<double>(d2));
}

// _bz_ArrayExpr<P_expr1> * long double
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<long double>,
      Multiply<typename P_expr1::T_numtype, long double > > >
operator*(_bz_ArrayExpr<P_expr1> d1, 
      long double d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<long double>, 
      Multiply<typename P_expr1::T_numtype, long double> >
      (d1, 
      _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
// _bz_ArrayExpr<P_expr1> * complex<T2>
template<class P_expr1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<complex<T2> > ,
      Multiply<typename P_expr1::T_numtype, complex<T2>  > > >
operator*(_bz_ArrayExpr<P_expr1> d1, 
      complex<T2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<complex<T2> > , 
      Multiply<typename P_expr1::T_numtype, complex<T2> > >
      (d1, 
      _bz_ArrayExprConstant<complex<T2> > (d2));
}
#endif // BZ_HAVE_COMPLEX

// IndexPlaceholder<N_index1> * Array<T_numtype2, N_rank2>
template<int N_index1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Multiply<int, T_numtype2 > > >
operator*(IndexPlaceholder<N_index1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Multiply<int, T_numtype2> >
      (d1, 
      d2.begin());
}

// IndexPlaceholder<N_index1> * _bz_ArrayExpr<P_expr2>
template<int N_index1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>,
      Multiply<int, typename P_expr2::T_numtype > > >
operator*(IndexPlaceholder<N_index1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>, 
      Multiply<int, typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> * IndexPlaceholder<N_index2>
template<int N_index1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>,
      Multiply<int, int > > >
operator*(IndexPlaceholder<N_index1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>, 
      Multiply<int, int> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> * int
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>,
      Multiply<int, int > > >
operator*(IndexPlaceholder<N_index1> d1, 
      int d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>, 
      Multiply<int, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// IndexPlaceholder<N_index1> * float
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<float>,
      Multiply<int, float > > >
operator*(IndexPlaceholder<N_index1> d1, 
      float d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<float>, 
      Multiply<int, float> >
      (d1, 
      _bz_ArrayExprConstant<float>(d2));
}

// IndexPlaceholder<N_index1> * double
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<double>,
      Multiply<int, double > > >
operator*(IndexPlaceholder<N_index1> d1, 
      double d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<double>, 
      Multiply<int, double> >
      (d1, 
      _bz_ArrayExprConstant<double>(d2));
}

// IndexPlaceholder<N_index1> * long double
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<long double>,
      Multiply<int, long double > > >
operator*(IndexPlaceholder<N_index1> d1, 
      long double d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<long double>, 
      Multiply<int, long double> >
      (d1, 
      _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
// IndexPlaceholder<N_index1> * complex<T2>
template<int N_index1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<complex<T2> > ,
      Multiply<int, complex<T2>  > > >
operator*(IndexPlaceholder<N_index1> d1, 
      complex<T2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<complex<T2> > , 
      Multiply<int, complex<T2> > >
      (d1, 
      _bz_ArrayExprConstant<complex<T2> > (d2));
}
#endif // BZ_HAVE_COMPLEX

// int * Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Multiply<int, T_numtype2 > > >
operator*(int d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Multiply<int, T_numtype2> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2.begin());
}

// int * _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>,
      Multiply<int, typename P_expr2::T_numtype > > >
operator*(int d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>, 
      Multiply<int, typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2);
}

// int * IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      IndexPlaceholder<N_index2>,
      Multiply<int, int > > >
operator*(int d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      IndexPlaceholder<N_index2>, 
      Multiply<int, int> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2);
}

// float * Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Multiply<float, T_numtype2 > > >
operator*(float d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Multiply<float, T_numtype2> >
      (_bz_ArrayExprConstant<float>(d1), 
      d2.begin());
}

// float * _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, 
      _bz_ArrayExpr<P_expr2>,
      Multiply<float, typename P_expr2::T_numtype > > >
operator*(float d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, 
      _bz_ArrayExpr<P_expr2>, 
      Multiply<float, typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<float>(d1), 
      d2);
}

// float * IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, 
      IndexPlaceholder<N_index2>,
      Multiply<float, int > > >
operator*(float d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, 
      IndexPlaceholder<N_index2>, 
      Multiply<float, int> >
      (_bz_ArrayExprConstant<float>(d1), 
      d2);
}

// double * Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Multiply<double, T_numtype2 > > >
operator*(double d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Multiply<double, T_numtype2> >
      (_bz_ArrayExprConstant<double>(d1), 
      d2.begin());
}

// double * _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, 
      _bz_ArrayExpr<P_expr2>,
      Multiply<double, typename P_expr2::T_numtype > > >
operator*(double d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, 
      _bz_ArrayExpr<P_expr2>, 
      Multiply<double, typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<double>(d1), 
      d2);
}

// double * IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, 
      IndexPlaceholder<N_index2>,
      Multiply<double, int > > >
operator*(double d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, 
      IndexPlaceholder<N_index2>, 
      Multiply<double, int> >
      (_bz_ArrayExprConstant<double>(d1), 
      d2);
}

// long double * Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Multiply<long double, T_numtype2 > > >
operator*(long double d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Multiply<long double, T_numtype2> >
      (_bz_ArrayExprConstant<long double>(d1), 
      d2.begin());
}

// long double * _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, 
      _bz_ArrayExpr<P_expr2>,
      Multiply<long double, typename P_expr2::T_numtype > > >
operator*(long double d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, 
      _bz_ArrayExpr<P_expr2>, 
      Multiply<long double, typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<long double>(d1), 
      d2);
}

// long double * IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, 
      IndexPlaceholder<N_index2>,
      Multiply<long double, int > > >
operator*(long double d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, 
      IndexPlaceholder<N_index2>, 
      Multiply<long double, int> >
      (_bz_ArrayExprConstant<long double>(d1), 
      d2);
}

#ifdef BZ_HAVE_COMPLEX
// complex<T1> * Array<T_numtype2, N_rank2>
template<class T1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , 
      ArrayIterator<T_numtype2, N_rank2>,
      Multiply<complex<T1> , T_numtype2 > > >
operator*(complex<T1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , 
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
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , 
      _bz_ArrayExpr<P_expr2>,
      Multiply<complex<T1> , typename P_expr2::T_numtype > > >
operator*(complex<T1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , 
      _bz_ArrayExpr<P_expr2>, 
      Multiply<complex<T1> , typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<complex<T1> > (d1), 
      d2);
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX
// complex<T1> * IndexPlaceholder<N_index2>
template<class T1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , 
      IndexPlaceholder<N_index2>,
      Multiply<complex<T1> , int > > >
operator*(complex<T1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , 
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
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Divide<T_numtype1, T_numtype2 > > >
operator/(const Array<T_numtype1, N_rank1>& d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Divide<T_numtype1, T_numtype2> >
      (d1.begin(), 
      d2.begin());
}

// Array<T_numtype1, N_rank1> / _bz_ArrayExpr<P_expr2>
template<class T_numtype1, int N_rank1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>,
      Divide<T_numtype1, typename P_expr2::T_numtype > > >
operator/(const Array<T_numtype1, N_rank1>& d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>, 
      Divide<T_numtype1, typename P_expr2::T_numtype> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> / IndexPlaceholder<N_index2>
template<class T_numtype1, int N_rank1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>,
      Divide<T_numtype1, int > > >
operator/(const Array<T_numtype1, N_rank1>& d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>, 
      Divide<T_numtype1, int> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> / int
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>,
      Divide<T_numtype1, int > > >
operator/(const Array<T_numtype1, N_rank1>& d1, 
      int d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>, 
      Divide<T_numtype1, int> >
      (d1.begin(), 
      _bz_ArrayExprConstant<int>(d2));
}

// Array<T_numtype1, N_rank1> / float
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<float>,
      Divide<T_numtype1, float > > >
operator/(const Array<T_numtype1, N_rank1>& d1, 
      float d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<float>, 
      Divide<T_numtype1, float> >
      (d1.begin(), 
      _bz_ArrayExprConstant<float>(d2));
}

// Array<T_numtype1, N_rank1> / double
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<double>,
      Divide<T_numtype1, double > > >
operator/(const Array<T_numtype1, N_rank1>& d1, 
      double d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<double>, 
      Divide<T_numtype1, double> >
      (d1.begin(), 
      _bz_ArrayExprConstant<double>(d2));
}

// Array<T_numtype1, N_rank1> / long double
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<long double>,
      Divide<T_numtype1, long double > > >
operator/(const Array<T_numtype1, N_rank1>& d1, 
      long double d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<long double>, 
      Divide<T_numtype1, long double> >
      (d1.begin(), 
      _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
// Array<T_numtype1, N_rank1> / complex<T2>
template<class T_numtype1, int N_rank1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<complex<T2> > ,
      Divide<T_numtype1, complex<T2>  > > >
operator/(const Array<T_numtype1, N_rank1>& d1, 
      complex<T2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<complex<T2> > , 
      Divide<T_numtype1, complex<T2> > >
      (d1.begin(), 
      _bz_ArrayExprConstant<complex<T2> > (d2));
}
#endif // BZ_HAVE_COMPLEX

// _bz_ArrayExpr<P_expr1> / Array<T_numtype2, N_rank2>
template<class P_expr1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Divide<typename P_expr1::T_numtype, T_numtype2 > > >
operator/(_bz_ArrayExpr<P_expr1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Divide<typename P_expr1::T_numtype, T_numtype2> >
      (d1, 
      d2.begin());
}

// _bz_ArrayExpr<P_expr1> / _bz_ArrayExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>,
      Divide<typename P_expr1::T_numtype, typename P_expr2::T_numtype > > >
operator/(_bz_ArrayExpr<P_expr1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>, 
      Divide<typename P_expr1::T_numtype, typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> / IndexPlaceholder<N_index2>
template<class P_expr1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>,
      Divide<typename P_expr1::T_numtype, int > > >
operator/(_bz_ArrayExpr<P_expr1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>, 
      Divide<typename P_expr1::T_numtype, int> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> / int
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>,
      Divide<typename P_expr1::T_numtype, int > > >
operator/(_bz_ArrayExpr<P_expr1> d1, 
      int d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>, 
      Divide<typename P_expr1::T_numtype, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// _bz_ArrayExpr<P_expr1> / float
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<float>,
      Divide<typename P_expr1::T_numtype, float > > >
operator/(_bz_ArrayExpr<P_expr1> d1, 
      float d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<float>, 
      Divide<typename P_expr1::T_numtype, float> >
      (d1, 
      _bz_ArrayExprConstant<float>(d2));
}

// _bz_ArrayExpr<P_expr1> / double
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<double>,
      Divide<typename P_expr1::T_numtype, double > > >
operator/(_bz_ArrayExpr<P_expr1> d1, 
      double d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<double>, 
      Divide<typename P_expr1::T_numtype, double> >
      (d1, 
      _bz_ArrayExprConstant<double>(d2));
}

// _bz_ArrayExpr<P_expr1> / long double
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<long double>,
      Divide<typename P_expr1::T_numtype, long double > > >
operator/(_bz_ArrayExpr<P_expr1> d1, 
      long double d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<long double>, 
      Divide<typename P_expr1::T_numtype, long double> >
      (d1, 
      _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
// _bz_ArrayExpr<P_expr1> / complex<T2>
template<class P_expr1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<complex<T2> > ,
      Divide<typename P_expr1::T_numtype, complex<T2>  > > >
operator/(_bz_ArrayExpr<P_expr1> d1, 
      complex<T2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<complex<T2> > , 
      Divide<typename P_expr1::T_numtype, complex<T2> > >
      (d1, 
      _bz_ArrayExprConstant<complex<T2> > (d2));
}
#endif // BZ_HAVE_COMPLEX

// IndexPlaceholder<N_index1> / Array<T_numtype2, N_rank2>
template<int N_index1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Divide<int, T_numtype2 > > >
operator/(IndexPlaceholder<N_index1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Divide<int, T_numtype2> >
      (d1, 
      d2.begin());
}

// IndexPlaceholder<N_index1> / _bz_ArrayExpr<P_expr2>
template<int N_index1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>,
      Divide<int, typename P_expr2::T_numtype > > >
operator/(IndexPlaceholder<N_index1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>, 
      Divide<int, typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> / IndexPlaceholder<N_index2>
template<int N_index1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>,
      Divide<int, int > > >
operator/(IndexPlaceholder<N_index1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>, 
      Divide<int, int> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> / int
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>,
      Divide<int, int > > >
operator/(IndexPlaceholder<N_index1> d1, 
      int d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>, 
      Divide<int, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// IndexPlaceholder<N_index1> / float
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<float>,
      Divide<int, float > > >
operator/(IndexPlaceholder<N_index1> d1, 
      float d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<float>, 
      Divide<int, float> >
      (d1, 
      _bz_ArrayExprConstant<float>(d2));
}

// IndexPlaceholder<N_index1> / double
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<double>,
      Divide<int, double > > >
operator/(IndexPlaceholder<N_index1> d1, 
      double d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<double>, 
      Divide<int, double> >
      (d1, 
      _bz_ArrayExprConstant<double>(d2));
}

// IndexPlaceholder<N_index1> / long double
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<long double>,
      Divide<int, long double > > >
operator/(IndexPlaceholder<N_index1> d1, 
      long double d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<long double>, 
      Divide<int, long double> >
      (d1, 
      _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
// IndexPlaceholder<N_index1> / complex<T2>
template<int N_index1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<complex<T2> > ,
      Divide<int, complex<T2>  > > >
operator/(IndexPlaceholder<N_index1> d1, 
      complex<T2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<complex<T2> > , 
      Divide<int, complex<T2> > >
      (d1, 
      _bz_ArrayExprConstant<complex<T2> > (d2));
}
#endif // BZ_HAVE_COMPLEX

// int / Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Divide<int, T_numtype2 > > >
operator/(int d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Divide<int, T_numtype2> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2.begin());
}

// int / _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>,
      Divide<int, typename P_expr2::T_numtype > > >
operator/(int d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>, 
      Divide<int, typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2);
}

// int / IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      IndexPlaceholder<N_index2>,
      Divide<int, int > > >
operator/(int d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      IndexPlaceholder<N_index2>, 
      Divide<int, int> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2);
}

// float / Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Divide<float, T_numtype2 > > >
operator/(float d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Divide<float, T_numtype2> >
      (_bz_ArrayExprConstant<float>(d1), 
      d2.begin());
}

// float / _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, 
      _bz_ArrayExpr<P_expr2>,
      Divide<float, typename P_expr2::T_numtype > > >
operator/(float d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, 
      _bz_ArrayExpr<P_expr2>, 
      Divide<float, typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<float>(d1), 
      d2);
}

// float / IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, 
      IndexPlaceholder<N_index2>,
      Divide<float, int > > >
operator/(float d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, 
      IndexPlaceholder<N_index2>, 
      Divide<float, int> >
      (_bz_ArrayExprConstant<float>(d1), 
      d2);
}

// double / Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Divide<double, T_numtype2 > > >
operator/(double d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Divide<double, T_numtype2> >
      (_bz_ArrayExprConstant<double>(d1), 
      d2.begin());
}

// double / _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, 
      _bz_ArrayExpr<P_expr2>,
      Divide<double, typename P_expr2::T_numtype > > >
operator/(double d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, 
      _bz_ArrayExpr<P_expr2>, 
      Divide<double, typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<double>(d1), 
      d2);
}

// double / IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, 
      IndexPlaceholder<N_index2>,
      Divide<double, int > > >
operator/(double d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, 
      IndexPlaceholder<N_index2>, 
      Divide<double, int> >
      (_bz_ArrayExprConstant<double>(d1), 
      d2);
}

// long double / Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Divide<long double, T_numtype2 > > >
operator/(long double d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Divide<long double, T_numtype2> >
      (_bz_ArrayExprConstant<long double>(d1), 
      d2.begin());
}

// long double / _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, 
      _bz_ArrayExpr<P_expr2>,
      Divide<long double, typename P_expr2::T_numtype > > >
operator/(long double d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, 
      _bz_ArrayExpr<P_expr2>, 
      Divide<long double, typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<long double>(d1), 
      d2);
}

// long double / IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, 
      IndexPlaceholder<N_index2>,
      Divide<long double, int > > >
operator/(long double d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, 
      IndexPlaceholder<N_index2>, 
      Divide<long double, int> >
      (_bz_ArrayExprConstant<long double>(d1), 
      d2);
}

#ifdef BZ_HAVE_COMPLEX
// complex<T1> / Array<T_numtype2, N_rank2>
template<class T1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , 
      ArrayIterator<T_numtype2, N_rank2>,
      Divide<complex<T1> , T_numtype2 > > >
operator/(complex<T1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , 
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
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , 
      _bz_ArrayExpr<P_expr2>,
      Divide<complex<T1> , typename P_expr2::T_numtype > > >
operator/(complex<T1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , 
      _bz_ArrayExpr<P_expr2>, 
      Divide<complex<T1> , typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<complex<T1> > (d1), 
      d2);
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX
// complex<T1> / IndexPlaceholder<N_index2>
template<class T1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , 
      IndexPlaceholder<N_index2>,
      Divide<complex<T1> , int > > >
operator/(complex<T1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , 
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
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Modulo<T_numtype1, T_numtype2 > > >
operator%(const Array<T_numtype1, N_rank1>& d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Modulo<T_numtype1, T_numtype2> >
      (d1.begin(), 
      d2.begin());
}

// Array<T_numtype1, N_rank1> % _bz_ArrayExpr<P_expr2>
template<class T_numtype1, int N_rank1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>,
      Modulo<T_numtype1, typename P_expr2::T_numtype > > >
operator%(const Array<T_numtype1, N_rank1>& d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>, 
      Modulo<T_numtype1, typename P_expr2::T_numtype> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> % IndexPlaceholder<N_index2>
template<class T_numtype1, int N_rank1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>,
      Modulo<T_numtype1, int > > >
operator%(const Array<T_numtype1, N_rank1>& d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>, 
      Modulo<T_numtype1, int> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> % int
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>,
      Modulo<T_numtype1, int > > >
operator%(const Array<T_numtype1, N_rank1>& d1, 
      int d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>, 
      Modulo<T_numtype1, int> >
      (d1.begin(), 
      _bz_ArrayExprConstant<int>(d2));
}

// _bz_ArrayExpr<P_expr1> % Array<T_numtype2, N_rank2>
template<class P_expr1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Modulo<typename P_expr1::T_numtype, T_numtype2 > > >
operator%(_bz_ArrayExpr<P_expr1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Modulo<typename P_expr1::T_numtype, T_numtype2> >
      (d1, 
      d2.begin());
}

// _bz_ArrayExpr<P_expr1> % _bz_ArrayExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>,
      Modulo<typename P_expr1::T_numtype, typename P_expr2::T_numtype > > >
operator%(_bz_ArrayExpr<P_expr1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>, 
      Modulo<typename P_expr1::T_numtype, typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> % IndexPlaceholder<N_index2>
template<class P_expr1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>,
      Modulo<typename P_expr1::T_numtype, int > > >
operator%(_bz_ArrayExpr<P_expr1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>, 
      Modulo<typename P_expr1::T_numtype, int> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> % int
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>,
      Modulo<typename P_expr1::T_numtype, int > > >
operator%(_bz_ArrayExpr<P_expr1> d1, 
      int d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>, 
      Modulo<typename P_expr1::T_numtype, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// IndexPlaceholder<N_index1> % Array<T_numtype2, N_rank2>
template<int N_index1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Modulo<int, T_numtype2 > > >
operator%(IndexPlaceholder<N_index1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Modulo<int, T_numtype2> >
      (d1, 
      d2.begin());
}

// IndexPlaceholder<N_index1> % _bz_ArrayExpr<P_expr2>
template<int N_index1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>,
      Modulo<int, typename P_expr2::T_numtype > > >
operator%(IndexPlaceholder<N_index1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>, 
      Modulo<int, typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> % IndexPlaceholder<N_index2>
template<int N_index1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>,
      Modulo<int, int > > >
operator%(IndexPlaceholder<N_index1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>, 
      Modulo<int, int> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> % int
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>,
      Modulo<int, int > > >
operator%(IndexPlaceholder<N_index1> d1, 
      int d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>, 
      Modulo<int, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// int % Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Modulo<int, T_numtype2 > > >
operator%(int d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Modulo<int, T_numtype2> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2.begin());
}

// int % _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>,
      Modulo<int, typename P_expr2::T_numtype > > >
operator%(int d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>, 
      Modulo<int, typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2);
}

// int % IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      IndexPlaceholder<N_index2>,
      Modulo<int, int > > >
operator%(int d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
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
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Greater<T_numtype1, T_numtype2 > > >
operator>(const Array<T_numtype1, N_rank1>& d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Greater<T_numtype1, T_numtype2> >
      (d1.begin(), 
      d2.begin());
}

// Array<T_numtype1, N_rank1> > _bz_ArrayExpr<P_expr2>
template<class T_numtype1, int N_rank1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>,
      Greater<T_numtype1, typename P_expr2::T_numtype > > >
operator>(const Array<T_numtype1, N_rank1>& d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>, 
      Greater<T_numtype1, typename P_expr2::T_numtype> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> > IndexPlaceholder<N_index2>
template<class T_numtype1, int N_rank1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>,
      Greater<T_numtype1, int > > >
operator>(const Array<T_numtype1, N_rank1>& d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>, 
      Greater<T_numtype1, int> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> > int
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>,
      Greater<T_numtype1, int > > >
operator>(const Array<T_numtype1, N_rank1>& d1, 
      int d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>, 
      Greater<T_numtype1, int> >
      (d1.begin(), 
      _bz_ArrayExprConstant<int>(d2));
}

// Array<T_numtype1, N_rank1> > float
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<float>,
      Greater<T_numtype1, float > > >
operator>(const Array<T_numtype1, N_rank1>& d1, 
      float d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<float>, 
      Greater<T_numtype1, float> >
      (d1.begin(), 
      _bz_ArrayExprConstant<float>(d2));
}

// Array<T_numtype1, N_rank1> > double
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<double>,
      Greater<T_numtype1, double > > >
operator>(const Array<T_numtype1, N_rank1>& d1, 
      double d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<double>, 
      Greater<T_numtype1, double> >
      (d1.begin(), 
      _bz_ArrayExprConstant<double>(d2));
}

// Array<T_numtype1, N_rank1> > long double
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<long double>,
      Greater<T_numtype1, long double > > >
operator>(const Array<T_numtype1, N_rank1>& d1, 
      long double d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<long double>, 
      Greater<T_numtype1, long double> >
      (d1.begin(), 
      _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
// Array<T_numtype1, N_rank1> > complex<T2>
template<class T_numtype1, int N_rank1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<complex<T2> > ,
      Greater<T_numtype1, complex<T2>  > > >
operator>(const Array<T_numtype1, N_rank1>& d1, 
      complex<T2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<complex<T2> > , 
      Greater<T_numtype1, complex<T2> > >
      (d1.begin(), 
      _bz_ArrayExprConstant<complex<T2> > (d2));
}
#endif // BZ_HAVE_COMPLEX

// _bz_ArrayExpr<P_expr1> > Array<T_numtype2, N_rank2>
template<class P_expr1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Greater<typename P_expr1::T_numtype, T_numtype2 > > >
operator>(_bz_ArrayExpr<P_expr1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Greater<typename P_expr1::T_numtype, T_numtype2> >
      (d1, 
      d2.begin());
}

// _bz_ArrayExpr<P_expr1> > _bz_ArrayExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>,
      Greater<typename P_expr1::T_numtype, typename P_expr2::T_numtype > > >
operator>(_bz_ArrayExpr<P_expr1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>, 
      Greater<typename P_expr1::T_numtype, typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> > IndexPlaceholder<N_index2>
template<class P_expr1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>,
      Greater<typename P_expr1::T_numtype, int > > >
operator>(_bz_ArrayExpr<P_expr1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>, 
      Greater<typename P_expr1::T_numtype, int> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> > int
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>,
      Greater<typename P_expr1::T_numtype, int > > >
operator>(_bz_ArrayExpr<P_expr1> d1, 
      int d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>, 
      Greater<typename P_expr1::T_numtype, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// _bz_ArrayExpr<P_expr1> > float
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<float>,
      Greater<typename P_expr1::T_numtype, float > > >
operator>(_bz_ArrayExpr<P_expr1> d1, 
      float d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<float>, 
      Greater<typename P_expr1::T_numtype, float> >
      (d1, 
      _bz_ArrayExprConstant<float>(d2));
}

// _bz_ArrayExpr<P_expr1> > double
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<double>,
      Greater<typename P_expr1::T_numtype, double > > >
operator>(_bz_ArrayExpr<P_expr1> d1, 
      double d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<double>, 
      Greater<typename P_expr1::T_numtype, double> >
      (d1, 
      _bz_ArrayExprConstant<double>(d2));
}

// _bz_ArrayExpr<P_expr1> > long double
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<long double>,
      Greater<typename P_expr1::T_numtype, long double > > >
operator>(_bz_ArrayExpr<P_expr1> d1, 
      long double d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<long double>, 
      Greater<typename P_expr1::T_numtype, long double> >
      (d1, 
      _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
// _bz_ArrayExpr<P_expr1> > complex<T2>
template<class P_expr1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<complex<T2> > ,
      Greater<typename P_expr1::T_numtype, complex<T2>  > > >
operator>(_bz_ArrayExpr<P_expr1> d1, 
      complex<T2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<complex<T2> > , 
      Greater<typename P_expr1::T_numtype, complex<T2> > >
      (d1, 
      _bz_ArrayExprConstant<complex<T2> > (d2));
}
#endif // BZ_HAVE_COMPLEX

// IndexPlaceholder<N_index1> > Array<T_numtype2, N_rank2>
template<int N_index1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Greater<int, T_numtype2 > > >
operator>(IndexPlaceholder<N_index1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Greater<int, T_numtype2> >
      (d1, 
      d2.begin());
}

// IndexPlaceholder<N_index1> > _bz_ArrayExpr<P_expr2>
template<int N_index1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>,
      Greater<int, typename P_expr2::T_numtype > > >
operator>(IndexPlaceholder<N_index1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>, 
      Greater<int, typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> > IndexPlaceholder<N_index2>
template<int N_index1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>,
      Greater<int, int > > >
operator>(IndexPlaceholder<N_index1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>, 
      Greater<int, int> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> > int
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>,
      Greater<int, int > > >
operator>(IndexPlaceholder<N_index1> d1, 
      int d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>, 
      Greater<int, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// IndexPlaceholder<N_index1> > float
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<float>,
      Greater<int, float > > >
operator>(IndexPlaceholder<N_index1> d1, 
      float d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<float>, 
      Greater<int, float> >
      (d1, 
      _bz_ArrayExprConstant<float>(d2));
}

// IndexPlaceholder<N_index1> > double
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<double>,
      Greater<int, double > > >
operator>(IndexPlaceholder<N_index1> d1, 
      double d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<double>, 
      Greater<int, double> >
      (d1, 
      _bz_ArrayExprConstant<double>(d2));
}

// IndexPlaceholder<N_index1> > long double
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<long double>,
      Greater<int, long double > > >
operator>(IndexPlaceholder<N_index1> d1, 
      long double d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<long double>, 
      Greater<int, long double> >
      (d1, 
      _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
// IndexPlaceholder<N_index1> > complex<T2>
template<int N_index1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<complex<T2> > ,
      Greater<int, complex<T2>  > > >
operator>(IndexPlaceholder<N_index1> d1, 
      complex<T2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<complex<T2> > , 
      Greater<int, complex<T2> > >
      (d1, 
      _bz_ArrayExprConstant<complex<T2> > (d2));
}
#endif // BZ_HAVE_COMPLEX

// int > Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Greater<int, T_numtype2 > > >
operator>(int d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Greater<int, T_numtype2> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2.begin());
}

// int > _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>,
      Greater<int, typename P_expr2::T_numtype > > >
operator>(int d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>, 
      Greater<int, typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2);
}

// int > IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      IndexPlaceholder<N_index2>,
      Greater<int, int > > >
operator>(int d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      IndexPlaceholder<N_index2>, 
      Greater<int, int> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2);
}

// float > Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Greater<float, T_numtype2 > > >
operator>(float d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Greater<float, T_numtype2> >
      (_bz_ArrayExprConstant<float>(d1), 
      d2.begin());
}

// float > _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, 
      _bz_ArrayExpr<P_expr2>,
      Greater<float, typename P_expr2::T_numtype > > >
operator>(float d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, 
      _bz_ArrayExpr<P_expr2>, 
      Greater<float, typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<float>(d1), 
      d2);
}

// float > IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, 
      IndexPlaceholder<N_index2>,
      Greater<float, int > > >
operator>(float d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, 
      IndexPlaceholder<N_index2>, 
      Greater<float, int> >
      (_bz_ArrayExprConstant<float>(d1), 
      d2);
}

// double > Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Greater<double, T_numtype2 > > >
operator>(double d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Greater<double, T_numtype2> >
      (_bz_ArrayExprConstant<double>(d1), 
      d2.begin());
}

// double > _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, 
      _bz_ArrayExpr<P_expr2>,
      Greater<double, typename P_expr2::T_numtype > > >
operator>(double d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, 
      _bz_ArrayExpr<P_expr2>, 
      Greater<double, typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<double>(d1), 
      d2);
}

// double > IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, 
      IndexPlaceholder<N_index2>,
      Greater<double, int > > >
operator>(double d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, 
      IndexPlaceholder<N_index2>, 
      Greater<double, int> >
      (_bz_ArrayExprConstant<double>(d1), 
      d2);
}

// long double > Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Greater<long double, T_numtype2 > > >
operator>(long double d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Greater<long double, T_numtype2> >
      (_bz_ArrayExprConstant<long double>(d1), 
      d2.begin());
}

// long double > _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, 
      _bz_ArrayExpr<P_expr2>,
      Greater<long double, typename P_expr2::T_numtype > > >
operator>(long double d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, 
      _bz_ArrayExpr<P_expr2>, 
      Greater<long double, typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<long double>(d1), 
      d2);
}

// long double > IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, 
      IndexPlaceholder<N_index2>,
      Greater<long double, int > > >
operator>(long double d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, 
      IndexPlaceholder<N_index2>, 
      Greater<long double, int> >
      (_bz_ArrayExprConstant<long double>(d1), 
      d2);
}

#ifdef BZ_HAVE_COMPLEX
// complex<T1> > Array<T_numtype2, N_rank2>
template<class T1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , 
      ArrayIterator<T_numtype2, N_rank2>,
      Greater<complex<T1> , T_numtype2 > > >
operator>(complex<T1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , 
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
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , 
      _bz_ArrayExpr<P_expr2>,
      Greater<complex<T1> , typename P_expr2::T_numtype > > >
operator>(complex<T1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , 
      _bz_ArrayExpr<P_expr2>, 
      Greater<complex<T1> , typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<complex<T1> > (d1), 
      d2);
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX
// complex<T1> > IndexPlaceholder<N_index2>
template<class T1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , 
      IndexPlaceholder<N_index2>,
      Greater<complex<T1> , int > > >
operator>(complex<T1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , 
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
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Less<T_numtype1, T_numtype2 > > >
operator<(const Array<T_numtype1, N_rank1>& d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Less<T_numtype1, T_numtype2> >
      (d1.begin(), 
      d2.begin());
}

// Array<T_numtype1, N_rank1> < _bz_ArrayExpr<P_expr2>
template<class T_numtype1, int N_rank1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>,
      Less<T_numtype1, typename P_expr2::T_numtype > > >
operator<(const Array<T_numtype1, N_rank1>& d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>, 
      Less<T_numtype1, typename P_expr2::T_numtype> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> < IndexPlaceholder<N_index2>
template<class T_numtype1, int N_rank1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>,
      Less<T_numtype1, int > > >
operator<(const Array<T_numtype1, N_rank1>& d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>, 
      Less<T_numtype1, int> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> < int
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>,
      Less<T_numtype1, int > > >
operator<(const Array<T_numtype1, N_rank1>& d1, 
      int d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>, 
      Less<T_numtype1, int> >
      (d1.begin(), 
      _bz_ArrayExprConstant<int>(d2));
}

// Array<T_numtype1, N_rank1> < float
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<float>,
      Less<T_numtype1, float > > >
operator<(const Array<T_numtype1, N_rank1>& d1, 
      float d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<float>, 
      Less<T_numtype1, float> >
      (d1.begin(), 
      _bz_ArrayExprConstant<float>(d2));
}

// Array<T_numtype1, N_rank1> < double
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<double>,
      Less<T_numtype1, double > > >
operator<(const Array<T_numtype1, N_rank1>& d1, 
      double d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<double>, 
      Less<T_numtype1, double> >
      (d1.begin(), 
      _bz_ArrayExprConstant<double>(d2));
}

// Array<T_numtype1, N_rank1> < long double
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<long double>,
      Less<T_numtype1, long double > > >
operator<(const Array<T_numtype1, N_rank1>& d1, 
      long double d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<long double>, 
      Less<T_numtype1, long double> >
      (d1.begin(), 
      _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
// Array<T_numtype1, N_rank1> < complex<T2>
template<class T_numtype1, int N_rank1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<complex<T2> > ,
      Less<T_numtype1, complex<T2>  > > >
operator<(const Array<T_numtype1, N_rank1>& d1, 
      complex<T2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<complex<T2> > , 
      Less<T_numtype1, complex<T2> > >
      (d1.begin(), 
      _bz_ArrayExprConstant<complex<T2> > (d2));
}
#endif // BZ_HAVE_COMPLEX

// _bz_ArrayExpr<P_expr1> < Array<T_numtype2, N_rank2>
template<class P_expr1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Less<typename P_expr1::T_numtype, T_numtype2 > > >
operator<(_bz_ArrayExpr<P_expr1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Less<typename P_expr1::T_numtype, T_numtype2> >
      (d1, 
      d2.begin());
}

// _bz_ArrayExpr<P_expr1> < _bz_ArrayExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>,
      Less<typename P_expr1::T_numtype, typename P_expr2::T_numtype > > >
operator<(_bz_ArrayExpr<P_expr1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>, 
      Less<typename P_expr1::T_numtype, typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> < IndexPlaceholder<N_index2>
template<class P_expr1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>,
      Less<typename P_expr1::T_numtype, int > > >
operator<(_bz_ArrayExpr<P_expr1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>, 
      Less<typename P_expr1::T_numtype, int> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> < int
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>,
      Less<typename P_expr1::T_numtype, int > > >
operator<(_bz_ArrayExpr<P_expr1> d1, 
      int d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>, 
      Less<typename P_expr1::T_numtype, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// _bz_ArrayExpr<P_expr1> < float
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<float>,
      Less<typename P_expr1::T_numtype, float > > >
operator<(_bz_ArrayExpr<P_expr1> d1, 
      float d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<float>, 
      Less<typename P_expr1::T_numtype, float> >
      (d1, 
      _bz_ArrayExprConstant<float>(d2));
}

// _bz_ArrayExpr<P_expr1> < double
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<double>,
      Less<typename P_expr1::T_numtype, double > > >
operator<(_bz_ArrayExpr<P_expr1> d1, 
      double d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<double>, 
      Less<typename P_expr1::T_numtype, double> >
      (d1, 
      _bz_ArrayExprConstant<double>(d2));
}

// _bz_ArrayExpr<P_expr1> < long double
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<long double>,
      Less<typename P_expr1::T_numtype, long double > > >
operator<(_bz_ArrayExpr<P_expr1> d1, 
      long double d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<long double>, 
      Less<typename P_expr1::T_numtype, long double> >
      (d1, 
      _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
// _bz_ArrayExpr<P_expr1> < complex<T2>
template<class P_expr1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<complex<T2> > ,
      Less<typename P_expr1::T_numtype, complex<T2>  > > >
operator<(_bz_ArrayExpr<P_expr1> d1, 
      complex<T2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<complex<T2> > , 
      Less<typename P_expr1::T_numtype, complex<T2> > >
      (d1, 
      _bz_ArrayExprConstant<complex<T2> > (d2));
}
#endif // BZ_HAVE_COMPLEX

// IndexPlaceholder<N_index1> < Array<T_numtype2, N_rank2>
template<int N_index1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Less<int, T_numtype2 > > >
operator<(IndexPlaceholder<N_index1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Less<int, T_numtype2> >
      (d1, 
      d2.begin());
}

// IndexPlaceholder<N_index1> < _bz_ArrayExpr<P_expr2>
template<int N_index1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>,
      Less<int, typename P_expr2::T_numtype > > >
operator<(IndexPlaceholder<N_index1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>, 
      Less<int, typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> < IndexPlaceholder<N_index2>
template<int N_index1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>,
      Less<int, int > > >
operator<(IndexPlaceholder<N_index1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>, 
      Less<int, int> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> < int
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>,
      Less<int, int > > >
operator<(IndexPlaceholder<N_index1> d1, 
      int d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>, 
      Less<int, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// IndexPlaceholder<N_index1> < float
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<float>,
      Less<int, float > > >
operator<(IndexPlaceholder<N_index1> d1, 
      float d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<float>, 
      Less<int, float> >
      (d1, 
      _bz_ArrayExprConstant<float>(d2));
}

// IndexPlaceholder<N_index1> < double
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<double>,
      Less<int, double > > >
operator<(IndexPlaceholder<N_index1> d1, 
      double d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<double>, 
      Less<int, double> >
      (d1, 
      _bz_ArrayExprConstant<double>(d2));
}

// IndexPlaceholder<N_index1> < long double
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<long double>,
      Less<int, long double > > >
operator<(IndexPlaceholder<N_index1> d1, 
      long double d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<long double>, 
      Less<int, long double> >
      (d1, 
      _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
// IndexPlaceholder<N_index1> < complex<T2>
template<int N_index1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<complex<T2> > ,
      Less<int, complex<T2>  > > >
operator<(IndexPlaceholder<N_index1> d1, 
      complex<T2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<complex<T2> > , 
      Less<int, complex<T2> > >
      (d1, 
      _bz_ArrayExprConstant<complex<T2> > (d2));
}
#endif // BZ_HAVE_COMPLEX

// int < Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Less<int, T_numtype2 > > >
operator<(int d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Less<int, T_numtype2> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2.begin());
}

// int < _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>,
      Less<int, typename P_expr2::T_numtype > > >
operator<(int d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>, 
      Less<int, typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2);
}

// int < IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      IndexPlaceholder<N_index2>,
      Less<int, int > > >
operator<(int d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      IndexPlaceholder<N_index2>, 
      Less<int, int> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2);
}

// float < Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Less<float, T_numtype2 > > >
operator<(float d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Less<float, T_numtype2> >
      (_bz_ArrayExprConstant<float>(d1), 
      d2.begin());
}

// float < _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, 
      _bz_ArrayExpr<P_expr2>,
      Less<float, typename P_expr2::T_numtype > > >
operator<(float d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, 
      _bz_ArrayExpr<P_expr2>, 
      Less<float, typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<float>(d1), 
      d2);
}

// float < IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, 
      IndexPlaceholder<N_index2>,
      Less<float, int > > >
operator<(float d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, 
      IndexPlaceholder<N_index2>, 
      Less<float, int> >
      (_bz_ArrayExprConstant<float>(d1), 
      d2);
}

// double < Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Less<double, T_numtype2 > > >
operator<(double d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Less<double, T_numtype2> >
      (_bz_ArrayExprConstant<double>(d1), 
      d2.begin());
}

// double < _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, 
      _bz_ArrayExpr<P_expr2>,
      Less<double, typename P_expr2::T_numtype > > >
operator<(double d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, 
      _bz_ArrayExpr<P_expr2>, 
      Less<double, typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<double>(d1), 
      d2);
}

// double < IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, 
      IndexPlaceholder<N_index2>,
      Less<double, int > > >
operator<(double d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, 
      IndexPlaceholder<N_index2>, 
      Less<double, int> >
      (_bz_ArrayExprConstant<double>(d1), 
      d2);
}

// long double < Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Less<long double, T_numtype2 > > >
operator<(long double d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Less<long double, T_numtype2> >
      (_bz_ArrayExprConstant<long double>(d1), 
      d2.begin());
}

// long double < _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, 
      _bz_ArrayExpr<P_expr2>,
      Less<long double, typename P_expr2::T_numtype > > >
operator<(long double d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, 
      _bz_ArrayExpr<P_expr2>, 
      Less<long double, typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<long double>(d1), 
      d2);
}

// long double < IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, 
      IndexPlaceholder<N_index2>,
      Less<long double, int > > >
operator<(long double d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, 
      IndexPlaceholder<N_index2>, 
      Less<long double, int> >
      (_bz_ArrayExprConstant<long double>(d1), 
      d2);
}

#ifdef BZ_HAVE_COMPLEX
// complex<T1> < Array<T_numtype2, N_rank2>
template<class T1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , 
      ArrayIterator<T_numtype2, N_rank2>,
      Less<complex<T1> , T_numtype2 > > >
operator<(complex<T1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , 
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
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , 
      _bz_ArrayExpr<P_expr2>,
      Less<complex<T1> , typename P_expr2::T_numtype > > >
operator<(complex<T1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , 
      _bz_ArrayExpr<P_expr2>, 
      Less<complex<T1> , typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<complex<T1> > (d1), 
      d2);
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX
// complex<T1> < IndexPlaceholder<N_index2>
template<class T1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , 
      IndexPlaceholder<N_index2>,
      Less<complex<T1> , int > > >
operator<(complex<T1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , 
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
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      GreaterOrEqual<T_numtype1, T_numtype2 > > >
operator>=(const Array<T_numtype1, N_rank1>& d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      GreaterOrEqual<T_numtype1, T_numtype2> >
      (d1.begin(), 
      d2.begin());
}

// Array<T_numtype1, N_rank1> >= _bz_ArrayExpr<P_expr2>
template<class T_numtype1, int N_rank1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>,
      GreaterOrEqual<T_numtype1, typename P_expr2::T_numtype > > >
operator>=(const Array<T_numtype1, N_rank1>& d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>, 
      GreaterOrEqual<T_numtype1, typename P_expr2::T_numtype> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> >= IndexPlaceholder<N_index2>
template<class T_numtype1, int N_rank1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>,
      GreaterOrEqual<T_numtype1, int > > >
operator>=(const Array<T_numtype1, N_rank1>& d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>, 
      GreaterOrEqual<T_numtype1, int> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> >= int
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>,
      GreaterOrEqual<T_numtype1, int > > >
operator>=(const Array<T_numtype1, N_rank1>& d1, 
      int d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>, 
      GreaterOrEqual<T_numtype1, int> >
      (d1.begin(), 
      _bz_ArrayExprConstant<int>(d2));
}

// Array<T_numtype1, N_rank1> >= float
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<float>,
      GreaterOrEqual<T_numtype1, float > > >
operator>=(const Array<T_numtype1, N_rank1>& d1, 
      float d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<float>, 
      GreaterOrEqual<T_numtype1, float> >
      (d1.begin(), 
      _bz_ArrayExprConstant<float>(d2));
}

// Array<T_numtype1, N_rank1> >= double
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<double>,
      GreaterOrEqual<T_numtype1, double > > >
operator>=(const Array<T_numtype1, N_rank1>& d1, 
      double d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<double>, 
      GreaterOrEqual<T_numtype1, double> >
      (d1.begin(), 
      _bz_ArrayExprConstant<double>(d2));
}

// Array<T_numtype1, N_rank1> >= long double
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<long double>,
      GreaterOrEqual<T_numtype1, long double > > >
operator>=(const Array<T_numtype1, N_rank1>& d1, 
      long double d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<long double>, 
      GreaterOrEqual<T_numtype1, long double> >
      (d1.begin(), 
      _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
// Array<T_numtype1, N_rank1> >= complex<T2>
template<class T_numtype1, int N_rank1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<complex<T2> > ,
      GreaterOrEqual<T_numtype1, complex<T2>  > > >
operator>=(const Array<T_numtype1, N_rank1>& d1, 
      complex<T2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<complex<T2> > , 
      GreaterOrEqual<T_numtype1, complex<T2> > >
      (d1.begin(), 
      _bz_ArrayExprConstant<complex<T2> > (d2));
}
#endif // BZ_HAVE_COMPLEX

// _bz_ArrayExpr<P_expr1> >= Array<T_numtype2, N_rank2>
template<class P_expr1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      GreaterOrEqual<typename P_expr1::T_numtype, T_numtype2 > > >
operator>=(_bz_ArrayExpr<P_expr1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      GreaterOrEqual<typename P_expr1::T_numtype, T_numtype2> >
      (d1, 
      d2.begin());
}

// _bz_ArrayExpr<P_expr1> >= _bz_ArrayExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>,
      GreaterOrEqual<typename P_expr1::T_numtype, typename P_expr2::T_numtype > > >
operator>=(_bz_ArrayExpr<P_expr1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>, 
      GreaterOrEqual<typename P_expr1::T_numtype, typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> >= IndexPlaceholder<N_index2>
template<class P_expr1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>,
      GreaterOrEqual<typename P_expr1::T_numtype, int > > >
operator>=(_bz_ArrayExpr<P_expr1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>, 
      GreaterOrEqual<typename P_expr1::T_numtype, int> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> >= int
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>,
      GreaterOrEqual<typename P_expr1::T_numtype, int > > >
operator>=(_bz_ArrayExpr<P_expr1> d1, 
      int d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>, 
      GreaterOrEqual<typename P_expr1::T_numtype, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// _bz_ArrayExpr<P_expr1> >= float
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<float>,
      GreaterOrEqual<typename P_expr1::T_numtype, float > > >
operator>=(_bz_ArrayExpr<P_expr1> d1, 
      float d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<float>, 
      GreaterOrEqual<typename P_expr1::T_numtype, float> >
      (d1, 
      _bz_ArrayExprConstant<float>(d2));
}

// _bz_ArrayExpr<P_expr1> >= double
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<double>,
      GreaterOrEqual<typename P_expr1::T_numtype, double > > >
operator>=(_bz_ArrayExpr<P_expr1> d1, 
      double d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<double>, 
      GreaterOrEqual<typename P_expr1::T_numtype, double> >
      (d1, 
      _bz_ArrayExprConstant<double>(d2));
}

// _bz_ArrayExpr<P_expr1> >= long double
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<long double>,
      GreaterOrEqual<typename P_expr1::T_numtype, long double > > >
operator>=(_bz_ArrayExpr<P_expr1> d1, 
      long double d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<long double>, 
      GreaterOrEqual<typename P_expr1::T_numtype, long double> >
      (d1, 
      _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
// _bz_ArrayExpr<P_expr1> >= complex<T2>
template<class P_expr1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<complex<T2> > ,
      GreaterOrEqual<typename P_expr1::T_numtype, complex<T2>  > > >
operator>=(_bz_ArrayExpr<P_expr1> d1, 
      complex<T2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<complex<T2> > , 
      GreaterOrEqual<typename P_expr1::T_numtype, complex<T2> > >
      (d1, 
      _bz_ArrayExprConstant<complex<T2> > (d2));
}
#endif // BZ_HAVE_COMPLEX

// IndexPlaceholder<N_index1> >= Array<T_numtype2, N_rank2>
template<int N_index1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      GreaterOrEqual<int, T_numtype2 > > >
operator>=(IndexPlaceholder<N_index1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      GreaterOrEqual<int, T_numtype2> >
      (d1, 
      d2.begin());
}

// IndexPlaceholder<N_index1> >= _bz_ArrayExpr<P_expr2>
template<int N_index1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>,
      GreaterOrEqual<int, typename P_expr2::T_numtype > > >
operator>=(IndexPlaceholder<N_index1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>, 
      GreaterOrEqual<int, typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> >= IndexPlaceholder<N_index2>
template<int N_index1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>,
      GreaterOrEqual<int, int > > >
operator>=(IndexPlaceholder<N_index1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>, 
      GreaterOrEqual<int, int> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> >= int
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>,
      GreaterOrEqual<int, int > > >
operator>=(IndexPlaceholder<N_index1> d1, 
      int d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>, 
      GreaterOrEqual<int, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// IndexPlaceholder<N_index1> >= float
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<float>,
      GreaterOrEqual<int, float > > >
operator>=(IndexPlaceholder<N_index1> d1, 
      float d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<float>, 
      GreaterOrEqual<int, float> >
      (d1, 
      _bz_ArrayExprConstant<float>(d2));
}

// IndexPlaceholder<N_index1> >= double
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<double>,
      GreaterOrEqual<int, double > > >
operator>=(IndexPlaceholder<N_index1> d1, 
      double d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<double>, 
      GreaterOrEqual<int, double> >
      (d1, 
      _bz_ArrayExprConstant<double>(d2));
}

// IndexPlaceholder<N_index1> >= long double
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<long double>,
      GreaterOrEqual<int, long double > > >
operator>=(IndexPlaceholder<N_index1> d1, 
      long double d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<long double>, 
      GreaterOrEqual<int, long double> >
      (d1, 
      _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
// IndexPlaceholder<N_index1> >= complex<T2>
template<int N_index1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<complex<T2> > ,
      GreaterOrEqual<int, complex<T2>  > > >
operator>=(IndexPlaceholder<N_index1> d1, 
      complex<T2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<complex<T2> > , 
      GreaterOrEqual<int, complex<T2> > >
      (d1, 
      _bz_ArrayExprConstant<complex<T2> > (d2));
}
#endif // BZ_HAVE_COMPLEX

// int >= Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>,
      GreaterOrEqual<int, T_numtype2 > > >
operator>=(int d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      GreaterOrEqual<int, T_numtype2> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2.begin());
}

// int >= _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>,
      GreaterOrEqual<int, typename P_expr2::T_numtype > > >
operator>=(int d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>, 
      GreaterOrEqual<int, typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2);
}

// int >= IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      IndexPlaceholder<N_index2>,
      GreaterOrEqual<int, int > > >
operator>=(int d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      IndexPlaceholder<N_index2>, 
      GreaterOrEqual<int, int> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2);
}

// float >= Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, 
      ArrayIterator<T_numtype2, N_rank2>,
      GreaterOrEqual<float, T_numtype2 > > >
operator>=(float d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      GreaterOrEqual<float, T_numtype2> >
      (_bz_ArrayExprConstant<float>(d1), 
      d2.begin());
}

// float >= _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, 
      _bz_ArrayExpr<P_expr2>,
      GreaterOrEqual<float, typename P_expr2::T_numtype > > >
operator>=(float d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, 
      _bz_ArrayExpr<P_expr2>, 
      GreaterOrEqual<float, typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<float>(d1), 
      d2);
}

// float >= IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, 
      IndexPlaceholder<N_index2>,
      GreaterOrEqual<float, int > > >
operator>=(float d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, 
      IndexPlaceholder<N_index2>, 
      GreaterOrEqual<float, int> >
      (_bz_ArrayExprConstant<float>(d1), 
      d2);
}

// double >= Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, 
      ArrayIterator<T_numtype2, N_rank2>,
      GreaterOrEqual<double, T_numtype2 > > >
operator>=(double d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      GreaterOrEqual<double, T_numtype2> >
      (_bz_ArrayExprConstant<double>(d1), 
      d2.begin());
}

// double >= _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, 
      _bz_ArrayExpr<P_expr2>,
      GreaterOrEqual<double, typename P_expr2::T_numtype > > >
operator>=(double d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, 
      _bz_ArrayExpr<P_expr2>, 
      GreaterOrEqual<double, typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<double>(d1), 
      d2);
}

// double >= IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, 
      IndexPlaceholder<N_index2>,
      GreaterOrEqual<double, int > > >
operator>=(double d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, 
      IndexPlaceholder<N_index2>, 
      GreaterOrEqual<double, int> >
      (_bz_ArrayExprConstant<double>(d1), 
      d2);
}

// long double >= Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, 
      ArrayIterator<T_numtype2, N_rank2>,
      GreaterOrEqual<long double, T_numtype2 > > >
operator>=(long double d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      GreaterOrEqual<long double, T_numtype2> >
      (_bz_ArrayExprConstant<long double>(d1), 
      d2.begin());
}

// long double >= _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, 
      _bz_ArrayExpr<P_expr2>,
      GreaterOrEqual<long double, typename P_expr2::T_numtype > > >
operator>=(long double d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, 
      _bz_ArrayExpr<P_expr2>, 
      GreaterOrEqual<long double, typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<long double>(d1), 
      d2);
}

// long double >= IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, 
      IndexPlaceholder<N_index2>,
      GreaterOrEqual<long double, int > > >
operator>=(long double d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, 
      IndexPlaceholder<N_index2>, 
      GreaterOrEqual<long double, int> >
      (_bz_ArrayExprConstant<long double>(d1), 
      d2);
}

#ifdef BZ_HAVE_COMPLEX
// complex<T1> >= Array<T_numtype2, N_rank2>
template<class T1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , 
      ArrayIterator<T_numtype2, N_rank2>,
      GreaterOrEqual<complex<T1> , T_numtype2 > > >
operator>=(complex<T1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , 
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
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , 
      _bz_ArrayExpr<P_expr2>,
      GreaterOrEqual<complex<T1> , typename P_expr2::T_numtype > > >
operator>=(complex<T1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , 
      _bz_ArrayExpr<P_expr2>, 
      GreaterOrEqual<complex<T1> , typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<complex<T1> > (d1), 
      d2);
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX
// complex<T1> >= IndexPlaceholder<N_index2>
template<class T1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , 
      IndexPlaceholder<N_index2>,
      GreaterOrEqual<complex<T1> , int > > >
operator>=(complex<T1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , 
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
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      LessOrEqual<T_numtype1, T_numtype2 > > >
operator<=(const Array<T_numtype1, N_rank1>& d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      LessOrEqual<T_numtype1, T_numtype2> >
      (d1.begin(), 
      d2.begin());
}

// Array<T_numtype1, N_rank1> <= _bz_ArrayExpr<P_expr2>
template<class T_numtype1, int N_rank1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>,
      LessOrEqual<T_numtype1, typename P_expr2::T_numtype > > >
operator<=(const Array<T_numtype1, N_rank1>& d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>, 
      LessOrEqual<T_numtype1, typename P_expr2::T_numtype> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> <= IndexPlaceholder<N_index2>
template<class T_numtype1, int N_rank1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>,
      LessOrEqual<T_numtype1, int > > >
operator<=(const Array<T_numtype1, N_rank1>& d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>, 
      LessOrEqual<T_numtype1, int> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> <= int
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>,
      LessOrEqual<T_numtype1, int > > >
operator<=(const Array<T_numtype1, N_rank1>& d1, 
      int d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>, 
      LessOrEqual<T_numtype1, int> >
      (d1.begin(), 
      _bz_ArrayExprConstant<int>(d2));
}

// Array<T_numtype1, N_rank1> <= float
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<float>,
      LessOrEqual<T_numtype1, float > > >
operator<=(const Array<T_numtype1, N_rank1>& d1, 
      float d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<float>, 
      LessOrEqual<T_numtype1, float> >
      (d1.begin(), 
      _bz_ArrayExprConstant<float>(d2));
}

// Array<T_numtype1, N_rank1> <= double
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<double>,
      LessOrEqual<T_numtype1, double > > >
operator<=(const Array<T_numtype1, N_rank1>& d1, 
      double d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<double>, 
      LessOrEqual<T_numtype1, double> >
      (d1.begin(), 
      _bz_ArrayExprConstant<double>(d2));
}

// Array<T_numtype1, N_rank1> <= long double
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<long double>,
      LessOrEqual<T_numtype1, long double > > >
operator<=(const Array<T_numtype1, N_rank1>& d1, 
      long double d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<long double>, 
      LessOrEqual<T_numtype1, long double> >
      (d1.begin(), 
      _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
// Array<T_numtype1, N_rank1> <= complex<T2>
template<class T_numtype1, int N_rank1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<complex<T2> > ,
      LessOrEqual<T_numtype1, complex<T2>  > > >
operator<=(const Array<T_numtype1, N_rank1>& d1, 
      complex<T2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<complex<T2> > , 
      LessOrEqual<T_numtype1, complex<T2> > >
      (d1.begin(), 
      _bz_ArrayExprConstant<complex<T2> > (d2));
}
#endif // BZ_HAVE_COMPLEX

// _bz_ArrayExpr<P_expr1> <= Array<T_numtype2, N_rank2>
template<class P_expr1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      LessOrEqual<typename P_expr1::T_numtype, T_numtype2 > > >
operator<=(_bz_ArrayExpr<P_expr1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      LessOrEqual<typename P_expr1::T_numtype, T_numtype2> >
      (d1, 
      d2.begin());
}

// _bz_ArrayExpr<P_expr1> <= _bz_ArrayExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>,
      LessOrEqual<typename P_expr1::T_numtype, typename P_expr2::T_numtype > > >
operator<=(_bz_ArrayExpr<P_expr1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>, 
      LessOrEqual<typename P_expr1::T_numtype, typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> <= IndexPlaceholder<N_index2>
template<class P_expr1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>,
      LessOrEqual<typename P_expr1::T_numtype, int > > >
operator<=(_bz_ArrayExpr<P_expr1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>, 
      LessOrEqual<typename P_expr1::T_numtype, int> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> <= int
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>,
      LessOrEqual<typename P_expr1::T_numtype, int > > >
operator<=(_bz_ArrayExpr<P_expr1> d1, 
      int d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>, 
      LessOrEqual<typename P_expr1::T_numtype, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// _bz_ArrayExpr<P_expr1> <= float
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<float>,
      LessOrEqual<typename P_expr1::T_numtype, float > > >
operator<=(_bz_ArrayExpr<P_expr1> d1, 
      float d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<float>, 
      LessOrEqual<typename P_expr1::T_numtype, float> >
      (d1, 
      _bz_ArrayExprConstant<float>(d2));
}

// _bz_ArrayExpr<P_expr1> <= double
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<double>,
      LessOrEqual<typename P_expr1::T_numtype, double > > >
operator<=(_bz_ArrayExpr<P_expr1> d1, 
      double d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<double>, 
      LessOrEqual<typename P_expr1::T_numtype, double> >
      (d1, 
      _bz_ArrayExprConstant<double>(d2));
}

// _bz_ArrayExpr<P_expr1> <= long double
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<long double>,
      LessOrEqual<typename P_expr1::T_numtype, long double > > >
operator<=(_bz_ArrayExpr<P_expr1> d1, 
      long double d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<long double>, 
      LessOrEqual<typename P_expr1::T_numtype, long double> >
      (d1, 
      _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
// _bz_ArrayExpr<P_expr1> <= complex<T2>
template<class P_expr1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<complex<T2> > ,
      LessOrEqual<typename P_expr1::T_numtype, complex<T2>  > > >
operator<=(_bz_ArrayExpr<P_expr1> d1, 
      complex<T2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<complex<T2> > , 
      LessOrEqual<typename P_expr1::T_numtype, complex<T2> > >
      (d1, 
      _bz_ArrayExprConstant<complex<T2> > (d2));
}
#endif // BZ_HAVE_COMPLEX

// IndexPlaceholder<N_index1> <= Array<T_numtype2, N_rank2>
template<int N_index1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      LessOrEqual<int, T_numtype2 > > >
operator<=(IndexPlaceholder<N_index1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      LessOrEqual<int, T_numtype2> >
      (d1, 
      d2.begin());
}

// IndexPlaceholder<N_index1> <= _bz_ArrayExpr<P_expr2>
template<int N_index1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>,
      LessOrEqual<int, typename P_expr2::T_numtype > > >
operator<=(IndexPlaceholder<N_index1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>, 
      LessOrEqual<int, typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> <= IndexPlaceholder<N_index2>
template<int N_index1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>,
      LessOrEqual<int, int > > >
operator<=(IndexPlaceholder<N_index1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>, 
      LessOrEqual<int, int> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> <= int
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>,
      LessOrEqual<int, int > > >
operator<=(IndexPlaceholder<N_index1> d1, 
      int d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>, 
      LessOrEqual<int, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// IndexPlaceholder<N_index1> <= float
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<float>,
      LessOrEqual<int, float > > >
operator<=(IndexPlaceholder<N_index1> d1, 
      float d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<float>, 
      LessOrEqual<int, float> >
      (d1, 
      _bz_ArrayExprConstant<float>(d2));
}

// IndexPlaceholder<N_index1> <= double
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<double>,
      LessOrEqual<int, double > > >
operator<=(IndexPlaceholder<N_index1> d1, 
      double d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<double>, 
      LessOrEqual<int, double> >
      (d1, 
      _bz_ArrayExprConstant<double>(d2));
}

// IndexPlaceholder<N_index1> <= long double
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<long double>,
      LessOrEqual<int, long double > > >
operator<=(IndexPlaceholder<N_index1> d1, 
      long double d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<long double>, 
      LessOrEqual<int, long double> >
      (d1, 
      _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
// IndexPlaceholder<N_index1> <= complex<T2>
template<int N_index1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<complex<T2> > ,
      LessOrEqual<int, complex<T2>  > > >
operator<=(IndexPlaceholder<N_index1> d1, 
      complex<T2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<complex<T2> > , 
      LessOrEqual<int, complex<T2> > >
      (d1, 
      _bz_ArrayExprConstant<complex<T2> > (d2));
}
#endif // BZ_HAVE_COMPLEX

// int <= Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>,
      LessOrEqual<int, T_numtype2 > > >
operator<=(int d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      LessOrEqual<int, T_numtype2> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2.begin());
}

// int <= _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>,
      LessOrEqual<int, typename P_expr2::T_numtype > > >
operator<=(int d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>, 
      LessOrEqual<int, typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2);
}

// int <= IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      IndexPlaceholder<N_index2>,
      LessOrEqual<int, int > > >
operator<=(int d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      IndexPlaceholder<N_index2>, 
      LessOrEqual<int, int> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2);
}

// float <= Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, 
      ArrayIterator<T_numtype2, N_rank2>,
      LessOrEqual<float, T_numtype2 > > >
operator<=(float d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      LessOrEqual<float, T_numtype2> >
      (_bz_ArrayExprConstant<float>(d1), 
      d2.begin());
}

// float <= _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, 
      _bz_ArrayExpr<P_expr2>,
      LessOrEqual<float, typename P_expr2::T_numtype > > >
operator<=(float d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, 
      _bz_ArrayExpr<P_expr2>, 
      LessOrEqual<float, typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<float>(d1), 
      d2);
}

// float <= IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, 
      IndexPlaceholder<N_index2>,
      LessOrEqual<float, int > > >
operator<=(float d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, 
      IndexPlaceholder<N_index2>, 
      LessOrEqual<float, int> >
      (_bz_ArrayExprConstant<float>(d1), 
      d2);
}

// double <= Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, 
      ArrayIterator<T_numtype2, N_rank2>,
      LessOrEqual<double, T_numtype2 > > >
operator<=(double d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      LessOrEqual<double, T_numtype2> >
      (_bz_ArrayExprConstant<double>(d1), 
      d2.begin());
}

// double <= _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, 
      _bz_ArrayExpr<P_expr2>,
      LessOrEqual<double, typename P_expr2::T_numtype > > >
operator<=(double d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, 
      _bz_ArrayExpr<P_expr2>, 
      LessOrEqual<double, typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<double>(d1), 
      d2);
}

// double <= IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, 
      IndexPlaceholder<N_index2>,
      LessOrEqual<double, int > > >
operator<=(double d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, 
      IndexPlaceholder<N_index2>, 
      LessOrEqual<double, int> >
      (_bz_ArrayExprConstant<double>(d1), 
      d2);
}

// long double <= Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, 
      ArrayIterator<T_numtype2, N_rank2>,
      LessOrEqual<long double, T_numtype2 > > >
operator<=(long double d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      LessOrEqual<long double, T_numtype2> >
      (_bz_ArrayExprConstant<long double>(d1), 
      d2.begin());
}

// long double <= _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, 
      _bz_ArrayExpr<P_expr2>,
      LessOrEqual<long double, typename P_expr2::T_numtype > > >
operator<=(long double d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, 
      _bz_ArrayExpr<P_expr2>, 
      LessOrEqual<long double, typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<long double>(d1), 
      d2);
}

// long double <= IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, 
      IndexPlaceholder<N_index2>,
      LessOrEqual<long double, int > > >
operator<=(long double d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, 
      IndexPlaceholder<N_index2>, 
      LessOrEqual<long double, int> >
      (_bz_ArrayExprConstant<long double>(d1), 
      d2);
}

#ifdef BZ_HAVE_COMPLEX
// complex<T1> <= Array<T_numtype2, N_rank2>
template<class T1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , 
      ArrayIterator<T_numtype2, N_rank2>,
      LessOrEqual<complex<T1> , T_numtype2 > > >
operator<=(complex<T1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , 
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
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , 
      _bz_ArrayExpr<P_expr2>,
      LessOrEqual<complex<T1> , typename P_expr2::T_numtype > > >
operator<=(complex<T1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , 
      _bz_ArrayExpr<P_expr2>, 
      LessOrEqual<complex<T1> , typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<complex<T1> > (d1), 
      d2);
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX
// complex<T1> <= IndexPlaceholder<N_index2>
template<class T1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , 
      IndexPlaceholder<N_index2>,
      LessOrEqual<complex<T1> , int > > >
operator<=(complex<T1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , 
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
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Equal<T_numtype1, T_numtype2 > > >
operator==(const Array<T_numtype1, N_rank1>& d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Equal<T_numtype1, T_numtype2> >
      (d1.begin(), 
      d2.begin());
}

// Array<T_numtype1, N_rank1> == _bz_ArrayExpr<P_expr2>
template<class T_numtype1, int N_rank1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>,
      Equal<T_numtype1, typename P_expr2::T_numtype > > >
operator==(const Array<T_numtype1, N_rank1>& d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>, 
      Equal<T_numtype1, typename P_expr2::T_numtype> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> == IndexPlaceholder<N_index2>
template<class T_numtype1, int N_rank1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>,
      Equal<T_numtype1, int > > >
operator==(const Array<T_numtype1, N_rank1>& d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>, 
      Equal<T_numtype1, int> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> == int
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>,
      Equal<T_numtype1, int > > >
operator==(const Array<T_numtype1, N_rank1>& d1, 
      int d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>, 
      Equal<T_numtype1, int> >
      (d1.begin(), 
      _bz_ArrayExprConstant<int>(d2));
}

// Array<T_numtype1, N_rank1> == float
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<float>,
      Equal<T_numtype1, float > > >
operator==(const Array<T_numtype1, N_rank1>& d1, 
      float d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<float>, 
      Equal<T_numtype1, float> >
      (d1.begin(), 
      _bz_ArrayExprConstant<float>(d2));
}

// Array<T_numtype1, N_rank1> == double
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<double>,
      Equal<T_numtype1, double > > >
operator==(const Array<T_numtype1, N_rank1>& d1, 
      double d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<double>, 
      Equal<T_numtype1, double> >
      (d1.begin(), 
      _bz_ArrayExprConstant<double>(d2));
}

// Array<T_numtype1, N_rank1> == long double
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<long double>,
      Equal<T_numtype1, long double > > >
operator==(const Array<T_numtype1, N_rank1>& d1, 
      long double d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<long double>, 
      Equal<T_numtype1, long double> >
      (d1.begin(), 
      _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
// Array<T_numtype1, N_rank1> == complex<T2>
template<class T_numtype1, int N_rank1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<complex<T2> > ,
      Equal<T_numtype1, complex<T2>  > > >
operator==(const Array<T_numtype1, N_rank1>& d1, 
      complex<T2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<complex<T2> > , 
      Equal<T_numtype1, complex<T2> > >
      (d1.begin(), 
      _bz_ArrayExprConstant<complex<T2> > (d2));
}
#endif // BZ_HAVE_COMPLEX

// _bz_ArrayExpr<P_expr1> == Array<T_numtype2, N_rank2>
template<class P_expr1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Equal<typename P_expr1::T_numtype, T_numtype2 > > >
operator==(_bz_ArrayExpr<P_expr1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Equal<typename P_expr1::T_numtype, T_numtype2> >
      (d1, 
      d2.begin());
}

// _bz_ArrayExpr<P_expr1> == _bz_ArrayExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>,
      Equal<typename P_expr1::T_numtype, typename P_expr2::T_numtype > > >
operator==(_bz_ArrayExpr<P_expr1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>, 
      Equal<typename P_expr1::T_numtype, typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> == IndexPlaceholder<N_index2>
template<class P_expr1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>,
      Equal<typename P_expr1::T_numtype, int > > >
operator==(_bz_ArrayExpr<P_expr1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>, 
      Equal<typename P_expr1::T_numtype, int> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> == int
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>,
      Equal<typename P_expr1::T_numtype, int > > >
operator==(_bz_ArrayExpr<P_expr1> d1, 
      int d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>, 
      Equal<typename P_expr1::T_numtype, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// _bz_ArrayExpr<P_expr1> == float
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<float>,
      Equal<typename P_expr1::T_numtype, float > > >
operator==(_bz_ArrayExpr<P_expr1> d1, 
      float d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<float>, 
      Equal<typename P_expr1::T_numtype, float> >
      (d1, 
      _bz_ArrayExprConstant<float>(d2));
}

// _bz_ArrayExpr<P_expr1> == double
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<double>,
      Equal<typename P_expr1::T_numtype, double > > >
operator==(_bz_ArrayExpr<P_expr1> d1, 
      double d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<double>, 
      Equal<typename P_expr1::T_numtype, double> >
      (d1, 
      _bz_ArrayExprConstant<double>(d2));
}

// _bz_ArrayExpr<P_expr1> == long double
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<long double>,
      Equal<typename P_expr1::T_numtype, long double > > >
operator==(_bz_ArrayExpr<P_expr1> d1, 
      long double d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<long double>, 
      Equal<typename P_expr1::T_numtype, long double> >
      (d1, 
      _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
// _bz_ArrayExpr<P_expr1> == complex<T2>
template<class P_expr1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<complex<T2> > ,
      Equal<typename P_expr1::T_numtype, complex<T2>  > > >
operator==(_bz_ArrayExpr<P_expr1> d1, 
      complex<T2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<complex<T2> > , 
      Equal<typename P_expr1::T_numtype, complex<T2> > >
      (d1, 
      _bz_ArrayExprConstant<complex<T2> > (d2));
}
#endif // BZ_HAVE_COMPLEX

// IndexPlaceholder<N_index1> == Array<T_numtype2, N_rank2>
template<int N_index1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Equal<int, T_numtype2 > > >
operator==(IndexPlaceholder<N_index1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Equal<int, T_numtype2> >
      (d1, 
      d2.begin());
}

// IndexPlaceholder<N_index1> == _bz_ArrayExpr<P_expr2>
template<int N_index1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>,
      Equal<int, typename P_expr2::T_numtype > > >
operator==(IndexPlaceholder<N_index1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>, 
      Equal<int, typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> == IndexPlaceholder<N_index2>
template<int N_index1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>,
      Equal<int, int > > >
operator==(IndexPlaceholder<N_index1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>, 
      Equal<int, int> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> == int
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>,
      Equal<int, int > > >
operator==(IndexPlaceholder<N_index1> d1, 
      int d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>, 
      Equal<int, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// IndexPlaceholder<N_index1> == float
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<float>,
      Equal<int, float > > >
operator==(IndexPlaceholder<N_index1> d1, 
      float d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<float>, 
      Equal<int, float> >
      (d1, 
      _bz_ArrayExprConstant<float>(d2));
}

// IndexPlaceholder<N_index1> == double
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<double>,
      Equal<int, double > > >
operator==(IndexPlaceholder<N_index1> d1, 
      double d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<double>, 
      Equal<int, double> >
      (d1, 
      _bz_ArrayExprConstant<double>(d2));
}

// IndexPlaceholder<N_index1> == long double
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<long double>,
      Equal<int, long double > > >
operator==(IndexPlaceholder<N_index1> d1, 
      long double d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<long double>, 
      Equal<int, long double> >
      (d1, 
      _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
// IndexPlaceholder<N_index1> == complex<T2>
template<int N_index1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<complex<T2> > ,
      Equal<int, complex<T2>  > > >
operator==(IndexPlaceholder<N_index1> d1, 
      complex<T2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<complex<T2> > , 
      Equal<int, complex<T2> > >
      (d1, 
      _bz_ArrayExprConstant<complex<T2> > (d2));
}
#endif // BZ_HAVE_COMPLEX

// int == Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Equal<int, T_numtype2 > > >
operator==(int d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Equal<int, T_numtype2> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2.begin());
}

// int == _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>,
      Equal<int, typename P_expr2::T_numtype > > >
operator==(int d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>, 
      Equal<int, typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2);
}

// int == IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      IndexPlaceholder<N_index2>,
      Equal<int, int > > >
operator==(int d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      IndexPlaceholder<N_index2>, 
      Equal<int, int> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2);
}

// float == Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Equal<float, T_numtype2 > > >
operator==(float d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Equal<float, T_numtype2> >
      (_bz_ArrayExprConstant<float>(d1), 
      d2.begin());
}

// float == _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, 
      _bz_ArrayExpr<P_expr2>,
      Equal<float, typename P_expr2::T_numtype > > >
operator==(float d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, 
      _bz_ArrayExpr<P_expr2>, 
      Equal<float, typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<float>(d1), 
      d2);
}

// float == IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, 
      IndexPlaceholder<N_index2>,
      Equal<float, int > > >
operator==(float d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, 
      IndexPlaceholder<N_index2>, 
      Equal<float, int> >
      (_bz_ArrayExprConstant<float>(d1), 
      d2);
}

// double == Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Equal<double, T_numtype2 > > >
operator==(double d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Equal<double, T_numtype2> >
      (_bz_ArrayExprConstant<double>(d1), 
      d2.begin());
}

// double == _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, 
      _bz_ArrayExpr<P_expr2>,
      Equal<double, typename P_expr2::T_numtype > > >
operator==(double d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, 
      _bz_ArrayExpr<P_expr2>, 
      Equal<double, typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<double>(d1), 
      d2);
}

// double == IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, 
      IndexPlaceholder<N_index2>,
      Equal<double, int > > >
operator==(double d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, 
      IndexPlaceholder<N_index2>, 
      Equal<double, int> >
      (_bz_ArrayExprConstant<double>(d1), 
      d2);
}

// long double == Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, 
      ArrayIterator<T_numtype2, N_rank2>,
      Equal<long double, T_numtype2 > > >
operator==(long double d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      Equal<long double, T_numtype2> >
      (_bz_ArrayExprConstant<long double>(d1), 
      d2.begin());
}

// long double == _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, 
      _bz_ArrayExpr<P_expr2>,
      Equal<long double, typename P_expr2::T_numtype > > >
operator==(long double d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, 
      _bz_ArrayExpr<P_expr2>, 
      Equal<long double, typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<long double>(d1), 
      d2);
}

// long double == IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, 
      IndexPlaceholder<N_index2>,
      Equal<long double, int > > >
operator==(long double d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, 
      IndexPlaceholder<N_index2>, 
      Equal<long double, int> >
      (_bz_ArrayExprConstant<long double>(d1), 
      d2);
}

#ifdef BZ_HAVE_COMPLEX
// complex<T1> == Array<T_numtype2, N_rank2>
template<class T1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , 
      ArrayIterator<T_numtype2, N_rank2>,
      Equal<complex<T1> , T_numtype2 > > >
operator==(complex<T1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , 
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
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , 
      _bz_ArrayExpr<P_expr2>,
      Equal<complex<T1> , typename P_expr2::T_numtype > > >
operator==(complex<T1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , 
      _bz_ArrayExpr<P_expr2>, 
      Equal<complex<T1> , typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<complex<T1> > (d1), 
      d2);
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX
// complex<T1> == IndexPlaceholder<N_index2>
template<class T1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , 
      IndexPlaceholder<N_index2>,
      Equal<complex<T1> , int > > >
operator==(complex<T1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , 
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
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      NotEqual<T_numtype1, T_numtype2 > > >
operator!=(const Array<T_numtype1, N_rank1>& d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      NotEqual<T_numtype1, T_numtype2> >
      (d1.begin(), 
      d2.begin());
}

// Array<T_numtype1, N_rank1> != _bz_ArrayExpr<P_expr2>
template<class T_numtype1, int N_rank1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>,
      NotEqual<T_numtype1, typename P_expr2::T_numtype > > >
operator!=(const Array<T_numtype1, N_rank1>& d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>, 
      NotEqual<T_numtype1, typename P_expr2::T_numtype> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> != IndexPlaceholder<N_index2>
template<class T_numtype1, int N_rank1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>,
      NotEqual<T_numtype1, int > > >
operator!=(const Array<T_numtype1, N_rank1>& d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>, 
      NotEqual<T_numtype1, int> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> != int
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>,
      NotEqual<T_numtype1, int > > >
operator!=(const Array<T_numtype1, N_rank1>& d1, 
      int d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>, 
      NotEqual<T_numtype1, int> >
      (d1.begin(), 
      _bz_ArrayExprConstant<int>(d2));
}

// Array<T_numtype1, N_rank1> != float
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<float>,
      NotEqual<T_numtype1, float > > >
operator!=(const Array<T_numtype1, N_rank1>& d1, 
      float d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<float>, 
      NotEqual<T_numtype1, float> >
      (d1.begin(), 
      _bz_ArrayExprConstant<float>(d2));
}

// Array<T_numtype1, N_rank1> != double
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<double>,
      NotEqual<T_numtype1, double > > >
operator!=(const Array<T_numtype1, N_rank1>& d1, 
      double d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<double>, 
      NotEqual<T_numtype1, double> >
      (d1.begin(), 
      _bz_ArrayExprConstant<double>(d2));
}

// Array<T_numtype1, N_rank1> != long double
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<long double>,
      NotEqual<T_numtype1, long double > > >
operator!=(const Array<T_numtype1, N_rank1>& d1, 
      long double d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<long double>, 
      NotEqual<T_numtype1, long double> >
      (d1.begin(), 
      _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
// Array<T_numtype1, N_rank1> != complex<T2>
template<class T_numtype1, int N_rank1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<complex<T2> > ,
      NotEqual<T_numtype1, complex<T2>  > > >
operator!=(const Array<T_numtype1, N_rank1>& d1, 
      complex<T2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<complex<T2> > , 
      NotEqual<T_numtype1, complex<T2> > >
      (d1.begin(), 
      _bz_ArrayExprConstant<complex<T2> > (d2));
}
#endif // BZ_HAVE_COMPLEX

// _bz_ArrayExpr<P_expr1> != Array<T_numtype2, N_rank2>
template<class P_expr1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      NotEqual<typename P_expr1::T_numtype, T_numtype2 > > >
operator!=(_bz_ArrayExpr<P_expr1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      NotEqual<typename P_expr1::T_numtype, T_numtype2> >
      (d1, 
      d2.begin());
}

// _bz_ArrayExpr<P_expr1> != _bz_ArrayExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>,
      NotEqual<typename P_expr1::T_numtype, typename P_expr2::T_numtype > > >
operator!=(_bz_ArrayExpr<P_expr1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>, 
      NotEqual<typename P_expr1::T_numtype, typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> != IndexPlaceholder<N_index2>
template<class P_expr1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>,
      NotEqual<typename P_expr1::T_numtype, int > > >
operator!=(_bz_ArrayExpr<P_expr1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>, 
      NotEqual<typename P_expr1::T_numtype, int> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> != int
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>,
      NotEqual<typename P_expr1::T_numtype, int > > >
operator!=(_bz_ArrayExpr<P_expr1> d1, 
      int d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>, 
      NotEqual<typename P_expr1::T_numtype, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// _bz_ArrayExpr<P_expr1> != float
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<float>,
      NotEqual<typename P_expr1::T_numtype, float > > >
operator!=(_bz_ArrayExpr<P_expr1> d1, 
      float d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<float>, 
      NotEqual<typename P_expr1::T_numtype, float> >
      (d1, 
      _bz_ArrayExprConstant<float>(d2));
}

// _bz_ArrayExpr<P_expr1> != double
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<double>,
      NotEqual<typename P_expr1::T_numtype, double > > >
operator!=(_bz_ArrayExpr<P_expr1> d1, 
      double d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<double>, 
      NotEqual<typename P_expr1::T_numtype, double> >
      (d1, 
      _bz_ArrayExprConstant<double>(d2));
}

// _bz_ArrayExpr<P_expr1> != long double
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<long double>,
      NotEqual<typename P_expr1::T_numtype, long double > > >
operator!=(_bz_ArrayExpr<P_expr1> d1, 
      long double d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<long double>, 
      NotEqual<typename P_expr1::T_numtype, long double> >
      (d1, 
      _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
// _bz_ArrayExpr<P_expr1> != complex<T2>
template<class P_expr1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<complex<T2> > ,
      NotEqual<typename P_expr1::T_numtype, complex<T2>  > > >
operator!=(_bz_ArrayExpr<P_expr1> d1, 
      complex<T2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<complex<T2> > , 
      NotEqual<typename P_expr1::T_numtype, complex<T2> > >
      (d1, 
      _bz_ArrayExprConstant<complex<T2> > (d2));
}
#endif // BZ_HAVE_COMPLEX

// IndexPlaceholder<N_index1> != Array<T_numtype2, N_rank2>
template<int N_index1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      NotEqual<int, T_numtype2 > > >
operator!=(IndexPlaceholder<N_index1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      NotEqual<int, T_numtype2> >
      (d1, 
      d2.begin());
}

// IndexPlaceholder<N_index1> != _bz_ArrayExpr<P_expr2>
template<int N_index1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>,
      NotEqual<int, typename P_expr2::T_numtype > > >
operator!=(IndexPlaceholder<N_index1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>, 
      NotEqual<int, typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> != IndexPlaceholder<N_index2>
template<int N_index1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>,
      NotEqual<int, int > > >
operator!=(IndexPlaceholder<N_index1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>, 
      NotEqual<int, int> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> != int
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>,
      NotEqual<int, int > > >
operator!=(IndexPlaceholder<N_index1> d1, 
      int d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>, 
      NotEqual<int, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// IndexPlaceholder<N_index1> != float
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<float>,
      NotEqual<int, float > > >
operator!=(IndexPlaceholder<N_index1> d1, 
      float d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<float>, 
      NotEqual<int, float> >
      (d1, 
      _bz_ArrayExprConstant<float>(d2));
}

// IndexPlaceholder<N_index1> != double
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<double>,
      NotEqual<int, double > > >
operator!=(IndexPlaceholder<N_index1> d1, 
      double d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<double>, 
      NotEqual<int, double> >
      (d1, 
      _bz_ArrayExprConstant<double>(d2));
}

// IndexPlaceholder<N_index1> != long double
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<long double>,
      NotEqual<int, long double > > >
operator!=(IndexPlaceholder<N_index1> d1, 
      long double d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<long double>, 
      NotEqual<int, long double> >
      (d1, 
      _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
// IndexPlaceholder<N_index1> != complex<T2>
template<int N_index1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<complex<T2> > ,
      NotEqual<int, complex<T2>  > > >
operator!=(IndexPlaceholder<N_index1> d1, 
      complex<T2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<complex<T2> > , 
      NotEqual<int, complex<T2> > >
      (d1, 
      _bz_ArrayExprConstant<complex<T2> > (d2));
}
#endif // BZ_HAVE_COMPLEX

// int != Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>,
      NotEqual<int, T_numtype2 > > >
operator!=(int d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      NotEqual<int, T_numtype2> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2.begin());
}

// int != _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>,
      NotEqual<int, typename P_expr2::T_numtype > > >
operator!=(int d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>, 
      NotEqual<int, typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2);
}

// int != IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      IndexPlaceholder<N_index2>,
      NotEqual<int, int > > >
operator!=(int d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      IndexPlaceholder<N_index2>, 
      NotEqual<int, int> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2);
}

// float != Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, 
      ArrayIterator<T_numtype2, N_rank2>,
      NotEqual<float, T_numtype2 > > >
operator!=(float d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      NotEqual<float, T_numtype2> >
      (_bz_ArrayExprConstant<float>(d1), 
      d2.begin());
}

// float != _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, 
      _bz_ArrayExpr<P_expr2>,
      NotEqual<float, typename P_expr2::T_numtype > > >
operator!=(float d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, 
      _bz_ArrayExpr<P_expr2>, 
      NotEqual<float, typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<float>(d1), 
      d2);
}

// float != IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, 
      IndexPlaceholder<N_index2>,
      NotEqual<float, int > > >
operator!=(float d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, 
      IndexPlaceholder<N_index2>, 
      NotEqual<float, int> >
      (_bz_ArrayExprConstant<float>(d1), 
      d2);
}

// double != Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, 
      ArrayIterator<T_numtype2, N_rank2>,
      NotEqual<double, T_numtype2 > > >
operator!=(double d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      NotEqual<double, T_numtype2> >
      (_bz_ArrayExprConstant<double>(d1), 
      d2.begin());
}

// double != _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, 
      _bz_ArrayExpr<P_expr2>,
      NotEqual<double, typename P_expr2::T_numtype > > >
operator!=(double d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, 
      _bz_ArrayExpr<P_expr2>, 
      NotEqual<double, typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<double>(d1), 
      d2);
}

// double != IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, 
      IndexPlaceholder<N_index2>,
      NotEqual<double, int > > >
operator!=(double d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, 
      IndexPlaceholder<N_index2>, 
      NotEqual<double, int> >
      (_bz_ArrayExprConstant<double>(d1), 
      d2);
}

// long double != Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, 
      ArrayIterator<T_numtype2, N_rank2>,
      NotEqual<long double, T_numtype2 > > >
operator!=(long double d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      NotEqual<long double, T_numtype2> >
      (_bz_ArrayExprConstant<long double>(d1), 
      d2.begin());
}

// long double != _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, 
      _bz_ArrayExpr<P_expr2>,
      NotEqual<long double, typename P_expr2::T_numtype > > >
operator!=(long double d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, 
      _bz_ArrayExpr<P_expr2>, 
      NotEqual<long double, typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<long double>(d1), 
      d2);
}

// long double != IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, 
      IndexPlaceholder<N_index2>,
      NotEqual<long double, int > > >
operator!=(long double d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, 
      IndexPlaceholder<N_index2>, 
      NotEqual<long double, int> >
      (_bz_ArrayExprConstant<long double>(d1), 
      d2);
}

#ifdef BZ_HAVE_COMPLEX
// complex<T1> != Array<T_numtype2, N_rank2>
template<class T1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , 
      ArrayIterator<T_numtype2, N_rank2>,
      NotEqual<complex<T1> , T_numtype2 > > >
operator!=(complex<T1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , 
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
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , 
      _bz_ArrayExpr<P_expr2>,
      NotEqual<complex<T1> , typename P_expr2::T_numtype > > >
operator!=(complex<T1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , 
      _bz_ArrayExpr<P_expr2>, 
      NotEqual<complex<T1> , typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<complex<T1> > (d1), 
      d2);
}
#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX
// complex<T1> != IndexPlaceholder<N_index2>
template<class T1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , 
      IndexPlaceholder<N_index2>,
      NotEqual<complex<T1> , int > > >
operator!=(complex<T1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , 
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
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      LogicalAnd<T_numtype1, T_numtype2 > > >
operator&&(const Array<T_numtype1, N_rank1>& d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      LogicalAnd<T_numtype1, T_numtype2> >
      (d1.begin(), 
      d2.begin());
}

// Array<T_numtype1, N_rank1> && _bz_ArrayExpr<P_expr2>
template<class T_numtype1, int N_rank1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>,
      LogicalAnd<T_numtype1, typename P_expr2::T_numtype > > >
operator&&(const Array<T_numtype1, N_rank1>& d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>, 
      LogicalAnd<T_numtype1, typename P_expr2::T_numtype> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> && IndexPlaceholder<N_index2>
template<class T_numtype1, int N_rank1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>,
      LogicalAnd<T_numtype1, int > > >
operator&&(const Array<T_numtype1, N_rank1>& d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>, 
      LogicalAnd<T_numtype1, int> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> && int
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>,
      LogicalAnd<T_numtype1, int > > >
operator&&(const Array<T_numtype1, N_rank1>& d1, 
      int d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>, 
      LogicalAnd<T_numtype1, int> >
      (d1.begin(), 
      _bz_ArrayExprConstant<int>(d2));
}

// _bz_ArrayExpr<P_expr1> && Array<T_numtype2, N_rank2>
template<class P_expr1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      LogicalAnd<typename P_expr1::T_numtype, T_numtype2 > > >
operator&&(_bz_ArrayExpr<P_expr1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      LogicalAnd<typename P_expr1::T_numtype, T_numtype2> >
      (d1, 
      d2.begin());
}

// _bz_ArrayExpr<P_expr1> && _bz_ArrayExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>,
      LogicalAnd<typename P_expr1::T_numtype, typename P_expr2::T_numtype > > >
operator&&(_bz_ArrayExpr<P_expr1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>, 
      LogicalAnd<typename P_expr1::T_numtype, typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> && IndexPlaceholder<N_index2>
template<class P_expr1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>,
      LogicalAnd<typename P_expr1::T_numtype, int > > >
operator&&(_bz_ArrayExpr<P_expr1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>, 
      LogicalAnd<typename P_expr1::T_numtype, int> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> && int
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>,
      LogicalAnd<typename P_expr1::T_numtype, int > > >
operator&&(_bz_ArrayExpr<P_expr1> d1, 
      int d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>, 
      LogicalAnd<typename P_expr1::T_numtype, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// IndexPlaceholder<N_index1> && Array<T_numtype2, N_rank2>
template<int N_index1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      LogicalAnd<int, T_numtype2 > > >
operator&&(IndexPlaceholder<N_index1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      LogicalAnd<int, T_numtype2> >
      (d1, 
      d2.begin());
}

// IndexPlaceholder<N_index1> && _bz_ArrayExpr<P_expr2>
template<int N_index1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>,
      LogicalAnd<int, typename P_expr2::T_numtype > > >
operator&&(IndexPlaceholder<N_index1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>, 
      LogicalAnd<int, typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> && IndexPlaceholder<N_index2>
template<int N_index1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>,
      LogicalAnd<int, int > > >
operator&&(IndexPlaceholder<N_index1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>, 
      LogicalAnd<int, int> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> && int
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>,
      LogicalAnd<int, int > > >
operator&&(IndexPlaceholder<N_index1> d1, 
      int d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>, 
      LogicalAnd<int, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// int && Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>,
      LogicalAnd<int, T_numtype2 > > >
operator&&(int d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      LogicalAnd<int, T_numtype2> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2.begin());
}

// int && _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>,
      LogicalAnd<int, typename P_expr2::T_numtype > > >
operator&&(int d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>, 
      LogicalAnd<int, typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2);
}

// int && IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      IndexPlaceholder<N_index2>,
      LogicalAnd<int, int > > >
operator&&(int d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
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
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      LogicalOr<T_numtype1, T_numtype2 > > >
operator||(const Array<T_numtype1, N_rank1>& d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      LogicalOr<T_numtype1, T_numtype2> >
      (d1.begin(), 
      d2.begin());
}

// Array<T_numtype1, N_rank1> || _bz_ArrayExpr<P_expr2>
template<class T_numtype1, int N_rank1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>,
      LogicalOr<T_numtype1, typename P_expr2::T_numtype > > >
operator||(const Array<T_numtype1, N_rank1>& d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>, 
      LogicalOr<T_numtype1, typename P_expr2::T_numtype> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> || IndexPlaceholder<N_index2>
template<class T_numtype1, int N_rank1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>,
      LogicalOr<T_numtype1, int > > >
operator||(const Array<T_numtype1, N_rank1>& d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>, 
      LogicalOr<T_numtype1, int> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> || int
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>,
      LogicalOr<T_numtype1, int > > >
operator||(const Array<T_numtype1, N_rank1>& d1, 
      int d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>, 
      LogicalOr<T_numtype1, int> >
      (d1.begin(), 
      _bz_ArrayExprConstant<int>(d2));
}

// _bz_ArrayExpr<P_expr1> || Array<T_numtype2, N_rank2>
template<class P_expr1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      LogicalOr<typename P_expr1::T_numtype, T_numtype2 > > >
operator||(_bz_ArrayExpr<P_expr1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      LogicalOr<typename P_expr1::T_numtype, T_numtype2> >
      (d1, 
      d2.begin());
}

// _bz_ArrayExpr<P_expr1> || _bz_ArrayExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>,
      LogicalOr<typename P_expr1::T_numtype, typename P_expr2::T_numtype > > >
operator||(_bz_ArrayExpr<P_expr1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>, 
      LogicalOr<typename P_expr1::T_numtype, typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> || IndexPlaceholder<N_index2>
template<class P_expr1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>,
      LogicalOr<typename P_expr1::T_numtype, int > > >
operator||(_bz_ArrayExpr<P_expr1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>, 
      LogicalOr<typename P_expr1::T_numtype, int> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> || int
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>,
      LogicalOr<typename P_expr1::T_numtype, int > > >
operator||(_bz_ArrayExpr<P_expr1> d1, 
      int d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>, 
      LogicalOr<typename P_expr1::T_numtype, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// IndexPlaceholder<N_index1> || Array<T_numtype2, N_rank2>
template<int N_index1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      LogicalOr<int, T_numtype2 > > >
operator||(IndexPlaceholder<N_index1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      LogicalOr<int, T_numtype2> >
      (d1, 
      d2.begin());
}

// IndexPlaceholder<N_index1> || _bz_ArrayExpr<P_expr2>
template<int N_index1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>,
      LogicalOr<int, typename P_expr2::T_numtype > > >
operator||(IndexPlaceholder<N_index1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>, 
      LogicalOr<int, typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> || IndexPlaceholder<N_index2>
template<int N_index1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>,
      LogicalOr<int, int > > >
operator||(IndexPlaceholder<N_index1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>, 
      LogicalOr<int, int> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> || int
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>,
      LogicalOr<int, int > > >
operator||(IndexPlaceholder<N_index1> d1, 
      int d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>, 
      LogicalOr<int, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// int || Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>,
      LogicalOr<int, T_numtype2 > > >
operator||(int d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      LogicalOr<int, T_numtype2> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2.begin());
}

// int || _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>,
      LogicalOr<int, typename P_expr2::T_numtype > > >
operator||(int d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>, 
      LogicalOr<int, typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2);
}

// int || IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      IndexPlaceholder<N_index2>,
      LogicalOr<int, int > > >
operator||(int d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
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
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      BitwiseXor<T_numtype1, T_numtype2 > > >
operator^(const Array<T_numtype1, N_rank1>& d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      BitwiseXor<T_numtype1, T_numtype2> >
      (d1.begin(), 
      d2.begin());
}

// Array<T_numtype1, N_rank1> ^ _bz_ArrayExpr<P_expr2>
template<class T_numtype1, int N_rank1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>,
      BitwiseXor<T_numtype1, typename P_expr2::T_numtype > > >
operator^(const Array<T_numtype1, N_rank1>& d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>, 
      BitwiseXor<T_numtype1, typename P_expr2::T_numtype> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> ^ IndexPlaceholder<N_index2>
template<class T_numtype1, int N_rank1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>,
      BitwiseXor<T_numtype1, int > > >
operator^(const Array<T_numtype1, N_rank1>& d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>, 
      BitwiseXor<T_numtype1, int> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> ^ int
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>,
      BitwiseXor<T_numtype1, int > > >
operator^(const Array<T_numtype1, N_rank1>& d1, 
      int d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>, 
      BitwiseXor<T_numtype1, int> >
      (d1.begin(), 
      _bz_ArrayExprConstant<int>(d2));
}

// _bz_ArrayExpr<P_expr1> ^ Array<T_numtype2, N_rank2>
template<class P_expr1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      BitwiseXor<typename P_expr1::T_numtype, T_numtype2 > > >
operator^(_bz_ArrayExpr<P_expr1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      BitwiseXor<typename P_expr1::T_numtype, T_numtype2> >
      (d1, 
      d2.begin());
}

// _bz_ArrayExpr<P_expr1> ^ _bz_ArrayExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>,
      BitwiseXor<typename P_expr1::T_numtype, typename P_expr2::T_numtype > > >
operator^(_bz_ArrayExpr<P_expr1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>, 
      BitwiseXor<typename P_expr1::T_numtype, typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> ^ IndexPlaceholder<N_index2>
template<class P_expr1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>,
      BitwiseXor<typename P_expr1::T_numtype, int > > >
operator^(_bz_ArrayExpr<P_expr1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>, 
      BitwiseXor<typename P_expr1::T_numtype, int> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> ^ int
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>,
      BitwiseXor<typename P_expr1::T_numtype, int > > >
operator^(_bz_ArrayExpr<P_expr1> d1, 
      int d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>, 
      BitwiseXor<typename P_expr1::T_numtype, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// IndexPlaceholder<N_index1> ^ Array<T_numtype2, N_rank2>
template<int N_index1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      BitwiseXor<int, T_numtype2 > > >
operator^(IndexPlaceholder<N_index1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      BitwiseXor<int, T_numtype2> >
      (d1, 
      d2.begin());
}

// IndexPlaceholder<N_index1> ^ _bz_ArrayExpr<P_expr2>
template<int N_index1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>,
      BitwiseXor<int, typename P_expr2::T_numtype > > >
operator^(IndexPlaceholder<N_index1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>, 
      BitwiseXor<int, typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> ^ IndexPlaceholder<N_index2>
template<int N_index1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>,
      BitwiseXor<int, int > > >
operator^(IndexPlaceholder<N_index1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>, 
      BitwiseXor<int, int> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> ^ int
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>,
      BitwiseXor<int, int > > >
operator^(IndexPlaceholder<N_index1> d1, 
      int d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>, 
      BitwiseXor<int, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// int ^ Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>,
      BitwiseXor<int, T_numtype2 > > >
operator^(int d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      BitwiseXor<int, T_numtype2> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2.begin());
}

// int ^ _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>,
      BitwiseXor<int, typename P_expr2::T_numtype > > >
operator^(int d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>, 
      BitwiseXor<int, typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2);
}

// int ^ IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      IndexPlaceholder<N_index2>,
      BitwiseXor<int, int > > >
operator^(int d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
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
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      BitwiseAnd<T_numtype1, T_numtype2 > > >
operator&(const Array<T_numtype1, N_rank1>& d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      BitwiseAnd<T_numtype1, T_numtype2> >
      (d1.begin(), 
      d2.begin());
}

// Array<T_numtype1, N_rank1> & _bz_ArrayExpr<P_expr2>
template<class T_numtype1, int N_rank1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>,
      BitwiseAnd<T_numtype1, typename P_expr2::T_numtype > > >
operator&(const Array<T_numtype1, N_rank1>& d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>, 
      BitwiseAnd<T_numtype1, typename P_expr2::T_numtype> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> & IndexPlaceholder<N_index2>
template<class T_numtype1, int N_rank1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>,
      BitwiseAnd<T_numtype1, int > > >
operator&(const Array<T_numtype1, N_rank1>& d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>, 
      BitwiseAnd<T_numtype1, int> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> & int
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>,
      BitwiseAnd<T_numtype1, int > > >
operator&(const Array<T_numtype1, N_rank1>& d1, 
      int d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>, 
      BitwiseAnd<T_numtype1, int> >
      (d1.begin(), 
      _bz_ArrayExprConstant<int>(d2));
}

// _bz_ArrayExpr<P_expr1> & Array<T_numtype2, N_rank2>
template<class P_expr1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      BitwiseAnd<typename P_expr1::T_numtype, T_numtype2 > > >
operator&(_bz_ArrayExpr<P_expr1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      BitwiseAnd<typename P_expr1::T_numtype, T_numtype2> >
      (d1, 
      d2.begin());
}

// _bz_ArrayExpr<P_expr1> & _bz_ArrayExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>,
      BitwiseAnd<typename P_expr1::T_numtype, typename P_expr2::T_numtype > > >
operator&(_bz_ArrayExpr<P_expr1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>, 
      BitwiseAnd<typename P_expr1::T_numtype, typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> & IndexPlaceholder<N_index2>
template<class P_expr1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>,
      BitwiseAnd<typename P_expr1::T_numtype, int > > >
operator&(_bz_ArrayExpr<P_expr1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>, 
      BitwiseAnd<typename P_expr1::T_numtype, int> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> & int
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>,
      BitwiseAnd<typename P_expr1::T_numtype, int > > >
operator&(_bz_ArrayExpr<P_expr1> d1, 
      int d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>, 
      BitwiseAnd<typename P_expr1::T_numtype, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// IndexPlaceholder<N_index1> & Array<T_numtype2, N_rank2>
template<int N_index1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      BitwiseAnd<int, T_numtype2 > > >
operator&(IndexPlaceholder<N_index1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      BitwiseAnd<int, T_numtype2> >
      (d1, 
      d2.begin());
}

// IndexPlaceholder<N_index1> & _bz_ArrayExpr<P_expr2>
template<int N_index1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>,
      BitwiseAnd<int, typename P_expr2::T_numtype > > >
operator&(IndexPlaceholder<N_index1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>, 
      BitwiseAnd<int, typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> & IndexPlaceholder<N_index2>
template<int N_index1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>,
      BitwiseAnd<int, int > > >
operator&(IndexPlaceholder<N_index1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>, 
      BitwiseAnd<int, int> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> & int
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>,
      BitwiseAnd<int, int > > >
operator&(IndexPlaceholder<N_index1> d1, 
      int d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>, 
      BitwiseAnd<int, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// int & Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>,
      BitwiseAnd<int, T_numtype2 > > >
operator&(int d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      BitwiseAnd<int, T_numtype2> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2.begin());
}

// int & _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>,
      BitwiseAnd<int, typename P_expr2::T_numtype > > >
operator&(int d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>, 
      BitwiseAnd<int, typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2);
}

// int & IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      IndexPlaceholder<N_index2>,
      BitwiseAnd<int, int > > >
operator&(int d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
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
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      BitwiseOr<T_numtype1, T_numtype2 > > >
operator|(const Array<T_numtype1, N_rank1>& d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      BitwiseOr<T_numtype1, T_numtype2> >
      (d1.begin(), 
      d2.begin());
}

// Array<T_numtype1, N_rank1> | _bz_ArrayExpr<P_expr2>
template<class T_numtype1, int N_rank1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>,
      BitwiseOr<T_numtype1, typename P_expr2::T_numtype > > >
operator|(const Array<T_numtype1, N_rank1>& d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>, 
      BitwiseOr<T_numtype1, typename P_expr2::T_numtype> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> | IndexPlaceholder<N_index2>
template<class T_numtype1, int N_rank1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>,
      BitwiseOr<T_numtype1, int > > >
operator|(const Array<T_numtype1, N_rank1>& d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>, 
      BitwiseOr<T_numtype1, int> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> | int
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>,
      BitwiseOr<T_numtype1, int > > >
operator|(const Array<T_numtype1, N_rank1>& d1, 
      int d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>, 
      BitwiseOr<T_numtype1, int> >
      (d1.begin(), 
      _bz_ArrayExprConstant<int>(d2));
}

// _bz_ArrayExpr<P_expr1> | Array<T_numtype2, N_rank2>
template<class P_expr1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      BitwiseOr<typename P_expr1::T_numtype, T_numtype2 > > >
operator|(_bz_ArrayExpr<P_expr1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      BitwiseOr<typename P_expr1::T_numtype, T_numtype2> >
      (d1, 
      d2.begin());
}

// _bz_ArrayExpr<P_expr1> | _bz_ArrayExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>,
      BitwiseOr<typename P_expr1::T_numtype, typename P_expr2::T_numtype > > >
operator|(_bz_ArrayExpr<P_expr1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>, 
      BitwiseOr<typename P_expr1::T_numtype, typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> | IndexPlaceholder<N_index2>
template<class P_expr1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>,
      BitwiseOr<typename P_expr1::T_numtype, int > > >
operator|(_bz_ArrayExpr<P_expr1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>, 
      BitwiseOr<typename P_expr1::T_numtype, int> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> | int
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>,
      BitwiseOr<typename P_expr1::T_numtype, int > > >
operator|(_bz_ArrayExpr<P_expr1> d1, 
      int d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>, 
      BitwiseOr<typename P_expr1::T_numtype, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// IndexPlaceholder<N_index1> | Array<T_numtype2, N_rank2>
template<int N_index1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      BitwiseOr<int, T_numtype2 > > >
operator|(IndexPlaceholder<N_index1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      BitwiseOr<int, T_numtype2> >
      (d1, 
      d2.begin());
}

// IndexPlaceholder<N_index1> | _bz_ArrayExpr<P_expr2>
template<int N_index1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>,
      BitwiseOr<int, typename P_expr2::T_numtype > > >
operator|(IndexPlaceholder<N_index1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>, 
      BitwiseOr<int, typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> | IndexPlaceholder<N_index2>
template<int N_index1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>,
      BitwiseOr<int, int > > >
operator|(IndexPlaceholder<N_index1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>, 
      BitwiseOr<int, int> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> | int
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>,
      BitwiseOr<int, int > > >
operator|(IndexPlaceholder<N_index1> d1, 
      int d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>, 
      BitwiseOr<int, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// int | Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>,
      BitwiseOr<int, T_numtype2 > > >
operator|(int d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      BitwiseOr<int, T_numtype2> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2.begin());
}

// int | _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>,
      BitwiseOr<int, typename P_expr2::T_numtype > > >
operator|(int d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>, 
      BitwiseOr<int, typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2);
}

// int | IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      IndexPlaceholder<N_index2>,
      BitwiseOr<int, int > > >
operator|(int d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
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
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      ShiftRight<T_numtype1, T_numtype2 > > >
operator>>(const Array<T_numtype1, N_rank1>& d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      ShiftRight<T_numtype1, T_numtype2> >
      (d1.begin(), 
      d2.begin());
}

// Array<T_numtype1, N_rank1> >> _bz_ArrayExpr<P_expr2>
template<class T_numtype1, int N_rank1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>,
      ShiftRight<T_numtype1, typename P_expr2::T_numtype > > >
operator>>(const Array<T_numtype1, N_rank1>& d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>, 
      ShiftRight<T_numtype1, typename P_expr2::T_numtype> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> >> IndexPlaceholder<N_index2>
template<class T_numtype1, int N_rank1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>,
      ShiftRight<T_numtype1, int > > >
operator>>(const Array<T_numtype1, N_rank1>& d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>, 
      ShiftRight<T_numtype1, int> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> >> int
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>,
      ShiftRight<T_numtype1, int > > >
operator>>(const Array<T_numtype1, N_rank1>& d1, 
      int d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>, 
      ShiftRight<T_numtype1, int> >
      (d1.begin(), 
      _bz_ArrayExprConstant<int>(d2));
}

// _bz_ArrayExpr<P_expr1> >> Array<T_numtype2, N_rank2>
template<class P_expr1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      ShiftRight<typename P_expr1::T_numtype, T_numtype2 > > >
operator>>(_bz_ArrayExpr<P_expr1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      ShiftRight<typename P_expr1::T_numtype, T_numtype2> >
      (d1, 
      d2.begin());
}

// _bz_ArrayExpr<P_expr1> >> _bz_ArrayExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>,
      ShiftRight<typename P_expr1::T_numtype, typename P_expr2::T_numtype > > >
operator>>(_bz_ArrayExpr<P_expr1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>, 
      ShiftRight<typename P_expr1::T_numtype, typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> >> IndexPlaceholder<N_index2>
template<class P_expr1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>,
      ShiftRight<typename P_expr1::T_numtype, int > > >
operator>>(_bz_ArrayExpr<P_expr1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>, 
      ShiftRight<typename P_expr1::T_numtype, int> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> >> int
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>,
      ShiftRight<typename P_expr1::T_numtype, int > > >
operator>>(_bz_ArrayExpr<P_expr1> d1, 
      int d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>, 
      ShiftRight<typename P_expr1::T_numtype, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// IndexPlaceholder<N_index1> >> Array<T_numtype2, N_rank2>
template<int N_index1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      ShiftRight<int, T_numtype2 > > >
operator>>(IndexPlaceholder<N_index1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      ShiftRight<int, T_numtype2> >
      (d1, 
      d2.begin());
}

// IndexPlaceholder<N_index1> >> _bz_ArrayExpr<P_expr2>
template<int N_index1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>,
      ShiftRight<int, typename P_expr2::T_numtype > > >
operator>>(IndexPlaceholder<N_index1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>, 
      ShiftRight<int, typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> >> IndexPlaceholder<N_index2>
template<int N_index1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>,
      ShiftRight<int, int > > >
operator>>(IndexPlaceholder<N_index1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>, 
      ShiftRight<int, int> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> >> int
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>,
      ShiftRight<int, int > > >
operator>>(IndexPlaceholder<N_index1> d1, 
      int d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>, 
      ShiftRight<int, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// int >> Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>,
      ShiftRight<int, T_numtype2 > > >
operator>>(int d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      ShiftRight<int, T_numtype2> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2.begin());
}

// int >> _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>,
      ShiftRight<int, typename P_expr2::T_numtype > > >
operator>>(int d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>, 
      ShiftRight<int, typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2);
}

// int >> IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      IndexPlaceholder<N_index2>,
      ShiftRight<int, int > > >
operator>>(int d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
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
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      ShiftLeft<T_numtype1, T_numtype2 > > >
operator<<(const Array<T_numtype1, N_rank1>& d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      ShiftLeft<T_numtype1, T_numtype2> >
      (d1.begin(), 
      d2.begin());
}

// Array<T_numtype1, N_rank1> << _bz_ArrayExpr<P_expr2>
template<class T_numtype1, int N_rank1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>,
      ShiftLeft<T_numtype1, typename P_expr2::T_numtype > > >
operator<<(const Array<T_numtype1, N_rank1>& d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>, 
      ShiftLeft<T_numtype1, typename P_expr2::T_numtype> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> << IndexPlaceholder<N_index2>
template<class T_numtype1, int N_rank1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>,
      ShiftLeft<T_numtype1, int > > >
operator<<(const Array<T_numtype1, N_rank1>& d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>, 
      ShiftLeft<T_numtype1, int> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> << int
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>,
      ShiftLeft<T_numtype1, int > > >
operator<<(const Array<T_numtype1, N_rank1>& d1, 
      int d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>, 
      ShiftLeft<T_numtype1, int> >
      (d1.begin(), 
      _bz_ArrayExprConstant<int>(d2));
}

// _bz_ArrayExpr<P_expr1> << Array<T_numtype2, N_rank2>
template<class P_expr1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      ShiftLeft<typename P_expr1::T_numtype, T_numtype2 > > >
operator<<(_bz_ArrayExpr<P_expr1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      ShiftLeft<typename P_expr1::T_numtype, T_numtype2> >
      (d1, 
      d2.begin());
}

// _bz_ArrayExpr<P_expr1> << _bz_ArrayExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>,
      ShiftLeft<typename P_expr1::T_numtype, typename P_expr2::T_numtype > > >
operator<<(_bz_ArrayExpr<P_expr1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>, 
      ShiftLeft<typename P_expr1::T_numtype, typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> << IndexPlaceholder<N_index2>
template<class P_expr1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>,
      ShiftLeft<typename P_expr1::T_numtype, int > > >
operator<<(_bz_ArrayExpr<P_expr1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>, 
      ShiftLeft<typename P_expr1::T_numtype, int> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> << int
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>,
      ShiftLeft<typename P_expr1::T_numtype, int > > >
operator<<(_bz_ArrayExpr<P_expr1> d1, 
      int d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>, 
      ShiftLeft<typename P_expr1::T_numtype, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// IndexPlaceholder<N_index1> << Array<T_numtype2, N_rank2>
template<int N_index1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      ShiftLeft<int, T_numtype2 > > >
operator<<(IndexPlaceholder<N_index1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      ShiftLeft<int, T_numtype2> >
      (d1, 
      d2.begin());
}

// IndexPlaceholder<N_index1> << _bz_ArrayExpr<P_expr2>
template<int N_index1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>,
      ShiftLeft<int, typename P_expr2::T_numtype > > >
operator<<(IndexPlaceholder<N_index1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>, 
      ShiftLeft<int, typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> << IndexPlaceholder<N_index2>
template<int N_index1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>,
      ShiftLeft<int, int > > >
operator<<(IndexPlaceholder<N_index1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>, 
      ShiftLeft<int, int> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> << int
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>,
      ShiftLeft<int, int > > >
operator<<(IndexPlaceholder<N_index1> d1, 
      int d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>, 
      ShiftLeft<int, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// int << Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>,
      ShiftLeft<int, T_numtype2 > > >
operator<<(int d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      ShiftLeft<int, T_numtype2> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2.begin());
}

// int << _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>,
      ShiftLeft<int, typename P_expr2::T_numtype > > >
operator<<(int d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>, 
      ShiftLeft<int, typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2);
}

// int << IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      IndexPlaceholder<N_index2>,
      ShiftLeft<int, int > > >
operator<<(int d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      IndexPlaceholder<N_index2>, 
      ShiftLeft<int, int> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2);
}
/****************************************************************************
 * Minimum Operators
 ****************************************************************************/

// Array<T_numtype1, N_rank1> min Array<T_numtype2, N_rank2>
template<class T_numtype1, int N_rank1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      _bz_Min<T_numtype1, T_numtype2 > > >
min(const Array<T_numtype1, N_rank1>& d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      _bz_Min<T_numtype1, T_numtype2> >
      (d1.begin(), 
      d2.begin());
}

// Array<T_numtype1, N_rank1> min _bz_ArrayExpr<P_expr2>
template<class T_numtype1, int N_rank1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>,
      _bz_Min<T_numtype1, typename P_expr2::T_numtype > > >
min(const Array<T_numtype1, N_rank1>& d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>, 
      _bz_Min<T_numtype1, typename P_expr2::T_numtype> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> min IndexPlaceholder<N_index2>
template<class T_numtype1, int N_rank1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>,
      _bz_Min<T_numtype1, int > > >
min(const Array<T_numtype1, N_rank1>& d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>, 
      _bz_Min<T_numtype1, int> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> min int
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>,
      _bz_Min<T_numtype1, int > > >
min(const Array<T_numtype1, N_rank1>& d1, 
      int d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>, 
      _bz_Min<T_numtype1, int> >
      (d1.begin(), 
      _bz_ArrayExprConstant<int>(d2));
}

// _bz_ArrayExpr<P_expr1> min Array<T_numtype2, N_rank2>
template<class P_expr1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      _bz_Min<typename P_expr1::T_numtype, T_numtype2 > > >
min(_bz_ArrayExpr<P_expr1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      _bz_Min<typename P_expr1::T_numtype, T_numtype2> >
      (d1, 
      d2.begin());
}

// _bz_ArrayExpr<P_expr1> min _bz_ArrayExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>,
      _bz_Min<typename P_expr1::T_numtype, typename P_expr2::T_numtype > > >
min(_bz_ArrayExpr<P_expr1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>, 
      _bz_Min<typename P_expr1::T_numtype, typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> min IndexPlaceholder<N_index2>
template<class P_expr1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>,
      _bz_Min<typename P_expr1::T_numtype, int > > >
min(_bz_ArrayExpr<P_expr1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>, 
      _bz_Min<typename P_expr1::T_numtype, int> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> min int
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>,
      _bz_Min<typename P_expr1::T_numtype, int > > >
min(_bz_ArrayExpr<P_expr1> d1, 
      int d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>, 
      _bz_Min<typename P_expr1::T_numtype, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// IndexPlaceholder<N_index1> min Array<T_numtype2, N_rank2>
template<int N_index1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      _bz_Min<int, T_numtype2 > > >
min(IndexPlaceholder<N_index1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      _bz_Min<int, T_numtype2> >
      (d1, 
      d2.begin());
}

// IndexPlaceholder<N_index1> min _bz_ArrayExpr<P_expr2>
template<int N_index1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>,
      _bz_Min<int, typename P_expr2::T_numtype > > >
min(IndexPlaceholder<N_index1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>, 
      _bz_Min<int, typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> min IndexPlaceholder<N_index2>
template<int N_index1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>,
      _bz_Min<int, int > > >
min(IndexPlaceholder<N_index1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>, 
      _bz_Min<int, int> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> min int
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>,
      _bz_Min<int, int > > >
min(IndexPlaceholder<N_index1> d1, 
      int d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>, 
      _bz_Min<int, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// int min Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>,
      _bz_Min<int, T_numtype2 > > >
min(int d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      _bz_Min<int, T_numtype2> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2.begin());
}

// int min _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>,
      _bz_Min<int, typename P_expr2::T_numtype > > >
min(int d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>, 
      _bz_Min<int, typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2);
}

// int min IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      IndexPlaceholder<N_index2>,
      _bz_Min<int, int > > >
min(int d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      IndexPlaceholder<N_index2>, 
      _bz_Min<int, int> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2);
}
/****************************************************************************
 * Maximum Operators
 ****************************************************************************/

// Array<T_numtype1, N_rank1> max Array<T_numtype2, N_rank2>
template<class T_numtype1, int N_rank1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      _bz_Max<T_numtype1, T_numtype2 > > >
max(const Array<T_numtype1, N_rank1>& d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      _bz_Max<T_numtype1, T_numtype2> >
      (d1.begin(), 
      d2.begin());
}

// Array<T_numtype1, N_rank1> max _bz_ArrayExpr<P_expr2>
template<class T_numtype1, int N_rank1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>,
      _bz_Max<T_numtype1, typename P_expr2::T_numtype > > >
max(const Array<T_numtype1, N_rank1>& d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExpr<P_expr2>, 
      _bz_Max<T_numtype1, typename P_expr2::T_numtype> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> max IndexPlaceholder<N_index2>
template<class T_numtype1, int N_rank1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>,
      _bz_Max<T_numtype1, int > > >
max(const Array<T_numtype1, N_rank1>& d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      IndexPlaceholder<N_index2>, 
      _bz_Max<T_numtype1, int> >
      (d1.begin(), 
      d2);
}

// Array<T_numtype1, N_rank1> max int
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>,
      _bz_Max<T_numtype1, int > > >
max(const Array<T_numtype1, N_rank1>& d1, 
      int d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, 
      _bz_ArrayExprConstant<int>, 
      _bz_Max<T_numtype1, int> >
      (d1.begin(), 
      _bz_ArrayExprConstant<int>(d2));
}

// _bz_ArrayExpr<P_expr1> max Array<T_numtype2, N_rank2>
template<class P_expr1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      _bz_Max<typename P_expr1::T_numtype, T_numtype2 > > >
max(_bz_ArrayExpr<P_expr1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      _bz_Max<typename P_expr1::T_numtype, T_numtype2> >
      (d1, 
      d2.begin());
}

// _bz_ArrayExpr<P_expr1> max _bz_ArrayExpr<P_expr2>
template<class P_expr1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>,
      _bz_Max<typename P_expr1::T_numtype, typename P_expr2::T_numtype > > >
max(_bz_ArrayExpr<P_expr1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExpr<P_expr2>, 
      _bz_Max<typename P_expr1::T_numtype, typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> max IndexPlaceholder<N_index2>
template<class P_expr1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>,
      _bz_Max<typename P_expr1::T_numtype, int > > >
max(_bz_ArrayExpr<P_expr1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      IndexPlaceholder<N_index2>, 
      _bz_Max<typename P_expr1::T_numtype, int> >
      (d1, 
      d2);
}

// _bz_ArrayExpr<P_expr1> max int
template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>,
      _bz_Max<typename P_expr1::T_numtype, int > > >
max(_bz_ArrayExpr<P_expr1> d1, 
      int d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, 
      _bz_ArrayExprConstant<int>, 
      _bz_Max<typename P_expr1::T_numtype, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// IndexPlaceholder<N_index1> max Array<T_numtype2, N_rank2>
template<int N_index1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>,
      _bz_Max<int, T_numtype2 > > >
max(IndexPlaceholder<N_index1> d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      _bz_Max<int, T_numtype2> >
      (d1, 
      d2.begin());
}

// IndexPlaceholder<N_index1> max _bz_ArrayExpr<P_expr2>
template<int N_index1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>,
      _bz_Max<int, typename P_expr2::T_numtype > > >
max(IndexPlaceholder<N_index1> d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExpr<P_expr2>, 
      _bz_Max<int, typename P_expr2::T_numtype> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> max IndexPlaceholder<N_index2>
template<int N_index1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>,
      _bz_Max<int, int > > >
max(IndexPlaceholder<N_index1> d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      IndexPlaceholder<N_index2>, 
      _bz_Max<int, int> >
      (d1, 
      d2);
}

// IndexPlaceholder<N_index1> max int
template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>,
      _bz_Max<int, int > > >
max(IndexPlaceholder<N_index1> d1, 
      int d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, 
      _bz_ArrayExprConstant<int>, 
      _bz_Max<int, int> >
      (d1, 
      _bz_ArrayExprConstant<int>(d2));
}

// int max Array<T_numtype2, N_rank2>
template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>,
      _bz_Max<int, T_numtype2 > > >
max(int d1, 
      const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      ArrayIterator<T_numtype2, N_rank2>, 
      _bz_Max<int, T_numtype2> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2.begin());
}

// int max _bz_ArrayExpr<P_expr2>
template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>,
      _bz_Max<int, typename P_expr2::T_numtype > > >
max(int d1, 
      _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      _bz_ArrayExpr<P_expr2>, 
      _bz_Max<int, typename P_expr2::T_numtype> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2);
}

// int max IndexPlaceholder<N_index2>
template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      IndexPlaceholder<N_index2>,
      _bz_Max<int, int > > >
max(int d1, 
      IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<int>, 
      IndexPlaceholder<N_index2>, 
      _bz_Max<int, int> >
      (_bz_ArrayExprConstant<int>(d1), 
      d2);
}
BZ_NAMESPACE_END

#endif
