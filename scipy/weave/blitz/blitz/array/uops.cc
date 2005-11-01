/***************************************************************************
 * blitz/../array/uops.cc	Expression templates for arrays, unary functions
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
// genarruops.cpp Dec 30 2003 16:49:07

#ifndef BZ_ARRAYUOPS_CC
#define BZ_ARRAYUOPS_CC

#ifndef BZ_ARRAYEXPR_H
 #error <blitz/array/uops.cc> must be included after <blitz/arrayexpr.h>
#endif // BZ_ARRAYEXPR_H

BZ_NAMESPACE(blitz)

/****************************************************************************
 * abs
 ****************************************************************************/

template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_abs<T_numtype1> > >
abs(const Array<T_numtype1, N_rank1>& d1)
{
    return _bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_abs<T_numtype1> >(d1.begin());
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_abs<typename P_expr1::T_numtype> > >
abs(_bz_ArrayExpr<P_expr1> d1)
{
    return _bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_abs<typename P_expr1::T_numtype> >(d1);
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_abs<int> > >
abs(IndexPlaceholder<N_index1> d1)
{
    return _bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_abs<int> >(d1);
}


/****************************************************************************
 * acos
 ****************************************************************************/

template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_acos<T_numtype1> > >
acos(const Array<T_numtype1, N_rank1>& d1)
{
    return _bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_acos<T_numtype1> >(d1.begin());
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_acos<typename P_expr1::T_numtype> > >
acos(_bz_ArrayExpr<P_expr1> d1)
{
    return _bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_acos<typename P_expr1::T_numtype> >(d1);
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_acos<int> > >
acos(IndexPlaceholder<N_index1> d1)
{
    return _bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_acos<int> >(d1);
}


/****************************************************************************
 * acosh
 ****************************************************************************/

#ifdef BZ_HAVE_IEEE_MATH
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_acosh<T_numtype1> > >
acosh(const Array<T_numtype1, N_rank1>& d1)
{
    return _bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_acosh<T_numtype1> >(d1.begin());
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_acosh<typename P_expr1::T_numtype> > >
acosh(_bz_ArrayExpr<P_expr1> d1)
{
    return _bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_acosh<typename P_expr1::T_numtype> >(d1);
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_acosh<int> > >
acosh(IndexPlaceholder<N_index1> d1)
{
    return _bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_acosh<int> >(d1);
}

#endif

/****************************************************************************
 * asin
 ****************************************************************************/

template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_asin<T_numtype1> > >
asin(const Array<T_numtype1, N_rank1>& d1)
{
    return _bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_asin<T_numtype1> >(d1.begin());
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_asin<typename P_expr1::T_numtype> > >
asin(_bz_ArrayExpr<P_expr1> d1)
{
    return _bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_asin<typename P_expr1::T_numtype> >(d1);
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_asin<int> > >
asin(IndexPlaceholder<N_index1> d1)
{
    return _bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_asin<int> >(d1);
}


/****************************************************************************
 * asinh
 ****************************************************************************/

#ifdef BZ_HAVE_IEEE_MATH
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_asinh<T_numtype1> > >
asinh(const Array<T_numtype1, N_rank1>& d1)
{
    return _bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_asinh<T_numtype1> >(d1.begin());
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_asinh<typename P_expr1::T_numtype> > >
asinh(_bz_ArrayExpr<P_expr1> d1)
{
    return _bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_asinh<typename P_expr1::T_numtype> >(d1);
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_asinh<int> > >
asinh(IndexPlaceholder<N_index1> d1)
{
    return _bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_asinh<int> >(d1);
}

#endif

/****************************************************************************
 * atan
 ****************************************************************************/

template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_atan<T_numtype1> > >
atan(const Array<T_numtype1, N_rank1>& d1)
{
    return _bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_atan<T_numtype1> >(d1.begin());
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_atan<typename P_expr1::T_numtype> > >
atan(_bz_ArrayExpr<P_expr1> d1)
{
    return _bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_atan<typename P_expr1::T_numtype> >(d1);
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_atan<int> > >
atan(IndexPlaceholder<N_index1> d1)
{
    return _bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_atan<int> >(d1);
}


/****************************************************************************
 * atanh
 ****************************************************************************/

#ifdef BZ_HAVE_IEEE_MATH
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_atanh<T_numtype1> > >
atanh(const Array<T_numtype1, N_rank1>& d1)
{
    return _bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_atanh<T_numtype1> >(d1.begin());
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_atanh<typename P_expr1::T_numtype> > >
atanh(_bz_ArrayExpr<P_expr1> d1)
{
    return _bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_atanh<typename P_expr1::T_numtype> >(d1);
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_atanh<int> > >
atanh(IndexPlaceholder<N_index1> d1)
{
    return _bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_atanh<int> >(d1);
}

#endif

/****************************************************************************
 * atan2
 ****************************************************************************/

template<class T_numtype1, int N_rank1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_atan2<T_numtype1,T_numtype2> > >
atan2(const Array<T_numtype1, N_rank1>& d1, const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_atan2<T_numtype1,T_numtype2> >(d1.begin(), d2.begin());
}

template<class T_numtype1, int N_rank1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExpr<P_expr2>,
    _bz_atan2<T_numtype1,typename P_expr2::T_numtype> > >
atan2(const Array<T_numtype1, N_rank1>& d1, _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExpr<P_expr2>,
    _bz_atan2<T_numtype1,typename P_expr2::T_numtype> >(d1.begin(), d2);
}

template<class T_numtype1, int N_rank1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, IndexPlaceholder<N_index2>,
    _bz_atan2<T_numtype1,int> > >
atan2(const Array<T_numtype1, N_rank1>& d1, IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, IndexPlaceholder<N_index2>,
    _bz_atan2<T_numtype1,int> >(d1.begin(), d2);
}

template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<float>,
    _bz_atan2<T_numtype1,float> > >
atan2(const Array<T_numtype1, N_rank1>& d1, float d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<float>,
    _bz_atan2<T_numtype1,float> >(d1.begin(), _bz_ArrayExprConstant<float>(d2));
}

template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<double>,
    _bz_atan2<T_numtype1,double> > >
atan2(const Array<T_numtype1, N_rank1>& d1, double d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<double>,
    _bz_atan2<T_numtype1,double> >(d1.begin(), _bz_ArrayExprConstant<double>(d2));
}

template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<long double>,
    _bz_atan2<T_numtype1,long double> > >
atan2(const Array<T_numtype1, N_rank1>& d1, long double d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<long double>,
    _bz_atan2<T_numtype1,long double> >(d1.begin(), _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
template<class T_numtype1, int N_rank1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<complex<T2> > ,
    _bz_atan2<T_numtype1,complex<T2> > > >
atan2(const Array<T_numtype1, N_rank1>& d1, complex<T2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<complex<T2> > ,
    _bz_atan2<T_numtype1,complex<T2> > >(d1.begin(), _bz_ArrayExprConstant<complex<T2> > (d2));
}

#endif // BZ_HAVE_COMPLEX

template<class P_expr1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_atan2<typename P_expr1::T_numtype,T_numtype2> > >
atan2(_bz_ArrayExpr<P_expr1> d1, const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_atan2<typename P_expr1::T_numtype,T_numtype2> >(d1, d2.begin());
}

template<class P_expr1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExpr<P_expr2>,
    _bz_atan2<typename P_expr1::T_numtype,typename P_expr2::T_numtype> > >
atan2(_bz_ArrayExpr<P_expr1> d1, _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExpr<P_expr2>,
    _bz_atan2<typename P_expr1::T_numtype,typename P_expr2::T_numtype> >(d1, d2);
}

template<class P_expr1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, IndexPlaceholder<N_index2>,
    _bz_atan2<typename P_expr1::T_numtype,int> > >
atan2(_bz_ArrayExpr<P_expr1> d1, IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, IndexPlaceholder<N_index2>,
    _bz_atan2<typename P_expr1::T_numtype,int> >(d1, d2);
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<float>,
    _bz_atan2<typename P_expr1::T_numtype,float> > >
atan2(_bz_ArrayExpr<P_expr1> d1, float d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<float>,
    _bz_atan2<typename P_expr1::T_numtype,float> >(d1, _bz_ArrayExprConstant<float>(d2));
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<double>,
    _bz_atan2<typename P_expr1::T_numtype,double> > >
atan2(_bz_ArrayExpr<P_expr1> d1, double d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<double>,
    _bz_atan2<typename P_expr1::T_numtype,double> >(d1, _bz_ArrayExprConstant<double>(d2));
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<long double>,
    _bz_atan2<typename P_expr1::T_numtype,long double> > >
atan2(_bz_ArrayExpr<P_expr1> d1, long double d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<long double>,
    _bz_atan2<typename P_expr1::T_numtype,long double> >(d1, _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
template<class P_expr1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<complex<T2> > ,
    _bz_atan2<typename P_expr1::T_numtype,complex<T2> > > >
atan2(_bz_ArrayExpr<P_expr1> d1, complex<T2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<complex<T2> > ,
    _bz_atan2<typename P_expr1::T_numtype,complex<T2> > >(d1, _bz_ArrayExprConstant<complex<T2> > (d2));
}

#endif // BZ_HAVE_COMPLEX

template<int N_index1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_atan2<int,T_numtype2> > >
atan2(IndexPlaceholder<N_index1> d1, const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_atan2<int,T_numtype2> >(d1, d2.begin());
}

template<int N_index1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExpr<P_expr2>,
    _bz_atan2<int,typename P_expr2::T_numtype> > >
atan2(IndexPlaceholder<N_index1> d1, _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExpr<P_expr2>,
    _bz_atan2<int,typename P_expr2::T_numtype> >(d1, d2);
}

template<int N_index1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, IndexPlaceholder<N_index2>,
    _bz_atan2<int,int> > >
atan2(IndexPlaceholder<N_index1> d1, IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, IndexPlaceholder<N_index2>,
    _bz_atan2<int,int> >(d1, d2);
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<float>,
    _bz_atan2<int,float> > >
atan2(IndexPlaceholder<N_index1> d1, float d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<float>,
    _bz_atan2<int,float> >(d1, _bz_ArrayExprConstant<float>(d2));
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<double>,
    _bz_atan2<int,double> > >
atan2(IndexPlaceholder<N_index1> d1, double d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<double>,
    _bz_atan2<int,double> >(d1, _bz_ArrayExprConstant<double>(d2));
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<long double>,
    _bz_atan2<int,long double> > >
atan2(IndexPlaceholder<N_index1> d1, long double d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<long double>,
    _bz_atan2<int,long double> >(d1, _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
template<int N_index1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<complex<T2> > ,
    _bz_atan2<int,complex<T2> > > >
atan2(IndexPlaceholder<N_index1> d1, complex<T2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<complex<T2> > ,
    _bz_atan2<int,complex<T2> > >(d1, _bz_ArrayExprConstant<complex<T2> > (d2));
}

#endif // BZ_HAVE_COMPLEX

template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_atan2<float,T_numtype2> > >
atan2(float d1, const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_atan2<float,T_numtype2> >(_bz_ArrayExprConstant<float>(d1), d2.begin());
}

template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, _bz_ArrayExpr<P_expr2>,
    _bz_atan2<float,typename P_expr2::T_numtype> > >
atan2(float d1, _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, _bz_ArrayExpr<P_expr2>,
    _bz_atan2<float,typename P_expr2::T_numtype> >(_bz_ArrayExprConstant<float>(d1), d2);
}

template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, IndexPlaceholder<N_index2>,
    _bz_atan2<float,int> > >
atan2(float d1, IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, IndexPlaceholder<N_index2>,
    _bz_atan2<float,int> >(_bz_ArrayExprConstant<float>(d1), d2);
}

template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_atan2<double,T_numtype2> > >
atan2(double d1, const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_atan2<double,T_numtype2> >(_bz_ArrayExprConstant<double>(d1), d2.begin());
}

template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, _bz_ArrayExpr<P_expr2>,
    _bz_atan2<double,typename P_expr2::T_numtype> > >
atan2(double d1, _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, _bz_ArrayExpr<P_expr2>,
    _bz_atan2<double,typename P_expr2::T_numtype> >(_bz_ArrayExprConstant<double>(d1), d2);
}

template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, IndexPlaceholder<N_index2>,
    _bz_atan2<double,int> > >
atan2(double d1, IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, IndexPlaceholder<N_index2>,
    _bz_atan2<double,int> >(_bz_ArrayExprConstant<double>(d1), d2);
}

template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_atan2<long double,T_numtype2> > >
atan2(long double d1, const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_atan2<long double,T_numtype2> >(_bz_ArrayExprConstant<long double>(d1), d2.begin());
}

template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, _bz_ArrayExpr<P_expr2>,
    _bz_atan2<long double,typename P_expr2::T_numtype> > >
atan2(long double d1, _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, _bz_ArrayExpr<P_expr2>,
    _bz_atan2<long double,typename P_expr2::T_numtype> >(_bz_ArrayExprConstant<long double>(d1), d2);
}

template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, IndexPlaceholder<N_index2>,
    _bz_atan2<long double,int> > >
atan2(long double d1, IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, IndexPlaceholder<N_index2>,
    _bz_atan2<long double,int> >(_bz_ArrayExprConstant<long double>(d1), d2);
}

#ifdef BZ_HAVE_COMPLEX
template<class T1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , ArrayIterator<T_numtype2, N_rank2>,
    _bz_atan2<complex<T1> ,T_numtype2> > >
atan2(complex<T1> d1, const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , ArrayIterator<T_numtype2, N_rank2>,
    _bz_atan2<complex<T1> ,T_numtype2> >(_bz_ArrayExprConstant<complex<T1> > (d1), d2.begin());
}

#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX
template<class T1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , _bz_ArrayExpr<P_expr2>,
    _bz_atan2<complex<T1> ,typename P_expr2::T_numtype> > >
atan2(complex<T1> d1, _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , _bz_ArrayExpr<P_expr2>,
    _bz_atan2<complex<T1> ,typename P_expr2::T_numtype> >(_bz_ArrayExprConstant<complex<T1> > (d1), d2);
}

#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX
template<class T1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , IndexPlaceholder<N_index2>,
    _bz_atan2<complex<T1> ,int> > >
atan2(complex<T1> d1, IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , IndexPlaceholder<N_index2>,
    _bz_atan2<complex<T1> ,int> >(_bz_ArrayExprConstant<complex<T1> > (d1), d2);
}

#endif // BZ_HAVE_COMPLEX


/****************************************************************************
 * _class
 ****************************************************************************/

#ifdef BZ_HAVE_SYSTEM_V_MATH
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz__class<T_numtype1> > >
_class(const Array<T_numtype1, N_rank1>& d1)
{
    return _bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz__class<T_numtype1> >(d1.begin());
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz__class<typename P_expr1::T_numtype> > >
_class(_bz_ArrayExpr<P_expr1> d1)
{
    return _bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz__class<typename P_expr1::T_numtype> >(d1);
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz__class<int> > >
_class(IndexPlaceholder<N_index1> d1)
{
    return _bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz__class<int> >(d1);
}

#endif

/****************************************************************************
 * cbrt
 ****************************************************************************/

#ifdef BZ_HAVE_IEEE_MATH
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_cbrt<T_numtype1> > >
cbrt(const Array<T_numtype1, N_rank1>& d1)
{
    return _bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_cbrt<T_numtype1> >(d1.begin());
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_cbrt<typename P_expr1::T_numtype> > >
cbrt(_bz_ArrayExpr<P_expr1> d1)
{
    return _bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_cbrt<typename P_expr1::T_numtype> >(d1);
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_cbrt<int> > >
cbrt(IndexPlaceholder<N_index1> d1)
{
    return _bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_cbrt<int> >(d1);
}

#endif

/****************************************************************************
 * ceil
 ****************************************************************************/

template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_ceil<T_numtype1> > >
ceil(const Array<T_numtype1, N_rank1>& d1)
{
    return _bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_ceil<T_numtype1> >(d1.begin());
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_ceil<typename P_expr1::T_numtype> > >
ceil(_bz_ArrayExpr<P_expr1> d1)
{
    return _bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_ceil<typename P_expr1::T_numtype> >(d1);
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_ceil<int> > >
ceil(IndexPlaceholder<N_index1> d1)
{
    return _bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_ceil<int> >(d1);
}


/****************************************************************************
 * cexp
 ****************************************************************************/

template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_cexp<T_numtype1> > >
cexp(const Array<T_numtype1, N_rank1>& d1)
{
    return _bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_cexp<T_numtype1> >(d1.begin());
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_cexp<typename P_expr1::T_numtype> > >
cexp(_bz_ArrayExpr<P_expr1> d1)
{
    return _bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_cexp<typename P_expr1::T_numtype> >(d1);
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_cexp<int> > >
cexp(IndexPlaceholder<N_index1> d1)
{
    return _bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_cexp<int> >(d1);
}


/****************************************************************************
 * cos
 ****************************************************************************/

template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_cos<T_numtype1> > >
cos(const Array<T_numtype1, N_rank1>& d1)
{
    return _bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_cos<T_numtype1> >(d1.begin());
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_cos<typename P_expr1::T_numtype> > >
cos(_bz_ArrayExpr<P_expr1> d1)
{
    return _bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_cos<typename P_expr1::T_numtype> >(d1);
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_cos<int> > >
cos(IndexPlaceholder<N_index1> d1)
{
    return _bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_cos<int> >(d1);
}


/****************************************************************************
 * cosh
 ****************************************************************************/

template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_cosh<T_numtype1> > >
cosh(const Array<T_numtype1, N_rank1>& d1)
{
    return _bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_cosh<T_numtype1> >(d1.begin());
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_cosh<typename P_expr1::T_numtype> > >
cosh(_bz_ArrayExpr<P_expr1> d1)
{
    return _bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_cosh<typename P_expr1::T_numtype> >(d1);
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_cosh<int> > >
cosh(IndexPlaceholder<N_index1> d1)
{
    return _bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_cosh<int> >(d1);
}


/****************************************************************************
 * copysign
 ****************************************************************************/

#ifdef BZ_HAVE_SYSTEM_V_MATH
template<class T_numtype1, int N_rank1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_copysign<T_numtype1,T_numtype2> > >
copysign(const Array<T_numtype1, N_rank1>& d1, const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_copysign<T_numtype1,T_numtype2> >(d1.begin(), d2.begin());
}

template<class T_numtype1, int N_rank1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExpr<P_expr2>,
    _bz_copysign<T_numtype1,typename P_expr2::T_numtype> > >
copysign(const Array<T_numtype1, N_rank1>& d1, _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExpr<P_expr2>,
    _bz_copysign<T_numtype1,typename P_expr2::T_numtype> >(d1.begin(), d2);
}

template<class T_numtype1, int N_rank1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, IndexPlaceholder<N_index2>,
    _bz_copysign<T_numtype1,int> > >
copysign(const Array<T_numtype1, N_rank1>& d1, IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, IndexPlaceholder<N_index2>,
    _bz_copysign<T_numtype1,int> >(d1.begin(), d2);
}

template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<float>,
    _bz_copysign<T_numtype1,float> > >
copysign(const Array<T_numtype1, N_rank1>& d1, float d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<float>,
    _bz_copysign<T_numtype1,float> >(d1.begin(), _bz_ArrayExprConstant<float>(d2));
}

template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<double>,
    _bz_copysign<T_numtype1,double> > >
copysign(const Array<T_numtype1, N_rank1>& d1, double d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<double>,
    _bz_copysign<T_numtype1,double> >(d1.begin(), _bz_ArrayExprConstant<double>(d2));
}

template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<long double>,
    _bz_copysign<T_numtype1,long double> > >
copysign(const Array<T_numtype1, N_rank1>& d1, long double d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<long double>,
    _bz_copysign<T_numtype1,long double> >(d1.begin(), _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
template<class T_numtype1, int N_rank1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<complex<T2> > ,
    _bz_copysign<T_numtype1,complex<T2> > > >
copysign(const Array<T_numtype1, N_rank1>& d1, complex<T2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<complex<T2> > ,
    _bz_copysign<T_numtype1,complex<T2> > >(d1.begin(), _bz_ArrayExprConstant<complex<T2> > (d2));
}

#endif // BZ_HAVE_COMPLEX

template<class P_expr1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_copysign<typename P_expr1::T_numtype,T_numtype2> > >
copysign(_bz_ArrayExpr<P_expr1> d1, const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_copysign<typename P_expr1::T_numtype,T_numtype2> >(d1, d2.begin());
}

template<class P_expr1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExpr<P_expr2>,
    _bz_copysign<typename P_expr1::T_numtype,typename P_expr2::T_numtype> > >
copysign(_bz_ArrayExpr<P_expr1> d1, _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExpr<P_expr2>,
    _bz_copysign<typename P_expr1::T_numtype,typename P_expr2::T_numtype> >(d1, d2);
}

template<class P_expr1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, IndexPlaceholder<N_index2>,
    _bz_copysign<typename P_expr1::T_numtype,int> > >
copysign(_bz_ArrayExpr<P_expr1> d1, IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, IndexPlaceholder<N_index2>,
    _bz_copysign<typename P_expr1::T_numtype,int> >(d1, d2);
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<float>,
    _bz_copysign<typename P_expr1::T_numtype,float> > >
copysign(_bz_ArrayExpr<P_expr1> d1, float d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<float>,
    _bz_copysign<typename P_expr1::T_numtype,float> >(d1, _bz_ArrayExprConstant<float>(d2));
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<double>,
    _bz_copysign<typename P_expr1::T_numtype,double> > >
copysign(_bz_ArrayExpr<P_expr1> d1, double d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<double>,
    _bz_copysign<typename P_expr1::T_numtype,double> >(d1, _bz_ArrayExprConstant<double>(d2));
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<long double>,
    _bz_copysign<typename P_expr1::T_numtype,long double> > >
copysign(_bz_ArrayExpr<P_expr1> d1, long double d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<long double>,
    _bz_copysign<typename P_expr1::T_numtype,long double> >(d1, _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
template<class P_expr1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<complex<T2> > ,
    _bz_copysign<typename P_expr1::T_numtype,complex<T2> > > >
copysign(_bz_ArrayExpr<P_expr1> d1, complex<T2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<complex<T2> > ,
    _bz_copysign<typename P_expr1::T_numtype,complex<T2> > >(d1, _bz_ArrayExprConstant<complex<T2> > (d2));
}

#endif // BZ_HAVE_COMPLEX

template<int N_index1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_copysign<int,T_numtype2> > >
copysign(IndexPlaceholder<N_index1> d1, const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_copysign<int,T_numtype2> >(d1, d2.begin());
}

template<int N_index1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExpr<P_expr2>,
    _bz_copysign<int,typename P_expr2::T_numtype> > >
copysign(IndexPlaceholder<N_index1> d1, _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExpr<P_expr2>,
    _bz_copysign<int,typename P_expr2::T_numtype> >(d1, d2);
}

template<int N_index1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, IndexPlaceholder<N_index2>,
    _bz_copysign<int,int> > >
copysign(IndexPlaceholder<N_index1> d1, IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, IndexPlaceholder<N_index2>,
    _bz_copysign<int,int> >(d1, d2);
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<float>,
    _bz_copysign<int,float> > >
copysign(IndexPlaceholder<N_index1> d1, float d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<float>,
    _bz_copysign<int,float> >(d1, _bz_ArrayExprConstant<float>(d2));
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<double>,
    _bz_copysign<int,double> > >
copysign(IndexPlaceholder<N_index1> d1, double d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<double>,
    _bz_copysign<int,double> >(d1, _bz_ArrayExprConstant<double>(d2));
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<long double>,
    _bz_copysign<int,long double> > >
copysign(IndexPlaceholder<N_index1> d1, long double d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<long double>,
    _bz_copysign<int,long double> >(d1, _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
template<int N_index1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<complex<T2> > ,
    _bz_copysign<int,complex<T2> > > >
copysign(IndexPlaceholder<N_index1> d1, complex<T2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<complex<T2> > ,
    _bz_copysign<int,complex<T2> > >(d1, _bz_ArrayExprConstant<complex<T2> > (d2));
}

#endif // BZ_HAVE_COMPLEX

template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_copysign<float,T_numtype2> > >
copysign(float d1, const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_copysign<float,T_numtype2> >(_bz_ArrayExprConstant<float>(d1), d2.begin());
}

template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, _bz_ArrayExpr<P_expr2>,
    _bz_copysign<float,typename P_expr2::T_numtype> > >
copysign(float d1, _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, _bz_ArrayExpr<P_expr2>,
    _bz_copysign<float,typename P_expr2::T_numtype> >(_bz_ArrayExprConstant<float>(d1), d2);
}

template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, IndexPlaceholder<N_index2>,
    _bz_copysign<float,int> > >
copysign(float d1, IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, IndexPlaceholder<N_index2>,
    _bz_copysign<float,int> >(_bz_ArrayExprConstant<float>(d1), d2);
}

template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_copysign<double,T_numtype2> > >
copysign(double d1, const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_copysign<double,T_numtype2> >(_bz_ArrayExprConstant<double>(d1), d2.begin());
}

template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, _bz_ArrayExpr<P_expr2>,
    _bz_copysign<double,typename P_expr2::T_numtype> > >
copysign(double d1, _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, _bz_ArrayExpr<P_expr2>,
    _bz_copysign<double,typename P_expr2::T_numtype> >(_bz_ArrayExprConstant<double>(d1), d2);
}

template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, IndexPlaceholder<N_index2>,
    _bz_copysign<double,int> > >
copysign(double d1, IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, IndexPlaceholder<N_index2>,
    _bz_copysign<double,int> >(_bz_ArrayExprConstant<double>(d1), d2);
}

template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_copysign<long double,T_numtype2> > >
copysign(long double d1, const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_copysign<long double,T_numtype2> >(_bz_ArrayExprConstant<long double>(d1), d2.begin());
}

template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, _bz_ArrayExpr<P_expr2>,
    _bz_copysign<long double,typename P_expr2::T_numtype> > >
copysign(long double d1, _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, _bz_ArrayExpr<P_expr2>,
    _bz_copysign<long double,typename P_expr2::T_numtype> >(_bz_ArrayExprConstant<long double>(d1), d2);
}

template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, IndexPlaceholder<N_index2>,
    _bz_copysign<long double,int> > >
copysign(long double d1, IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, IndexPlaceholder<N_index2>,
    _bz_copysign<long double,int> >(_bz_ArrayExprConstant<long double>(d1), d2);
}

#ifdef BZ_HAVE_COMPLEX
template<class T1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , ArrayIterator<T_numtype2, N_rank2>,
    _bz_copysign<complex<T1> ,T_numtype2> > >
copysign(complex<T1> d1, const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , ArrayIterator<T_numtype2, N_rank2>,
    _bz_copysign<complex<T1> ,T_numtype2> >(_bz_ArrayExprConstant<complex<T1> > (d1), d2.begin());
}

#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX
template<class T1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , _bz_ArrayExpr<P_expr2>,
    _bz_copysign<complex<T1> ,typename P_expr2::T_numtype> > >
copysign(complex<T1> d1, _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , _bz_ArrayExpr<P_expr2>,
    _bz_copysign<complex<T1> ,typename P_expr2::T_numtype> >(_bz_ArrayExprConstant<complex<T1> > (d1), d2);
}

#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX
template<class T1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , IndexPlaceholder<N_index2>,
    _bz_copysign<complex<T1> ,int> > >
copysign(complex<T1> d1, IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , IndexPlaceholder<N_index2>,
    _bz_copysign<complex<T1> ,int> >(_bz_ArrayExprConstant<complex<T1> > (d1), d2);
}

#endif // BZ_HAVE_COMPLEX

#endif

/****************************************************************************
 * csqrt
 ****************************************************************************/

template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_csqrt<T_numtype1> > >
csqrt(const Array<T_numtype1, N_rank1>& d1)
{
    return _bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_csqrt<T_numtype1> >(d1.begin());
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_csqrt<typename P_expr1::T_numtype> > >
csqrt(_bz_ArrayExpr<P_expr1> d1)
{
    return _bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_csqrt<typename P_expr1::T_numtype> >(d1);
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_csqrt<int> > >
csqrt(IndexPlaceholder<N_index1> d1)
{
    return _bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_csqrt<int> >(d1);
}


/****************************************************************************
 * drem
 ****************************************************************************/

#ifdef BZ_HAVE_SYSTEM_V_MATH
template<class T_numtype1, int N_rank1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_drem<T_numtype1,T_numtype2> > >
drem(const Array<T_numtype1, N_rank1>& d1, const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_drem<T_numtype1,T_numtype2> >(d1.begin(), d2.begin());
}

template<class T_numtype1, int N_rank1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExpr<P_expr2>,
    _bz_drem<T_numtype1,typename P_expr2::T_numtype> > >
drem(const Array<T_numtype1, N_rank1>& d1, _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExpr<P_expr2>,
    _bz_drem<T_numtype1,typename P_expr2::T_numtype> >(d1.begin(), d2);
}

template<class T_numtype1, int N_rank1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, IndexPlaceholder<N_index2>,
    _bz_drem<T_numtype1,int> > >
drem(const Array<T_numtype1, N_rank1>& d1, IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, IndexPlaceholder<N_index2>,
    _bz_drem<T_numtype1,int> >(d1.begin(), d2);
}

template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<float>,
    _bz_drem<T_numtype1,float> > >
drem(const Array<T_numtype1, N_rank1>& d1, float d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<float>,
    _bz_drem<T_numtype1,float> >(d1.begin(), _bz_ArrayExprConstant<float>(d2));
}

template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<double>,
    _bz_drem<T_numtype1,double> > >
drem(const Array<T_numtype1, N_rank1>& d1, double d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<double>,
    _bz_drem<T_numtype1,double> >(d1.begin(), _bz_ArrayExprConstant<double>(d2));
}

template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<long double>,
    _bz_drem<T_numtype1,long double> > >
drem(const Array<T_numtype1, N_rank1>& d1, long double d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<long double>,
    _bz_drem<T_numtype1,long double> >(d1.begin(), _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
template<class T_numtype1, int N_rank1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<complex<T2> > ,
    _bz_drem<T_numtype1,complex<T2> > > >
drem(const Array<T_numtype1, N_rank1>& d1, complex<T2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<complex<T2> > ,
    _bz_drem<T_numtype1,complex<T2> > >(d1.begin(), _bz_ArrayExprConstant<complex<T2> > (d2));
}

#endif // BZ_HAVE_COMPLEX

template<class P_expr1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_drem<typename P_expr1::T_numtype,T_numtype2> > >
drem(_bz_ArrayExpr<P_expr1> d1, const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_drem<typename P_expr1::T_numtype,T_numtype2> >(d1, d2.begin());
}

template<class P_expr1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExpr<P_expr2>,
    _bz_drem<typename P_expr1::T_numtype,typename P_expr2::T_numtype> > >
drem(_bz_ArrayExpr<P_expr1> d1, _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExpr<P_expr2>,
    _bz_drem<typename P_expr1::T_numtype,typename P_expr2::T_numtype> >(d1, d2);
}

template<class P_expr1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, IndexPlaceholder<N_index2>,
    _bz_drem<typename P_expr1::T_numtype,int> > >
drem(_bz_ArrayExpr<P_expr1> d1, IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, IndexPlaceholder<N_index2>,
    _bz_drem<typename P_expr1::T_numtype,int> >(d1, d2);
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<float>,
    _bz_drem<typename P_expr1::T_numtype,float> > >
drem(_bz_ArrayExpr<P_expr1> d1, float d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<float>,
    _bz_drem<typename P_expr1::T_numtype,float> >(d1, _bz_ArrayExprConstant<float>(d2));
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<double>,
    _bz_drem<typename P_expr1::T_numtype,double> > >
drem(_bz_ArrayExpr<P_expr1> d1, double d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<double>,
    _bz_drem<typename P_expr1::T_numtype,double> >(d1, _bz_ArrayExprConstant<double>(d2));
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<long double>,
    _bz_drem<typename P_expr1::T_numtype,long double> > >
drem(_bz_ArrayExpr<P_expr1> d1, long double d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<long double>,
    _bz_drem<typename P_expr1::T_numtype,long double> >(d1, _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
template<class P_expr1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<complex<T2> > ,
    _bz_drem<typename P_expr1::T_numtype,complex<T2> > > >
drem(_bz_ArrayExpr<P_expr1> d1, complex<T2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<complex<T2> > ,
    _bz_drem<typename P_expr1::T_numtype,complex<T2> > >(d1, _bz_ArrayExprConstant<complex<T2> > (d2));
}

#endif // BZ_HAVE_COMPLEX

template<int N_index1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_drem<int,T_numtype2> > >
drem(IndexPlaceholder<N_index1> d1, const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_drem<int,T_numtype2> >(d1, d2.begin());
}

template<int N_index1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExpr<P_expr2>,
    _bz_drem<int,typename P_expr2::T_numtype> > >
drem(IndexPlaceholder<N_index1> d1, _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExpr<P_expr2>,
    _bz_drem<int,typename P_expr2::T_numtype> >(d1, d2);
}

template<int N_index1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, IndexPlaceholder<N_index2>,
    _bz_drem<int,int> > >
drem(IndexPlaceholder<N_index1> d1, IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, IndexPlaceholder<N_index2>,
    _bz_drem<int,int> >(d1, d2);
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<float>,
    _bz_drem<int,float> > >
drem(IndexPlaceholder<N_index1> d1, float d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<float>,
    _bz_drem<int,float> >(d1, _bz_ArrayExprConstant<float>(d2));
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<double>,
    _bz_drem<int,double> > >
drem(IndexPlaceholder<N_index1> d1, double d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<double>,
    _bz_drem<int,double> >(d1, _bz_ArrayExprConstant<double>(d2));
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<long double>,
    _bz_drem<int,long double> > >
drem(IndexPlaceholder<N_index1> d1, long double d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<long double>,
    _bz_drem<int,long double> >(d1, _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
template<int N_index1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<complex<T2> > ,
    _bz_drem<int,complex<T2> > > >
drem(IndexPlaceholder<N_index1> d1, complex<T2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<complex<T2> > ,
    _bz_drem<int,complex<T2> > >(d1, _bz_ArrayExprConstant<complex<T2> > (d2));
}

#endif // BZ_HAVE_COMPLEX

template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_drem<float,T_numtype2> > >
drem(float d1, const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_drem<float,T_numtype2> >(_bz_ArrayExprConstant<float>(d1), d2.begin());
}

template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, _bz_ArrayExpr<P_expr2>,
    _bz_drem<float,typename P_expr2::T_numtype> > >
drem(float d1, _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, _bz_ArrayExpr<P_expr2>,
    _bz_drem<float,typename P_expr2::T_numtype> >(_bz_ArrayExprConstant<float>(d1), d2);
}

template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, IndexPlaceholder<N_index2>,
    _bz_drem<float,int> > >
drem(float d1, IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, IndexPlaceholder<N_index2>,
    _bz_drem<float,int> >(_bz_ArrayExprConstant<float>(d1), d2);
}

template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_drem<double,T_numtype2> > >
drem(double d1, const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_drem<double,T_numtype2> >(_bz_ArrayExprConstant<double>(d1), d2.begin());
}

template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, _bz_ArrayExpr<P_expr2>,
    _bz_drem<double,typename P_expr2::T_numtype> > >
drem(double d1, _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, _bz_ArrayExpr<P_expr2>,
    _bz_drem<double,typename P_expr2::T_numtype> >(_bz_ArrayExprConstant<double>(d1), d2);
}

template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, IndexPlaceholder<N_index2>,
    _bz_drem<double,int> > >
drem(double d1, IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, IndexPlaceholder<N_index2>,
    _bz_drem<double,int> >(_bz_ArrayExprConstant<double>(d1), d2);
}

template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_drem<long double,T_numtype2> > >
drem(long double d1, const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_drem<long double,T_numtype2> >(_bz_ArrayExprConstant<long double>(d1), d2.begin());
}

template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, _bz_ArrayExpr<P_expr2>,
    _bz_drem<long double,typename P_expr2::T_numtype> > >
drem(long double d1, _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, _bz_ArrayExpr<P_expr2>,
    _bz_drem<long double,typename P_expr2::T_numtype> >(_bz_ArrayExprConstant<long double>(d1), d2);
}

template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, IndexPlaceholder<N_index2>,
    _bz_drem<long double,int> > >
drem(long double d1, IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, IndexPlaceholder<N_index2>,
    _bz_drem<long double,int> >(_bz_ArrayExprConstant<long double>(d1), d2);
}

#ifdef BZ_HAVE_COMPLEX
template<class T1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , ArrayIterator<T_numtype2, N_rank2>,
    _bz_drem<complex<T1> ,T_numtype2> > >
drem(complex<T1> d1, const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , ArrayIterator<T_numtype2, N_rank2>,
    _bz_drem<complex<T1> ,T_numtype2> >(_bz_ArrayExprConstant<complex<T1> > (d1), d2.begin());
}

#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX
template<class T1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , _bz_ArrayExpr<P_expr2>,
    _bz_drem<complex<T1> ,typename P_expr2::T_numtype> > >
drem(complex<T1> d1, _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , _bz_ArrayExpr<P_expr2>,
    _bz_drem<complex<T1> ,typename P_expr2::T_numtype> >(_bz_ArrayExprConstant<complex<T1> > (d1), d2);
}

#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX
template<class T1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , IndexPlaceholder<N_index2>,
    _bz_drem<complex<T1> ,int> > >
drem(complex<T1> d1, IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , IndexPlaceholder<N_index2>,
    _bz_drem<complex<T1> ,int> >(_bz_ArrayExprConstant<complex<T1> > (d1), d2);
}

#endif // BZ_HAVE_COMPLEX

#endif

/****************************************************************************
 * exp
 ****************************************************************************/

template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_exp<T_numtype1> > >
exp(const Array<T_numtype1, N_rank1>& d1)
{
    return _bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_exp<T_numtype1> >(d1.begin());
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_exp<typename P_expr1::T_numtype> > >
exp(_bz_ArrayExpr<P_expr1> d1)
{
    return _bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_exp<typename P_expr1::T_numtype> >(d1);
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_exp<int> > >
exp(IndexPlaceholder<N_index1> d1)
{
    return _bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_exp<int> >(d1);
}


/****************************************************************************
 * expm1
 ****************************************************************************/

#ifdef BZ_HAVE_IEEE_MATH
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_expm1<T_numtype1> > >
expm1(const Array<T_numtype1, N_rank1>& d1)
{
    return _bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_expm1<T_numtype1> >(d1.begin());
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_expm1<typename P_expr1::T_numtype> > >
expm1(_bz_ArrayExpr<P_expr1> d1)
{
    return _bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_expm1<typename P_expr1::T_numtype> >(d1);
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_expm1<int> > >
expm1(IndexPlaceholder<N_index1> d1)
{
    return _bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_expm1<int> >(d1);
}

#endif

/****************************************************************************
 * erf
 ****************************************************************************/

#ifdef BZ_HAVE_IEEE_MATH
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_erf<T_numtype1> > >
erf(const Array<T_numtype1, N_rank1>& d1)
{
    return _bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_erf<T_numtype1> >(d1.begin());
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_erf<typename P_expr1::T_numtype> > >
erf(_bz_ArrayExpr<P_expr1> d1)
{
    return _bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_erf<typename P_expr1::T_numtype> >(d1);
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_erf<int> > >
erf(IndexPlaceholder<N_index1> d1)
{
    return _bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_erf<int> >(d1);
}

#endif

/****************************************************************************
 * erfc
 ****************************************************************************/

#ifdef BZ_HAVE_IEEE_MATH
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_erfc<T_numtype1> > >
erfc(const Array<T_numtype1, N_rank1>& d1)
{
    return _bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_erfc<T_numtype1> >(d1.begin());
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_erfc<typename P_expr1::T_numtype> > >
erfc(_bz_ArrayExpr<P_expr1> d1)
{
    return _bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_erfc<typename P_expr1::T_numtype> >(d1);
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_erfc<int> > >
erfc(IndexPlaceholder<N_index1> d1)
{
    return _bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_erfc<int> >(d1);
}

#endif

/****************************************************************************
 * fabs
 ****************************************************************************/

template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_abs<T_numtype1> > >
fabs(const Array<T_numtype1, N_rank1>& d1)
{
    return _bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_abs<T_numtype1> >(d1.begin());
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_abs<typename P_expr1::T_numtype> > >
fabs(_bz_ArrayExpr<P_expr1> d1)
{
    return _bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_abs<typename P_expr1::T_numtype> >(d1);
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_abs<int> > >
fabs(IndexPlaceholder<N_index1> d1)
{
    return _bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_abs<int> >(d1);
}


/****************************************************************************
 * floor
 ****************************************************************************/

template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_floor<T_numtype1> > >
floor(const Array<T_numtype1, N_rank1>& d1)
{
    return _bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_floor<T_numtype1> >(d1.begin());
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_floor<typename P_expr1::T_numtype> > >
floor(_bz_ArrayExpr<P_expr1> d1)
{
    return _bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_floor<typename P_expr1::T_numtype> >(d1);
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_floor<int> > >
floor(IndexPlaceholder<N_index1> d1)
{
    return _bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_floor<int> >(d1);
}


/****************************************************************************
 * fmod
 ****************************************************************************/

#ifdef BZ_HAVE_SYSTEM_V_MATH
template<class T_numtype1, int N_rank1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_fmod<T_numtype1,T_numtype2> > >
fmod(const Array<T_numtype1, N_rank1>& d1, const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_fmod<T_numtype1,T_numtype2> >(d1.begin(), d2.begin());
}

template<class T_numtype1, int N_rank1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExpr<P_expr2>,
    _bz_fmod<T_numtype1,typename P_expr2::T_numtype> > >
fmod(const Array<T_numtype1, N_rank1>& d1, _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExpr<P_expr2>,
    _bz_fmod<T_numtype1,typename P_expr2::T_numtype> >(d1.begin(), d2);
}

template<class T_numtype1, int N_rank1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, IndexPlaceholder<N_index2>,
    _bz_fmod<T_numtype1,int> > >
fmod(const Array<T_numtype1, N_rank1>& d1, IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, IndexPlaceholder<N_index2>,
    _bz_fmod<T_numtype1,int> >(d1.begin(), d2);
}

template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<float>,
    _bz_fmod<T_numtype1,float> > >
fmod(const Array<T_numtype1, N_rank1>& d1, float d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<float>,
    _bz_fmod<T_numtype1,float> >(d1.begin(), _bz_ArrayExprConstant<float>(d2));
}

template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<double>,
    _bz_fmod<T_numtype1,double> > >
fmod(const Array<T_numtype1, N_rank1>& d1, double d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<double>,
    _bz_fmod<T_numtype1,double> >(d1.begin(), _bz_ArrayExprConstant<double>(d2));
}

template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<long double>,
    _bz_fmod<T_numtype1,long double> > >
fmod(const Array<T_numtype1, N_rank1>& d1, long double d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<long double>,
    _bz_fmod<T_numtype1,long double> >(d1.begin(), _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
template<class T_numtype1, int N_rank1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<complex<T2> > ,
    _bz_fmod<T_numtype1,complex<T2> > > >
fmod(const Array<T_numtype1, N_rank1>& d1, complex<T2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<complex<T2> > ,
    _bz_fmod<T_numtype1,complex<T2> > >(d1.begin(), _bz_ArrayExprConstant<complex<T2> > (d2));
}

#endif // BZ_HAVE_COMPLEX

template<class P_expr1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_fmod<typename P_expr1::T_numtype,T_numtype2> > >
fmod(_bz_ArrayExpr<P_expr1> d1, const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_fmod<typename P_expr1::T_numtype,T_numtype2> >(d1, d2.begin());
}

template<class P_expr1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExpr<P_expr2>,
    _bz_fmod<typename P_expr1::T_numtype,typename P_expr2::T_numtype> > >
fmod(_bz_ArrayExpr<P_expr1> d1, _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExpr<P_expr2>,
    _bz_fmod<typename P_expr1::T_numtype,typename P_expr2::T_numtype> >(d1, d2);
}

template<class P_expr1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, IndexPlaceholder<N_index2>,
    _bz_fmod<typename P_expr1::T_numtype,int> > >
fmod(_bz_ArrayExpr<P_expr1> d1, IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, IndexPlaceholder<N_index2>,
    _bz_fmod<typename P_expr1::T_numtype,int> >(d1, d2);
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<float>,
    _bz_fmod<typename P_expr1::T_numtype,float> > >
fmod(_bz_ArrayExpr<P_expr1> d1, float d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<float>,
    _bz_fmod<typename P_expr1::T_numtype,float> >(d1, _bz_ArrayExprConstant<float>(d2));
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<double>,
    _bz_fmod<typename P_expr1::T_numtype,double> > >
fmod(_bz_ArrayExpr<P_expr1> d1, double d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<double>,
    _bz_fmod<typename P_expr1::T_numtype,double> >(d1, _bz_ArrayExprConstant<double>(d2));
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<long double>,
    _bz_fmod<typename P_expr1::T_numtype,long double> > >
fmod(_bz_ArrayExpr<P_expr1> d1, long double d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<long double>,
    _bz_fmod<typename P_expr1::T_numtype,long double> >(d1, _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
template<class P_expr1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<complex<T2> > ,
    _bz_fmod<typename P_expr1::T_numtype,complex<T2> > > >
fmod(_bz_ArrayExpr<P_expr1> d1, complex<T2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<complex<T2> > ,
    _bz_fmod<typename P_expr1::T_numtype,complex<T2> > >(d1, _bz_ArrayExprConstant<complex<T2> > (d2));
}

#endif // BZ_HAVE_COMPLEX

template<int N_index1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_fmod<int,T_numtype2> > >
fmod(IndexPlaceholder<N_index1> d1, const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_fmod<int,T_numtype2> >(d1, d2.begin());
}

template<int N_index1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExpr<P_expr2>,
    _bz_fmod<int,typename P_expr2::T_numtype> > >
fmod(IndexPlaceholder<N_index1> d1, _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExpr<P_expr2>,
    _bz_fmod<int,typename P_expr2::T_numtype> >(d1, d2);
}

template<int N_index1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, IndexPlaceholder<N_index2>,
    _bz_fmod<int,int> > >
fmod(IndexPlaceholder<N_index1> d1, IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, IndexPlaceholder<N_index2>,
    _bz_fmod<int,int> >(d1, d2);
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<float>,
    _bz_fmod<int,float> > >
fmod(IndexPlaceholder<N_index1> d1, float d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<float>,
    _bz_fmod<int,float> >(d1, _bz_ArrayExprConstant<float>(d2));
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<double>,
    _bz_fmod<int,double> > >
fmod(IndexPlaceholder<N_index1> d1, double d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<double>,
    _bz_fmod<int,double> >(d1, _bz_ArrayExprConstant<double>(d2));
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<long double>,
    _bz_fmod<int,long double> > >
fmod(IndexPlaceholder<N_index1> d1, long double d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<long double>,
    _bz_fmod<int,long double> >(d1, _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
template<int N_index1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<complex<T2> > ,
    _bz_fmod<int,complex<T2> > > >
fmod(IndexPlaceholder<N_index1> d1, complex<T2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<complex<T2> > ,
    _bz_fmod<int,complex<T2> > >(d1, _bz_ArrayExprConstant<complex<T2> > (d2));
}

#endif // BZ_HAVE_COMPLEX

template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_fmod<float,T_numtype2> > >
fmod(float d1, const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_fmod<float,T_numtype2> >(_bz_ArrayExprConstant<float>(d1), d2.begin());
}

template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, _bz_ArrayExpr<P_expr2>,
    _bz_fmod<float,typename P_expr2::T_numtype> > >
fmod(float d1, _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, _bz_ArrayExpr<P_expr2>,
    _bz_fmod<float,typename P_expr2::T_numtype> >(_bz_ArrayExprConstant<float>(d1), d2);
}

template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, IndexPlaceholder<N_index2>,
    _bz_fmod<float,int> > >
fmod(float d1, IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, IndexPlaceholder<N_index2>,
    _bz_fmod<float,int> >(_bz_ArrayExprConstant<float>(d1), d2);
}

template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_fmod<double,T_numtype2> > >
fmod(double d1, const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_fmod<double,T_numtype2> >(_bz_ArrayExprConstant<double>(d1), d2.begin());
}

template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, _bz_ArrayExpr<P_expr2>,
    _bz_fmod<double,typename P_expr2::T_numtype> > >
fmod(double d1, _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, _bz_ArrayExpr<P_expr2>,
    _bz_fmod<double,typename P_expr2::T_numtype> >(_bz_ArrayExprConstant<double>(d1), d2);
}

template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, IndexPlaceholder<N_index2>,
    _bz_fmod<double,int> > >
fmod(double d1, IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, IndexPlaceholder<N_index2>,
    _bz_fmod<double,int> >(_bz_ArrayExprConstant<double>(d1), d2);
}

template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_fmod<long double,T_numtype2> > >
fmod(long double d1, const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_fmod<long double,T_numtype2> >(_bz_ArrayExprConstant<long double>(d1), d2.begin());
}

template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, _bz_ArrayExpr<P_expr2>,
    _bz_fmod<long double,typename P_expr2::T_numtype> > >
fmod(long double d1, _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, _bz_ArrayExpr<P_expr2>,
    _bz_fmod<long double,typename P_expr2::T_numtype> >(_bz_ArrayExprConstant<long double>(d1), d2);
}

template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, IndexPlaceholder<N_index2>,
    _bz_fmod<long double,int> > >
fmod(long double d1, IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, IndexPlaceholder<N_index2>,
    _bz_fmod<long double,int> >(_bz_ArrayExprConstant<long double>(d1), d2);
}

#ifdef BZ_HAVE_COMPLEX
template<class T1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , ArrayIterator<T_numtype2, N_rank2>,
    _bz_fmod<complex<T1> ,T_numtype2> > >
fmod(complex<T1> d1, const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , ArrayIterator<T_numtype2, N_rank2>,
    _bz_fmod<complex<T1> ,T_numtype2> >(_bz_ArrayExprConstant<complex<T1> > (d1), d2.begin());
}

#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX
template<class T1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , _bz_ArrayExpr<P_expr2>,
    _bz_fmod<complex<T1> ,typename P_expr2::T_numtype> > >
fmod(complex<T1> d1, _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , _bz_ArrayExpr<P_expr2>,
    _bz_fmod<complex<T1> ,typename P_expr2::T_numtype> >(_bz_ArrayExprConstant<complex<T1> > (d1), d2);
}

#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX
template<class T1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , IndexPlaceholder<N_index2>,
    _bz_fmod<complex<T1> ,int> > >
fmod(complex<T1> d1, IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , IndexPlaceholder<N_index2>,
    _bz_fmod<complex<T1> ,int> >(_bz_ArrayExprConstant<complex<T1> > (d1), d2);
}

#endif // BZ_HAVE_COMPLEX

#endif

/****************************************************************************
 * hypot
 ****************************************************************************/

#ifdef BZ_HAVE_SYSTEM_V_MATH
template<class T_numtype1, int N_rank1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_hypot<T_numtype1,T_numtype2> > >
hypot(const Array<T_numtype1, N_rank1>& d1, const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_hypot<T_numtype1,T_numtype2> >(d1.begin(), d2.begin());
}

template<class T_numtype1, int N_rank1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExpr<P_expr2>,
    _bz_hypot<T_numtype1,typename P_expr2::T_numtype> > >
hypot(const Array<T_numtype1, N_rank1>& d1, _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExpr<P_expr2>,
    _bz_hypot<T_numtype1,typename P_expr2::T_numtype> >(d1.begin(), d2);
}

template<class T_numtype1, int N_rank1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, IndexPlaceholder<N_index2>,
    _bz_hypot<T_numtype1,int> > >
hypot(const Array<T_numtype1, N_rank1>& d1, IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, IndexPlaceholder<N_index2>,
    _bz_hypot<T_numtype1,int> >(d1.begin(), d2);
}

template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<float>,
    _bz_hypot<T_numtype1,float> > >
hypot(const Array<T_numtype1, N_rank1>& d1, float d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<float>,
    _bz_hypot<T_numtype1,float> >(d1.begin(), _bz_ArrayExprConstant<float>(d2));
}

template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<double>,
    _bz_hypot<T_numtype1,double> > >
hypot(const Array<T_numtype1, N_rank1>& d1, double d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<double>,
    _bz_hypot<T_numtype1,double> >(d1.begin(), _bz_ArrayExprConstant<double>(d2));
}

template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<long double>,
    _bz_hypot<T_numtype1,long double> > >
hypot(const Array<T_numtype1, N_rank1>& d1, long double d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<long double>,
    _bz_hypot<T_numtype1,long double> >(d1.begin(), _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
template<class T_numtype1, int N_rank1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<complex<T2> > ,
    _bz_hypot<T_numtype1,complex<T2> > > >
hypot(const Array<T_numtype1, N_rank1>& d1, complex<T2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<complex<T2> > ,
    _bz_hypot<T_numtype1,complex<T2> > >(d1.begin(), _bz_ArrayExprConstant<complex<T2> > (d2));
}

#endif // BZ_HAVE_COMPLEX

template<class P_expr1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_hypot<typename P_expr1::T_numtype,T_numtype2> > >
hypot(_bz_ArrayExpr<P_expr1> d1, const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_hypot<typename P_expr1::T_numtype,T_numtype2> >(d1, d2.begin());
}

template<class P_expr1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExpr<P_expr2>,
    _bz_hypot<typename P_expr1::T_numtype,typename P_expr2::T_numtype> > >
hypot(_bz_ArrayExpr<P_expr1> d1, _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExpr<P_expr2>,
    _bz_hypot<typename P_expr1::T_numtype,typename P_expr2::T_numtype> >(d1, d2);
}

template<class P_expr1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, IndexPlaceholder<N_index2>,
    _bz_hypot<typename P_expr1::T_numtype,int> > >
hypot(_bz_ArrayExpr<P_expr1> d1, IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, IndexPlaceholder<N_index2>,
    _bz_hypot<typename P_expr1::T_numtype,int> >(d1, d2);
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<float>,
    _bz_hypot<typename P_expr1::T_numtype,float> > >
hypot(_bz_ArrayExpr<P_expr1> d1, float d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<float>,
    _bz_hypot<typename P_expr1::T_numtype,float> >(d1, _bz_ArrayExprConstant<float>(d2));
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<double>,
    _bz_hypot<typename P_expr1::T_numtype,double> > >
hypot(_bz_ArrayExpr<P_expr1> d1, double d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<double>,
    _bz_hypot<typename P_expr1::T_numtype,double> >(d1, _bz_ArrayExprConstant<double>(d2));
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<long double>,
    _bz_hypot<typename P_expr1::T_numtype,long double> > >
hypot(_bz_ArrayExpr<P_expr1> d1, long double d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<long double>,
    _bz_hypot<typename P_expr1::T_numtype,long double> >(d1, _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
template<class P_expr1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<complex<T2> > ,
    _bz_hypot<typename P_expr1::T_numtype,complex<T2> > > >
hypot(_bz_ArrayExpr<P_expr1> d1, complex<T2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<complex<T2> > ,
    _bz_hypot<typename P_expr1::T_numtype,complex<T2> > >(d1, _bz_ArrayExprConstant<complex<T2> > (d2));
}

#endif // BZ_HAVE_COMPLEX

template<int N_index1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_hypot<int,T_numtype2> > >
hypot(IndexPlaceholder<N_index1> d1, const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_hypot<int,T_numtype2> >(d1, d2.begin());
}

template<int N_index1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExpr<P_expr2>,
    _bz_hypot<int,typename P_expr2::T_numtype> > >
hypot(IndexPlaceholder<N_index1> d1, _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExpr<P_expr2>,
    _bz_hypot<int,typename P_expr2::T_numtype> >(d1, d2);
}

template<int N_index1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, IndexPlaceholder<N_index2>,
    _bz_hypot<int,int> > >
hypot(IndexPlaceholder<N_index1> d1, IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, IndexPlaceholder<N_index2>,
    _bz_hypot<int,int> >(d1, d2);
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<float>,
    _bz_hypot<int,float> > >
hypot(IndexPlaceholder<N_index1> d1, float d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<float>,
    _bz_hypot<int,float> >(d1, _bz_ArrayExprConstant<float>(d2));
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<double>,
    _bz_hypot<int,double> > >
hypot(IndexPlaceholder<N_index1> d1, double d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<double>,
    _bz_hypot<int,double> >(d1, _bz_ArrayExprConstant<double>(d2));
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<long double>,
    _bz_hypot<int,long double> > >
hypot(IndexPlaceholder<N_index1> d1, long double d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<long double>,
    _bz_hypot<int,long double> >(d1, _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
template<int N_index1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<complex<T2> > ,
    _bz_hypot<int,complex<T2> > > >
hypot(IndexPlaceholder<N_index1> d1, complex<T2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<complex<T2> > ,
    _bz_hypot<int,complex<T2> > >(d1, _bz_ArrayExprConstant<complex<T2> > (d2));
}

#endif // BZ_HAVE_COMPLEX

template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_hypot<float,T_numtype2> > >
hypot(float d1, const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_hypot<float,T_numtype2> >(_bz_ArrayExprConstant<float>(d1), d2.begin());
}

template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, _bz_ArrayExpr<P_expr2>,
    _bz_hypot<float,typename P_expr2::T_numtype> > >
hypot(float d1, _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, _bz_ArrayExpr<P_expr2>,
    _bz_hypot<float,typename P_expr2::T_numtype> >(_bz_ArrayExprConstant<float>(d1), d2);
}

template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, IndexPlaceholder<N_index2>,
    _bz_hypot<float,int> > >
hypot(float d1, IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, IndexPlaceholder<N_index2>,
    _bz_hypot<float,int> >(_bz_ArrayExprConstant<float>(d1), d2);
}

template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_hypot<double,T_numtype2> > >
hypot(double d1, const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_hypot<double,T_numtype2> >(_bz_ArrayExprConstant<double>(d1), d2.begin());
}

template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, _bz_ArrayExpr<P_expr2>,
    _bz_hypot<double,typename P_expr2::T_numtype> > >
hypot(double d1, _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, _bz_ArrayExpr<P_expr2>,
    _bz_hypot<double,typename P_expr2::T_numtype> >(_bz_ArrayExprConstant<double>(d1), d2);
}

template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, IndexPlaceholder<N_index2>,
    _bz_hypot<double,int> > >
hypot(double d1, IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, IndexPlaceholder<N_index2>,
    _bz_hypot<double,int> >(_bz_ArrayExprConstant<double>(d1), d2);
}

template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_hypot<long double,T_numtype2> > >
hypot(long double d1, const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_hypot<long double,T_numtype2> >(_bz_ArrayExprConstant<long double>(d1), d2.begin());
}

template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, _bz_ArrayExpr<P_expr2>,
    _bz_hypot<long double,typename P_expr2::T_numtype> > >
hypot(long double d1, _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, _bz_ArrayExpr<P_expr2>,
    _bz_hypot<long double,typename P_expr2::T_numtype> >(_bz_ArrayExprConstant<long double>(d1), d2);
}

template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, IndexPlaceholder<N_index2>,
    _bz_hypot<long double,int> > >
hypot(long double d1, IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, IndexPlaceholder<N_index2>,
    _bz_hypot<long double,int> >(_bz_ArrayExprConstant<long double>(d1), d2);
}

#ifdef BZ_HAVE_COMPLEX
template<class T1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , ArrayIterator<T_numtype2, N_rank2>,
    _bz_hypot<complex<T1> ,T_numtype2> > >
hypot(complex<T1> d1, const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , ArrayIterator<T_numtype2, N_rank2>,
    _bz_hypot<complex<T1> ,T_numtype2> >(_bz_ArrayExprConstant<complex<T1> > (d1), d2.begin());
}

#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX
template<class T1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , _bz_ArrayExpr<P_expr2>,
    _bz_hypot<complex<T1> ,typename P_expr2::T_numtype> > >
hypot(complex<T1> d1, _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , _bz_ArrayExpr<P_expr2>,
    _bz_hypot<complex<T1> ,typename P_expr2::T_numtype> >(_bz_ArrayExprConstant<complex<T1> > (d1), d2);
}

#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX
template<class T1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , IndexPlaceholder<N_index2>,
    _bz_hypot<complex<T1> ,int> > >
hypot(complex<T1> d1, IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , IndexPlaceholder<N_index2>,
    _bz_hypot<complex<T1> ,int> >(_bz_ArrayExprConstant<complex<T1> > (d1), d2);
}

#endif // BZ_HAVE_COMPLEX

#endif

/****************************************************************************
 * ilogb
 ****************************************************************************/

#ifdef BZ_HAVE_SYSTEM_V_MATH
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_ilogb<T_numtype1> > >
ilogb(const Array<T_numtype1, N_rank1>& d1)
{
    return _bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_ilogb<T_numtype1> >(d1.begin());
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_ilogb<typename P_expr1::T_numtype> > >
ilogb(_bz_ArrayExpr<P_expr1> d1)
{
    return _bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_ilogb<typename P_expr1::T_numtype> >(d1);
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_ilogb<int> > >
ilogb(IndexPlaceholder<N_index1> d1)
{
    return _bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_ilogb<int> >(d1);
}

#endif

/****************************************************************************
 * isnan
 ****************************************************************************/

#ifdef BZ_HAVE_IEEE_MATH
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_isnan<T_numtype1> > >
isnan(const Array<T_numtype1, N_rank1>& d1)
{
    return _bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_isnan<T_numtype1> >(d1.begin());
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_isnan<typename P_expr1::T_numtype> > >
isnan(_bz_ArrayExpr<P_expr1> d1)
{
    return _bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_isnan<typename P_expr1::T_numtype> >(d1);
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_isnan<int> > >
isnan(IndexPlaceholder<N_index1> d1)
{
    return _bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_isnan<int> >(d1);
}

#endif

/****************************************************************************
 * itrunc
 ****************************************************************************/

#ifdef BZ_HAVE_SYSTEM_V_MATH
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_itrunc<T_numtype1> > >
itrunc(const Array<T_numtype1, N_rank1>& d1)
{
    return _bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_itrunc<T_numtype1> >(d1.begin());
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_itrunc<typename P_expr1::T_numtype> > >
itrunc(_bz_ArrayExpr<P_expr1> d1)
{
    return _bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_itrunc<typename P_expr1::T_numtype> >(d1);
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_itrunc<int> > >
itrunc(IndexPlaceholder<N_index1> d1)
{
    return _bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_itrunc<int> >(d1);
}

#endif

/****************************************************************************
 * j0
 ****************************************************************************/

#ifdef BZ_HAVE_IEEE_MATH
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_j0<T_numtype1> > >
j0(const Array<T_numtype1, N_rank1>& d1)
{
    return _bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_j0<T_numtype1> >(d1.begin());
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_j0<typename P_expr1::T_numtype> > >
j0(_bz_ArrayExpr<P_expr1> d1)
{
    return _bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_j0<typename P_expr1::T_numtype> >(d1);
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_j0<int> > >
j0(IndexPlaceholder<N_index1> d1)
{
    return _bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_j0<int> >(d1);
}

#endif

/****************************************************************************
 * j1
 ****************************************************************************/

#ifdef BZ_HAVE_IEEE_MATH
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_j1<T_numtype1> > >
j1(const Array<T_numtype1, N_rank1>& d1)
{
    return _bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_j1<T_numtype1> >(d1.begin());
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_j1<typename P_expr1::T_numtype> > >
j1(_bz_ArrayExpr<P_expr1> d1)
{
    return _bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_j1<typename P_expr1::T_numtype> >(d1);
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_j1<int> > >
j1(IndexPlaceholder<N_index1> d1)
{
    return _bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_j1<int> >(d1);
}

#endif

/****************************************************************************
 * lgamma
 ****************************************************************************/

#ifdef BZ_HAVE_IEEE_MATH
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_lgamma<T_numtype1> > >
lgamma(const Array<T_numtype1, N_rank1>& d1)
{
    return _bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_lgamma<T_numtype1> >(d1.begin());
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_lgamma<typename P_expr1::T_numtype> > >
lgamma(_bz_ArrayExpr<P_expr1> d1)
{
    return _bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_lgamma<typename P_expr1::T_numtype> >(d1);
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_lgamma<int> > >
lgamma(IndexPlaceholder<N_index1> d1)
{
    return _bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_lgamma<int> >(d1);
}

#endif

/****************************************************************************
 * log
 ****************************************************************************/

template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_log<T_numtype1> > >
log(const Array<T_numtype1, N_rank1>& d1)
{
    return _bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_log<T_numtype1> >(d1.begin());
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_log<typename P_expr1::T_numtype> > >
log(_bz_ArrayExpr<P_expr1> d1)
{
    return _bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_log<typename P_expr1::T_numtype> >(d1);
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_log<int> > >
log(IndexPlaceholder<N_index1> d1)
{
    return _bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_log<int> >(d1);
}


/****************************************************************************
 * logb
 ****************************************************************************/

#ifdef BZ_HAVE_IEEE_MATH
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_logb<T_numtype1> > >
logb(const Array<T_numtype1, N_rank1>& d1)
{
    return _bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_logb<T_numtype1> >(d1.begin());
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_logb<typename P_expr1::T_numtype> > >
logb(_bz_ArrayExpr<P_expr1> d1)
{
    return _bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_logb<typename P_expr1::T_numtype> >(d1);
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_logb<int> > >
logb(IndexPlaceholder<N_index1> d1)
{
    return _bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_logb<int> >(d1);
}

#endif

/****************************************************************************
 * log1p
 ****************************************************************************/

#ifdef BZ_HAVE_IEEE_MATH
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_log1p<T_numtype1> > >
log1p(const Array<T_numtype1, N_rank1>& d1)
{
    return _bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_log1p<T_numtype1> >(d1.begin());
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_log1p<typename P_expr1::T_numtype> > >
log1p(_bz_ArrayExpr<P_expr1> d1)
{
    return _bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_log1p<typename P_expr1::T_numtype> >(d1);
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_log1p<int> > >
log1p(IndexPlaceholder<N_index1> d1)
{
    return _bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_log1p<int> >(d1);
}

#endif

/****************************************************************************
 * log10
 ****************************************************************************/

template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_log10<T_numtype1> > >
log10(const Array<T_numtype1, N_rank1>& d1)
{
    return _bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_log10<T_numtype1> >(d1.begin());
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_log10<typename P_expr1::T_numtype> > >
log10(_bz_ArrayExpr<P_expr1> d1)
{
    return _bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_log10<typename P_expr1::T_numtype> >(d1);
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_log10<int> > >
log10(IndexPlaceholder<N_index1> d1)
{
    return _bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_log10<int> >(d1);
}


/****************************************************************************
 * nearest
 ****************************************************************************/

#ifdef BZ_HAVE_SYSTEM_V_MATH
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_nearest<T_numtype1> > >
nearest(const Array<T_numtype1, N_rank1>& d1)
{
    return _bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_nearest<T_numtype1> >(d1.begin());
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_nearest<typename P_expr1::T_numtype> > >
nearest(_bz_ArrayExpr<P_expr1> d1)
{
    return _bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_nearest<typename P_expr1::T_numtype> >(d1);
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_nearest<int> > >
nearest(IndexPlaceholder<N_index1> d1)
{
    return _bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_nearest<int> >(d1);
}

#endif

/****************************************************************************
 * nextafter
 ****************************************************************************/

#ifdef BZ_HAVE_SYSTEM_V_MATH
template<class T_numtype1, int N_rank1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_nextafter<T_numtype1,T_numtype2> > >
nextafter(const Array<T_numtype1, N_rank1>& d1, const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_nextafter<T_numtype1,T_numtype2> >(d1.begin(), d2.begin());
}

template<class T_numtype1, int N_rank1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExpr<P_expr2>,
    _bz_nextafter<T_numtype1,typename P_expr2::T_numtype> > >
nextafter(const Array<T_numtype1, N_rank1>& d1, _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExpr<P_expr2>,
    _bz_nextafter<T_numtype1,typename P_expr2::T_numtype> >(d1.begin(), d2);
}

template<class T_numtype1, int N_rank1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, IndexPlaceholder<N_index2>,
    _bz_nextafter<T_numtype1,int> > >
nextafter(const Array<T_numtype1, N_rank1>& d1, IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, IndexPlaceholder<N_index2>,
    _bz_nextafter<T_numtype1,int> >(d1.begin(), d2);
}

template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<float>,
    _bz_nextafter<T_numtype1,float> > >
nextafter(const Array<T_numtype1, N_rank1>& d1, float d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<float>,
    _bz_nextafter<T_numtype1,float> >(d1.begin(), _bz_ArrayExprConstant<float>(d2));
}

template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<double>,
    _bz_nextafter<T_numtype1,double> > >
nextafter(const Array<T_numtype1, N_rank1>& d1, double d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<double>,
    _bz_nextafter<T_numtype1,double> >(d1.begin(), _bz_ArrayExprConstant<double>(d2));
}

template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<long double>,
    _bz_nextafter<T_numtype1,long double> > >
nextafter(const Array<T_numtype1, N_rank1>& d1, long double d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<long double>,
    _bz_nextafter<T_numtype1,long double> >(d1.begin(), _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
template<class T_numtype1, int N_rank1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<complex<T2> > ,
    _bz_nextafter<T_numtype1,complex<T2> > > >
nextafter(const Array<T_numtype1, N_rank1>& d1, complex<T2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<complex<T2> > ,
    _bz_nextafter<T_numtype1,complex<T2> > >(d1.begin(), _bz_ArrayExprConstant<complex<T2> > (d2));
}

#endif // BZ_HAVE_COMPLEX

template<class P_expr1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_nextafter<typename P_expr1::T_numtype,T_numtype2> > >
nextafter(_bz_ArrayExpr<P_expr1> d1, const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_nextafter<typename P_expr1::T_numtype,T_numtype2> >(d1, d2.begin());
}

template<class P_expr1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExpr<P_expr2>,
    _bz_nextafter<typename P_expr1::T_numtype,typename P_expr2::T_numtype> > >
nextafter(_bz_ArrayExpr<P_expr1> d1, _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExpr<P_expr2>,
    _bz_nextafter<typename P_expr1::T_numtype,typename P_expr2::T_numtype> >(d1, d2);
}

template<class P_expr1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, IndexPlaceholder<N_index2>,
    _bz_nextafter<typename P_expr1::T_numtype,int> > >
nextafter(_bz_ArrayExpr<P_expr1> d1, IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, IndexPlaceholder<N_index2>,
    _bz_nextafter<typename P_expr1::T_numtype,int> >(d1, d2);
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<float>,
    _bz_nextafter<typename P_expr1::T_numtype,float> > >
nextafter(_bz_ArrayExpr<P_expr1> d1, float d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<float>,
    _bz_nextafter<typename P_expr1::T_numtype,float> >(d1, _bz_ArrayExprConstant<float>(d2));
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<double>,
    _bz_nextafter<typename P_expr1::T_numtype,double> > >
nextafter(_bz_ArrayExpr<P_expr1> d1, double d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<double>,
    _bz_nextafter<typename P_expr1::T_numtype,double> >(d1, _bz_ArrayExprConstant<double>(d2));
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<long double>,
    _bz_nextafter<typename P_expr1::T_numtype,long double> > >
nextafter(_bz_ArrayExpr<P_expr1> d1, long double d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<long double>,
    _bz_nextafter<typename P_expr1::T_numtype,long double> >(d1, _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
template<class P_expr1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<complex<T2> > ,
    _bz_nextafter<typename P_expr1::T_numtype,complex<T2> > > >
nextafter(_bz_ArrayExpr<P_expr1> d1, complex<T2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<complex<T2> > ,
    _bz_nextafter<typename P_expr1::T_numtype,complex<T2> > >(d1, _bz_ArrayExprConstant<complex<T2> > (d2));
}

#endif // BZ_HAVE_COMPLEX

template<int N_index1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_nextafter<int,T_numtype2> > >
nextafter(IndexPlaceholder<N_index1> d1, const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_nextafter<int,T_numtype2> >(d1, d2.begin());
}

template<int N_index1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExpr<P_expr2>,
    _bz_nextafter<int,typename P_expr2::T_numtype> > >
nextafter(IndexPlaceholder<N_index1> d1, _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExpr<P_expr2>,
    _bz_nextafter<int,typename P_expr2::T_numtype> >(d1, d2);
}

template<int N_index1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, IndexPlaceholder<N_index2>,
    _bz_nextafter<int,int> > >
nextafter(IndexPlaceholder<N_index1> d1, IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, IndexPlaceholder<N_index2>,
    _bz_nextafter<int,int> >(d1, d2);
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<float>,
    _bz_nextafter<int,float> > >
nextafter(IndexPlaceholder<N_index1> d1, float d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<float>,
    _bz_nextafter<int,float> >(d1, _bz_ArrayExprConstant<float>(d2));
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<double>,
    _bz_nextafter<int,double> > >
nextafter(IndexPlaceholder<N_index1> d1, double d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<double>,
    _bz_nextafter<int,double> >(d1, _bz_ArrayExprConstant<double>(d2));
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<long double>,
    _bz_nextafter<int,long double> > >
nextafter(IndexPlaceholder<N_index1> d1, long double d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<long double>,
    _bz_nextafter<int,long double> >(d1, _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
template<int N_index1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<complex<T2> > ,
    _bz_nextafter<int,complex<T2> > > >
nextafter(IndexPlaceholder<N_index1> d1, complex<T2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<complex<T2> > ,
    _bz_nextafter<int,complex<T2> > >(d1, _bz_ArrayExprConstant<complex<T2> > (d2));
}

#endif // BZ_HAVE_COMPLEX

template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_nextafter<float,T_numtype2> > >
nextafter(float d1, const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_nextafter<float,T_numtype2> >(_bz_ArrayExprConstant<float>(d1), d2.begin());
}

template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, _bz_ArrayExpr<P_expr2>,
    _bz_nextafter<float,typename P_expr2::T_numtype> > >
nextafter(float d1, _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, _bz_ArrayExpr<P_expr2>,
    _bz_nextafter<float,typename P_expr2::T_numtype> >(_bz_ArrayExprConstant<float>(d1), d2);
}

template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, IndexPlaceholder<N_index2>,
    _bz_nextafter<float,int> > >
nextafter(float d1, IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, IndexPlaceholder<N_index2>,
    _bz_nextafter<float,int> >(_bz_ArrayExprConstant<float>(d1), d2);
}

template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_nextafter<double,T_numtype2> > >
nextafter(double d1, const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_nextafter<double,T_numtype2> >(_bz_ArrayExprConstant<double>(d1), d2.begin());
}

template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, _bz_ArrayExpr<P_expr2>,
    _bz_nextafter<double,typename P_expr2::T_numtype> > >
nextafter(double d1, _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, _bz_ArrayExpr<P_expr2>,
    _bz_nextafter<double,typename P_expr2::T_numtype> >(_bz_ArrayExprConstant<double>(d1), d2);
}

template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, IndexPlaceholder<N_index2>,
    _bz_nextafter<double,int> > >
nextafter(double d1, IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, IndexPlaceholder<N_index2>,
    _bz_nextafter<double,int> >(_bz_ArrayExprConstant<double>(d1), d2);
}

template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_nextafter<long double,T_numtype2> > >
nextafter(long double d1, const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_nextafter<long double,T_numtype2> >(_bz_ArrayExprConstant<long double>(d1), d2.begin());
}

template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, _bz_ArrayExpr<P_expr2>,
    _bz_nextafter<long double,typename P_expr2::T_numtype> > >
nextafter(long double d1, _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, _bz_ArrayExpr<P_expr2>,
    _bz_nextafter<long double,typename P_expr2::T_numtype> >(_bz_ArrayExprConstant<long double>(d1), d2);
}

template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, IndexPlaceholder<N_index2>,
    _bz_nextafter<long double,int> > >
nextafter(long double d1, IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, IndexPlaceholder<N_index2>,
    _bz_nextafter<long double,int> >(_bz_ArrayExprConstant<long double>(d1), d2);
}

#ifdef BZ_HAVE_COMPLEX
template<class T1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , ArrayIterator<T_numtype2, N_rank2>,
    _bz_nextafter<complex<T1> ,T_numtype2> > >
nextafter(complex<T1> d1, const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , ArrayIterator<T_numtype2, N_rank2>,
    _bz_nextafter<complex<T1> ,T_numtype2> >(_bz_ArrayExprConstant<complex<T1> > (d1), d2.begin());
}

#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX
template<class T1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , _bz_ArrayExpr<P_expr2>,
    _bz_nextafter<complex<T1> ,typename P_expr2::T_numtype> > >
nextafter(complex<T1> d1, _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , _bz_ArrayExpr<P_expr2>,
    _bz_nextafter<complex<T1> ,typename P_expr2::T_numtype> >(_bz_ArrayExprConstant<complex<T1> > (d1), d2);
}

#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX
template<class T1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , IndexPlaceholder<N_index2>,
    _bz_nextafter<complex<T1> ,int> > >
nextafter(complex<T1> d1, IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , IndexPlaceholder<N_index2>,
    _bz_nextafter<complex<T1> ,int> >(_bz_ArrayExprConstant<complex<T1> > (d1), d2);
}

#endif // BZ_HAVE_COMPLEX

#endif

/****************************************************************************
 * pow
 ****************************************************************************/

template<class T_numtype1, int N_rank1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_pow<T_numtype1,T_numtype2> > >
pow(const Array<T_numtype1, N_rank1>& d1, const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_pow<T_numtype1,T_numtype2> >(d1.begin(), d2.begin());
}

template<class T_numtype1, int N_rank1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExpr<P_expr2>,
    _bz_pow<T_numtype1,typename P_expr2::T_numtype> > >
pow(const Array<T_numtype1, N_rank1>& d1, _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExpr<P_expr2>,
    _bz_pow<T_numtype1,typename P_expr2::T_numtype> >(d1.begin(), d2);
}

template<class T_numtype1, int N_rank1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, IndexPlaceholder<N_index2>,
    _bz_pow<T_numtype1,int> > >
pow(const Array<T_numtype1, N_rank1>& d1, IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, IndexPlaceholder<N_index2>,
    _bz_pow<T_numtype1,int> >(d1.begin(), d2);
}

template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<float>,
    _bz_pow<T_numtype1,float> > >
pow(const Array<T_numtype1, N_rank1>& d1, float d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<float>,
    _bz_pow<T_numtype1,float> >(d1.begin(), _bz_ArrayExprConstant<float>(d2));
}

template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<double>,
    _bz_pow<T_numtype1,double> > >
pow(const Array<T_numtype1, N_rank1>& d1, double d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<double>,
    _bz_pow<T_numtype1,double> >(d1.begin(), _bz_ArrayExprConstant<double>(d2));
}

template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<long double>,
    _bz_pow<T_numtype1,long double> > >
pow(const Array<T_numtype1, N_rank1>& d1, long double d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<long double>,
    _bz_pow<T_numtype1,long double> >(d1.begin(), _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
template<class T_numtype1, int N_rank1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<complex<T2> > ,
    _bz_pow<T_numtype1,complex<T2> > > >
pow(const Array<T_numtype1, N_rank1>& d1, complex<T2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<complex<T2> > ,
    _bz_pow<T_numtype1,complex<T2> > >(d1.begin(), _bz_ArrayExprConstant<complex<T2> > (d2));
}

#endif // BZ_HAVE_COMPLEX

template<class P_expr1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_pow<typename P_expr1::T_numtype,T_numtype2> > >
pow(_bz_ArrayExpr<P_expr1> d1, const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_pow<typename P_expr1::T_numtype,T_numtype2> >(d1, d2.begin());
}

template<class P_expr1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExpr<P_expr2>,
    _bz_pow<typename P_expr1::T_numtype,typename P_expr2::T_numtype> > >
pow(_bz_ArrayExpr<P_expr1> d1, _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExpr<P_expr2>,
    _bz_pow<typename P_expr1::T_numtype,typename P_expr2::T_numtype> >(d1, d2);
}

template<class P_expr1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, IndexPlaceholder<N_index2>,
    _bz_pow<typename P_expr1::T_numtype,int> > >
pow(_bz_ArrayExpr<P_expr1> d1, IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, IndexPlaceholder<N_index2>,
    _bz_pow<typename P_expr1::T_numtype,int> >(d1, d2);
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<float>,
    _bz_pow<typename P_expr1::T_numtype,float> > >
pow(_bz_ArrayExpr<P_expr1> d1, float d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<float>,
    _bz_pow<typename P_expr1::T_numtype,float> >(d1, _bz_ArrayExprConstant<float>(d2));
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<double>,
    _bz_pow<typename P_expr1::T_numtype,double> > >
pow(_bz_ArrayExpr<P_expr1> d1, double d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<double>,
    _bz_pow<typename P_expr1::T_numtype,double> >(d1, _bz_ArrayExprConstant<double>(d2));
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<long double>,
    _bz_pow<typename P_expr1::T_numtype,long double> > >
pow(_bz_ArrayExpr<P_expr1> d1, long double d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<long double>,
    _bz_pow<typename P_expr1::T_numtype,long double> >(d1, _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
template<class P_expr1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<complex<T2> > ,
    _bz_pow<typename P_expr1::T_numtype,complex<T2> > > >
pow(_bz_ArrayExpr<P_expr1> d1, complex<T2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<complex<T2> > ,
    _bz_pow<typename P_expr1::T_numtype,complex<T2> > >(d1, _bz_ArrayExprConstant<complex<T2> > (d2));
}

#endif // BZ_HAVE_COMPLEX

template<int N_index1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_pow<int,T_numtype2> > >
pow(IndexPlaceholder<N_index1> d1, const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_pow<int,T_numtype2> >(d1, d2.begin());
}

template<int N_index1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExpr<P_expr2>,
    _bz_pow<int,typename P_expr2::T_numtype> > >
pow(IndexPlaceholder<N_index1> d1, _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExpr<P_expr2>,
    _bz_pow<int,typename P_expr2::T_numtype> >(d1, d2);
}

template<int N_index1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, IndexPlaceholder<N_index2>,
    _bz_pow<int,int> > >
pow(IndexPlaceholder<N_index1> d1, IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, IndexPlaceholder<N_index2>,
    _bz_pow<int,int> >(d1, d2);
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<float>,
    _bz_pow<int,float> > >
pow(IndexPlaceholder<N_index1> d1, float d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<float>,
    _bz_pow<int,float> >(d1, _bz_ArrayExprConstant<float>(d2));
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<double>,
    _bz_pow<int,double> > >
pow(IndexPlaceholder<N_index1> d1, double d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<double>,
    _bz_pow<int,double> >(d1, _bz_ArrayExprConstant<double>(d2));
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<long double>,
    _bz_pow<int,long double> > >
pow(IndexPlaceholder<N_index1> d1, long double d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<long double>,
    _bz_pow<int,long double> >(d1, _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
template<int N_index1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<complex<T2> > ,
    _bz_pow<int,complex<T2> > > >
pow(IndexPlaceholder<N_index1> d1, complex<T2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<complex<T2> > ,
    _bz_pow<int,complex<T2> > >(d1, _bz_ArrayExprConstant<complex<T2> > (d2));
}

#endif // BZ_HAVE_COMPLEX

template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_pow<float,T_numtype2> > >
pow(float d1, const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_pow<float,T_numtype2> >(_bz_ArrayExprConstant<float>(d1), d2.begin());
}

template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, _bz_ArrayExpr<P_expr2>,
    _bz_pow<float,typename P_expr2::T_numtype> > >
pow(float d1, _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, _bz_ArrayExpr<P_expr2>,
    _bz_pow<float,typename P_expr2::T_numtype> >(_bz_ArrayExprConstant<float>(d1), d2);
}

template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, IndexPlaceholder<N_index2>,
    _bz_pow<float,int> > >
pow(float d1, IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, IndexPlaceholder<N_index2>,
    _bz_pow<float,int> >(_bz_ArrayExprConstant<float>(d1), d2);
}

template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_pow<double,T_numtype2> > >
pow(double d1, const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_pow<double,T_numtype2> >(_bz_ArrayExprConstant<double>(d1), d2.begin());
}

template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, _bz_ArrayExpr<P_expr2>,
    _bz_pow<double,typename P_expr2::T_numtype> > >
pow(double d1, _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, _bz_ArrayExpr<P_expr2>,
    _bz_pow<double,typename P_expr2::T_numtype> >(_bz_ArrayExprConstant<double>(d1), d2);
}

template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, IndexPlaceholder<N_index2>,
    _bz_pow<double,int> > >
pow(double d1, IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, IndexPlaceholder<N_index2>,
    _bz_pow<double,int> >(_bz_ArrayExprConstant<double>(d1), d2);
}

template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_pow<long double,T_numtype2> > >
pow(long double d1, const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_pow<long double,T_numtype2> >(_bz_ArrayExprConstant<long double>(d1), d2.begin());
}

template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, _bz_ArrayExpr<P_expr2>,
    _bz_pow<long double,typename P_expr2::T_numtype> > >
pow(long double d1, _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, _bz_ArrayExpr<P_expr2>,
    _bz_pow<long double,typename P_expr2::T_numtype> >(_bz_ArrayExprConstant<long double>(d1), d2);
}

template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, IndexPlaceholder<N_index2>,
    _bz_pow<long double,int> > >
pow(long double d1, IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, IndexPlaceholder<N_index2>,
    _bz_pow<long double,int> >(_bz_ArrayExprConstant<long double>(d1), d2);
}

#ifdef BZ_HAVE_COMPLEX
template<class T1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , ArrayIterator<T_numtype2, N_rank2>,
    _bz_pow<complex<T1> ,T_numtype2> > >
pow(complex<T1> d1, const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , ArrayIterator<T_numtype2, N_rank2>,
    _bz_pow<complex<T1> ,T_numtype2> >(_bz_ArrayExprConstant<complex<T1> > (d1), d2.begin());
}

#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX
template<class T1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , _bz_ArrayExpr<P_expr2>,
    _bz_pow<complex<T1> ,typename P_expr2::T_numtype> > >
pow(complex<T1> d1, _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , _bz_ArrayExpr<P_expr2>,
    _bz_pow<complex<T1> ,typename P_expr2::T_numtype> >(_bz_ArrayExprConstant<complex<T1> > (d1), d2);
}

#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX
template<class T1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , IndexPlaceholder<N_index2>,
    _bz_pow<complex<T1> ,int> > >
pow(complex<T1> d1, IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , IndexPlaceholder<N_index2>,
    _bz_pow<complex<T1> ,int> >(_bz_ArrayExprConstant<complex<T1> > (d1), d2);
}

#endif // BZ_HAVE_COMPLEX


/****************************************************************************
 * pow2
 ****************************************************************************/

template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_pow2<T_numtype1> > >
pow2(const Array<T_numtype1, N_rank1>& d1)
{
    return _bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_pow2<T_numtype1> >(d1.begin());
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_pow2<typename P_expr1::T_numtype> > >
pow2(_bz_ArrayExpr<P_expr1> d1)
{
    return _bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_pow2<typename P_expr1::T_numtype> >(d1);
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_pow2<int> > >
pow2(IndexPlaceholder<N_index1> d1)
{
    return _bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_pow2<int> >(d1);
}


/****************************************************************************
 * pow3
 ****************************************************************************/

template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_pow3<T_numtype1> > >
pow3(const Array<T_numtype1, N_rank1>& d1)
{
    return _bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_pow3<T_numtype1> >(d1.begin());
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_pow3<typename P_expr1::T_numtype> > >
pow3(_bz_ArrayExpr<P_expr1> d1)
{
    return _bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_pow3<typename P_expr1::T_numtype> >(d1);
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_pow3<int> > >
pow3(IndexPlaceholder<N_index1> d1)
{
    return _bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_pow3<int> >(d1);
}


/****************************************************************************
 * pow4
 ****************************************************************************/

template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_pow4<T_numtype1> > >
pow4(const Array<T_numtype1, N_rank1>& d1)
{
    return _bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_pow4<T_numtype1> >(d1.begin());
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_pow4<typename P_expr1::T_numtype> > >
pow4(_bz_ArrayExpr<P_expr1> d1)
{
    return _bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_pow4<typename P_expr1::T_numtype> >(d1);
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_pow4<int> > >
pow4(IndexPlaceholder<N_index1> d1)
{
    return _bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_pow4<int> >(d1);
}


/****************************************************************************
 * pow5
 ****************************************************************************/

template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_pow5<T_numtype1> > >
pow5(const Array<T_numtype1, N_rank1>& d1)
{
    return _bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_pow5<T_numtype1> >(d1.begin());
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_pow5<typename P_expr1::T_numtype> > >
pow5(_bz_ArrayExpr<P_expr1> d1)
{
    return _bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_pow5<typename P_expr1::T_numtype> >(d1);
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_pow5<int> > >
pow5(IndexPlaceholder<N_index1> d1)
{
    return _bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_pow5<int> >(d1);
}


/****************************************************************************
 * pow6
 ****************************************************************************/

template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_pow6<T_numtype1> > >
pow6(const Array<T_numtype1, N_rank1>& d1)
{
    return _bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_pow6<T_numtype1> >(d1.begin());
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_pow6<typename P_expr1::T_numtype> > >
pow6(_bz_ArrayExpr<P_expr1> d1)
{
    return _bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_pow6<typename P_expr1::T_numtype> >(d1);
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_pow6<int> > >
pow6(IndexPlaceholder<N_index1> d1)
{
    return _bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_pow6<int> >(d1);
}


/****************************************************************************
 * pow7
 ****************************************************************************/

template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_pow7<T_numtype1> > >
pow7(const Array<T_numtype1, N_rank1>& d1)
{
    return _bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_pow7<T_numtype1> >(d1.begin());
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_pow7<typename P_expr1::T_numtype> > >
pow7(_bz_ArrayExpr<P_expr1> d1)
{
    return _bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_pow7<typename P_expr1::T_numtype> >(d1);
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_pow7<int> > >
pow7(IndexPlaceholder<N_index1> d1)
{
    return _bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_pow7<int> >(d1);
}


/****************************************************************************
 * pow8
 ****************************************************************************/

template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_pow8<T_numtype1> > >
pow8(const Array<T_numtype1, N_rank1>& d1)
{
    return _bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_pow8<T_numtype1> >(d1.begin());
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_pow8<typename P_expr1::T_numtype> > >
pow8(_bz_ArrayExpr<P_expr1> d1)
{
    return _bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_pow8<typename P_expr1::T_numtype> >(d1);
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_pow8<int> > >
pow8(IndexPlaceholder<N_index1> d1)
{
    return _bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_pow8<int> >(d1);
}


/****************************************************************************
 * remainder
 ****************************************************************************/

#ifdef BZ_HAVE_SYSTEM_V_MATH
template<class T_numtype1, int N_rank1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_remainder<T_numtype1,T_numtype2> > >
remainder(const Array<T_numtype1, N_rank1>& d1, const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_remainder<T_numtype1,T_numtype2> >(d1.begin(), d2.begin());
}

template<class T_numtype1, int N_rank1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExpr<P_expr2>,
    _bz_remainder<T_numtype1,typename P_expr2::T_numtype> > >
remainder(const Array<T_numtype1, N_rank1>& d1, _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExpr<P_expr2>,
    _bz_remainder<T_numtype1,typename P_expr2::T_numtype> >(d1.begin(), d2);
}

template<class T_numtype1, int N_rank1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, IndexPlaceholder<N_index2>,
    _bz_remainder<T_numtype1,int> > >
remainder(const Array<T_numtype1, N_rank1>& d1, IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, IndexPlaceholder<N_index2>,
    _bz_remainder<T_numtype1,int> >(d1.begin(), d2);
}

template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<float>,
    _bz_remainder<T_numtype1,float> > >
remainder(const Array<T_numtype1, N_rank1>& d1, float d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<float>,
    _bz_remainder<T_numtype1,float> >(d1.begin(), _bz_ArrayExprConstant<float>(d2));
}

template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<double>,
    _bz_remainder<T_numtype1,double> > >
remainder(const Array<T_numtype1, N_rank1>& d1, double d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<double>,
    _bz_remainder<T_numtype1,double> >(d1.begin(), _bz_ArrayExprConstant<double>(d2));
}

template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<long double>,
    _bz_remainder<T_numtype1,long double> > >
remainder(const Array<T_numtype1, N_rank1>& d1, long double d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<long double>,
    _bz_remainder<T_numtype1,long double> >(d1.begin(), _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
template<class T_numtype1, int N_rank1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<complex<T2> > ,
    _bz_remainder<T_numtype1,complex<T2> > > >
remainder(const Array<T_numtype1, N_rank1>& d1, complex<T2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<complex<T2> > ,
    _bz_remainder<T_numtype1,complex<T2> > >(d1.begin(), _bz_ArrayExprConstant<complex<T2> > (d2));
}

#endif // BZ_HAVE_COMPLEX

template<class P_expr1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_remainder<typename P_expr1::T_numtype,T_numtype2> > >
remainder(_bz_ArrayExpr<P_expr1> d1, const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_remainder<typename P_expr1::T_numtype,T_numtype2> >(d1, d2.begin());
}

template<class P_expr1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExpr<P_expr2>,
    _bz_remainder<typename P_expr1::T_numtype,typename P_expr2::T_numtype> > >
remainder(_bz_ArrayExpr<P_expr1> d1, _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExpr<P_expr2>,
    _bz_remainder<typename P_expr1::T_numtype,typename P_expr2::T_numtype> >(d1, d2);
}

template<class P_expr1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, IndexPlaceholder<N_index2>,
    _bz_remainder<typename P_expr1::T_numtype,int> > >
remainder(_bz_ArrayExpr<P_expr1> d1, IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, IndexPlaceholder<N_index2>,
    _bz_remainder<typename P_expr1::T_numtype,int> >(d1, d2);
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<float>,
    _bz_remainder<typename P_expr1::T_numtype,float> > >
remainder(_bz_ArrayExpr<P_expr1> d1, float d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<float>,
    _bz_remainder<typename P_expr1::T_numtype,float> >(d1, _bz_ArrayExprConstant<float>(d2));
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<double>,
    _bz_remainder<typename P_expr1::T_numtype,double> > >
remainder(_bz_ArrayExpr<P_expr1> d1, double d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<double>,
    _bz_remainder<typename P_expr1::T_numtype,double> >(d1, _bz_ArrayExprConstant<double>(d2));
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<long double>,
    _bz_remainder<typename P_expr1::T_numtype,long double> > >
remainder(_bz_ArrayExpr<P_expr1> d1, long double d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<long double>,
    _bz_remainder<typename P_expr1::T_numtype,long double> >(d1, _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
template<class P_expr1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<complex<T2> > ,
    _bz_remainder<typename P_expr1::T_numtype,complex<T2> > > >
remainder(_bz_ArrayExpr<P_expr1> d1, complex<T2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<complex<T2> > ,
    _bz_remainder<typename P_expr1::T_numtype,complex<T2> > >(d1, _bz_ArrayExprConstant<complex<T2> > (d2));
}

#endif // BZ_HAVE_COMPLEX

template<int N_index1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_remainder<int,T_numtype2> > >
remainder(IndexPlaceholder<N_index1> d1, const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_remainder<int,T_numtype2> >(d1, d2.begin());
}

template<int N_index1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExpr<P_expr2>,
    _bz_remainder<int,typename P_expr2::T_numtype> > >
remainder(IndexPlaceholder<N_index1> d1, _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExpr<P_expr2>,
    _bz_remainder<int,typename P_expr2::T_numtype> >(d1, d2);
}

template<int N_index1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, IndexPlaceholder<N_index2>,
    _bz_remainder<int,int> > >
remainder(IndexPlaceholder<N_index1> d1, IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, IndexPlaceholder<N_index2>,
    _bz_remainder<int,int> >(d1, d2);
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<float>,
    _bz_remainder<int,float> > >
remainder(IndexPlaceholder<N_index1> d1, float d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<float>,
    _bz_remainder<int,float> >(d1, _bz_ArrayExprConstant<float>(d2));
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<double>,
    _bz_remainder<int,double> > >
remainder(IndexPlaceholder<N_index1> d1, double d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<double>,
    _bz_remainder<int,double> >(d1, _bz_ArrayExprConstant<double>(d2));
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<long double>,
    _bz_remainder<int,long double> > >
remainder(IndexPlaceholder<N_index1> d1, long double d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<long double>,
    _bz_remainder<int,long double> >(d1, _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
template<int N_index1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<complex<T2> > ,
    _bz_remainder<int,complex<T2> > > >
remainder(IndexPlaceholder<N_index1> d1, complex<T2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<complex<T2> > ,
    _bz_remainder<int,complex<T2> > >(d1, _bz_ArrayExprConstant<complex<T2> > (d2));
}

#endif // BZ_HAVE_COMPLEX

template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_remainder<float,T_numtype2> > >
remainder(float d1, const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_remainder<float,T_numtype2> >(_bz_ArrayExprConstant<float>(d1), d2.begin());
}

template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, _bz_ArrayExpr<P_expr2>,
    _bz_remainder<float,typename P_expr2::T_numtype> > >
remainder(float d1, _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, _bz_ArrayExpr<P_expr2>,
    _bz_remainder<float,typename P_expr2::T_numtype> >(_bz_ArrayExprConstant<float>(d1), d2);
}

template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, IndexPlaceholder<N_index2>,
    _bz_remainder<float,int> > >
remainder(float d1, IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, IndexPlaceholder<N_index2>,
    _bz_remainder<float,int> >(_bz_ArrayExprConstant<float>(d1), d2);
}

template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_remainder<double,T_numtype2> > >
remainder(double d1, const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_remainder<double,T_numtype2> >(_bz_ArrayExprConstant<double>(d1), d2.begin());
}

template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, _bz_ArrayExpr<P_expr2>,
    _bz_remainder<double,typename P_expr2::T_numtype> > >
remainder(double d1, _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, _bz_ArrayExpr<P_expr2>,
    _bz_remainder<double,typename P_expr2::T_numtype> >(_bz_ArrayExprConstant<double>(d1), d2);
}

template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, IndexPlaceholder<N_index2>,
    _bz_remainder<double,int> > >
remainder(double d1, IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, IndexPlaceholder<N_index2>,
    _bz_remainder<double,int> >(_bz_ArrayExprConstant<double>(d1), d2);
}

template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_remainder<long double,T_numtype2> > >
remainder(long double d1, const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_remainder<long double,T_numtype2> >(_bz_ArrayExprConstant<long double>(d1), d2.begin());
}

template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, _bz_ArrayExpr<P_expr2>,
    _bz_remainder<long double,typename P_expr2::T_numtype> > >
remainder(long double d1, _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, _bz_ArrayExpr<P_expr2>,
    _bz_remainder<long double,typename P_expr2::T_numtype> >(_bz_ArrayExprConstant<long double>(d1), d2);
}

template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, IndexPlaceholder<N_index2>,
    _bz_remainder<long double,int> > >
remainder(long double d1, IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, IndexPlaceholder<N_index2>,
    _bz_remainder<long double,int> >(_bz_ArrayExprConstant<long double>(d1), d2);
}

#ifdef BZ_HAVE_COMPLEX
template<class T1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , ArrayIterator<T_numtype2, N_rank2>,
    _bz_remainder<complex<T1> ,T_numtype2> > >
remainder(complex<T1> d1, const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , ArrayIterator<T_numtype2, N_rank2>,
    _bz_remainder<complex<T1> ,T_numtype2> >(_bz_ArrayExprConstant<complex<T1> > (d1), d2.begin());
}

#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX
template<class T1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , _bz_ArrayExpr<P_expr2>,
    _bz_remainder<complex<T1> ,typename P_expr2::T_numtype> > >
remainder(complex<T1> d1, _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , _bz_ArrayExpr<P_expr2>,
    _bz_remainder<complex<T1> ,typename P_expr2::T_numtype> >(_bz_ArrayExprConstant<complex<T1> > (d1), d2);
}

#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX
template<class T1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , IndexPlaceholder<N_index2>,
    _bz_remainder<complex<T1> ,int> > >
remainder(complex<T1> d1, IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , IndexPlaceholder<N_index2>,
    _bz_remainder<complex<T1> ,int> >(_bz_ArrayExprConstant<complex<T1> > (d1), d2);
}

#endif // BZ_HAVE_COMPLEX

#endif

/****************************************************************************
 * rint
 ****************************************************************************/

#ifdef BZ_HAVE_IEEE_MATH
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_rint<T_numtype1> > >
rint(const Array<T_numtype1, N_rank1>& d1)
{
    return _bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_rint<T_numtype1> >(d1.begin());
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_rint<typename P_expr1::T_numtype> > >
rint(_bz_ArrayExpr<P_expr1> d1)
{
    return _bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_rint<typename P_expr1::T_numtype> >(d1);
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_rint<int> > >
rint(IndexPlaceholder<N_index1> d1)
{
    return _bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_rint<int> >(d1);
}

#endif

/****************************************************************************
 * rsqrt
 ****************************************************************************/

#ifdef BZ_HAVE_SYSTEM_V_MATH
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_rsqrt<T_numtype1> > >
rsqrt(const Array<T_numtype1, N_rank1>& d1)
{
    return _bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_rsqrt<T_numtype1> >(d1.begin());
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_rsqrt<typename P_expr1::T_numtype> > >
rsqrt(_bz_ArrayExpr<P_expr1> d1)
{
    return _bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_rsqrt<typename P_expr1::T_numtype> >(d1);
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_rsqrt<int> > >
rsqrt(IndexPlaceholder<N_index1> d1)
{
    return _bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_rsqrt<int> >(d1);
}

#endif

/****************************************************************************
 * scalb
 ****************************************************************************/

#ifdef BZ_HAVE_SYSTEM_V_MATH
template<class T_numtype1, int N_rank1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_scalb<T_numtype1,T_numtype2> > >
scalb(const Array<T_numtype1, N_rank1>& d1, const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_scalb<T_numtype1,T_numtype2> >(d1.begin(), d2.begin());
}

template<class T_numtype1, int N_rank1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExpr<P_expr2>,
    _bz_scalb<T_numtype1,typename P_expr2::T_numtype> > >
scalb(const Array<T_numtype1, N_rank1>& d1, _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExpr<P_expr2>,
    _bz_scalb<T_numtype1,typename P_expr2::T_numtype> >(d1.begin(), d2);
}

template<class T_numtype1, int N_rank1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, IndexPlaceholder<N_index2>,
    _bz_scalb<T_numtype1,int> > >
scalb(const Array<T_numtype1, N_rank1>& d1, IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, IndexPlaceholder<N_index2>,
    _bz_scalb<T_numtype1,int> >(d1.begin(), d2);
}

template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<float>,
    _bz_scalb<T_numtype1,float> > >
scalb(const Array<T_numtype1, N_rank1>& d1, float d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<float>,
    _bz_scalb<T_numtype1,float> >(d1.begin(), _bz_ArrayExprConstant<float>(d2));
}

template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<double>,
    _bz_scalb<T_numtype1,double> > >
scalb(const Array<T_numtype1, N_rank1>& d1, double d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<double>,
    _bz_scalb<T_numtype1,double> >(d1.begin(), _bz_ArrayExprConstant<double>(d2));
}

template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<long double>,
    _bz_scalb<T_numtype1,long double> > >
scalb(const Array<T_numtype1, N_rank1>& d1, long double d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<long double>,
    _bz_scalb<T_numtype1,long double> >(d1.begin(), _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
template<class T_numtype1, int N_rank1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<complex<T2> > ,
    _bz_scalb<T_numtype1,complex<T2> > > >
scalb(const Array<T_numtype1, N_rank1>& d1, complex<T2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<complex<T2> > ,
    _bz_scalb<T_numtype1,complex<T2> > >(d1.begin(), _bz_ArrayExprConstant<complex<T2> > (d2));
}

#endif // BZ_HAVE_COMPLEX

template<class P_expr1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_scalb<typename P_expr1::T_numtype,T_numtype2> > >
scalb(_bz_ArrayExpr<P_expr1> d1, const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_scalb<typename P_expr1::T_numtype,T_numtype2> >(d1, d2.begin());
}

template<class P_expr1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExpr<P_expr2>,
    _bz_scalb<typename P_expr1::T_numtype,typename P_expr2::T_numtype> > >
scalb(_bz_ArrayExpr<P_expr1> d1, _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExpr<P_expr2>,
    _bz_scalb<typename P_expr1::T_numtype,typename P_expr2::T_numtype> >(d1, d2);
}

template<class P_expr1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, IndexPlaceholder<N_index2>,
    _bz_scalb<typename P_expr1::T_numtype,int> > >
scalb(_bz_ArrayExpr<P_expr1> d1, IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, IndexPlaceholder<N_index2>,
    _bz_scalb<typename P_expr1::T_numtype,int> >(d1, d2);
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<float>,
    _bz_scalb<typename P_expr1::T_numtype,float> > >
scalb(_bz_ArrayExpr<P_expr1> d1, float d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<float>,
    _bz_scalb<typename P_expr1::T_numtype,float> >(d1, _bz_ArrayExprConstant<float>(d2));
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<double>,
    _bz_scalb<typename P_expr1::T_numtype,double> > >
scalb(_bz_ArrayExpr<P_expr1> d1, double d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<double>,
    _bz_scalb<typename P_expr1::T_numtype,double> >(d1, _bz_ArrayExprConstant<double>(d2));
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<long double>,
    _bz_scalb<typename P_expr1::T_numtype,long double> > >
scalb(_bz_ArrayExpr<P_expr1> d1, long double d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<long double>,
    _bz_scalb<typename P_expr1::T_numtype,long double> >(d1, _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
template<class P_expr1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<complex<T2> > ,
    _bz_scalb<typename P_expr1::T_numtype,complex<T2> > > >
scalb(_bz_ArrayExpr<P_expr1> d1, complex<T2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<complex<T2> > ,
    _bz_scalb<typename P_expr1::T_numtype,complex<T2> > >(d1, _bz_ArrayExprConstant<complex<T2> > (d2));
}

#endif // BZ_HAVE_COMPLEX

template<int N_index1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_scalb<int,T_numtype2> > >
scalb(IndexPlaceholder<N_index1> d1, const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_scalb<int,T_numtype2> >(d1, d2.begin());
}

template<int N_index1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExpr<P_expr2>,
    _bz_scalb<int,typename P_expr2::T_numtype> > >
scalb(IndexPlaceholder<N_index1> d1, _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExpr<P_expr2>,
    _bz_scalb<int,typename P_expr2::T_numtype> >(d1, d2);
}

template<int N_index1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, IndexPlaceholder<N_index2>,
    _bz_scalb<int,int> > >
scalb(IndexPlaceholder<N_index1> d1, IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, IndexPlaceholder<N_index2>,
    _bz_scalb<int,int> >(d1, d2);
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<float>,
    _bz_scalb<int,float> > >
scalb(IndexPlaceholder<N_index1> d1, float d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<float>,
    _bz_scalb<int,float> >(d1, _bz_ArrayExprConstant<float>(d2));
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<double>,
    _bz_scalb<int,double> > >
scalb(IndexPlaceholder<N_index1> d1, double d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<double>,
    _bz_scalb<int,double> >(d1, _bz_ArrayExprConstant<double>(d2));
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<long double>,
    _bz_scalb<int,long double> > >
scalb(IndexPlaceholder<N_index1> d1, long double d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<long double>,
    _bz_scalb<int,long double> >(d1, _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
template<int N_index1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<complex<T2> > ,
    _bz_scalb<int,complex<T2> > > >
scalb(IndexPlaceholder<N_index1> d1, complex<T2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<complex<T2> > ,
    _bz_scalb<int,complex<T2> > >(d1, _bz_ArrayExprConstant<complex<T2> > (d2));
}

#endif // BZ_HAVE_COMPLEX

template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_scalb<float,T_numtype2> > >
scalb(float d1, const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_scalb<float,T_numtype2> >(_bz_ArrayExprConstant<float>(d1), d2.begin());
}

template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, _bz_ArrayExpr<P_expr2>,
    _bz_scalb<float,typename P_expr2::T_numtype> > >
scalb(float d1, _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, _bz_ArrayExpr<P_expr2>,
    _bz_scalb<float,typename P_expr2::T_numtype> >(_bz_ArrayExprConstant<float>(d1), d2);
}

template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, IndexPlaceholder<N_index2>,
    _bz_scalb<float,int> > >
scalb(float d1, IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, IndexPlaceholder<N_index2>,
    _bz_scalb<float,int> >(_bz_ArrayExprConstant<float>(d1), d2);
}

template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_scalb<double,T_numtype2> > >
scalb(double d1, const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_scalb<double,T_numtype2> >(_bz_ArrayExprConstant<double>(d1), d2.begin());
}

template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, _bz_ArrayExpr<P_expr2>,
    _bz_scalb<double,typename P_expr2::T_numtype> > >
scalb(double d1, _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, _bz_ArrayExpr<P_expr2>,
    _bz_scalb<double,typename P_expr2::T_numtype> >(_bz_ArrayExprConstant<double>(d1), d2);
}

template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, IndexPlaceholder<N_index2>,
    _bz_scalb<double,int> > >
scalb(double d1, IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, IndexPlaceholder<N_index2>,
    _bz_scalb<double,int> >(_bz_ArrayExprConstant<double>(d1), d2);
}

template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_scalb<long double,T_numtype2> > >
scalb(long double d1, const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_scalb<long double,T_numtype2> >(_bz_ArrayExprConstant<long double>(d1), d2.begin());
}

template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, _bz_ArrayExpr<P_expr2>,
    _bz_scalb<long double,typename P_expr2::T_numtype> > >
scalb(long double d1, _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, _bz_ArrayExpr<P_expr2>,
    _bz_scalb<long double,typename P_expr2::T_numtype> >(_bz_ArrayExprConstant<long double>(d1), d2);
}

template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, IndexPlaceholder<N_index2>,
    _bz_scalb<long double,int> > >
scalb(long double d1, IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, IndexPlaceholder<N_index2>,
    _bz_scalb<long double,int> >(_bz_ArrayExprConstant<long double>(d1), d2);
}

#ifdef BZ_HAVE_COMPLEX
template<class T1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , ArrayIterator<T_numtype2, N_rank2>,
    _bz_scalb<complex<T1> ,T_numtype2> > >
scalb(complex<T1> d1, const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , ArrayIterator<T_numtype2, N_rank2>,
    _bz_scalb<complex<T1> ,T_numtype2> >(_bz_ArrayExprConstant<complex<T1> > (d1), d2.begin());
}

#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX
template<class T1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , _bz_ArrayExpr<P_expr2>,
    _bz_scalb<complex<T1> ,typename P_expr2::T_numtype> > >
scalb(complex<T1> d1, _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , _bz_ArrayExpr<P_expr2>,
    _bz_scalb<complex<T1> ,typename P_expr2::T_numtype> >(_bz_ArrayExprConstant<complex<T1> > (d1), d2);
}

#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX
template<class T1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , IndexPlaceholder<N_index2>,
    _bz_scalb<complex<T1> ,int> > >
scalb(complex<T1> d1, IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , IndexPlaceholder<N_index2>,
    _bz_scalb<complex<T1> ,int> >(_bz_ArrayExprConstant<complex<T1> > (d1), d2);
}

#endif // BZ_HAVE_COMPLEX

#endif

/****************************************************************************
 * sin
 ****************************************************************************/

template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_sin<T_numtype1> > >
sin(const Array<T_numtype1, N_rank1>& d1)
{
    return _bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_sin<T_numtype1> >(d1.begin());
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_sin<typename P_expr1::T_numtype> > >
sin(_bz_ArrayExpr<P_expr1> d1)
{
    return _bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_sin<typename P_expr1::T_numtype> >(d1);
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_sin<int> > >
sin(IndexPlaceholder<N_index1> d1)
{
    return _bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_sin<int> >(d1);
}


/****************************************************************************
 * sinh
 ****************************************************************************/

template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_sinh<T_numtype1> > >
sinh(const Array<T_numtype1, N_rank1>& d1)
{
    return _bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_sinh<T_numtype1> >(d1.begin());
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_sinh<typename P_expr1::T_numtype> > >
sinh(_bz_ArrayExpr<P_expr1> d1)
{
    return _bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_sinh<typename P_expr1::T_numtype> >(d1);
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_sinh<int> > >
sinh(IndexPlaceholder<N_index1> d1)
{
    return _bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_sinh<int> >(d1);
}


/****************************************************************************
 * sqr
 ****************************************************************************/

template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_sqr<T_numtype1> > >
sqr(const Array<T_numtype1, N_rank1>& d1)
{
    return _bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_sqr<T_numtype1> >(d1.begin());
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_sqr<typename P_expr1::T_numtype> > >
sqr(_bz_ArrayExpr<P_expr1> d1)
{
    return _bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_sqr<typename P_expr1::T_numtype> >(d1);
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_sqr<int> > >
sqr(IndexPlaceholder<N_index1> d1)
{
    return _bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_sqr<int> >(d1);
}


/****************************************************************************
 * sqrt
 ****************************************************************************/

template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_sqrt<T_numtype1> > >
sqrt(const Array<T_numtype1, N_rank1>& d1)
{
    return _bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_sqrt<T_numtype1> >(d1.begin());
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_sqrt<typename P_expr1::T_numtype> > >
sqrt(_bz_ArrayExpr<P_expr1> d1)
{
    return _bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_sqrt<typename P_expr1::T_numtype> >(d1);
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_sqrt<int> > >
sqrt(IndexPlaceholder<N_index1> d1)
{
    return _bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_sqrt<int> >(d1);
}


/****************************************************************************
 * tan
 ****************************************************************************/

template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_tan<T_numtype1> > >
tan(const Array<T_numtype1, N_rank1>& d1)
{
    return _bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_tan<T_numtype1> >(d1.begin());
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_tan<typename P_expr1::T_numtype> > >
tan(_bz_ArrayExpr<P_expr1> d1)
{
    return _bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_tan<typename P_expr1::T_numtype> >(d1);
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_tan<int> > >
tan(IndexPlaceholder<N_index1> d1)
{
    return _bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_tan<int> >(d1);
}


/****************************************************************************
 * tanh
 ****************************************************************************/

template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_tanh<T_numtype1> > >
tanh(const Array<T_numtype1, N_rank1>& d1)
{
    return _bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_tanh<T_numtype1> >(d1.begin());
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_tanh<typename P_expr1::T_numtype> > >
tanh(_bz_ArrayExpr<P_expr1> d1)
{
    return _bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_tanh<typename P_expr1::T_numtype> >(d1);
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_tanh<int> > >
tanh(IndexPlaceholder<N_index1> d1)
{
    return _bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_tanh<int> >(d1);
}


/****************************************************************************
 * uitrunc
 ****************************************************************************/

#ifdef BZ_HAVE_SYSTEM_V_MATH
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_uitrunc<T_numtype1> > >
uitrunc(const Array<T_numtype1, N_rank1>& d1)
{
    return _bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_uitrunc<T_numtype1> >(d1.begin());
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_uitrunc<typename P_expr1::T_numtype> > >
uitrunc(_bz_ArrayExpr<P_expr1> d1)
{
    return _bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_uitrunc<typename P_expr1::T_numtype> >(d1);
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_uitrunc<int> > >
uitrunc(IndexPlaceholder<N_index1> d1)
{
    return _bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_uitrunc<int> >(d1);
}

#endif

/****************************************************************************
 * unordered
 ****************************************************************************/

#ifdef BZ_HAVE_SYSTEM_V_MATH
template<class T_numtype1, int N_rank1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_unordered<T_numtype1,T_numtype2> > >
unordered(const Array<T_numtype1, N_rank1>& d1, const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_unordered<T_numtype1,T_numtype2> >(d1.begin(), d2.begin());
}

template<class T_numtype1, int N_rank1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExpr<P_expr2>,
    _bz_unordered<T_numtype1,typename P_expr2::T_numtype> > >
unordered(const Array<T_numtype1, N_rank1>& d1, _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExpr<P_expr2>,
    _bz_unordered<T_numtype1,typename P_expr2::T_numtype> >(d1.begin(), d2);
}

template<class T_numtype1, int N_rank1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, IndexPlaceholder<N_index2>,
    _bz_unordered<T_numtype1,int> > >
unordered(const Array<T_numtype1, N_rank1>& d1, IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, IndexPlaceholder<N_index2>,
    _bz_unordered<T_numtype1,int> >(d1.begin(), d2);
}

template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<float>,
    _bz_unordered<T_numtype1,float> > >
unordered(const Array<T_numtype1, N_rank1>& d1, float d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<float>,
    _bz_unordered<T_numtype1,float> >(d1.begin(), _bz_ArrayExprConstant<float>(d2));
}

template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<double>,
    _bz_unordered<T_numtype1,double> > >
unordered(const Array<T_numtype1, N_rank1>& d1, double d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<double>,
    _bz_unordered<T_numtype1,double> >(d1.begin(), _bz_ArrayExprConstant<double>(d2));
}

template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<long double>,
    _bz_unordered<T_numtype1,long double> > >
unordered(const Array<T_numtype1, N_rank1>& d1, long double d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<long double>,
    _bz_unordered<T_numtype1,long double> >(d1.begin(), _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
template<class T_numtype1, int N_rank1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<complex<T2> > ,
    _bz_unordered<T_numtype1,complex<T2> > > >
unordered(const Array<T_numtype1, N_rank1>& d1, complex<T2> d2)
{
    return _bz_ArrayExprBinaryOp<ArrayIterator<T_numtype1, N_rank1>, _bz_ArrayExprConstant<complex<T2> > ,
    _bz_unordered<T_numtype1,complex<T2> > >(d1.begin(), _bz_ArrayExprConstant<complex<T2> > (d2));
}

#endif // BZ_HAVE_COMPLEX

template<class P_expr1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_unordered<typename P_expr1::T_numtype,T_numtype2> > >
unordered(_bz_ArrayExpr<P_expr1> d1, const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_unordered<typename P_expr1::T_numtype,T_numtype2> >(d1, d2.begin());
}

template<class P_expr1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExpr<P_expr2>,
    _bz_unordered<typename P_expr1::T_numtype,typename P_expr2::T_numtype> > >
unordered(_bz_ArrayExpr<P_expr1> d1, _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExpr<P_expr2>,
    _bz_unordered<typename P_expr1::T_numtype,typename P_expr2::T_numtype> >(d1, d2);
}

template<class P_expr1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, IndexPlaceholder<N_index2>,
    _bz_unordered<typename P_expr1::T_numtype,int> > >
unordered(_bz_ArrayExpr<P_expr1> d1, IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, IndexPlaceholder<N_index2>,
    _bz_unordered<typename P_expr1::T_numtype,int> >(d1, d2);
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<float>,
    _bz_unordered<typename P_expr1::T_numtype,float> > >
unordered(_bz_ArrayExpr<P_expr1> d1, float d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<float>,
    _bz_unordered<typename P_expr1::T_numtype,float> >(d1, _bz_ArrayExprConstant<float>(d2));
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<double>,
    _bz_unordered<typename P_expr1::T_numtype,double> > >
unordered(_bz_ArrayExpr<P_expr1> d1, double d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<double>,
    _bz_unordered<typename P_expr1::T_numtype,double> >(d1, _bz_ArrayExprConstant<double>(d2));
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<long double>,
    _bz_unordered<typename P_expr1::T_numtype,long double> > >
unordered(_bz_ArrayExpr<P_expr1> d1, long double d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<long double>,
    _bz_unordered<typename P_expr1::T_numtype,long double> >(d1, _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
template<class P_expr1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<complex<T2> > ,
    _bz_unordered<typename P_expr1::T_numtype,complex<T2> > > >
unordered(_bz_ArrayExpr<P_expr1> d1, complex<T2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExpr<P_expr1>, _bz_ArrayExprConstant<complex<T2> > ,
    _bz_unordered<typename P_expr1::T_numtype,complex<T2> > >(d1, _bz_ArrayExprConstant<complex<T2> > (d2));
}

#endif // BZ_HAVE_COMPLEX

template<int N_index1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_unordered<int,T_numtype2> > >
unordered(IndexPlaceholder<N_index1> d1, const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_unordered<int,T_numtype2> >(d1, d2.begin());
}

template<int N_index1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExpr<P_expr2>,
    _bz_unordered<int,typename P_expr2::T_numtype> > >
unordered(IndexPlaceholder<N_index1> d1, _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExpr<P_expr2>,
    _bz_unordered<int,typename P_expr2::T_numtype> >(d1, d2);
}

template<int N_index1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, IndexPlaceholder<N_index2>,
    _bz_unordered<int,int> > >
unordered(IndexPlaceholder<N_index1> d1, IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, IndexPlaceholder<N_index2>,
    _bz_unordered<int,int> >(d1, d2);
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<float>,
    _bz_unordered<int,float> > >
unordered(IndexPlaceholder<N_index1> d1, float d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<float>,
    _bz_unordered<int,float> >(d1, _bz_ArrayExprConstant<float>(d2));
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<double>,
    _bz_unordered<int,double> > >
unordered(IndexPlaceholder<N_index1> d1, double d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<double>,
    _bz_unordered<int,double> >(d1, _bz_ArrayExprConstant<double>(d2));
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<long double>,
    _bz_unordered<int,long double> > >
unordered(IndexPlaceholder<N_index1> d1, long double d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<long double>,
    _bz_unordered<int,long double> >(d1, _bz_ArrayExprConstant<long double>(d2));
}

#ifdef BZ_HAVE_COMPLEX
template<int N_index1, class T2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<complex<T2> > ,
    _bz_unordered<int,complex<T2> > > >
unordered(IndexPlaceholder<N_index1> d1, complex<T2> d2)
{
    return _bz_ArrayExprBinaryOp<IndexPlaceholder<N_index1>, _bz_ArrayExprConstant<complex<T2> > ,
    _bz_unordered<int,complex<T2> > >(d1, _bz_ArrayExprConstant<complex<T2> > (d2));
}

#endif // BZ_HAVE_COMPLEX

template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_unordered<float,T_numtype2> > >
unordered(float d1, const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_unordered<float,T_numtype2> >(_bz_ArrayExprConstant<float>(d1), d2.begin());
}

template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, _bz_ArrayExpr<P_expr2>,
    _bz_unordered<float,typename P_expr2::T_numtype> > >
unordered(float d1, _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, _bz_ArrayExpr<P_expr2>,
    _bz_unordered<float,typename P_expr2::T_numtype> >(_bz_ArrayExprConstant<float>(d1), d2);
}

template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, IndexPlaceholder<N_index2>,
    _bz_unordered<float,int> > >
unordered(float d1, IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<float>, IndexPlaceholder<N_index2>,
    _bz_unordered<float,int> >(_bz_ArrayExprConstant<float>(d1), d2);
}

template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_unordered<double,T_numtype2> > >
unordered(double d1, const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_unordered<double,T_numtype2> >(_bz_ArrayExprConstant<double>(d1), d2.begin());
}

template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, _bz_ArrayExpr<P_expr2>,
    _bz_unordered<double,typename P_expr2::T_numtype> > >
unordered(double d1, _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, _bz_ArrayExpr<P_expr2>,
    _bz_unordered<double,typename P_expr2::T_numtype> >(_bz_ArrayExprConstant<double>(d1), d2);
}

template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, IndexPlaceholder<N_index2>,
    _bz_unordered<double,int> > >
unordered(double d1, IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<double>, IndexPlaceholder<N_index2>,
    _bz_unordered<double,int> >(_bz_ArrayExprConstant<double>(d1), d2);
}

template<class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_unordered<long double,T_numtype2> > >
unordered(long double d1, const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, ArrayIterator<T_numtype2, N_rank2>,
    _bz_unordered<long double,T_numtype2> >(_bz_ArrayExprConstant<long double>(d1), d2.begin());
}

template<class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, _bz_ArrayExpr<P_expr2>,
    _bz_unordered<long double,typename P_expr2::T_numtype> > >
unordered(long double d1, _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, _bz_ArrayExpr<P_expr2>,
    _bz_unordered<long double,typename P_expr2::T_numtype> >(_bz_ArrayExprConstant<long double>(d1), d2);
}

template<int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, IndexPlaceholder<N_index2>,
    _bz_unordered<long double,int> > >
unordered(long double d1, IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<long double>, IndexPlaceholder<N_index2>,
    _bz_unordered<long double,int> >(_bz_ArrayExprConstant<long double>(d1), d2);
}

#ifdef BZ_HAVE_COMPLEX
template<class T1, class T_numtype2, int N_rank2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , ArrayIterator<T_numtype2, N_rank2>,
    _bz_unordered<complex<T1> ,T_numtype2> > >
unordered(complex<T1> d1, const Array<T_numtype2, N_rank2>& d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , ArrayIterator<T_numtype2, N_rank2>,
    _bz_unordered<complex<T1> ,T_numtype2> >(_bz_ArrayExprConstant<complex<T1> > (d1), d2.begin());
}

#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX
template<class T1, class P_expr2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , _bz_ArrayExpr<P_expr2>,
    _bz_unordered<complex<T1> ,typename P_expr2::T_numtype> > >
unordered(complex<T1> d1, _bz_ArrayExpr<P_expr2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , _bz_ArrayExpr<P_expr2>,
    _bz_unordered<complex<T1> ,typename P_expr2::T_numtype> >(_bz_ArrayExprConstant<complex<T1> > (d1), d2);
}

#endif // BZ_HAVE_COMPLEX

#ifdef BZ_HAVE_COMPLEX
template<class T1, int N_index2>
inline
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , IndexPlaceholder<N_index2>,
    _bz_unordered<complex<T1> ,int> > >
unordered(complex<T1> d1, IndexPlaceholder<N_index2> d2)
{
    return _bz_ArrayExprBinaryOp<_bz_ArrayExprConstant<complex<T1> > , IndexPlaceholder<N_index2>,
    _bz_unordered<complex<T1> ,int> >(_bz_ArrayExprConstant<complex<T1> > (d1), d2);
}

#endif // BZ_HAVE_COMPLEX

#endif

/****************************************************************************
 * y0
 ****************************************************************************/

#ifdef BZ_HAVE_IEEE_MATH
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_y0<T_numtype1> > >
y0(const Array<T_numtype1, N_rank1>& d1)
{
    return _bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_y0<T_numtype1> >(d1.begin());
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_y0<typename P_expr1::T_numtype> > >
y0(_bz_ArrayExpr<P_expr1> d1)
{
    return _bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_y0<typename P_expr1::T_numtype> >(d1);
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_y0<int> > >
y0(IndexPlaceholder<N_index1> d1)
{
    return _bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_y0<int> >(d1);
}

#endif

/****************************************************************************
 * y1
 ****************************************************************************/

#ifdef BZ_HAVE_IEEE_MATH
template<class T_numtype1, int N_rank1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_y1<T_numtype1> > >
y1(const Array<T_numtype1, N_rank1>& d1)
{
    return _bz_ArrayExprUnaryOp<ArrayIterator<T_numtype1, N_rank1>,
    _bz_y1<T_numtype1> >(d1.begin());
}

template<class P_expr1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_y1<typename P_expr1::T_numtype> > >
y1(_bz_ArrayExpr<P_expr1> d1)
{
    return _bz_ArrayExprUnaryOp<_bz_ArrayExpr<P_expr1>,
    _bz_y1<typename P_expr1::T_numtype> >(d1);
}

template<int N_index1>
inline
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_y1<int> > >
y1(IndexPlaceholder<N_index1> d1)
{
    return _bz_ArrayExprUnaryOp<IndexPlaceholder<N_index1>,
    _bz_y1<int> >(d1);
}

#endif

BZ_NAMESPACE_END

#endif
