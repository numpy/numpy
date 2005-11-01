/***************************************************************************
 * blitz/../vecwhere.cc	where(X,Y,Z) function for vectors
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
// genvecwhere.cpp Oct  6 2005 15:58:49

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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2, 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2, 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2, 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2, 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2, 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2, 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<int>(d2), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<int>(d2), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<int>(d2), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<float>(d2), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<float>(d2), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<float>(d2), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<double>(d2), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<double>(d2), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<double>(d2), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<long double>(d2), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<long double>(d2), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<long double>(d2), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<complex<T2> > (d2), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<complex<T2> > (d2), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<complex<T2> > (d2), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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
      d2.beginFast(), 
      d3.beginFast()));
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
      d2.beginFast(), 
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
      d2.beginFast(), 
      d3.beginFast()));
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
      d2.beginFast(), 
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
      d2.beginFast(), 
      d3.beginFast()));
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
      d2.beginFast(), 
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
      d2.beginFast(), 
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
      d2.beginFast(), 
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
      d2.beginFast(), 
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
      d2.beginFast(), 
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
      d3.beginFast()));
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
      d3.beginFast()));
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
      d3.beginFast()));
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
      d2.beginFast(), 
      d3.beginFast()));
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
      d2.beginFast(), 
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
      d2.beginFast(), 
      d3.beginFast()));
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
      d2.beginFast(), 
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
      d2.beginFast(), 
      d3.beginFast()));
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
      d2.beginFast(), 
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
      d2.beginFast(), 
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
      d2.beginFast(), 
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
      d2.beginFast(), 
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
      d2.beginFast(), 
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
      d3.beginFast()));
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
      d3.beginFast()));
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
      d3.beginFast()));
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
      d2.beginFast(), 
      d3.beginFast()));
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
      d2.beginFast(), 
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
      d2.beginFast(), 
      d3.beginFast()));
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
      d2.beginFast(), 
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
      d2.beginFast(), 
      d3.beginFast()));
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
      d2.beginFast(), 
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
      d2.beginFast(), 
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
      d2.beginFast(), 
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
      d2.beginFast(), 
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
      d2.beginFast(), 
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
      d3.beginFast()));
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
      d3.beginFast()));
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
      d3.beginFast()));
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
      d3.beginFast()));
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
      d3.beginFast()));
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
      d3.beginFast()));
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
      d3.beginFast()));
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
      d3.beginFast()));
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
      d3.beginFast()));
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
      d3.beginFast()));
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
      d3.beginFast()));
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
      d3.beginFast()));
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
      d3.beginFast()));
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
      d3.beginFast()));
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
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2, 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2, 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2, 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2, 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2, 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2, 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<int>(d2), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<int>(d2), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<int>(d2), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<float>(d2), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<float>(d2), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<float>(d2), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<double>(d2), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<double>(d2), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<double>(d2), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<long double>(d2), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<long double>(d2), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<long double>(d2), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<complex<T2> > (d2), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<complex<T2> > (d2), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<complex<T2> > (d2), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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
      d2.beginFast(), 
      d3.beginFast()));
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
      d2.beginFast(), 
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
      d2.beginFast(), 
      d3.beginFast()));
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
      d2.beginFast(), 
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
      d2.beginFast(), 
      d3.beginFast()));
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
      d2.beginFast(), 
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
      d2.beginFast(), 
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
      d2.beginFast(), 
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
      d2.beginFast(), 
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
      d2.beginFast(), 
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
      d3.beginFast()));
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
      d3.beginFast()));
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
      d3.beginFast()));
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
      d2.beginFast(), 
      d3.beginFast()));
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
      d2.beginFast(), 
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
      d2.beginFast(), 
      d3.beginFast()));
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
      d2.beginFast(), 
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
      d2.beginFast(), 
      d3.beginFast()));
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
      d2.beginFast(), 
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
      d2.beginFast(), 
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
      d2.beginFast(), 
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
      d2.beginFast(), 
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
      d2.beginFast(), 
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
      d3.beginFast()));
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
      d3.beginFast()));
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
      d3.beginFast()));
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
      d2.beginFast(), 
      d3.beginFast()));
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
      d2.beginFast(), 
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
      d2.beginFast(), 
      d3.beginFast()));
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
      d2.beginFast(), 
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
      d2.beginFast(), 
      d3.beginFast()));
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
      d2.beginFast(), 
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
      d2.beginFast(), 
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
      d2.beginFast(), 
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
      d2.beginFast(), 
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
      d2.beginFast(), 
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
      d3.beginFast()));
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
      d3.beginFast()));
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
      d3.beginFast()));
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
      d3.beginFast()));
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
      d3.beginFast()));
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
      d3.beginFast()));
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
      d3.beginFast()));
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
      d3.beginFast()));
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
      d3.beginFast()));
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
      d3.beginFast()));
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
      d3.beginFast()));
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
      d3.beginFast()));
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
      d3.beginFast()));
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
      d3.beginFast()));
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
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2, 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2, 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2, 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2, 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2, 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2, 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      d2.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<int>(d2), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<int>(d2), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<int>(d2), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<float>(d2), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<float>(d2), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<float>(d2), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<double>(d2), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<double>(d2), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<double>(d2), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<long double>(d2), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<long double>(d2), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<long double>(d2), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<complex<T2> > (d2), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<complex<T2> > (d2), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<complex<T2> > (d2), 
      d3.beginFast()));
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

    return _bz_VecExpr<T_expr>(T_expr(d1.beginFast(), 
      _bz_VecExprConstant<complex<T2> > (d2), 
      _bz_VecExprConstant<complex<T3> > (d3)));
}
#endif // BZ_HAVE_COMPLEX

BZ_NAMESPACE_END

#endif
