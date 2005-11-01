// -*- C++ -*-
/***************************************************************************
 * blitz/array/newet-macros.h  Macros for new e.t. implementation
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
#ifndef BZ_NEWET_MACROS_H
#define BZ_NEWET_MACROS_H

#include <blitz/array/asexpr.h>

BZ_NAMESPACE(blitz)

#ifdef BZ_HAVE_TEMPLATES_AS_TEMPLATE_ARGUMENTS

/*
 * Unary functions and operators
 */

#define BZ_DECLARE_ARRAY_ET_UNARY(name,functor)                           \
                                                                          \
template <typename T1>                                                    \
_bz_inline_et                                                             \
typename BzUnaryExprResult<functor,T1>::T_result                          \
name(const ETBase<T1>& d1)                                                \
{                                                                         \
    typedef typename BzUnaryExprResult<functor,T1>::T_result result;      \
    return result(asExpr<T1>::getExpr(d1.unwrap()));                      \
}

/*
 * Array expression templates: the macro BZ_DECLARE_ARRAY_ET_BINARY(X,Y)
 * declares a function or operator which takes two operands.
 * X is the function name (or operator), and Y is the functor object
 * which implements the operation.
 */

#define BZ_DECLARE_ARRAY_ET_BINARY(name, applic)                          \
                                                                          \
template <typename T1,typename T2>                                        \
_bz_inline_et                                                             \
typename BzBinaryExprResult<applic,T1,T2>::T_result                       \
name(const ETBase<T1>& d1,const ETBase<T2>& d2)                           \
{                                                                         \
    typedef typename BzBinaryExprResult<applic,T1,T2>::T_result result;   \
    return result(asExpr<T1>::getExpr(d1.unwrap()),                       \
                  asExpr<T2>::getExpr(d2.unwrap()));                      \
}

#define BZ_DECLARE_ARRAY_ET_BINARY_TINYVEC(name, applic)                  \
                                                                          \
template <typename T1, typename T2, int N>                                \
_bz_inline_et                                                             \
typename BzBinaryExprResult<applic,TinyVector<T2,N>,T1>::T_result         \
name(const TinyVector<T2,N> d1, const ETBase<T1>& d2)                     \
{                                                                         \
    typedef typename                                                      \
        BzBinaryExprResult<applic,TinyVector<T2,N>,T1>::T_result result;  \
    return result(asExpr<TinyVector<T2,N> >::getExpr(d1),                 \
                  asExpr<T1>::getExpr(d2.unwrap()));                      \
}                                                                         \
                                                                          \
template <typename T1, typename T2, int N>                                \
_bz_inline_et                                                             \
typename BzBinaryExprResult<applic,T1,TinyVector<T2,N> >::T_result        \
name(const ETBase<T1>& d1, const TinyVector<T2,N> d2)                     \
{                                                                         \
    typedef typename                                                      \
        BzBinaryExprResult<applic,T1,TinyVector<T2,N> >::T_result result; \
    return result(asExpr<T1>::getExpr(d1.unwrap()),                       \
                  asExpr<TinyVector<T2,N> >::getExpr(d2));                \
}

#define BZ_DECLARE_ARRAY_ET_BINARY_SCALAR(name, applic, sca)              \
                                                                          \
template<typename T>                                                      \
_bz_inline_et                                                             \
typename BzBinaryExprResult<applic,sca,T>::T_result                       \
name(const sca d1, const ETBase<T>& d2)                                   \
{                                                                         \
    typedef typename BzBinaryExprResult<applic,sca,T>::T_result result;   \
    return result(asExpr<sca >::getExpr(d1),                              \
                  asExpr<T>::getExpr(d2.unwrap()));                       \
}                                                                         \
                                                                          \
template<typename T>                                                      \
_bz_inline_et                                                             \
typename BzBinaryExprResult<applic,T,sca >::T_result                      \
name(const ETBase<T>& d1, const sca d2)                                   \
{                                                                         \
    typedef typename BzBinaryExprResult<applic,T,sca >::T_result result;  \
    return result(asExpr<T>::getExpr(d1.unwrap()),                        \
                  asExpr<sca >::getExpr(d2));                             \
}

/*
 * Array expression templates: the macro BZ_DECLARE_ARRAY_ET_TERNARY(X,Y)
 * declares a function or operator which takes three operands.
 * X is the function name (or operator), and Y is the functor object
 * which implements the operation.
 */

#define BZ_DECLARE_ARRAY_ET_TERNARY(name, applic)                             \
                                                                              \
template <typename T1, typename T2, typename T3>                              \
_bz_inline_et                                                                 \
typename BzTernaryExprResult<applic, T1, T2, T3>::T_result                    \
name(const ETBase<T1>& d1, const ETBase<T2>& d2, const ETBase<T3>& d3)        \
{                                                                             \
    typedef typename BzTernaryExprResult<applic,T1,T2,T3>::T_result result;   \
    return result(asExpr<T1>::getExpr(d1.unwrap()),                           \
                  asExpr<T2>::getExpr(d2.unwrap()),                           \
                  asExpr<T3>::getExpr(d3.unwrap()));                          \
}
    
#else /* !BZ_HAVE_TEMPLATES_AS_TEMPLATE_ARGUMENTS */

/*
 * Unary functions and operators
 */

#define BZ_DECLARE_ARRAY_ET_UNARY(name, functor)                          \
                                                                          \
template<typename T1>                                                     \
_bz_inline_et                                                             \
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<                                       \
    _bz_typename asExpr<T1>::T_expr,                                      \
    functor<_bz_typename asExpr<T1>::T_expr::T_numtype> > >               \
name(const ETBase<T1>& d1)                                                \
{                                                                         \
    return _bz_ArrayExpr<_bz_ArrayExprUnaryOp<                            \
        _bz_typename asExpr<T1>::T_expr,                                  \
        functor<_bz_typename asExpr<T1>::T_expr::T_numtype> > >           \
        (asExpr<T1>::getExpr(d1.unwrap()));                               \
}

/*
 * Array expression templates: the macro BZ_DECLARE_ARRAY_ET_BINARY(X,Y)
 * declares a function or operator which takes two operands.
 * X is the function name (or operator), and Y is the functor object
 * which implements the operation.
 */

#define BZ_DECLARE_ARRAY_ET_BINARY(name, applic)                          \
                                                                          \
template<typename T1, typename T2>                                        \
_bz_inline_et                                                             \
_bz_ArrayExpr<_bz_ArrayExprBinaryOp<                                      \
    _bz_typename asExpr<T1>::T_expr,                                      \
    _bz_typename asExpr<T2>::T_expr,                                      \
    applic<_bz_typename asExpr<T1>::T_expr::T_numtype,                    \
           _bz_typename asExpr<T2>::T_expr::T_numtype> > >                \
name(const ETBase<T1>& d1, const ETBase<T2>& d2)                          \
{                                                                         \
    return _bz_ArrayExpr<_bz_ArrayExprBinaryOp<                           \
        _bz_typename asExpr<T1>::T_expr,                                  \
        _bz_typename asExpr<T2>::T_expr,                                  \
        applic<_bz_typename asExpr<T1>::T_expr::T_numtype,                \
               _bz_typename asExpr<T2>::T_expr::T_numtype> > >            \
        (asExpr<T1>::getExpr(d1.unwrap()),                                \
         asExpr<T2>::getExpr(d2.unwrap()));                               \
}

#define BZ_DECLARE_ARRAY_ET_BINARY_TINYVEC(name, applic)                  \
                                                                          \
template <typename T1, typename T2, int N>                                \
_bz_inline_et                                                             \
_bz_ArrayExprBinaryOp<                                                    \
    _bz_typename asExpr<TinyVector<T2,N> >::T_expr,                       \
    _bz_typename asExpr<T1>::T_expr,                                      \
    applic<TinyVector<T2,N>,                                              \
    _bz_typename asExpr<T1>::T_expr::T_numtype> >                         \
name(const TinyVector<T2,N> d1, const ETBase<T1>& d2)                     \
{                                                                         \
    return _bz_ArrayExprBinaryOp<                                         \
        _bz_typename asExpr<TinyVector<T2,N> >::T_expr,                   \
        _bz_typename asExpr<T1>::T_expr,                                  \
        applic<TinyVector<T2,N>,                                          \
        _bz_typename asExpr<T1>::T_expr::T_numtype> >                     \
        (asExpr<TinyVector<T2,N> >::getExpr(d1),                          \
         asExpr<T1>::getExpr(d2.unwrap()));                               \
}                                                                         \
                                                                          \
template <typename T1, typename T2, int N>                                \
_bz_inline_et                                                             \
_bz_ArrayExprBinaryOp<                                                    \
    _bz_typename asExpr<T1>::T_expr,                                      \
    _bz_typename asExpr<TinyVector<T2,N> >::T_expr,                       \
    applic<_bz_typename asExpr<T1>::T_expr::T_numtype,                    \
    TinyVector<T2,N> > >                                                  \
name(const ETBase<T1>& d1, const TinyVector<T2,N> d2)                     \
{                                                                         \
    return _bz_ArrayExprBinaryOp<                                         \
        _bz_typename asExpr<T1>::T_expr,                                  \
        _bz_typename asExpr<TinyVector<T2,N> >::T_expr,                   \
        applic<_bz_typename asExpr<T1>::T_expr::T_numtype,                \
        TinyVector<T2,N> > >                                              \
        (asExpr<T1>::getExpr(d1.unwrap()),                                \
         asExpr<TinyVector<T2,N> >::getExpr(d2));                         \
}

#define BZ_DECLARE_ARRAY_ET_BINARY_SCALAR(name, applic, sca)              \
                                                                          \
template<typename T>                                                      \
_bz_inline_et                                                             \
_bz_ArrayExprBinaryOp<                                                    \
    asExpr<sca >::T_expr,                                                 \
    _bz_typename asExpr<T>::T_expr,                                       \
    applic<sca,_bz_typename asExpr<T>::T_expr::T_numtype> >               \
name(const sca d1, const ETBase<T>& d2)                                   \
{                                                                         \
    return _bz_ArrayExprBinaryOp<                                         \
        asExpr<sca >::T_expr,                                             \
        _bz_typename asExpr<T>::T_expr,                                   \
        applic<sca,_bz_typename asExpr<T>::T_expr::T_numtype> >           \
        (asExpr<sca >::getExpr(d1),                                       \
         asExpr<T>::getExpr(d2.unwrap()));                                \
}                                                                         \
                                                                          \
template<typename T>                                                      \
_bz_inline_et                                                             \
_bz_ArrayExprBinaryOp<                                                    \
    _bz_typename asExpr<T>::T_expr,                                       \
    asExpr<sca >::T_expr,                                                 \
    applic<_bz_typename asExpr<T>::T_expr::T_numtype,sca > >              \
name(const ETBase<T>& d1, const sca d2)                                   \
{                                                                         \
    return _bz_ArrayExprBinaryOp<                                         \
        _bz_typename asExpr<T>::T_expr,                                   \
        asExpr<sca >::T_expr,                                             \
        applic<_bz_typename asExpr<T>::T_expr::T_numtype,sca > >          \
        (asExpr<T>::getExpr(d1.unwrap()),                                 \
         asExpr<sca >::getExpr(d2));                                      \
}

/*
 * Array expression templates: the macro BZ_DECLARE_ARRAY_ET_TERNARY(X,Y)
 * declares a function or operator which takes three operands.
 * X is the function name (or operator), and Y is the functor object
 * which implements the operation.
 */

#define BZ_DECLARE_ARRAY_ET_TERNARY(name, applic)                         \
                                                                          \
template<typename T1, typename T2, typename T3>                           \
_bz_inline_et                                                             \
_bz_ArrayExpr<_bz_ArrayExprTernaryOp<                                     \
    _bz_typename asExpr<T1>::T_expr,                                      \
    _bz_typename asExpr<T2>::T_expr,                                      \
    _bz_typename asExpr<T3>::T_expr,                                      \
    applic<_bz_typename asExpr<T1>::T_expr::T_numtype,                    \
           _bz_typename asExpr<T2>::T_expr::T_numtype,                    \
           _bz_typename asExpr<T3>::T_expr::T_numtype> > >                \
name(const ETBase<T1>& d1, const ETBase<T2>& d2, const ETBase<T3>& d3)    \
{                                                                         \
    return _bz_ArrayExpr<_bz_ArrayExprTernaryOp<                          \
        _bz_typename asExpr<T1>::T_expr,                                  \
        _bz_typename asExpr<T2>::T_expr,                                  \
        _bz_typename asExpr<T3>::T_expr,                                  \
        applic<_bz_typename asExpr<T1>::T_expr::T_numtype,                \
               _bz_typename asExpr<T2>::T_expr::T_numtype,                \
               _bz_typename asExpr<T3>::T_expr::T_numtype> > >            \
        (asExpr<T1>::getExpr(d1.unwrap()),                                \
         asExpr<T2>::getExpr(d2.unwrap()),                                \
         asExpr<T3>::getExpr(d3.unwrap()));                               \
}

#endif /* BZ_HAVE_TEMPLATES_AS_TEMPLATE_ARGUMENTS */

/*
 * User-defined expression template routines
 */

#define BZ_DECLARE_FUNCTION(name)                                        \
BZ_DEFINE_UNARY_FUNC(name ## _impl,name)                                 \
BZ_DECLARE_ARRAY_ET_UNARY(name,name ## _impl)

#define BZ_DECLARE_FUNCTION_RET(name,return_type)                        \
BZ_DEFINE_UNARY_FUNC_RET(name ## _impl,name,return_type)                 \
BZ_DECLARE_ARRAY_ET_UNARY(name,name ## _impl)

#define BZ_DECLARE_FUNCTION2(name)                                       \
BZ_DEFINE_BINARY_FUNC(name ## _impl,name)                                \
BZ_DECLARE_ARRAY_ET_BINARY(name, name ## _impl)

#define BZ_DECLARE_FUNCTION2_RET(name,return_type)                       \
BZ_DEFINE_BINARY_FUNC_RET(name ## _impl,name,return_type)                \
BZ_DECLARE_ARRAY_ET_BINARY(name, name ## _impl)

#define BZ_DECLARE_FUNCTION2_SCALAR(name, sca)                           \
BZ_DECLARE_ARRAY_ET_BINARY_SCALAR(name, name ## _impl, sca)

#define BZ_DECLARE_FUNCTION3(name)                                       \
BZ_DEFINE_TERNARY_FUNC(name ## _impl,name)                               \
BZ_DECLARE_ARRAY_ET_TERNARY(name, name ## _impl)
    
#define BZ_DECLARE_FUNCTION3_RET(name,return_type)                       \
BZ_DEFINE_TERNARY_FUNC_RET(name ## _impl,name,return_type)               \
BZ_DECLARE_ARRAY_ET_TERNARY(name, name ## _impl)

BZ_NAMESPACE_END

#endif
