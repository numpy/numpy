#ifndef BZ_ARRAYMISC_CC
#define BZ_ARRAYMISC_CC

#ifndef BZ_ARRAY_H
 #error <blitz/array/misc.cc> must be included via <blitz/array.h>
#endif

BZ_NAMESPACE(blitz)

#define BZ_ARRAY_DECLARE_UOP(fn, fnobj)                                \
template<class T_numtype, int N_rank>                                  \
inline                                                                 \
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<FastArrayIterator<T_numtype,N_rank>, \
    fnobj<T_numtype> > >                                               \
fn(const Array<T_numtype,N_rank>& array)                               \
{                                                                      \
    return _bz_ArrayExprUnaryOp<FastArrayIterator<T_numtype,N_rank>,   \
        fnobj<T_numtype> >(array.beginFast());                         \
}                                                                      \
                                                                       \
template<class T_expr>                                                 \
inline                                                                 \
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<_bz_ArrayExpr<T_expr>,              \
    fnobj<_bz_typename T_expr::T_numtype> > >                          \
fn(BZ_ETPARM(_bz_ArrayExpr<T_expr>) expr)                              \
{                                                                      \
    return _bz_ArrayExprUnaryOp<_bz_ArrayExpr<T_expr>,                 \
        fnobj<_bz_typename T_expr::T_numtype> >(expr);                 \
}                                                                      
                                                                       
BZ_ARRAY_DECLARE_UOP(operator!, LogicalNot)
BZ_ARRAY_DECLARE_UOP(operator~, BitwiseNot)
BZ_ARRAY_DECLARE_UOP(operator-, Negate)

/*
 * cast() functions, for explicit type casting
 */

template<class T_numtype, int N_rank, class T_cast>
inline                                                                 
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<FastArrayIterator<T_numtype,N_rank>,   
    Cast<T_numtype, T_cast> > >
cast(const Array<T_numtype,N_rank>& array, T_cast)
{                                                                      
    return _bz_ArrayExprUnaryOp<FastArrayIterator<T_numtype,N_rank>,      
        Cast<T_numtype,T_cast> >(array.beginFast());                            
}                                                                      
                                                                       
template<class T_expr, class T_cast>
inline                                                                 
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<_bz_ArrayExpr<T_expr>,              
    Cast<_bz_typename T_expr::T_numtype,T_cast> > >                          
cast(BZ_ETPARM(_bz_ArrayExpr<T_expr>) expr, T_cast)
{                                                                      
    return _bz_ArrayExprUnaryOp<_bz_ArrayExpr<T_expr>,                 
        Cast<_bz_typename T_expr::T_numtype,T_cast> >(expr);                 
}                                                                      
                                                                       
BZ_NAMESPACE_END

#endif // BZ_ARRAYMISC_CC

