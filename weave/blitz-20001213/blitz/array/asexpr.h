#ifndef BZ_ARRAYASEXPR_H
#define BZ_ARRAYASEXPR_H

#ifndef BZ_ARRAY_H
 #error <blitz/array/asexpr.h> must be included via <blitz/array.h>
#endif

BZ_NAMESPACE(blitz)

// The traits class asExpr converts arbitrary things to
// expression templatable operands.

// Default to scalar.
template<class T>
struct asExpr {
    typedef _bz_ArrayExprConstant<T> T_expr;
};

// Already an expression template term
template<class T>
struct asExpr<_bz_ArrayExpr<T> > {
    typedef _bz_ArrayExpr<T> T_expr;
};

// An array operand
template<class T, int N>
struct asExpr<Array<T,N> > {
    typedef FastArrayIterator<T,N> T_expr;
};

// Index placeholder
template<int N>
struct asExpr<IndexPlaceholder<N> > {
    typedef IndexPlaceholder<N> T_expr;
};

BZ_NAMESPACE_END

#endif
