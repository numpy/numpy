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
 * Revision 1.1.1.1  2000/06/19 12:26:09  tveldhui
 * Imported sources
 *
 * Revision 1.5  1998/03/14 00:04:47  tveldhui
 * 0.2-alpha-05
 *
 * Revision 1.4  1997/07/16 14:51:20  tveldhui
 * Update: Alpha release 0.2 (Arrays)
 *
 * Revision 1.3  1997/01/24 14:42:00  tveldhui
 * Periodic RCS update
 *
 */

#ifndef BZ_VECPICK_CC
#define BZ_VECPICK_CC

#ifndef BZ_VECPICK_H
 #include <blitz/vecpick.h>
#endif

#ifndef BZ_UPDATE_H
 #include <blitz/update.h>
#endif

#ifndef BZ_RANDOM_H
 #include <blitz/random.h>
#endif

#ifndef BZ_VECEXPR_H
 #include <blitz/vecexpr.h>
#endif

#ifndef BZ_RANDOM_H
 #include <blitz/random.h>
#endif
BZ_NAMESPACE(blitz)

/*****************************************************************************
 * Assignment operators with vector expression operand
 */

template<class P_numtype> template<class P_expr, class P_updater>
inline
void VectorPick<P_numtype>::_bz_assign(P_expr expr, P_updater)
{
    BZPRECONDITION(expr.length(length()) == length());

    // If all vectors in the expression, plus the vector to which the
    // result is being assigned have unit stride, then avoid stride
    // calculations.
    if (_bz_hasFastAccess() && expr._bz_hasFastAccess())
    {
#ifndef BZ_PARTIAL_LOOP_UNROLL
        for (int i=0; i < length(); ++i)
            P_updater::update(vector_(index_(i)), expr._bz_fastAccess(i));
#else
        // Unwind the inner loop, five elements at a time.
        int leftover = length() % 5;

        int i=0;
        for (; i < leftover; ++i)
            P_updater::update(vector_(index_(i)), expr._bz_fastAccess(i));

        for (; i < length(); i += 5)
        {
            P_updater::update(vector_(index_(i)), expr._bz_fastAccess(i));
            P_updater::update(vector_(index_(i+1)), expr._bz_fastAccess(i+1));
            P_updater::update(vector_(index_(i+2)), expr._bz_fastAccess(i+2));
            P_updater::update(vector_(index_(i+3)), expr._bz_fastAccess(i+3));
            P_updater::update(vector_(index_(i+4)), expr._bz_fastAccess(i+4));
        }
#endif
    }
    else {
        // Not all unit strides -- have to access all the vector elements
        // as data_[i*stride_], which is slower.
        for (int i=0; i < length(); ++i)
            P_updater::update(vector_[index_[i]], expr[i]);
    }
}

template<class P_numtype> template<class P_expr>
inline VectorPick<P_numtype>& 
VectorPick<P_numtype>::operator=(_bz_VecExpr<P_expr> expr)
{
    BZPRECONDITION(expr.length(length()) == length());

    // If all vectors in the expression, plus the vector to which the
    // result is being assigned have unit stride, then avoid stride
    // calculations.
    if (_bz_hasFastAccess() && expr._bz_hasFastAccess())
    {
#ifndef BZ_PARTIAL_LOOP_UNROLL
        for (int i=0; i < length(); ++i)
            (*this)(i) = expr._bz_fastAccess(i);
#else
        // Unwind the inner loop, five elements at a time.
        int leftover = length() % 5;

        int i=0;
        for (; i < leftover; ++i)
            (*this)(i) = expr._bz_fastAccess(i);

        for (; i < length(); i += 5)
        {
            (*this)(i) = expr._bz_fastAccess(i);
            (*this)(i+1) = expr._bz_fastAccess(i+1);
            (*this)(i+2) = expr._bz_fastAccess(i+2);
            (*this)(i+3) = expr._bz_fastAccess(i+3);
            (*this)(i+4) = expr._bz_fastAccess(i+4);
        }
#endif
    }
    else {
        // Not all unit strides -- have to access all the vector elements
        // as data_[i*stride_], which is slower.
        for (int i=0; i < length(); ++i)
            (*this)[i] = expr[i];
    }
    return *this;
}


template<class P_numtype> template<class P_expr>
inline VectorPick<P_numtype>& 
VectorPick<P_numtype>::operator+=(_bz_VecExpr<P_expr> expr)
{
    _bz_assign(expr, _bz_plus_update<P_numtype,
        _bz_typename P_expr::T_numtype>());
    return *this;
}

template<class P_numtype> template<class P_expr>
inline VectorPick<P_numtype>& 
VectorPick<P_numtype>::operator-=(_bz_VecExpr<P_expr> expr)
{
    _bz_assign(expr, _bz_minus_update<P_numtype,
        _bz_typename P_expr::T_numtype>());
    return *this;
}

template<class P_numtype> template<class P_expr>
inline VectorPick<P_numtype>& 
VectorPick<P_numtype>::operator*=(_bz_VecExpr<P_expr> expr)
{
    _bz_assign(expr, _bz_multiply_update<P_numtype,
        _bz_typename P_expr::T_numtype>());
    return *this;
}

template<class P_numtype> template<class P_expr>
inline VectorPick<P_numtype>& 
VectorPick<P_numtype>::operator/=(_bz_VecExpr<P_expr> expr)
{
    _bz_assign(expr, _bz_divide_update<P_numtype,
        _bz_typename P_expr::T_numtype>());
    return *this;
}

template<class P_numtype> template<class P_expr>
inline VectorPick<P_numtype>& 
VectorPick<P_numtype>::operator%=(_bz_VecExpr<P_expr> expr)
{
    _bz_assign(expr, _bz_mod_update<P_numtype,
        _bz_typename P_expr::T_numtype>());
    return *this;
}

template<class P_numtype> template<class P_expr>
inline VectorPick<P_numtype>& 
VectorPick<P_numtype>::operator^=(_bz_VecExpr<P_expr> expr)
{
    _bz_assign(expr, _bz_xor_update<P_numtype,
        _bz_typename P_expr::T_numtype>());
    return *this;
}

template<class P_numtype> template<class P_expr>
inline VectorPick<P_numtype>& 
VectorPick<P_numtype>::operator&=(_bz_VecExpr<P_expr> expr)
{
    _bz_assign(expr, _bz_bitand_update<P_numtype,
        _bz_typename P_expr::T_numtype>());
    return *this;
}

template<class P_numtype> template<class P_expr>
inline VectorPick<P_numtype>& 
VectorPick<P_numtype>::operator|=(_bz_VecExpr<P_expr> expr)
{
    _bz_assign(expr, _bz_bitor_update<P_numtype,
        _bz_typename P_expr::T_numtype>());
    return *this;
}

template<class P_numtype> template<class P_expr>
inline VectorPick<P_numtype>& 
VectorPick<P_numtype>::operator>>=(_bz_VecExpr<P_expr> expr)
{
    _bz_assign(expr, _bz_shiftr_update<P_numtype,
        _bz_typename P_expr::T_numtype>());
    return *this;
}

template<class P_numtype> template<class P_expr>
inline VectorPick<P_numtype>& 
VectorPick<P_numtype>::operator<<=(_bz_VecExpr<P_expr> expr)
{
    _bz_assign(expr, _bz_shiftl_update<P_numtype,
        _bz_typename P_expr::T_numtype>());
    return *this;
}

/*****************************************************************************
 * Assignment operators with scalar operand
 */

template<class P_numtype>
inline VectorPick<P_numtype>& VectorPick<P_numtype>::operator=(P_numtype x)
{
    typedef _bz_VecExprConstant<P_numtype> T_expr;
    (*this) = _bz_VecExpr<T_expr>(T_expr(x));
    return *this;
}

template<class P_numtype>
inline VectorPick<P_numtype>& VectorPick<P_numtype>::operator+=(P_numtype x)
{
    typedef _bz_VecExprConstant<P_numtype> T_expr;
    (*this) += _bz_VecExpr<T_expr>(T_expr(x));
    return *this;
}

template<class P_numtype>
inline VectorPick<P_numtype>& VectorPick<P_numtype>::operator-=(P_numtype x)
{
    typedef _bz_VecExprConstant<P_numtype> T_expr;
    (*this) -= _bz_VecExpr<T_expr>(T_expr(x));
    return *this;
}

template<class P_numtype>
inline VectorPick<P_numtype>& VectorPick<P_numtype>::operator*=(P_numtype x)
{
    typedef _bz_VecExprConstant<P_numtype> T_expr;
    (*this) *= _bz_VecExpr<T_expr>(T_expr(x));
    return *this;
}

template<class P_numtype>
inline VectorPick<P_numtype>& VectorPick<P_numtype>::operator/=(P_numtype x)
{
    typedef _bz_VecExprConstant<P_numtype> T_expr;
    (*this) /= _bz_VecExpr<T_expr>(T_expr(x));
    return *this;
}

template<class P_numtype>
inline VectorPick<P_numtype>& VectorPick<P_numtype>::operator%=(P_numtype x)
{
    typedef _bz_VecExprConstant<P_numtype> T_expr;
    (*this) %= _bz_VecExpr<T_expr>(T_expr(x));
    return *this;
}

template<class P_numtype>
inline VectorPick<P_numtype>& VectorPick<P_numtype>::operator^=(P_numtype x)
{
    typedef _bz_VecExprConstant<P_numtype> T_expr;
    (*this) ^= _bz_VecExpr<T_expr>(T_expr(x));
    return *this;
}

template<class P_numtype>
inline VectorPick<P_numtype>& VectorPick<P_numtype>::operator&=(P_numtype x)
{
    typedef _bz_VecExprConstant<P_numtype> T_expr;
    (*this) &= _bz_VecExpr<T_expr>(T_expr(x));
    return *this;
}

template<class P_numtype>
inline VectorPick<P_numtype>& VectorPick<P_numtype>::operator|=(P_numtype x)
{
    typedef _bz_VecExprConstant<P_numtype> T_expr;
    (*this) |= _bz_VecExpr<T_expr>(T_expr(x));
    return *this;
}

template<class P_numtype>
inline VectorPick<P_numtype>& VectorPick<P_numtype>::operator>>=(int x)
{
    typedef _bz_VecExprConstant<int> T_expr;
    (*this) >>= _bz_VecExpr<T_expr>(T_expr(x));
    return *this;
}

template<class P_numtype>
inline VectorPick<P_numtype>& VectorPick<P_numtype>::operator<<=(int x)
{
    typedef _bz_VecExprConstant<P_numtype> T_expr;
    (*this) <<= _bz_VecExpr<T_expr>(T_expr(x));
    return *this;
}

/*****************************************************************************
 * Assignment operators with vector operand
 */

template<class P_numtype> template<class P_numtype2>
inline VectorPick<P_numtype>& 
VectorPick<P_numtype>::operator=(const Vector<P_numtype2>& x)
{
    (*this) = _bz_VecExpr<VectorIterConst<P_numtype2> >(x.begin());
    return *this;
}

template<class P_numtype> template<class P_numtype2>
inline VectorPick<P_numtype>&
VectorPick<P_numtype>::operator+=(const Vector<P_numtype2>& x)
{
    (*this) += _bz_VecExpr<VectorIterConst<P_numtype2> >(x.begin());
    return *this;
}

template<class P_numtype> template<class P_numtype2>
inline VectorPick<P_numtype>&
VectorPick<P_numtype>::operator-=(const Vector<P_numtype2>& x)
{
    (*this) -= _bz_VecExpr<VectorIterConst<P_numtype2> >(x.begin());
    return *this;
}

template<class P_numtype> template<class P_numtype2>
inline VectorPick<P_numtype>&
VectorPick<P_numtype>::operator*=(const Vector<P_numtype2>& x)
{
    (*this) *= _bz_VecExpr<VectorIterConst<P_numtype2> >(x.begin());
    return *this;
}

template<class P_numtype> template<class P_numtype2>
inline VectorPick<P_numtype>&
VectorPick<P_numtype>::operator/=(const Vector<P_numtype2>& x)
{
    (*this) /= _bz_VecExpr<VectorIterConst<P_numtype2> >(x.begin());
    return *this;
}

template<class P_numtype> template<class P_numtype2>
inline VectorPick<P_numtype>&
VectorPick<P_numtype>::operator%=(const Vector<P_numtype2>& x)
{
    (*this) %= _bz_VecExpr<VectorIterConst<P_numtype2> >(x.begin());
    return *this;
}

template<class P_numtype> template<class P_numtype2>
inline VectorPick<P_numtype>&
VectorPick<P_numtype>::operator^=(const Vector<P_numtype2>& x)
{
    (*this) ^= _bz_VecExpr<VectorIterConst<P_numtype2> >(x.begin());
    return *this;
}

template<class P_numtype> template<class P_numtype2>
inline VectorPick<P_numtype>&
VectorPick<P_numtype>::operator&=(const Vector<P_numtype2>& x)
{
    (*this) &= _bz_VecExpr<VectorIterConst<P_numtype2> >(x.begin());
    return *this;
}

template<class P_numtype> template<class P_numtype2>
inline VectorPick<P_numtype>&
VectorPick<P_numtype>::operator|=(const Vector<P_numtype2>& x)
{
    (*this) |= _bz_VecExpr<VectorIterConst<P_numtype2> >(x.begin());
    return *this;
}

template<class P_numtype> template<class P_numtype2>
inline VectorPick<P_numtype>&
VectorPick<P_numtype>::operator<<=(const Vector<P_numtype2>& x)
{
    (*this) <<= _bz_VecExpr<VectorIterConst<P_numtype2> >(x.begin());
    return *this;
}

template<class P_numtype> template<class P_numtype2>
inline VectorPick<P_numtype>&
VectorPick<P_numtype>::operator>>=(const Vector<P_numtype2>& x)
{
    (*this) >>= _bz_VecExpr<VectorIterConst<P_numtype2> >(x.begin());
    return *this;
}

/*****************************************************************************
 * Assignment operators with Range operand
 */

template<class P_numtype>
inline VectorPick<P_numtype>& VectorPick<P_numtype>::operator=(Range r)
{
    (*this) = _bz_VecExpr<Range>(r);
    return *this;
}

template<class P_numtype>
inline VectorPick<P_numtype>& VectorPick<P_numtype>::operator+=(Range r)
{
    (*this) += _bz_VecExpr<Range>(r);
    return *this;
}

template<class P_numtype>
inline VectorPick<P_numtype>& VectorPick<P_numtype>::operator-=(Range r)
{
    (*this) -= _bz_VecExpr<Range>(r);
    return *this;
}

template<class P_numtype>
inline VectorPick<P_numtype>& VectorPick<P_numtype>::operator*=(Range r)
{
    (*this) *= _bz_VecExpr<Range>(r);
    return *this;
}

template<class P_numtype>
inline VectorPick<P_numtype>& VectorPick<P_numtype>::operator/=(Range r)
{
    (*this) /= _bz_VecExpr<Range>(r);
    return *this;
}

template<class P_numtype>
inline VectorPick<P_numtype>& VectorPick<P_numtype>::operator%=(Range r)
{
    (*this) %= _bz_VecExpr<Range>(r);
    return *this;
}

template<class P_numtype>
inline VectorPick<P_numtype>& VectorPick<P_numtype>::operator^=(Range r)
{
    (*this) ^= _bz_VecExpr<Range>(r);
    return *this;
}

template<class P_numtype>
inline VectorPick<P_numtype>& VectorPick<P_numtype>::operator&=(Range r)
{
    (*this) &= _bz_VecExpr<Range>(r);
    return *this;
}

template<class P_numtype>
inline VectorPick<P_numtype>& VectorPick<P_numtype>::operator|=(Range r)
{
    (*this) |= _bz_VecExpr<Range>(r);
    return *this;
}

template<class P_numtype>
inline VectorPick<P_numtype>& VectorPick<P_numtype>::operator>>=(Range r)
{
    (*this) >>= _bz_VecExpr<Range>(r);
    return *this;
}

template<class P_numtype>
inline VectorPick<P_numtype>& VectorPick<P_numtype>::operator<<=(Range r)
{
    (*this) <<= _bz_VecExpr<Range>(r);
    return *this;
}

/*****************************************************************************
 * Assignment operators with VectorPick operand
 */

template<class P_numtype> template<class P_numtype2>
inline VectorPick<P_numtype>& VectorPick<P_numtype>::operator=(const 
    VectorPick<P_numtype2>& x)
{
    typedef VectorPickIterConst<P_numtype2> T_expr;
    (*this) = _bz_VecExpr<T_expr>(x.begin());
    return *this;
}

template<class P_numtype> template<class P_numtype2>
inline VectorPick<P_numtype>& VectorPick<P_numtype>::operator+=(const
    VectorPick<P_numtype2>& x)
{
    typedef VectorPickIterConst<P_numtype2> T_expr;
    (*this) += _bz_VecExpr<T_expr>(x.begin());
    return *this;
}


template<class P_numtype> template<class P_numtype2>
inline VectorPick<P_numtype>& VectorPick<P_numtype>::operator-=(const
    VectorPick<P_numtype2>& x)
{
    typedef VectorPickIterConst<P_numtype2> T_expr;
    (*this) -= _bz_VecExpr<T_expr>(x.begin());
    return *this;
}

template<class P_numtype> template<class P_numtype2>
inline VectorPick<P_numtype>& VectorPick<P_numtype>::operator*=(const
    VectorPick<P_numtype2>& x)
{
    typedef VectorPickIterConst<P_numtype2> T_expr;
    (*this) *= _bz_VecExpr<T_expr>(x.begin());
    return *this;
}

template<class P_numtype> template<class P_numtype2>
inline VectorPick<P_numtype>& VectorPick<P_numtype>::operator/=(const
    VectorPick<P_numtype2>& x)
{
    typedef VectorPickIterConst<P_numtype2> T_expr;
    (*this) /= _bz_VecExpr<T_expr>(x.begin());
    return *this;
}

template<class P_numtype> template<class P_numtype2>
inline VectorPick<P_numtype>& VectorPick<P_numtype>::operator%=(const
    VectorPick<P_numtype2>& x)
{
    typedef VectorPickIterConst<P_numtype2> T_expr;
    (*this) %= _bz_VecExpr<T_expr>(x.begin());
    return *this;
}

template<class P_numtype> template<class P_numtype2>
inline VectorPick<P_numtype>& VectorPick<P_numtype>::operator^=(const
    VectorPick<P_numtype2>& x)
{
    typedef VectorPickIterConst<P_numtype2> T_expr;
    (*this) ^= _bz_VecExpr<T_expr>(x.begin());
    return *this;
}

template<class P_numtype> template<class P_numtype2>
inline VectorPick<P_numtype>& VectorPick<P_numtype>::operator&=(const
    VectorPick<P_numtype2>& x)
{
    typedef VectorPickIterConst<P_numtype2> T_expr;
    (*this) &= _bz_VecExpr<T_expr>(x.begin());
    return *this;
}

template<class P_numtype> template<class P_numtype2>
inline VectorPick<P_numtype>& VectorPick<P_numtype>::operator|=(const
    VectorPick<P_numtype2>& x)
{
    typedef VectorPickIterConst<P_numtype2> T_expr;
    (*this) |= _bz_VecExpr<T_expr>(x.begin());
    return *this;
}

/*****************************************************************************
 * Assignment operators with Random operand
 */

template<class P_numtype> template<class P_distribution>
VectorPick<P_numtype>& 
VectorPick<P_numtype>::operator=(Random<P_distribution>& rand)
{
    (*this) = _bz_VecExpr<_bz_VecExprRandom<P_distribution> > 
        (_bz_VecExprRandom<P_distribution>(rand));
    return *this;
}

template<class P_numtype> template<class P_distribution>
VectorPick<P_numtype>& 
VectorPick<P_numtype>::operator+=(Random<P_distribution>& rand)
{
    (*this) += _bz_VecExpr<_bz_VecExprRandom<P_distribution> >
        (_bz_VecExprRandom<P_distribution>(rand));
    return *this;
}

template<class P_numtype> template<class P_distribution>
VectorPick<P_numtype>& 
VectorPick<P_numtype>::operator-=(Random<P_distribution>& rand)
{
    (*this) -= _bz_VecExpr<_bz_VecExprRandom<P_distribution> >
        (_bz_VecExprRandom<P_distribution>(rand));
    return *this;
}

template<class P_numtype> template<class P_distribution>
VectorPick<P_numtype>& 
VectorPick<P_numtype>::operator*=(Random<P_distribution>& rand)
{
    (*this) *= _bz_VecExpr<_bz_VecExprRandom<P_distribution> >
        (_bz_VecExprRandom<P_distribution>(rand));
    return *this;
}

template<class P_numtype> template<class P_distribution>
VectorPick<P_numtype>& 
VectorPick<P_numtype>::operator/=(Random<P_distribution>& rand)
{
    (*this) /= _bz_VecExpr<_bz_VecExprRandom<P_distribution> >
        (_bz_VecExprRandom<P_distribution>(rand));
    return *this;
}

template<class P_numtype> template<class P_distribution>
VectorPick<P_numtype>& 
VectorPick<P_numtype>::operator%=(Random<P_distribution>& rand)
{
    (*this) %= _bz_VecExpr<_bz_VecExprRandom<P_distribution> >
        (_bz_VecExprRandom<P_distribution>(rand));
    return *this;
}

template<class P_numtype> template<class P_distribution>
VectorPick<P_numtype>& 
VectorPick<P_numtype>::operator^=(Random<P_distribution>& rand)
{
    (*this) ^= _bz_VecExpr<_bz_VecExprRandom<P_distribution> >
        (_bz_VecExprRandom<P_distribution>(rand));
    return *this;
}

template<class P_numtype> template<class P_distribution>
VectorPick<P_numtype>& 
VectorPick<P_numtype>::operator&=(Random<P_distribution>& rand)
{
    (*this) &= _bz_VecExpr<_bz_VecExprRandom<P_distribution> >
        (_bz_VecExprRandom<P_distribution>(rand));
    return *this;
}

template<class P_numtype> template<class P_distribution>
VectorPick<P_numtype>& 
VectorPick<P_numtype>::operator|=(Random<P_distribution>& rand)
{
    (*this) |= _bz_VecExpr<_bz_VecExprRandom<P_distribution> >
        (_bz_VecExprRandom<P_distribution>(rand));
    return *this;
}

BZ_NAMESPACE_END

#endif // BZ_VECPICK_CC
