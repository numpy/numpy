/***************************************************************************
 * blitz/array/funcs.h   Math functions on arrays
 *
 * $Id$
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
 * Revision 1.5  2002/07/16 22:05:54  jcumming
 * Removed ET support for Array expressions involving ldexp(), jn() and yn()
 * functions.  These functions require specialized macros that allow one of
 * the function arguments to be an ordinary int.  Such macros have not yet
 * been added to <blitz/funcs.h>.
 *
 * Revision 1.4  2002/07/02 19:14:01  jcumming
 * Use new style of Array ET macros to declare unary and binary math functions
 * that act on Array types.
 *
 * Revision 1.3  2001/01/26 20:11:25  tveldhui
 * Changed isnan to blitz_isnan, to avoid conflicts with implementations
 * that define isnan as a preprocessor macro.
 *
 * Revision 1.2  2001/01/25 00:25:55  tveldhui
 * Ensured that source files have cvs logs.
 *
 */

#ifndef BZ_ARRAY_FUNCS_H
#define BZ_ARRAY_FUNCS_H

#ifndef BZ_FUNCS_H
 #include <blitz/funcs.h>
#endif

#ifndef BZ_NEWET_MACROS_H
 #include <blitz/array/newet-macros.h>
#endif

BZ_NAMESPACE(blitz)
    
// unary functions
    
BZ_DECLARE_ARRAY_ET_UNARY(abs,   Fn_abs)
BZ_DECLARE_ARRAY_ET_UNARY(acos,  Fn_acos)
BZ_DECLARE_ARRAY_ET_UNARY(asin,  Fn_asin)
BZ_DECLARE_ARRAY_ET_UNARY(atan,  Fn_atan)
BZ_DECLARE_ARRAY_ET_UNARY(ceil,  Fn_ceil)
BZ_DECLARE_ARRAY_ET_UNARY(cexp,  Fn_exp)
BZ_DECLARE_ARRAY_ET_UNARY(cos,   Fn_cos)
BZ_DECLARE_ARRAY_ET_UNARY(cosh,  Fn_cosh)
BZ_DECLARE_ARRAY_ET_UNARY(csqrt, Fn_sqrt)
BZ_DECLARE_ARRAY_ET_UNARY(cube,  Fn_cube)
BZ_DECLARE_ARRAY_ET_UNARY(exp,   Fn_exp)
BZ_DECLARE_ARRAY_ET_UNARY(fabs,  Fn_fabs)
BZ_DECLARE_ARRAY_ET_UNARY(floor, Fn_floor)
BZ_DECLARE_ARRAY_ET_UNARY(log,   Fn_log)
BZ_DECLARE_ARRAY_ET_UNARY(log10, Fn_log10)
BZ_DECLARE_ARRAY_ET_UNARY(pow2,  Fn_sqr)
BZ_DECLARE_ARRAY_ET_UNARY(pow3,  Fn_cube)
BZ_DECLARE_ARRAY_ET_UNARY(pow4,  Fn_pow4)
BZ_DECLARE_ARRAY_ET_UNARY(pow5,  Fn_pow5)
BZ_DECLARE_ARRAY_ET_UNARY(pow6,  Fn_pow6)
BZ_DECLARE_ARRAY_ET_UNARY(pow7,  Fn_pow7)
BZ_DECLARE_ARRAY_ET_UNARY(pow8,  Fn_pow8)
BZ_DECLARE_ARRAY_ET_UNARY(sin,   Fn_sin)
BZ_DECLARE_ARRAY_ET_UNARY(sinh,  Fn_sinh)
BZ_DECLARE_ARRAY_ET_UNARY(sqr,   Fn_sqr)
BZ_DECLARE_ARRAY_ET_UNARY(sqrt,  Fn_sqrt)
BZ_DECLARE_ARRAY_ET_UNARY(tan,   Fn_tan)
BZ_DECLARE_ARRAY_ET_UNARY(tanh,  Fn_tanh)

#ifdef BZ_HAVE_COMPLEX_MATH
BZ_DECLARE_ARRAY_ET_UNARY(arg,   Fn_arg)
BZ_DECLARE_ARRAY_ET_UNARY(conj,  Fn_conj)
BZ_DECLARE_ARRAY_ET_UNARY(imag,  Fn_imag)
BZ_DECLARE_ARRAY_ET_UNARY(norm,  Fn_norm)
BZ_DECLARE_ARRAY_ET_UNARY(real,  Fn_real)
#endif

#ifdef BZ_HAVE_IEEE_MATH
// finite and trunc omitted: blitz-bugs/archive/0189.html
BZ_DECLARE_ARRAY_ET_UNARY(acosh,  Fn_acosh)
BZ_DECLARE_ARRAY_ET_UNARY(asinh,  Fn_asinh)
BZ_DECLARE_ARRAY_ET_UNARY(atanh,  Fn_atanh)
BZ_DECLARE_ARRAY_ET_UNARY(cbrt,   Fn_cbrt)
BZ_DECLARE_ARRAY_ET_UNARY(erf,    Fn_erf)
BZ_DECLARE_ARRAY_ET_UNARY(erfc,   Fn_erfc)
BZ_DECLARE_ARRAY_ET_UNARY(expm1,  Fn_expm1)
// BZ_DECLARE_ARRAY_ET_UNARY(finite, Fn_finite)
BZ_DECLARE_ARRAY_ET_UNARY(ilogb,   Fn_ilogb)
BZ_DECLARE_ARRAY_ET_UNARY(blitz_isnan,  Fn_isnan)
BZ_DECLARE_ARRAY_ET_UNARY(j0,     Fn_j0)
BZ_DECLARE_ARRAY_ET_UNARY(j1,     Fn_j1)
BZ_DECLARE_ARRAY_ET_UNARY(lgamma, Fn_lgamma)
BZ_DECLARE_ARRAY_ET_UNARY(logb,   Fn_logb)
BZ_DECLARE_ARRAY_ET_UNARY(log1p,  Fn_log1p)
BZ_DECLARE_ARRAY_ET_UNARY(rint,   Fn_rint)
// BZ_DECLARE_ARRAY_ET_UNARY(trunc,  Fn_trunc)
BZ_DECLARE_ARRAY_ET_UNARY(y0,     Fn_y0)
BZ_DECLARE_ARRAY_ET_UNARY(y1,     Fn_y1)
#endif

#ifdef BZ_HAVE_SYSTEM_V_MATH
BZ_DECLARE_ARRAY_ET_UNARY(_class,  Fn__class)
BZ_DECLARE_ARRAY_ET_UNARY(itrunc,  Fn_itrunc)
BZ_DECLARE_ARRAY_ET_UNARY(nearest, Fn_nearest)
BZ_DECLARE_ARRAY_ET_UNARY(rsqrt,   Fn_rsqrt)
BZ_DECLARE_ARRAY_ET_UNARY(uitrunc, Fn_uitrunc)
#endif
    
// cast() function
    
template<class T_cast, class T1>
_bz_inline_et
_bz_ArrayExpr<_bz_ArrayExprUnaryOp<_bz_typename asExpr<T1>::T_expr,
    Cast<_bz_typename asExpr<T1>::T_expr::T_numtype, T_cast> > >
cast(const ETBase<T1>& expr)
{
    return _bz_ArrayExpr<_bz_ArrayExprUnaryOp<
        _bz_typename asExpr<T1>::T_expr,
        Cast<_bz_typename asExpr<T1>::T_expr::T_numtype,T_cast> > >
        (static_cast<const T1&>(expr));
}

// binary functions

BZ_DECLARE_ARRAY_ET_BINARY(atan2,     Fn_atan2)
BZ_DECLARE_ARRAY_ET_BINARY(fmod,      Fn_fmod)
BZ_DECLARE_ARRAY_ET_BINARY(pow,       Fn_pow)

#ifdef BZ_HAVE_COMPLEX_MATH
BZ_DECLARE_ARRAY_ET_BINARY(polar,     Fn_polar)
#endif
    
#ifdef BZ_HAVE_SYSTEM_V_MATH
BZ_DECLARE_ARRAY_ET_BINARY(copysign,  Fn_copysign)
BZ_DECLARE_ARRAY_ET_BINARY(drem,      Fn_drem)
BZ_DECLARE_ARRAY_ET_BINARY(hypot,     Fn_hypot)
BZ_DECLARE_ARRAY_ET_BINARY(nextafter, Fn_nextafter)
BZ_DECLARE_ARRAY_ET_BINARY(remainder, Fn_remainder)
BZ_DECLARE_ARRAY_ET_BINARY(scalb,     Fn_scalb)
BZ_DECLARE_ARRAY_ET_BINARY(unordered, Fn_unordered)
#endif

BZ_NAMESPACE_END

#endif // BZ_ARRAY_FUNCS_H
