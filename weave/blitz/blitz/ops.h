/***************************************************************************
 * blitz/ops.h           Function objects for math operators
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
 *************************************************************************
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
 * Revision 1.4  2002/07/02 19:11:06  jcumming
 * Rewrote and reorganized this file to make better use of macros to
 * generate all the functor classes needed to provide unary and binary
 * operators for the "new" style of expression templates.
 *
 * Revision 1.3  2002/03/06 16:06:19  patricg
 *
 * os replaced by str in the BitwiseNot template
 *
 * Revision 1.2  2001/01/24 20:22:50  tveldhui
 * Updated copyright date in headers.
 *
 * Revision 1.1.1.1  2000/06/19 12:26:12  tveldhui
 * Imported sources
 *
 * Revision 1.2  1998/03/14 00:04:47  tveldhui
 * 0.2-alpha-05
 *
 * Revision 1.1  1997/07/16 14:51:20  tveldhui
 * Update: Alpha release 0.2 (Arrays)
 *
 */

#ifndef BZ_OPS_H
#define BZ_OPS_H

#ifndef BZ_BLITZ_H
 #include <blitz/blitz.h>
#endif

#ifndef BZ_PROMOTE_H
 #include <blitz/promote.h>
#endif

#ifndef BZ_PRETTYPRINT_H
 #include <blitz/prettyprint.h>
#endif

BZ_NAMESPACE(blitz)

/*
 * Originally these function objects had no template arguments, e.g.
 *
 * struct Add {
 *     template<class T_numtype1, class T_numtype2>
 *     static inline BZ_PROMOTE(T_numtype1, T_numtype2)
 *     apply(T_numtype1 a, T_numtype2 b)
 *     { return a + b; }
 * };
 *
 * This made for neater expression templates syntax.  However, there are
 * some situations in which users may want to override type promotion
 * for certain operations.  For example, in theoretical physics, there
 * are U1 objects which when multiplied yield U1 objects, but when added
 * yield a different type.  To allow for this kind of behaviour, function
 * objects have been changed to take template parameters:
 *
 * template<class T_numtype1, class T_numtype2>
 * struct Add {
 *     typedef BZ_PROMOTE(T_numtype1, T_numtype2) T_numtype;
 *
 *     static inline T_numtype apply(T_numtype1 a, T_numtype2 b)
 *     { return a + b; }
 * };
 *
 * Type promotion is performed inside the function object.  The expression
 * templates code always looks inside the function object to determine
 * the type promotion, e.g. Add<int,float>::T_numtype
 *
 * Users are free to specialize these function objects for their own types.
 */
    
/* Unary operators that return same type as argument */
    
#define BZ_DEFINE_UNARY_OP(name,op)                         \
template<class T_numtype1>                                  \
struct name {                                               \
    typedef T_numtype1 T_numtype;                           \
                                                            \
    static inline T_numtype                                 \
    apply(T_numtype1 a)                                     \
    { return op a; }                                        \
							    \
    template<class T1>                                      \
    static inline void prettyPrint(string& str,             \
        prettyPrintFormat& format, const T1& t1)            \
    {                                                       \
        str += #op;                                         \
        t1.prettyPrint(str, format);                        \
    }                                                       \
};

BZ_DEFINE_UNARY_OP(BitwiseNot,~)
BZ_DEFINE_UNARY_OP(UnaryPlus,+)
BZ_DEFINE_UNARY_OP(UnaryMinus,-)
    
    
/* Unary operators that return a specified type */
    
#define BZ_DEFINE_UNARY_OP_RET(name,op,ret)                 \
template<class T_numtype1>                                  \
struct name {                                               \
    typedef ret T_numtype;                                  \
    static inline T_numtype                                 \
    apply(T_numtype1 a)                                     \
    { return op a; }                                        \
                                                            \
    template<class T1>                                      \
    static inline void prettyPrint(string& str,             \
        prettyPrintFormat& format, const T1& t1)            \
    {                                                       \
        str += #op;                                         \
        t1.prettyPrint(str, format);                        \
    }                                                       \
};

BZ_DEFINE_UNARY_OP_RET(LogicalNot,!,bool)
    
    
/* Binary operators that return type based on type promotion */
    
#define BZ_DEFINE_BINARY_OP(name,op)                        \
template<class T_numtype1, class T_numtype2>                \
struct name {                                               \
    typedef BZ_PROMOTE(T_numtype1, T_numtype2) T_numtype;   \
                                                            \
    static inline T_numtype                                 \
    apply(T_numtype1 a, T_numtype2 b)                       \
    { return a op b; }                                      \
							    \
    template<class T1, class T2>                            \
    static inline void prettyPrint(string& str,             \
        prettyPrintFormat& format, const T1& t1,            \
        const T2& t2)                                       \
    {                                                       \
        str += "(";                                         \
        t1.prettyPrint(str, format);                        \
        str += #op;                                         \
        t2.prettyPrint(str, format);                        \
        str += ")";                                         \
    }                                                       \
};

BZ_DEFINE_BINARY_OP(Add,+)
BZ_DEFINE_BINARY_OP(Subtract,-)
BZ_DEFINE_BINARY_OP(Multiply,*)
BZ_DEFINE_BINARY_OP(Divide,/)
BZ_DEFINE_BINARY_OP(Modulo,%)
BZ_DEFINE_BINARY_OP(BitwiseXor,^)
BZ_DEFINE_BINARY_OP(BitwiseAnd,&)
BZ_DEFINE_BINARY_OP(BitwiseOr,|)
BZ_DEFINE_BINARY_OP(ShiftRight,>>)
BZ_DEFINE_BINARY_OP(ShiftLeft,<<)
    
    
/* Binary operators that return a specified type */
    
#define BZ_DEFINE_BINARY_OP_RET(name,op,ret)                \
template<class T_numtype1, class T_numtype2>                \
struct name {                                               \
    typedef ret T_numtype;                                  \
    static inline T_numtype                                 \
    apply(T_numtype1 a, T_numtype2 b)                       \
    { return a op b; }                                      \
                                                            \
    template<class T1, class T2>                            \
    static inline void prettyPrint(string& str,             \
        prettyPrintFormat& format, const T1& t1,            \
        const T2& t2)                                       \
    {                                                       \
        str += "(";                                         \
        t1.prettyPrint(str, format);                        \
        str += #op;                                         \
        t2.prettyPrint(str, format);                        \
        str += ")";                                         \
    }                                                       \
};

BZ_DEFINE_BINARY_OP_RET(Greater,>,bool)
BZ_DEFINE_BINARY_OP_RET(Less,<,bool)
BZ_DEFINE_BINARY_OP_RET(GreaterOrEqual,>=,bool)
BZ_DEFINE_BINARY_OP_RET(LessOrEqual,<=,bool)
BZ_DEFINE_BINARY_OP_RET(Equal,==,bool)
BZ_DEFINE_BINARY_OP_RET(NotEqual,!=,bool)
BZ_DEFINE_BINARY_OP_RET(LogicalAnd,&&,bool)
BZ_DEFINE_BINARY_OP_RET(LogicalOr,||,bool)
    
    
BZ_NAMESPACE_END

#endif // BZ_OPS_H


