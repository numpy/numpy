/***************************************************************************
 * blitz/ops.h           Function objects for operators
 *
 * $Id$
 *
 * Copyright (C) 1997-1999 Todd Veldhuizen <tveldhui@oonumerics.org>
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
 * Revision 1.1  2002/01/03 19:50:34  eric
 * renaming compiler to weave
 *
 * Revision 1.1  2001/04/27 17:22:04  ej
 * first attempt to include needed pieces of blitz
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

#ifndef BZ_MATHFUNC_H
 #include <blitz/mathfunc.h>
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

#define BZ_DEFINE_OP(name,op,symbol)                        \
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
        str += symbol;                                      \
        t2.prettyPrint(str, format);                        \
        str += ")";                                         \
    }                                                       \
}

BZ_DEFINE_OP(Add,+,"+");
BZ_DEFINE_OP(Subtract,-,"-");
BZ_DEFINE_OP(Multiply,*,"*");
BZ_DEFINE_OP(Divide,/,"/");
BZ_DEFINE_OP(Modulo,%,"%");
BZ_DEFINE_OP(BitwiseXor,^,"^");
BZ_DEFINE_OP(BitwiseAnd,&,"&");
BZ_DEFINE_OP(BitwiseOr,|,"|");
BZ_DEFINE_OP(ShiftRight,>>,">>");
BZ_DEFINE_OP(ShiftLeft,<<,"<<");

#define BZ_DEFINE_BOOL_OP(name,op,symbol)                   \
template<class T_numtype1, class T_numtype2>                \
struct name {                                               \
    typedef bool T_numtype;                                 \
    static inline bool                                      \
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
        str += symbol;                                      \
        t2.prettyPrint(str, format);                        \
        str += ")";                                         \
    }                                                       \
}

BZ_DEFINE_BOOL_OP(Greater,>,">");
BZ_DEFINE_BOOL_OP(Less,<,"<");
BZ_DEFINE_BOOL_OP(GreaterOrEqual,>=,">=");
BZ_DEFINE_BOOL_OP(LessOrEqual,<=,"<=");
BZ_DEFINE_BOOL_OP(Equal,==,"==");
BZ_DEFINE_BOOL_OP(NotEqual,!=,"!=");
BZ_DEFINE_BOOL_OP(LogicalAnd,&&,"&&");
BZ_DEFINE_BOOL_OP(LogicalOr,||,"||");

template<class T_numtype1, class T_cast>
struct Cast {
    typedef T_cast T_numtype;
    static inline T_cast apply(T_numtype1 a)
    { return a; }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += BZ_DEBUG_TEMPLATE_AS_STRING_LITERAL(T_cast);
        str += "(";
        a.prettyPrint(str, format);
        str += ")";
    }
};

template<class T_numtype1>
struct LogicalNot {
    typedef bool T_numtype;
    static inline bool apply(T_numtype1 a)
    { return !a; }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "!";
        a.prettyPrint(str, format);
    }
};

template<class T_numtype1>
struct BitwiseNot {
    typedef T_numtype1 T_numtype;
    static inline T_numtype apply(T_numtype1 a)
    { return ~a; }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "~";
        a.prettyPrint(str,format);
    }
};

template<class T_numtype1>
struct Negate {
    typedef T_numtype1 T_numtype;
    static inline T_numtype apply(T_numtype1 a)
    { return -a; }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "-";
        a.prettyPrint(str, format);
    }
};
BZ_NAMESPACE_END

#endif


