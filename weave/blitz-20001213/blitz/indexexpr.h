/***************************************************************************
 * blitz/indexexpr.h     Declaration of the IndexPlaceholder<N> class
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
 *    p://seurat.uhttwaterloo.ca/blitz/
 *
 ***************************************************************************
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

#ifndef BZ_INDEXEXPR_H
#define BZ_INDEXEXPR_H

#ifndef BZ_TINYVEC_H
 #include <blitz/tinyvec.h>
#endif

#ifndef BZ_PRETTYPRINT_H
 #include <blitz/prettyprint.h>
#endif

#ifndef BZ_ETBASE_H
 #include <blitz/etbase.h>
#endif

BZ_NAMESPACE(blitz)

template<int N>
class IndexPlaceholder 
#ifdef BZ_NEW_EXPRESSION_TEMPLATES
  : public ETBase<IndexPlaceholder<N> > 
#endif
{
public:
    IndexPlaceholder()
    { }

    IndexPlaceholder(const IndexPlaceholder<N>&)
    { }

    ~IndexPlaceholder()
    { }

    void operator=(const IndexPlaceholder<N>&)
    { }

    typedef int T_numtype;
    typedef int T_ctorArg1;     // Dummy; not used
    typedef int T_ctorArg2;     // Ditto

    enum { numArrayOperands = 0, numIndexPlaceholders = 1,
        rank = N+1 };

    // If you have a precondition failure on this routine, it means
    // you are trying to use stack iteration mode on an expression
    // which contains an index placeholder.  You must use index 
    // iteration mode instead.
    int operator*()
    { 
        BZPRECONDITION(0); 
        return 0;
    }

#ifdef BZ_ARRAY_EXPR_PASS_INDEX_BY_VALUE
    template<int N_rank>
    T_numtype operator()(TinyVector<int, N_rank> i)
    { return i[N]; }
#else
    template<int N_rank>
    T_numtype operator()(const TinyVector<int, N_rank>& i)
    { return i[N]; }
#endif

    int ascending(int) const
    { return INT_MIN; }

    int ordering(int) const
    { return INT_MIN; }

    int lbound(int) const
    { return INT_MIN; }   // tiny(int());

    int ubound(int) const
    { return INT_MAX; }  // huge(int()); 

    // See operator*() note
    void push(int)
    { 
        BZPRECONDITION(0); 
    }

    // See operator*() note
    void pop(int)
    { 
        BZPRECONDITION(0); 
    }

    // See operator*() note
    void advance()
    { 
        BZPRECONDITION(0); 
    }

    // See operator*() note
    void advance(int)
    { 
        BZPRECONDITION(0); 
    }

    // See operator*() note
    void loadStride(int)
    { 
        BZPRECONDITION(0); 
    }

    _bz_bool isUnitStride(int rank) const
    { 
        BZPRECONDITION(0);
        return false;
    }

    void advanceUnitStride()
    { 
        BZPRECONDITION(0);
    }

    _bz_bool canCollapse(int,int) const
    {   
        BZPRECONDITION(0); 
        return _bz_false; 
    }

    T_numtype operator[](int)
    {
        BZPRECONDITION(0);
        return T_numtype();
    }

    T_numtype fastRead(int)
    {
        BZPRECONDITION(0);
        return T_numtype();
    }

    int suggestStride(int) const
    {
        BZPRECONDITION(0);
        return 0;
    }

    _bz_bool isStride(int,int) const
    {
        BZPRECONDITION(0);
        return _bz_true;
    }

    void prettyPrint(string& str, prettyPrintFormat& format) const
    {
        // NEEDS_WORK-- do real formatting for reductions
        str += "index-expr[NEEDS_WORK]";
    }

    template<class T_shape>
    _bz_bool shapeCheck(const T_shape& shape) const
    {
        return _bz_true;
    }
};

typedef IndexPlaceholder<0> firstIndex;
typedef IndexPlaceholder<1> secondIndex;
typedef IndexPlaceholder<2> thirdIndex;
typedef IndexPlaceholder<3> fourthIndex;
typedef IndexPlaceholder<4> fifthIndex;
typedef IndexPlaceholder<5> sixthIndex;
typedef IndexPlaceholder<6> seventhIndex;
typedef IndexPlaceholder<7> eighthIndex;
typedef IndexPlaceholder<8> ninthIndex;
typedef IndexPlaceholder<9> tenthIndex;
typedef IndexPlaceholder<10> eleventhIndex;

#ifndef BZ_NO_TENSOR_INDEX_OBJECTS

BZ_NAMESPACE(tensor)
    _bz_global blitz::IndexPlaceholder<0> i;
    _bz_global blitz::IndexPlaceholder<1> j;
    _bz_global blitz::IndexPlaceholder<2> k;
    _bz_global blitz::IndexPlaceholder<3> l;
    _bz_global blitz::IndexPlaceholder<4> m;
    _bz_global blitz::IndexPlaceholder<5> n;
    _bz_global blitz::IndexPlaceholder<6> o;
    _bz_global blitz::IndexPlaceholder<7> p;
    _bz_global blitz::IndexPlaceholder<8> q;
    _bz_global blitz::IndexPlaceholder<9> r;
    _bz_global blitz::IndexPlaceholder<10> s;
    _bz_global blitz::IndexPlaceholder<11> t;
BZ_NAMESPACE_END // tensor

#endif

BZ_NAMESPACE_END

#endif // BZ_INDEXEXPR_H

