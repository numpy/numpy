/***************************************************************************
 * blitz/indexexpr.h     Declaration of the IndexPlaceholder<N> class
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
 *    p://seurat.uhttwaterloo.ca/blitz/
 *
 ***************************************************************************
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
 * Revision 1.2  2001/01/24 20:22:49  tveldhui
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

