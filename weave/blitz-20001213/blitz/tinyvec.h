/***************************************************************************
 * blitz/tinyvec.h      Declaration of the TinyVector<T, N> class
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
 * Revision 1.5  2002/06/28 01:27:41  jcumming
 * Changed return type of lengthCheck() method from int to _bz_bool.
 *
 * Revision 1.4  2002/06/27 00:31:42  jcumming
 * Changed P_numtype to T_numtype inside class definition consistently.
 *
 * Revision 1.3  2002/06/26 23:51:13  jcumming
 * Explicitly specify second template argument for ListInitializationSwitch,
 * rather than relying on the default value.  This eliminates a compilation
 * problem using the xlC compiler.
 *
 * Revision 1.2  2001/01/24 20:22:50  tveldhui
 * Updated copyright date in headers.
 *
 * Revision 1.1.1.1  2000/06/19 12:26:11  tveldhui
 * Imported sources
 *
 * Revision 1.2  1998/03/14 00:04:47  tveldhui
 * 0.2-alpha-05
 *
 * Revision 1.1  1997/07/16 14:51:20  tveldhui
 * Update: Alpha release 0.2 (Arrays)
 *
 */

#ifndef BZ_TINYVEC_H
#define BZ_TINYVEC_H

#ifndef BZ_BLITZ_H
 #include <blitz/blitz.h>
#endif

#ifndef BZ_RANGE_H
 #include <blitz/range.h>
#endif

#ifndef BZ_LISTINIT_H
 #include <blitz/listinit.h>
#endif

#include <blitz/tiny.h>

BZ_NAMESPACE(blitz)

/*****************************************************************************
 * Forward declarations
 */

template<class P_numtype, int N_length, int N_stride BZ_TEMPLATE_DEFAULT(1) >
class TinyVectorIter;

template<class P_numtype, int N_length, int N_stride BZ_TEMPLATE_DEFAULT(1) >
class TinyVectorIterConst;

template<class P_numtype>
class Vector;

template<class P_expr>
class _bz_VecExpr;

template<class P_distribution>
class Random;

template<class P_numtype>
class VectorPick;

template<class T_numtype1, class T_numtype2, int N_rows, int N_columns,
    int N_vecStride>
class _bz_matrixVectorProduct;



/*****************************************************************************
 * Declaration of class TinyVector
 */

template<class P_numtype, int N_length>
class TinyVector {

public:
    //////////////////////////////////////////////
    // Public Types
    //////////////////////////////////////////////

    typedef P_numtype                                    T_numtype;
    typedef TinyVector<T_numtype,N_length>               T_vector;
    typedef TinyVectorIter<T_numtype,N_length,1>         T_iterator;
    typedef TinyVectorIterConst<T_numtype,N_length,1>    T_constIterator;
    typedef T_iterator iterator;
    typedef T_constIterator const_iterator;
    enum { numElements = N_length };

    TinyVector()
    { }

    ~TinyVector() 
    { }

    inline TinyVector(const TinyVector<T_numtype,N_length>& x);

    inline TinyVector(T_numtype initValue);

    TinyVector(T_numtype x0, T_numtype x1)
    {
        data_[0] = x0;
        data_[1] = x1;
    }

    TinyVector(T_numtype x0, T_numtype x1, T_numtype x2)
    {
        data_[0] = x0;
        data_[1] = x1;
        data_[2] = x2;
    }

    TinyVector(T_numtype x0, T_numtype x1, T_numtype x2,
        T_numtype x3)
    {
        data_[0] = x0;
        data_[1] = x1;
        data_[2] = x2;
        data_[3] = x3;
    }

    TinyVector(T_numtype x0, T_numtype x1, T_numtype x2,
        T_numtype x3, T_numtype x4)
    {
        data_[0] = x0;
        data_[1] = x1;
        data_[2] = x2;
        data_[3] = x3;
        data_[4] = x4;
    }

    TinyVector(T_numtype x0, T_numtype x1, T_numtype x2,
        T_numtype x3, T_numtype x4, T_numtype x5)
    {
        data_[0] = x0;
        data_[1] = x1;
        data_[2] = x2;
        data_[3] = x3;
        data_[4] = x4;
        data_[5] = x5;
    }

    TinyVector(T_numtype x0, T_numtype x1, T_numtype x2,
        T_numtype x3, T_numtype x4, T_numtype x5, T_numtype x6)
    {
        data_[0] = x0;
        data_[1] = x1;
        data_[2] = x2;
        data_[3] = x3;
        data_[4] = x4;
        data_[5] = x5;
        data_[6] = x6;
    }

    TinyVector(T_numtype x0, T_numtype x1, T_numtype x2,
        T_numtype x3, T_numtype x4, T_numtype x5, T_numtype x6,
        T_numtype x7)
    {
        data_[0] = x0;
        data_[1] = x1;
        data_[2] = x2;
        data_[3] = x3;
        data_[4] = x4;
        data_[5] = x5;
        data_[6] = x6;
        data_[7] = x7;
    }

    TinyVector(T_numtype x0, T_numtype x1, T_numtype x2,
        T_numtype x3, T_numtype x4, T_numtype x5, T_numtype x6,
        T_numtype x7, T_numtype x8)
    {
        data_[0] = x0;
        data_[1] = x1;
        data_[2] = x2;
        data_[3] = x3;
        data_[4] = x4;
        data_[5] = x5;
        data_[6] = x6;
        data_[7] = x7;
        data_[8] = x8;
    }

    TinyVector(T_numtype x0, T_numtype x1, T_numtype x2,
        T_numtype x3, T_numtype x4, T_numtype x5, T_numtype x6,
        T_numtype x7, T_numtype x8, T_numtype x9)
    {
        data_[0] = x0;
        data_[1] = x1;
        data_[2] = x2;
        data_[3] = x3;
        data_[4] = x4;
        data_[5] = x5;
        data_[6] = x6;
        data_[7] = x7;
        data_[8] = x8;
        data_[9] = x9;
    }

    TinyVector(T_numtype x0, T_numtype x1, T_numtype x2,
        T_numtype x3, T_numtype x4, T_numtype x5, T_numtype x6,
        T_numtype x7, T_numtype x8, T_numtype x9, T_numtype x10)
    {
        data_[0] = x0;
        data_[1] = x1;
        data_[2] = x2;
        data_[3] = x3;
        data_[4] = x4;
        data_[5] = x5;
        data_[6] = x6;
        data_[7] = x7;
        data_[8] = x8;
        data_[9] = x9;
        data_[10] = x10;
    }

    // Constructor added by Peter Nordlund
    template<class P_expr>
    inline TinyVector(_bz_VecExpr<P_expr> expr);

    T_iterator begin()
    { return T_iterator(*this); }

    T_constIterator begin() const
    { return T_constIterator(*this); }

    // T_iterator end();
    // T_constIterator end() const;

    T_numtype * _bz_restrict data()
    { return data_; }

    const T_numtype * _bz_restrict data() const
    { return data_; }

    T_numtype * _bz_restrict dataFirst()
    { return data_; }

    const T_numtype * _bz_restrict dataFirst() const
    { return data_; }

    unsigned length() const
    { return N_length; }

    /////////////////////////////////////////////
    // Library-internal member functions
    // These are undocumented and may change or
    // disappear in future releases.
    /////////////////////////////////////////////

    unsigned        _bz_suggestLength() const
    { return N_length; }

    _bz_bool        _bz_hasFastAccess() const
    { return _bz_true; }

    T_numtype& _bz_restrict     _bz_fastAccess(unsigned i)
    { return data_[i]; }

    T_numtype       _bz_fastAccess(unsigned i) const
    { return data_[i]; }

    template<class P_expr, class P_updater>
    void _bz_assign(P_expr, P_updater);

    _bz_VecExpr<T_constIterator> _bz_asVecExpr() const
    { return _bz_VecExpr<T_constIterator>(begin()); }
   
    //////////////////////////////////////////////
    // Subscripting operators
    //////////////////////////////////////////////

    _bz_bool lengthCheck(unsigned i) const
    {
        BZPRECHECK(i < N_length, 
            "TinyVector<" << BZ_DEBUG_TEMPLATE_AS_STRING_LITERAL(T_numtype) 
            << "," << N_length << "> index out of bounds: " << i);
        return _bz_true;
    }

    T_numtype operator()(unsigned i) const
    {
        BZPRECONDITION(lengthCheck(i));
        return data_[i];
    }

    T_numtype& _bz_restrict operator()(unsigned i)
    { 
        BZPRECONDITION(lengthCheck(i));
        return data_[i];
    }

    T_numtype operator[](unsigned i) const
    {
        BZPRECONDITION(lengthCheck(i));
        return data_[i];
    }

    T_numtype& _bz_restrict operator[](unsigned i)
    {
        BZPRECONDITION(lengthCheck(i));
        return data_[i];
    }

    //////////////////////////////////////////////
    // Assignment operators
    //////////////////////////////////////////////

    // Scalar operand
    ListInitializationSwitch<T_vector,T_numtype*> operator=(T_numtype x)
    {
        return ListInitializationSwitch<T_vector,T_numtype*>(*this, x);
    }

    T_vector& initialize(T_numtype);
    T_vector& operator+=(T_numtype);
    T_vector& operator-=(T_numtype);
    T_vector& operator*=(T_numtype);
    T_vector& operator/=(T_numtype);
    T_vector& operator%=(T_numtype);
    T_vector& operator^=(T_numtype);
    T_vector& operator&=(T_numtype);
    T_vector& operator|=(T_numtype);
    T_vector& operator>>=(int);
    T_vector& operator<<=(int);

    template<class P_numtype2> 
    T_vector& operator=(const TinyVector<P_numtype2, N_length> &);
    template<class P_numtype2>
    T_vector& operator+=(const TinyVector<P_numtype2, N_length> &);
    template<class P_numtype2>
    T_vector& operator-=(const TinyVector<P_numtype2, N_length> &);
    template<class P_numtype2>
    T_vector& operator*=(const TinyVector<P_numtype2, N_length> &);
    template<class P_numtype2>
    T_vector& operator/=(const TinyVector<P_numtype2, N_length> &);
    template<class P_numtype2>
    T_vector& operator%=(const TinyVector<P_numtype2, N_length> &);
    template<class P_numtype2>
    T_vector& operator^=(const TinyVector<P_numtype2, N_length> &);
    template<class P_numtype2>
    T_vector& operator&=(const TinyVector<P_numtype2, N_length> &);
    template<class P_numtype2>
    T_vector& operator|=(const TinyVector<P_numtype2, N_length> &);
    template<class P_numtype2>
    T_vector& operator>>=(const TinyVector<P_numtype2, N_length> &);
    template<class P_numtype2>
    T_vector& operator<<=(const TinyVector<P_numtype2, N_length> &);

    template<class P_numtype2> T_vector& operator=(const Vector<P_numtype2> &);
    template<class P_numtype2> T_vector& operator+=(const Vector<P_numtype2> &);
    template<class P_numtype2> T_vector& operator-=(const Vector<P_numtype2> &);
    template<class P_numtype2> T_vector& operator*=(const Vector<P_numtype2> &);
    template<class P_numtype2> T_vector& operator/=(const Vector<P_numtype2> &);
    template<class P_numtype2> T_vector& operator%=(const Vector<P_numtype2> &);
    template<class P_numtype2> T_vector& operator^=(const Vector<P_numtype2> &);
    template<class P_numtype2> T_vector& operator&=(const Vector<P_numtype2> &);
    template<class P_numtype2> T_vector& operator|=(const Vector<P_numtype2> &);
    template<class P_numtype2> T_vector& operator>>=(const Vector<P_numtype2> &);
    template<class P_numtype2> T_vector& operator<<=(const Vector<P_numtype2> &);

    // Vector expression operand
    template<class P_expr> T_vector& operator=(_bz_VecExpr<P_expr>);
    template<class P_expr> T_vector& operator+=(_bz_VecExpr<P_expr>);
    template<class P_expr> T_vector& operator-=(_bz_VecExpr<P_expr>);
    template<class P_expr> T_vector& operator*=(_bz_VecExpr<P_expr>);
    template<class P_expr> T_vector& operator/=(_bz_VecExpr<P_expr>);
    template<class P_expr> T_vector& operator%=(_bz_VecExpr<P_expr>);
    template<class P_expr> T_vector& operator^=(_bz_VecExpr<P_expr>);
    template<class P_expr> T_vector& operator&=(_bz_VecExpr<P_expr>);
    template<class P_expr> T_vector& operator|=(_bz_VecExpr<P_expr>);
    template<class P_expr> T_vector& operator>>=(_bz_VecExpr<P_expr>);
    template<class P_expr> T_vector& operator<<=(_bz_VecExpr<P_expr>);

    // VectorPick operand
    template<class P_numtype2>
    T_vector& operator=(const VectorPick<P_numtype2> &);
    template<class P_numtype2>
    T_vector& operator+=(const VectorPick<P_numtype2> &);
    template<class P_numtype2>
    T_vector& operator-=(const VectorPick<P_numtype2> &);
    template<class P_numtype2>
    T_vector& operator*=(const VectorPick<P_numtype2> &);
    template<class P_numtype2>
    T_vector& operator/=(const VectorPick<P_numtype2> &);
    template<class P_numtype2>
    T_vector& operator%=(const VectorPick<P_numtype2> &);
    template<class P_numtype2>
    T_vector& operator^=(const VectorPick<P_numtype2> &);
    template<class P_numtype2>
    T_vector& operator&=(const VectorPick<P_numtype2> &);
    template<class P_numtype2>
    T_vector& operator|=(const VectorPick<P_numtype2> &);
    template<class P_numtype2>
    T_vector& operator>>=(const VectorPick<P_numtype2> &);
    template<class P_numtype2>
    T_vector& operator<<=(const VectorPick<P_numtype2> &);

    // Range operand
    T_vector& operator=(Range);
    T_vector& operator+=(Range);
    T_vector& operator-=(Range);
    T_vector& operator*=(Range);
    T_vector& operator/=(Range);
    T_vector& operator%=(Range);
    T_vector& operator^=(Range);
    T_vector& operator&=(Range);
    T_vector& operator|=(Range);
    T_vector& operator>>=(Range);
    T_vector& operator<<=(Range);

    T_numtype* _bz_restrict getInitializationIterator()
    { return dataFirst(); }

private:
    T_numtype data_[N_length];
};


// Specialization for N = 0: KCC is giving some
// peculiar errors, perhaps this will fix.

template<class T>
class TinyVector<T,0> {
};

BZ_NAMESPACE_END

#include <blitz/tinyveciter.h>  // Iterators
#include <blitz/tvecglobs.h>    // Global functions
#include <blitz/vector.h>       // Expression templates
#include <blitz/tinyvec.cc>     // Member functions
#include <blitz/tinyvecio.cc>   // I/O functions

#endif // BZ_TINYVEC_H

