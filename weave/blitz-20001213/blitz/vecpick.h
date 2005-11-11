/***************************************************************************
 * blitz/vecpick.h      Declaration of the VectorPick<T_numtype> class
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
 * Revision 1.2  2001/01/24 20:22:50  tveldhui
 * Updated copyright date in headers.
 *
 * Revision 1.1.1.1  2000/06/19 12:26:10  tveldhui
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
 * Revision 1.2  1997/01/23 03:28:28  tveldhui
 * Periodic RCS update
 *
 * Revision 1.1  1997/01/13 22:19:58  tveldhui
 * Periodic RCS update
 *
 *
 */

#ifndef BZ_VECPICK_H
#define BZ_VECPICK_H

#ifndef BZ_VECTOR_H
 #include <blitz/vector.h>
#endif

BZ_NAMESPACE(blitz)

// Forward declarations

template<class P_numtype> class VectorPickIter;
template<class P_numtype> class VectorPickIterConst;

// Declaration of class VectorPick<P_numtype>

template<class P_numtype>
class VectorPick {

public:
    //////////////////////////////////////////////
    // Public Types
    //////////////////////////////////////////////

    typedef P_numtype                      T_numtype;
    typedef Vector<T_numtype>              T_vector;
    typedef Vector<int>                    T_indexVector;
    typedef VectorPick<T_numtype>          T_pick;
    typedef VectorPickIter<T_numtype>      T_iterator;
    typedef VectorPickIterConst<T_numtype> T_constIterator;

    //////////////////////////////////////////////
    // Constructors                             //
    //////////////////////////////////////////////

    VectorPick(T_vector& vector, T_indexVector& indexarg)
        : vector_(vector), index_(indexarg)
    { }

    VectorPick(const T_pick& vecpick)
        : vector_(const_cast<T_vector&>(vecpick.vector_)), 
          index_(const_cast<T_indexVector&>(vecpick.index_))
    { }

    VectorPick(T_pick& vecpick, Range r)
        : vector_(vecpick.vector_), index_(vecpick.index_[r])
    { }
 
    //////////////////////////////////////////////
    // Member functions
    //////////////////////////////////////////////

    T_iterator         begin()
    { return VectorPickIter<T_numtype>(*this); }

    T_constIterator    begin()      const
    { return VectorPickIterConst<T_numtype>(*this); }

    // T_vector           copy()       const;

    // T_iterator         end();

    // T_constIterator    end()        const;

    T_indexVector&     indexSet()
    { return index_; }
 
    const T_indexVector& indexSet()      const
    { return index_; }

    int           length()     const
    { return index_.length(); }

    void               setVector(Vector<T_numtype>& x)
    { vector_.reference(x); }

    void               setIndex(Vector<int>& index)
    { index_.reference(index); }

    T_vector&          vector()
    { return vector_; }

    const T_vector&    vector()     const
    { return vector_; }

    /////////////////////////////////////////////
    // Library-internal member functions
    // These are undocumented and may change or
    // disappear in future releases.
    /////////////////////////////////////////////

    int        _bz_suggestLength() const
    { return index_.length(); }

    _bz_bool        _bz_hasFastAccess() const
    { return vector_._bz_hasFastAccess() && index_._bz_hasFastAccess(); }

    T_numtype&      _bz_fastAccess(int i)
    { return vector_._bz_fastAccess(index_._bz_fastAccess(i)); }

    T_numtype       _bz_fastAccess(int i) const
    { return vector_._bz_fastAccess(index_._bz_fastAccess(i)); }

    _bz_VecExpr<T_constIterator> _bz_asVecExpr() const
    { return _bz_VecExpr<T_constIterator>(begin()); }

    //////////////////////////////////////////////
    // Subscripting operators
    //////////////////////////////////////////////

    T_numtype       operator()(int i) const
    { 
        BZPRECONDITION(index_.stride() == 1);
        BZPRECONDITION(vector_.stride() == 1);
        BZPRECONDITION(i < index_.length());
        BZPRECONDITION(index_[i] < vector_.length());
        return vector_(index_(i));
    }

    T_numtype&      operator()(int i)
    {
        BZPRECONDITION(index_.stride() == 1);
        BZPRECONDITION(vector_.stride() == 1);
        BZPRECONDITION(i < index_.length());
        BZPRECONDITION(index_[i] < vector_.length());
        return vector_(index_(i));
    }

    T_numtype       operator[](int i) const
    {
        BZPRECONDITION(index_.stride() == 1);
        BZPRECONDITION(vector_.stride() == 1);
        BZPRECONDITION(i < index_.length());
        BZPRECONDITION(index_[i] < vector_.length());
        return vector_[index_[i]];
    }

    T_numtype&      operator[](int i)
    {
        BZPRECONDITION(index_.stride() == 1);
        BZPRECONDITION(vector_.stride() == 1);
        BZPRECONDITION(i < index_.length());
        BZPRECONDITION(index_[i] < vector_.length());
        return vector_[index_[i]];
    }

    T_pick          operator()(Range r)
    {
        return T_pick(*this, index_[r]);
    }

    T_pick          operator[](Range r)
    {
        return T_pick(*this, index_[r]);
    }

    //////////////////////////////////////////////
    // Assignment operators
    //////////////////////////////////////////////

    // Scalar operand
    T_pick& operator=(T_numtype);
    T_pick& operator+=(T_numtype);
    T_pick& operator-=(T_numtype);
    T_pick& operator*=(T_numtype);
    T_pick& operator/=(T_numtype);
    T_pick& operator%=(T_numtype);
    T_pick& operator^=(T_numtype);
    T_pick& operator&=(T_numtype);
    T_pick& operator|=(T_numtype);
    T_pick& operator>>=(int);
    T_pick& operator<<=(int);

    // Vector operand
    template<class P_numtype2> T_pick& operator=(const Vector<P_numtype2> &);
    template<class P_numtype2> T_pick& operator+=(const Vector<P_numtype2> &);
    template<class P_numtype2> T_pick& operator-=(const Vector<P_numtype2> &);
    template<class P_numtype2> T_pick& operator*=(const Vector<P_numtype2> &);
    template<class P_numtype2> T_pick& operator/=(const Vector<P_numtype2> &);
    template<class P_numtype2> T_pick& operator%=(const Vector<P_numtype2> &);
    template<class P_numtype2> T_pick& operator^=(const Vector<P_numtype2> &);
    template<class P_numtype2> T_pick& operator&=(const Vector<P_numtype2> &);
    template<class P_numtype2> T_pick& operator|=(const Vector<P_numtype2> &);
    template<class P_numtype2> T_pick& operator>>=(const Vector<P_numtype2> &);
    template<class P_numtype2> T_pick& operator<<=(const Vector<P_numtype2> &);

    // Vector expression operand
    template<class P_expr> T_pick& operator=(_bz_VecExpr<P_expr>);
    template<class P_expr> T_pick& operator+=(_bz_VecExpr<P_expr>);
    template<class P_expr> T_pick& operator-=(_bz_VecExpr<P_expr>);
    template<class P_expr> T_pick& operator*=(_bz_VecExpr<P_expr>);
    template<class P_expr> T_pick& operator/=(_bz_VecExpr<P_expr>);
    template<class P_expr> T_pick& operator%=(_bz_VecExpr<P_expr>);
    template<class P_expr> T_pick& operator^=(_bz_VecExpr<P_expr>);
    template<class P_expr> T_pick& operator&=(_bz_VecExpr<P_expr>);
    template<class P_expr> T_pick& operator|=(_bz_VecExpr<P_expr>);
    template<class P_expr> T_pick& operator>>=(_bz_VecExpr<P_expr>);
    template<class P_expr> T_pick& operator<<=(_bz_VecExpr<P_expr>);

    // Range operand
    T_pick& operator=(Range);
    T_pick& operator+=(Range);
    T_pick& operator-=(Range);
    T_pick& operator*=(Range);
    T_pick& operator/=(Range);
    T_pick& operator%=(Range);
    T_pick& operator^=(Range);
    T_pick& operator&=(Range);
    T_pick& operator|=(Range);
    T_pick& operator>>=(Range);
    T_pick& operator<<=(Range);

    // Vector pick operand
    template<class P_numtype2> 
    T_pick& operator=(const VectorPick<P_numtype2> &);
    template<class P_numtype2> 
    T_pick& operator+=(const VectorPick<P_numtype2> &);
    template<class P_numtype2> 
    T_pick& operator-=(const VectorPick<P_numtype2> &);
    template<class P_numtype2> 
    T_pick& operator*=(const VectorPick<P_numtype2> &);
    template<class P_numtype2> 
    T_pick& operator/=(const VectorPick<P_numtype2> &);
    template<class P_numtype2> 
    T_pick& operator%=(const VectorPick<P_numtype2> &);
    template<class P_numtype2> 
    T_pick& operator^=(const VectorPick<P_numtype2> &);
    template<class P_numtype2> 
    T_pick& operator&=(const VectorPick<P_numtype2> &);
    template<class P_numtype2> 
    T_pick& operator|=(const VectorPick<P_numtype2> &);
    template<class P_numtype2> 
    T_pick& operator>>=(const VectorPick<P_numtype2> &);
    template<class P_numtype2> 
    T_pick& operator<<=(const VectorPick<P_numtype2> &);

    // Random operand
    template<class P_distribution>
    T_pick& operator=(Random<P_distribution>& random);
    template<class P_distribution>
    T_pick& operator+=(Random<P_distribution>& random);
    template<class P_distribution>
    T_pick& operator-=(Random<P_distribution>& random);
    template<class P_distribution>
    T_pick& operator*=(Random<P_distribution>& random);
    template<class P_distribution>
    T_pick& operator/=(Random<P_distribution>& random);
    template<class P_distribution>
    T_pick& operator%=(Random<P_distribution>& random);
    template<class P_distribution>
    T_pick& operator^=(Random<P_distribution>& random);
    template<class P_distribution>
    T_pick& operator&=(Random<P_distribution>& random);
    template<class P_distribution>
    T_pick& operator|=(Random<P_distribution>& random);

private:
    VectorPick() { }

    template<class P_expr, class P_updater>
    inline void _bz_assign(P_expr, P_updater);

private:
    T_vector vector_;
    T_indexVector index_;
};

BZ_NAMESPACE_END

#include <blitz/vecpick.cc>
#include <blitz/vecpickio.cc>
#include <blitz/vecpickiter.h>

#endif // BZ_VECPICK_H
