/***************************************************************************
 * blitz/vector.h      Declaration of the Vector<P_numtype> class
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
 * Revision 1.6  2002/05/27 19:30:27  jcumming
 * Removed use of this-> as means of accessing members of templated base class.
 * Instead provided using declarations for these members within the derived
 * class definitions to bring them into the scope of the derived class.
 *
 * Revision 1.5  2002/03/06 16:09:46  patricg
 * *** empty log message ***
 *
 * Revision 1.4  2001/01/26 18:30:50  tveldhui
 * More source code reorganization to reduce compile times.
 *
 * Revision 1.3  2001/01/24 22:51:50  tveldhui
 * Reorganized #include orders to avoid including the huge Vector e.t.
 * implementation when using Array.
 *
 * Revision 1.2  2001/01/24 20:22:50  tveldhui
 * Updated copyright date in headers.
 *
 * Revision 1.1.1.1  2000/06/19 12:26:10  tveldhui
 * Imported sources
 *
 * Revision 1.8  1998/03/14 00:04:47  tveldhui
 * 0.2-alpha-05
 *
 * Revision 1.7  1997/07/16 14:51:20  tveldhui
 * Update: Alpha release 0.2 (Arrays)
 *
 * Revision 1.6  1997/01/24 14:42:00  tveldhui
 * Periodic RCS update
 *
 * Revision 1.5  1997/01/23 03:28:28  tveldhui
 * Periodic RCS update
 *
 * Revision 1.4  1997/01/13 22:19:58  tveldhui
 * Periodic RCS update
 *
 * Revision 1.3  1996/11/11 17:29:13  tveldhui
 * Periodic RCS update
 *
 * Revision 1.2  1996/10/31 21:06:54  tveldhui
 * Did away with multiple template parameters.
 */

/*
 * KNOWN BUGS
 *
 * 1. operator[](Vector<int>) won't match; compiler complains of no
 *       suitable copy constructor for VectorPick<T>
 * 2. Vector<T>(_bz_VecExpr<E>) constructor generates warning
 * 3. operator+=,-=,..etc.(Random<D>) won't match; compiler complains of
 *       no suitable copy constructor for _bz_VecExpr(...).
 */

#ifndef BZ_VECTOR_H
#define BZ_VECTOR_H

#ifndef BZ_BLITZ_H
 #include <blitz/blitz.h>
#endif

#ifndef BZ_MEMBLOCK_H
 #include <blitz/memblock.h>
#endif

#ifndef BZ_RANGE_H
 #include <blitz/range.h>
#endif

#ifndef BZ_LISTINIT_H
 #include <blitz/listinit.h>
#endif

BZ_NAMESPACE(blitz)

// Forward declarations
template<class P_numtype> class VectorIter;
template<class P_numtype> class VectorIterConst;
template<class P_expr>    class _bz_VecExpr;       
template<class P_numtype> class VectorPick;
template<class P_numtype> class Random;

// Declaration of class Vector<P_numtype>

template<class P_numtype>
class Vector : protected MemoryBlockReference<P_numtype> {
  
private:
    typedef MemoryBlockReference<P_numtype> T_base;
    using T_base::data_;

public:
    //////////////////////////////////////////////
    // Public Types
    //////////////////////////////////////////////

    typedef P_numtype                  T_numtype;
    typedef Vector<T_numtype>          T_vector;
    typedef VectorIter<T_numtype>      T_iterator;
    typedef VectorIterConst<T_numtype> T_constIterator;
    typedef VectorPick<T_numtype>      T_pick;
    typedef Vector<int>                T_indexVector;

    //////////////////////////////////////////////
    // Constructors                             //
    //////////////////////////////////////////////
    // Most of the constructors are inlined so that
    // the setting of the stride_ data member will
    // be visible to the optimizer.

    Vector()
    { 
        length_ = 0;
        stride_ = 0;
    }

    // This constructor is provided inline because it involves
    // no memory allocation.
    Vector(Vector<T_numtype>& vec)
        : MemoryBlockReference<T_numtype>(vec)
    {
        length_ = vec.length_;
        stride_ = vec.stride_;
    }

    // Slightly unsafe cast-away-const version
    Vector(const Vector<T_numtype>& vec)
        : MemoryBlockReference<T_numtype>
           (const_cast<Vector<T_numtype>& >(vec))
    {
        length_ = vec.length_;
        stride_ = vec.stride_;
    }

    _bz_explicit Vector(int length)
        : MemoryBlockReference<T_numtype>(length)
    {
        length_ = length;
        stride_ = 1;
    }

    Vector(Vector<T_numtype>& vec, Range r)
        : MemoryBlockReference<T_numtype>(vec, 
            r.first() * vec.stride())
    {
        BZPRECONDITION((r.first() >= 0) && (r.first() < vec.length()));
        BZPRECONDITION((r.last(vec.length()-1) >= 0) 
            && (r.last(vec.length()-1) < vec.length()));
        length_ = (r.last(vec.length()-1) - r.first()) / r.stride() + 1;
        stride_ = vec.stride_ * r.stride();
    }

    Vector(int length, T_numtype initValue)
        : MemoryBlockReference<T_numtype>(length)
    {
        length_ = length;
        stride_ = 1;
        (*this) = initValue;
    }

    Vector(int length, T_numtype firstValue, T_numtype delta)
        : MemoryBlockReference<T_numtype>(length)
    {
        length_ = length;
        stride_ = 1;
        for (int i=0; i < length; ++i)
            data_[i] = firstValue + i * delta;
    }

    template<class P_distribution>
    Vector(int length, Random<P_distribution>& random)
        : MemoryBlockReference<T_numtype>(length)
    {
        length_ = length;
        stride_ = 1;
        (*this) = random;
    }

    template<class P_expr>
    Vector(_bz_VecExpr<P_expr> expr)
        : MemoryBlockReference<T_numtype>(expr._bz_suggestLength())
    {
        length_ = expr._bz_suggestLength();
        stride_ = 1;
        (*this) = expr;
    }

    // Create a vector view of an already allocated block of memory.
    // Note that the memory will not be freed when this vector is
    // destroyed.
    Vector(int length, T_numtype* _bz_restrict data, int stride = 1)
        : MemoryBlockReference<T_numtype>(length, data, neverDeleteData)
    {
        length_ = length;
        stride_ = stride;
    }

    // Create a vector containing a range of numbers
    Vector(Range r)
        : MemoryBlockReference<T_numtype>(r._bz_suggestLength())
    {
        length_ = r._bz_suggestLength();
        stride_ = 1;
        (*this) = _bz_VecExpr<Range>(r);
    }
    
    //////////////////////////////////////////////
    // Member functions
    //////////////////////////////////////////////

    // assertUnitStride() is provided as an optimizing trick.  When
    // vectors are constructed outside the function scope, the optimizer
    // is unaware that they have unit stride.  This function sets the
    // stride to 1 in the local scope so the optimizer can do copy
    // propagation & dead code elimination.  Obviously, you don't
    // want to use this routine unless you are certain that the
    // vectors have unit stride.
    void            assertUnitStride()
    {
        BZPRECONDITION(stride_ == 1);
        stride_ = 1;
    }

    T_iterator      begin()
    { return T_iterator(*this); }

    T_constIterator begin()  const
    { return T_constIterator(*this); }

    T_vector        copy()   const;

    // T_iterator      end();
    // T_constIterator end()    const;

    T_numtype * _bz_restrict data()  
    { return data_; }

    const T_numtype * _bz_restrict data() const
    { return data_; }

    _bz_bool        isUnitStride() const
    { return stride_ == 1; }

    int        length() const
    { return length_; }

    void            makeUnique();

    // int        storageSize() const;

    // void            storeToBuffer(void* buffer, int bufferLength) const;

    void            reference(T_vector&);

    void            resize(int length);

    void            resizeAndPreserve(int newLength);

    // int             restoreFromBuffer(void* buffer, int bufferLength);

    T_vector        reverse()
    { return T_vector(*this,Range(length()-1,0,-1)); }

    int             stride() const
    { return stride_; }

    operator _bz_VecExpr<VectorIterConst<T_numtype> > () const
    { return _bz_VecExpr<VectorIterConst<T_numtype> >(begin()); }

    /////////////////////////////////////////////
    // Library-internal member functions
    // These are undocumented and may change or
    // disappear in future releases.
    /////////////////////////////////////////////

    int        _bz_suggestLength() const
    { return length_; }

    _bz_bool        _bz_hasFastAccess() const
    { return stride_ == 1; }

    T_numtype&      _bz_fastAccess(int i)
    { return data_[i]; }

    T_numtype       _bz_fastAccess(int i) const
    { return data_[i]; }

    template<class P_expr, class P_updater>
    void            _bz_assign(P_expr, P_updater);

    _bz_VecExpr<T_constIterator> _bz_asVecExpr() const
    { return _bz_VecExpr<T_constIterator>(begin()); }

    //////////////////////////////////////////////
    // Subscripting operators
    //////////////////////////////////////////////

    // operator()(int) may be used only when the vector has unit
    // stride.  Otherwise, use operator[].
    T_numtype        operator()(int i) const
    {
        BZPRECONDITION(i < length_);
        BZPRECONDITION(stride_ == 1);
        return data_[i];
    }

    // operator()(int) may be used only when the vector has unit
    // stride.  Otherwise, use operator[].
    T_numtype& _bz_restrict operator()(int i) 
    {
        BZPRECONDITION(i < length_);
        BZPRECONDITION(stride_ == 1);
        return data_[i];
    }

    T_numtype        operator[](int i) const
    {
        BZPRECONDITION(i < length_);
        return data_[i * stride_];
    }

    T_numtype& _bz_restrict operator[](int i)
    {
        BZPRECONDITION(i < length_);
        return data_[i * stride_];
    }

    T_vector      operator()(Range r)
    {
        return T_vector(*this, r);
    }

    T_vector      operator[](Range r)
    {
        return T_vector(*this, r);
    }

    T_pick        operator()(T_indexVector i)
    {
        return T_pick(*this, i);
    }

    T_pick        operator[](T_indexVector i)
    {
        return T_pick(*this, i);
    }

    // T_vector      operator()(difference-equation-expression)

    //////////////////////////////////////////////
    // Assignment operators
    //////////////////////////////////////////////

    // Scalar operand
    ListInitializationSwitch<T_vector,T_iterator> operator=(T_numtype x)
    {
        return ListInitializationSwitch<T_vector,T_iterator>(*this, x);
    }

    T_iterator getInitializationIterator()
    { return begin(); }

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

    // Vector operand
   
    template<class P_numtype2> T_vector& operator=(const Vector<P_numtype2> &);

    // Specialization uses memcpy instead of element-by-element cast and
    // copy
    // NEEDS_WORK -- KCC won't accept this syntax; standard??
    // template<> T_vector& operator=(const T_vector&);

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

    // Random operand
    template<class P_distribution>
    T_vector& operator=(Random<P_distribution>& random);
    template<class P_distribution>
    T_vector& operator+=(Random<P_distribution>& random);
    template<class P_distribution>
    T_vector& operator-=(Random<P_distribution>& random);
    template<class P_distribution>
    T_vector& operator*=(Random<P_distribution>& random);
    template<class P_distribution>
    T_vector& operator/=(Random<P_distribution>& random);
    template<class P_distribution>
    T_vector& operator%=(Random<P_distribution>& random);
    template<class P_distribution>
    T_vector& operator^=(Random<P_distribution>& random);
    template<class P_distribution>
    T_vector& operator&=(Random<P_distribution>& random);
    template<class P_distribution>
    T_vector& operator|=(Random<P_distribution>& random);

    //////////////////////////////////////////////
    // Unary operators
    //////////////////////////////////////////////

//    T_vector& operator++();
//    void operator++(int);
//    T_vector& operator--();
//    void operator--(int);
    
private:
    int      length_;
    int      stride_;
};

// Global I/O functions

template<class P_numtype>
ostream& operator<<(ostream& os, const Vector<P_numtype>& x);

template<class P_expr>
ostream& operator<<(ostream& os, _bz_VecExpr<P_expr> expr);

BZ_NAMESPACE_END

#include <blitz/veciter.h>          // Iterators
#include <blitz/vecpick.h>          // VectorPick
#include <blitz/vecexpr.h>          // Expression template classes
#include <blitz/vecglobs.h>         // Global functions
#include <blitz/vector.cc>          // Member functions
#include <blitz/vecio.cc>           // IO functions

#endif // BZ_VECTOR_H
