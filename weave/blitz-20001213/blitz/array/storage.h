/***************************************************************************
 * blitz/array/storage.h  Memory layout of Arrays.
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
 * Revision 1.2  2002/09/12 07:02:06  eric
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
 * Revision 1.7  2002/08/30 22:10:04  jcumming
 * Added explicit assignment operator for GeneralArrayStorage class.
 *
 * Revision 1.6  2002/06/28 01:42:24  jcumming
 * Use _bz_bool and _bz_true where appropriate to avoid int/bool conversions.
 *
 * Revision 1.5  2002/05/27 19:45:43  jcumming
 * Removed use of this->.  Types and members from templated base class are now
 * declared in scope of derived classes.
 *
 * Revision 1.4  2002/03/06 17:08:36  patricg
 *
 * in
 * template<int N_rank>
 * class FortranArray : public GeneralArrayStorage<N_rank> {} and
 * template<int N_rank>
 * class ColumnMajorArray : public GeneralArrayStorage<N_rank> {}
 * ordering_, ascendingFlag_, base_ replaced by this->ordering_,
 * this->ascendingFlag_, this->base_
 * noInitializeFlag() replaced by
 * GeneralArrayStorage<N_rank>::noInitializeFlag()
 *
 * Revision 1.3  2001/01/25 00:25:56  tveldhui
 * Ensured that source files have cvs logs.
 *
 */

#ifndef BZ_ARRAY_STORAGE_H
#define BZ_ARRAY_STORAGE_H

BZ_NAMESPACE(blitz)

/*
 * Declaration of class GeneralStorage<N_rank>
 *
 * This class describes a storage format for an N-dimensional array.
 * The dimensions can be stored in an arbitrary order (for example, as
 * a C-style row major array or Fortran-style column major array, or
 * something else entirely).  Each dimension can be stored in either
 * ascending (the most common) or descending order.  Each dimension
 * can have its own base (starting index value: e.g. 0 for C-style arrays, 
 * 1 for Fortran arrays).
 *
 * GeneralArrayStorage<N> defaults to C-style arrays.  To implement
 * other storage formats, subclass and modify the constructor.  The
 * class FortranArray, below, is an example.
 *
 * Objects inheriting from GeneralArrayStorage<N> can be passed as
 * an optional constructor argument to Array objects.
 * e.g. Array<int,3> A(16,16,16, FortranArray<3>());
 * will create a 3-dimensional 16x16x16 Fortran-style array.
 */

template<int N_rank>
class GeneralArrayStorage {
public:
    class noInitializeFlag { };

    GeneralArrayStorage(noInitializeFlag)
    { }

    GeneralArrayStorage()
    {
        for (int i=0; i < N_rank; ++i)
          ordering_(i) = N_rank - 1 - i;
        ascendingFlag_ = _bz_true;
        base_ = 0;
    }

    GeneralArrayStorage(const GeneralArrayStorage<N_rank>& x)
        : ordering_(x.ordering_), ascendingFlag_(x.ascendingFlag_),
          base_(x.base_)
    { 
    }

    GeneralArrayStorage(TinyVector<int,N_rank> ordering,
        TinyVector<_bz_bool,N_rank> ascendingFlag)
      : ordering_(ordering), ascendingFlag_(ascendingFlag)
    {
        base_ = 0;
    }

    ~GeneralArrayStorage()
    { }

    GeneralArrayStorage<N_rank>& operator=(
        const GeneralArrayStorage<N_rank>& rhs)
    {
        ordering_ = rhs.ordering();
        ascendingFlag_ = rhs.ascendingFlag();
        base_ = rhs.base();
        return *this;
    }

    TinyVector<int, N_rank>& ordering()
    { return ordering_; }

    const TinyVector<int, N_rank>& ordering() const
    { return ordering_; }

    int ordering(int i) const
    { return ordering_[i]; }

    void setOrdering(int i, int order) 
    { ordering_[i] = order; }

    _bz_bool allRanksStoredAscending() const
    {
        _bz_bool result = _bz_true;
        for (int i=0; i < N_rank; ++i)
            result &= ascendingFlag_[i];
        return result;
    }

    _bz_bool isRankStoredAscending(int i) const
    { return ascendingFlag_[i]; }

    TinyVector<_bz_bool, N_rank>& ascendingFlag() 
    { return ascendingFlag_; }

    const TinyVector<_bz_bool, N_rank>& ascendingFlag() const
    { return ascendingFlag_; }

    void setAscendingFlag(int i, _bz_bool ascendingFlag) 
    { ascendingFlag_[i] = ascendingFlag; }

    TinyVector<int, N_rank>& base()
    { return base_; }

    const TinyVector<int, N_rank>& base() const
    { return base_; }

    int base(int i) const
    { return base_[i]; }

    void setBase(int i, int base)
    { base_[i] = base; }

    void setBase(const TinyVector<int, N_rank>& base)
    { base_ = base; }

protected:
    /*
     * ordering_[] specifies the order in which the array is stored in
     * memory.  For a newly allocated array, ordering_(0) will give the
     * rank with unit stride, and ordering_(N_rank-1) will be the rank
     * with largest stride.  An order like [2, 1, 0] corresponds to
     * C-style array storage; an order like [0, 1, 2] corresponds to
     * Fortran array storage.
     *
     * ascendingFlag_[] indicates whether the data in a rank is stored
     * in ascending or descending order.  Most of the time these values
     * will all be true (indicating ascending order).  Some peculiar 
     * formats (e.g. MS-Windows BMP image format) store the data in 
     * descending order.
     *  
     * base_[] gives the first valid index for each rank.  For a C-style
     * array, all the base_ elements will be zero; for a Fortran-style
     * array, they will be one.  base_[] can be set arbitrarily using
     * the Array constructor which takes a Range argument, e.g.
     * Array<float,2> A(Range(30,40),Range(23,33));
     * will create an array with base_[] = { 30, 23 }.
     */
    TinyVector<int,  N_rank> ordering_;
    TinyVector<_bz_bool, N_rank> ascendingFlag_;
    TinyVector<int,  N_rank> base_;
};

/*
 * Class FortranArray specializes GeneralArrayStorage to provide Fortran
 * style arrays (column major ordering, base of 1).  The noInitializeFlag()
 * passed to the base constructor indicates that the subclass will take
 * care of initializing the ordering_, ascendingFlag_ and base_ members.
 */

template<int N_rank>
class FortranArray : public GeneralArrayStorage<N_rank> {
private:
    typedef GeneralArrayStorage<N_rank> T_base;
    typedef _bz_typename T_base::noInitializeFlag noInitializeFlag;
    using T_base::ordering_;
    using T_base::ascendingFlag_;
    using T_base::base_;
public:
    FortranArray()
        : GeneralArrayStorage<N_rank>(noInitializeFlag())
    {
        for (int i=0; i < N_rank; ++i)
          ordering_(i) = i;
        ascendingFlag_ = _bz_true;
        base_ = 1;
    }
};


// This tag class can be used to provide a nicer notation for
// constructing Fortran-style arrays: instead of
//     Array<int,2> A(3, 3, FortranArray<2>());
// one can simply write:
//     Array<int,2> A(3, 3, fortranArray);
// where fortranArray is an object of type _bz_fortranTag.

class _bz_fortranTag {
public:
    operator GeneralArrayStorage<1>()
    { return FortranArray<1>(); }

    operator GeneralArrayStorage<2>()
    { return FortranArray<2>(); }

    operator GeneralArrayStorage<3>()
    { return FortranArray<3>(); }

    operator GeneralArrayStorage<4>()
    { return FortranArray<4>(); }

    operator GeneralArrayStorage<5>()
    { return FortranArray<5>(); }

    operator GeneralArrayStorage<6>()
    { return FortranArray<6>(); }

    operator GeneralArrayStorage<7>()
    { return FortranArray<7>(); }

    operator GeneralArrayStorage<8>()
    { return FortranArray<8>(); }

    operator GeneralArrayStorage<9>()
    { return FortranArray<9>(); }

    operator GeneralArrayStorage<10>()
    { return FortranArray<10>(); }

    operator GeneralArrayStorage<11>()
    { return FortranArray<11>(); }
};

// A global instance of this class will be placed in
// the blitz library (libblitz.a on unix machines).

_bz_global _bz_fortranTag fortranArray;


/*
 * Class ColumnMajorArray specializes GeneralArrayStorage to provide column
 * major arrays (column major ordering, base of 0).
 */

template<int N_rank>
class ColumnMajorArray : public GeneralArrayStorage<N_rank> {
private:
    typedef GeneralArrayStorage<N_rank> T_base;
    typedef _bz_typename T_base::noInitializeFlag noInitializeFlag;
    using T_base::ordering_;
    using T_base::ascendingFlag_;
    using T_base::base_;
public:
    ColumnMajorArray()
        : GeneralArrayStorage<N_rank>(noInitializeFlag())
    {
        ordering_ = Range(0, N_rank - 1);
        ascendingFlag_ = _bz_true;
        base_ = 0;
    }
};

BZ_NAMESPACE_END

#endif // BZ_ARRAY_STORAGE_H

