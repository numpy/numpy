/***************************************************************************
 * blitz/range.h      Declaration of the Range class
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
 * Revision 1.3  2001/01/26 18:30:50  tveldhui
 * More source code reorganization to reduce compile times.
 *
 * Revision 1.2  2001/01/24 20:22:50  tveldhui
 * Updated copyright date in headers.
 *
 * Revision 1.1.1.1  2000/06/19 12:26:09  tveldhui
 * Imported sources
 *
 * Revision 1.6  1998/03/14 00:04:47  tveldhui
 * 0.2-alpha-05
 *
 * Revision 1.5  1997/07/16 14:51:20  tveldhui
 * Update: Alpha release 0.2 (Arrays)
 *
 * Revision 1.4  1997/01/24 14:42:00  tveldhui
 * Periodic RCS update
 *
 * Revision 1.3  1997/01/23 03:28:28  tveldhui
 * Periodic RCS update
 *
 * Revision 1.2  1997/01/13 22:19:58  tveldhui
 * Periodic RCS update
 *
 * Revision 1.1  1996/11/11 17:29:13  tveldhui
 * Initial revision
 *
 *
 */

#ifndef BZ_RANGE_H
#define BZ_RANGE_H

#ifndef BZ_BLITZ_H
 #include <blitz/blitz.h>
#endif

#ifndef BZ_VECEXPRWRAP_H
 #include <blitz/vecexprwrap.h>      // _bz_VecExpr wrapper
#endif

#include <blitz/wrap-climits.h>                  // for INT_MIN

BZ_NAMESPACE(blitz)

// Examples: 
// Vector<double> x(7);
// Range::all()                    [0,1,2,3,4,5,6]
// Range(3,5)                      [3,4,5]
// Range(3,Range::toEnd)           [3,4,5,6]
// Range(Range::fromStart,3)       [0,1,2,3]
// Range(1,5,2);                   [1,3,5]

enum { fromStart = INT_MIN, toEnd = INT_MIN };

// Class Range
class Range {

public:
    // This declaration not yet supported by all compilers
    // const int fromStart = INT_MIN;
    // const int toEnd = INT_MIN;

    typedef int T_numtype;

    enum { fromStart = INT_MIN, toEnd = INT_MIN };

    Range()
    {
        first_ = fromStart;
        last_ = toEnd;
        stride_ = 1;
    }

    // Range(Range r): allow default copy constructor to be used
#ifdef BZ_MANUAL_VECEXPR_COPY_CONSTRUCTOR
    Range(const Range& r)
    {
        first_ = r.first_;
        last_ = r.last_;
        stride_ = r.stride_;
    }
#endif

    _bz_explicit Range(int slicePosition)
    {
        first_ = slicePosition;
        last_ = slicePosition;
        stride_ = 1;
    }

    Range(int first, int last, int stride=1)
        : first_(first), last_(last), stride_(stride)
    { 
        BZPRECHECK((first == fromStart) || (last == toEnd) ||
                       (first < last) && (stride > 0) ||
                       (first > last) && (stride < 0) ||
                       (first == last), (*this) << " is an invalid range.");
        BZPRECHECK((last-first) % stride == 0,
            (*this) << ": the stride must evenly divide the range");
    }

    int first(int lowRange = 0) const
    { 
        if (first_ == fromStart)
            return lowRange;
        return first_; 
    }

    int last(int highRange = 0) const
    {
        if (last_ == toEnd)
            return highRange;
        return last_;
    }

    unsigned length(int recommendedLength = 0) const
    {
        BZPRECONDITION(first_ != fromStart);
        BZPRECONDITION(last_ != toEnd);
        BZPRECONDITION((last_ - first_) % stride_ == 0);
        return (last_ - first_) / stride_ + 1;
    }

    int stride() const
    { return stride_; }

    _bz_bool isAscendingContiguous() const
    {
        return (first_ < last_) && (stride_ == 1);
    }

    void setRange(int first, int last, int stride=1)
    {
        BZPRECONDITION((first < last) && (stride > 0) ||
                       (first > last) && (stride < 0) ||
                       (first == last));
        BZPRECONDITION((last-first) % stride == 0);
        first_ = first;
        last_ = last;
        stride_ = stride;
    }

    static Range all() 
    { return Range(fromStart,toEnd,1); }

    bool isUnitStride() const
    { return stride_ == 1; }

    // Operators
    Range operator-(int shift) const
    { 
        BZPRECONDITION(first_ != fromStart);
        BZPRECONDITION(last_ != toEnd);
        return Range(first_ - shift, last_ - shift, stride_); 
    }

    Range operator+(int shift) const
    { 
        BZPRECONDITION(first_ != fromStart);
        BZPRECONDITION(last_ != toEnd);
        return Range(first_ + shift, last_ + shift, stride_); 
    }

    int operator[](unsigned i) const
    {
        return first_ + i * stride_;
    }

    int operator()(unsigned i) const
    {
        return first_ + i * stride_;
    }

    friend inline ostream& operator<<(ostream& os, const Range& range)
    {
        os << "Range(" << range.first() << "," << range.last() << ","
           << range.stride() << ")";

        return os;
    }

    /////////////////////////////////////////////
    // Library-internal member functions
    // These are undocumented and may change or
    // disappear in future releases.
    /////////////////////////////////////////////

    enum { _bz_staticLengthCount = 0,
           _bz_dynamicLengthCount = 0,
           _bz_staticLength = 0 };

    _bz_bool _bz_hasFastAccess() const
    { return stride_ == 1; }

    T_numtype _bz_fastAccess(unsigned i) const
    { return first_ + i; }

    unsigned _bz_suggestLength() const
    { 
        return length();
    }

    _bz_VecExpr<Range> _bz_asVecExpr() const
    { return _bz_VecExpr<Range>(*this); }

private:
    int first_, last_, stride_;
};

BZ_NAMESPACE_END

#endif // BZ_RANGE_H
