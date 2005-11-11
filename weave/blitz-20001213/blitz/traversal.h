/***************************************************************************
 * blitz/traversal.h      Declaration of the TraversalOrder classes
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
 * Revision 1.4  2002/03/07 08:38:20  patricg
 *
 * moved
 * template<int N_dimensions>
 * _bz_typename TraversalOrderCollection<N_dimensions>::T_set
 *     TraversalOrderCollection<N_dimensions>::traversals_;
 * after the declaration of
 * template<int N_dimensions> class TraversalOrderCollection
 *
 * Revision 1.3  2002/03/06 17:18:11  patricg
 *
 * template declaration
 * template<int N_dimensions>
 * _bz_typename TraversalOrderCollection<N_dimensions>::T_set
 * 	TraversalOrderCollection<N_dimensions>::traversals_;
 * in blitz/transversal.cc moved before template specialisation
 * template<>
 * class TraversalOrderCollection<0> {}
 * in blitz/transversal.h
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

// Fast traversal orders require the ISO/ANSI C++ standard library
// (particularly set).
#ifdef BZ_HAVE_STD

#ifndef BZ_TRAVERSAL_H
#define BZ_TRAVERSAL_H

#ifndef BZ_TINYVEC_H
 #include <blitz/tinyvec.h>
#endif

#ifndef BZ_VECTOR_H
 #include <blitz/vector.h>
#endif

#include <set>

BZ_NAMESPACE(blitz)

template<int N_dimensions>
class TraversalOrder {

public:
    typedef TinyVector<int, N_dimensions> T_coord;
    typedef Vector<T_coord>               T_traversal;

    TraversalOrder()
    {
        size_ = 0;
    }

    TraversalOrder(const T_coord& size, T_traversal& order)
        : size_(size), order_(order)
    { }

    TraversalOrder(const T_coord& size)
        : size_(size)
    { }

    T_coord operator[](int i) const
    { return order_[i]; }

    T_coord& operator[](int i)
    { return order_[i]; }

    int length() const
    { return order_.length(); }

    bool operator<(const TraversalOrder<N_dimensions>& x) const
    {
        for (int i=0; i < N_dimensions; ++i)
        {
            if (size_[i] < x.size_[i])
                return true;
            else if (size_[i] > x.size_[i])
                return false;
        }
        return false;
    }

    bool operator==(const TraversalOrder<N_dimensions>& x) const
    {
        for (int i=0; i < N_dimensions; ++i)
        {
            if (size_[i] != x.size_[i])
                return false;
        }

        return true;
    }

protected:
    T_traversal order_;
    T_coord     size_;
};

/*
 * This specialization is provided to avoid problems with zero-length
 * vectors.
 */
template<>
class TraversalOrder<0> {
public:
     TraversalOrder () {} // AJS
};

template<int N_dimensions>
class TraversalOrderCollection {
public:
    typedef TraversalOrder<N_dimensions>        T_traversal;
    typedef _bz_typename T_traversal::T_coord   T_coord;
    typedef set<T_traversal>                    T_set;
    typedef _bz_typename set<T_traversal>::const_iterator T_iterator;

    const T_traversal* find(const T_coord& size)
    {
        T_iterator iter = traversals_.find(T_traversal(size));
        if (iter != traversals_.end())
            return &(*iter);
        return 0;
    }

    void insert(T_traversal x)
    {
        traversals_.insert(x);
    }

protected:
    static T_set traversals_;
};

template<int N_dimensions>
_bz_typename TraversalOrderCollection<N_dimensions>::T_set
    TraversalOrderCollection<N_dimensions>::traversals_;

/*
 * This specialization is provided to avoid problems with zero-length
 * vectors.
 */

template<>
class TraversalOrderCollection<0> {
public:
    typedef int T_traversal;
    typedef int T_coord;
    typedef int T_set;
    typedef int T_iterator;

    const T_traversal* find(const T_coord& size)
    { return 0; }
};

BZ_NAMESPACE_END

#include <blitz/traversal.cc>

#endif // BZ_TRAVERSAL_H

#endif // BZ_HAVE_STD

