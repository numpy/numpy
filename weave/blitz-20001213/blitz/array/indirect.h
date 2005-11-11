/***************************************************************************
 * blitz/array/indirect.h  Array indirection
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
 * Revision 1.2  2001/01/25 00:25:55  tveldhui
 * Ensured that source files have cvs logs.
 *
 */

#ifndef BZ_ARRAY_INDIRECT_H
#define BZ_ARRAY_INDIRECT_H

#include <blitz/array/asexpr.h>
#include <blitz/array/cartesian.h>

BZ_NAMESPACE(blitz)

template<class T_array, class T_index>
class IndirectArray {

public:
    IndirectArray(T_array& array, T_index& index)
        : array_(array), index_(index)
    { }

    template<class T_expr>
    void operator=(T_expr expr);

protected:
    T_array& array_;
    T_index& index_;
};

// Forward declarations
template<class T_array, class T_arrayiter, class T_subdomain, class T_expr>
inline void applyOverSubdomain(const T_array& array, T_arrayiter& arrayIter,
    T_subdomain subdomain, T_expr expr);
template<class T_array, class T_arrayiter, int N_rank, class T_expr>
inline void applyOverSubdomain(const T_array& array, T_arrayiter& arrayIter,
    RectDomain<N_rank> subdomain,
    T_expr expr);

template<class T_array, class T_index> template<class T_rhs>
void IndirectArray<T_array, T_index>::operator=(T_rhs rhs)
{
    typedef _bz_typename asExpr<T_rhs>::T_expr T_expr;
    T_expr expr(rhs);

    _bz_typename T_array::T_iterator arrayIter(array_);

    _bz_typename T_index::iterator iter = index_.begin(),
                       end = index_.end();

    for (; iter != end; ++iter)
    {
        _bz_typename T_index::value_type subdomain = *iter;
        applyOverSubdomain(array_, arrayIter, subdomain, expr);
    }
}

template<class T_array, class T_arrayiter, class T_subdomain, class T_expr>
inline void applyOverSubdomain(const T_array& array, T_arrayiter& arrayIter, 
    T_subdomain subdomain, T_expr expr)
{
    BZPRECHECK(array.isInRange(subdomain),
        "In indirection using an STL container of TinyVector<int,"
        << array.rank() << ">, one of the" << endl << "positions is out of"
        " range: " << endl << subdomain << endl
        << "Array lower bounds: " << array.lbound() << endl
        << "Array upper bounds: " << array.ubound() << endl)

    arrayIter.moveTo(subdomain);
    expr.moveTo(subdomain);

    *const_cast<_bz_typename T_arrayiter::T_numtype*>(arrayIter.data()) = *expr;
}

// Specialization for RectDomain<N>
template<class T_array, class T_arrayiter, int N_rank, class T_expr>
inline void applyOverSubdomain(const T_array& array, T_arrayiter& arrayIter, 
    RectDomain<N_rank> subdomain,
    T_expr expr)
{
    typedef _bz_typename T_array::T_numtype T_numtype;

    // Assume that the RectDomain<N_rank> is a 1-D strip.
    // Find the dimension in which the strip is oriented.  This
    // variable is static so that we cache the value; likely to be
    // the same for all strips within a container.

    static int stripDim = 0;

    if (subdomain.lbound(stripDim) == subdomain.ubound(stripDim))
    {
        // Cached value was wrong, find the correct value of stripDim
        for (stripDim=0; stripDim < N_rank; ++stripDim)
          if (subdomain.lbound(stripDim) != subdomain.ubound(stripDim))
            break;

        // Handle case where the strip is just a single point
        if (stripDim == N_rank)
            stripDim = 0;
    }

#ifdef BZ_DEBUG
    // Check that this is in fact a 1D strip
    for (int i=0; i < N_rank; ++i)
      if ((i != stripDim) && (subdomain.lbound(i) != subdomain.ubound(i)))
        BZPRECHECK(0, "In indirection using an STL container of RectDomain<"
          << N_rank << ">, one of" << endl << "the RectDomain objects was not"
          " a one-dimensional strip:" << endl << "RectDomain<" << N_rank
          << ">::lbound() = " << subdomain.lbound() << endl
          << "RectDomain<" << N_rank << ">::ubound() = " << subdomain.ubound())
#endif

    // Check that the start and end position are in range
    BZPRECHECK(array.isInRange(subdomain.lbound()),
        "In indirection using an STL container of RectDomain<"
        << N_rank << ">, one of" << endl << "the RectDomain objects has a"
        " lbound which is out of range:" << endl
        << subdomain.lbound() << endl
        << "Array lower bounds: " << array.lbound() << endl
        << "Array upper bounds: " << array.ubound() << endl)

    BZPRECHECK(array.isInRange(subdomain.ubound()),
        "In indirection using an STL container of RectDomain<"
        << N_rank << ">, one of" << endl << "the RectDomain objects has a"
        " ubound which is out of range:" << endl
        << subdomain.lbound() << endl
        << "Array lower bounds: " << array.lbound() << endl
        << "Array upper bounds: " << array.ubound() << endl)

    // Position at the beginning of the strip
    arrayIter.moveTo(subdomain.lbound());
    expr.moveTo(subdomain.lbound());

    // Loop through the strip

#ifdef BZ_USE_FAST_READ_ARRAY_EXPR

    _bz_bool useUnitStride = arrayIter.isUnitStride(stripDim)
          && expr.isUnitStride(stripDim);

    int lbound = subdomain.lbound(stripDim); 
    int ubound = subdomain.ubound(stripDim);

    if (useUnitStride)
    {
        T_numtype* _bz_restrict data = const_cast<T_numtype*>(arrayIter.data());

        int length = ubound - lbound + 1;
        for (int i=0; i < length; ++i)
            data[i] = expr.fastRead(i);
    }
    else {
#endif

    arrayIter.loadStride(stripDim);
    expr.loadStride(stripDim);

    for (int i=lbound; i <= ubound; ++i)
    {
        *const_cast<_bz_typename T_arrayiter::T_numtype*>(arrayIter.data()) 
            = *expr;
        expr.advance();
        arrayIter.advance();
    }

#ifdef BZ_USE_FAST_READ_ARRAY_EXPR
    }
#endif
}

// Global functions for cartesian product of index sets
template<class T_container>
CartesianProduct<TinyVector<int,2>,T_container,2>
indexSet(const T_container& container0, const T_container& container1)
{
    return CartesianProduct<TinyVector<int,2>,T_container,2>(
        const_cast<T_container&>(container0), 
        const_cast<T_container&>(container1));
}

template<class T_container>
CartesianProduct<TinyVector<int,3>,T_container,3>
indexSet(const T_container& container0, const T_container& container1,
    const T_container& container2)
{
    return CartesianProduct<TinyVector<int,3>,T_container,3>(
        const_cast<T_container&>(container0), 
        const_cast<T_container&>(container1), 
        const_cast<T_container&>(container2));
}

// Mixture of singletons and containers, e.g. A[indexSet(I,3,K)]

// cp_findContainerType<T1,T2,T3,...,Tn>::T_container
// The set of parameters T1, T2, T3, ... Tn is a mixture of
// int and T_container.  This traits class finds the container
// type, and sets T_container.
//
// e.g. cp_findContainerType<int,int,list<int>,int>::T_container is list<int>
//      cp_findContainerType<int,deque<int>,deque<int>>::T_container 
//        is deque<int>

template<class T1, class T2, class T3=int, class T4=int>
struct cp_findContainerType {
    typedef T1 T_container;
};

template<class T2, class T3, class T4>
struct cp_findContainerType<int,T2,T3,T4> {
    typedef _bz_typename cp_findContainerType<T2,T3,T4>::T_container T_container;
};


// The cp_traits class handles promotion of singleton integers to
// containers.  It takes two template parameters:
//    T = argument type
//    T2 = container type
// If T is an integer, then a container of type T2 is created and the
// integer is inserted.  This container is returned.
// Otherwise, T is assumed to be the same type as T2, and the original
// container is returned.

template<class T, class T2>
struct cp_traits {
    typedef T T_container;

    static const T_container& make(const T& x)
    { return x; }
};

template<class T2>
struct cp_traits<int,T2> {
    typedef T2 T_container;

    static T2 make(int x)
    { 
        T2 singleton;
        singleton.push_back(x);
        return singleton;
    }
};

// These versions of indexSet() allow mixtures of integer
// and container arguments.  At least one integer must be
// specified.

template<class T1, class T2>
CartesianProduct<TinyVector<int,2>, _bz_typename 
    cp_findContainerType<T1,T2>::T_container,2> 
indexSet(const T1& c1, const T2& c2)
{
    typedef _bz_typename cp_findContainerType<T1,T2>::T_container
        T_container;

    return CartesianProduct<TinyVector<int,2>, T_container, 2>(
          cp_traits<T1,T_container>::make(c1),
          cp_traits<T2,T_container>::make(c2));
}

template<class T1, class T2, class T3>
CartesianProduct<TinyVector<int,3>, _bz_typename
    cp_findContainerType<T1,T2,T3>::T_container, 3>
indexSet(const T1& c1, const T2& c2, const T3& c3)
{
    typedef _bz_typename cp_findContainerType<T1,T2,T3>::T_container
        T_container;

    return CartesianProduct<TinyVector<int,3>, T_container, 3>(
          cp_traits<T1,T_container>::make(c1),
          cp_traits<T2,T_container>::make(c2),
          cp_traits<T3,T_container>::make(c3));
}

BZ_NAMESPACE_END

#endif // BZ_ARRAY_INDIRECT_H
