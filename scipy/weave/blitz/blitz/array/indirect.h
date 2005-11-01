/***************************************************************************
 * blitz/array/indirect.h  Array indirection
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
 ****************************************************************************/

#ifndef BZ_ARRAY_INDIRECT_H
#define BZ_ARRAY_INDIRECT_H

#include <blitz/array/asexpr.h>
#include <blitz/array/cartesian.h>

BZ_NAMESPACE(blitz)

template<typename T_array, typename T_index>
class IndirectArray {

public:
    IndirectArray(T_array& array, T_index& index)
        : array_(array), index_(index)
    { }

    template<typename T_expr>
    void operator=(T_expr expr);

protected:
    T_array& array_;
    T_index& index_;
};

// Forward declarations
template<typename T_array, typename T_arrayiter, typename T_subdomain, typename T_expr>
inline void applyOverSubdomain(const T_array& array, T_arrayiter& arrayIter,
    T_subdomain subdomain, T_expr expr);
template<typename T_array, typename T_arrayiter, int N_rank, typename T_expr>
inline void applyOverSubdomain(const T_array& array, T_arrayiter& arrayIter,
    RectDomain<N_rank> subdomain,
    T_expr expr);

template<typename T_array, typename T_index> template<typename T_rhs>
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

template<typename T_array, typename T_arrayiter, typename T_subdomain, typename T_expr>
inline void applyOverSubdomain(const T_array& BZ_DEBUG_PARAM(array), T_arrayiter& arrayIter, 
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
template<typename T_array, typename T_arrayiter, int N_rank, typename T_expr>
inline void applyOverSubdomain(const T_array& BZ_DEBUG_PARAM(array), T_arrayiter& arrayIter, 
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

    bool useUnitStride = arrayIter.isUnitStride(stripDim)
          && expr.isUnitStride(stripDim);

    int lbound = subdomain.lbound(stripDim); 
    int ubound = subdomain.ubound(stripDim);

    if (useUnitStride)
    {
        T_numtype* restrict data = const_cast<T_numtype*>(arrayIter.data());

        int length = ubound - lbound + 1;
        for (int i=0; i < length; ++i)
            *data++ = expr.fastRead(i);
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
template<typename T_container>
CartesianProduct<TinyVector<int,2>,T_container,2>
indexSet(const T_container& container0, const T_container& container1)
{
    return CartesianProduct<TinyVector<int,2>,T_container,2>(
        const_cast<T_container&>(container0), 
        const_cast<T_container&>(container1));
}

template<typename T_container>
CartesianProduct<TinyVector<int,3>,T_container,3>
indexSet(const T_container& container0, const T_container& container1,
    const T_container& container2)
{
    return CartesianProduct<TinyVector<int,3>,T_container,3>(
        const_cast<T_container&>(container0), 
        const_cast<T_container&>(container1), 
        const_cast<T_container&>(container2));
}

template<typename T_container>
CartesianProduct<TinyVector<int,4>,T_container,4>
indexSet(const T_container& container0, const T_container& container1,
    const T_container& container2, const T_container& container3)
{
    return CartesianProduct<TinyVector<int,4>,T_container,4>(
        const_cast<T_container&>(container0), 
        const_cast<T_container&>(container1), 
        const_cast<T_container&>(container2),
        const_cast<T_container&>(container3));
}

template<typename T_container>
CartesianProduct<TinyVector<int,5>,T_container,5>
indexSet(const T_container& container0, const T_container& container1,
    const T_container& container2, const T_container& container3,
    const T_container& container4)
{
    return CartesianProduct<TinyVector<int,5>,T_container,5>(
        const_cast<T_container&>(container0), 
        const_cast<T_container&>(container1), 
        const_cast<T_container&>(container2),
        const_cast<T_container&>(container3),
        const_cast<T_container&>(container4));
}

template<typename T_container>
CartesianProduct<TinyVector<int,6>,T_container,6>
indexSet(const T_container& container0, const T_container& container1,
    const T_container& container2, const T_container& container3,
    const T_container& container4, const T_container& container5)
{
    return CartesianProduct<TinyVector<int,6>,T_container,6>(
        const_cast<T_container&>(container0), 
        const_cast<T_container&>(container1), 
        const_cast<T_container&>(container2),
        const_cast<T_container&>(container3),
        const_cast<T_container&>(container4),
        const_cast<T_container&>(container5));
}

template<typename T_container>
CartesianProduct<TinyVector<int,7>,T_container,7>
indexSet(const T_container& container0, const T_container& container1,
    const T_container& container2, const T_container& container3,
    const T_container& container4, const T_container& container5,
    const T_container& container6)
{
    return CartesianProduct<TinyVector<int,7>,T_container,7>(
        const_cast<T_container&>(container0), 
        const_cast<T_container&>(container1), 
        const_cast<T_container&>(container2),
        const_cast<T_container&>(container3),
        const_cast<T_container&>(container4),
        const_cast<T_container&>(container5),
        const_cast<T_container&>(container6));
}

template<typename T_container>
CartesianProduct<TinyVector<int,8>,T_container,8>
indexSet(const T_container& container0, const T_container& container1,
    const T_container& container2, const T_container& container3,
    const T_container& container4, const T_container& container5,
    const T_container& container6, const T_container& container7)
{
    return CartesianProduct<TinyVector<int,8>,T_container,8>(
        const_cast<T_container&>(container0), 
        const_cast<T_container&>(container1), 
        const_cast<T_container&>(container2),
        const_cast<T_container&>(container3),
        const_cast<T_container&>(container4),
        const_cast<T_container&>(container5),
        const_cast<T_container&>(container6),
        const_cast<T_container&>(container7));
}

template<typename T_container>
CartesianProduct<TinyVector<int,9>,T_container,9>
indexSet(const T_container& container0, const T_container& container1,
    const T_container& container2, const T_container& container3,
    const T_container& container4, const T_container& container5,
    const T_container& container6, const T_container& container7,
    const T_container& container8)
{
    return CartesianProduct<TinyVector<int,9>,T_container,9>(
        const_cast<T_container&>(container0), 
        const_cast<T_container&>(container1), 
        const_cast<T_container&>(container2),
        const_cast<T_container&>(container3),
        const_cast<T_container&>(container4),
        const_cast<T_container&>(container5),
        const_cast<T_container&>(container6),
        const_cast<T_container&>(container7),
        const_cast<T_container&>(container8));
}

template<typename T_container>
CartesianProduct<TinyVector<int,10>,T_container,10>
indexSet(const T_container& container0, const T_container& container1,
    const T_container& container2, const T_container& container3,
    const T_container& container4, const T_container& container5,
    const T_container& container6, const T_container& container7,
    const T_container& container8, const T_container& container9)
{
    return CartesianProduct<TinyVector<int,10>,T_container,10>(
        const_cast<T_container&>(container0), 
        const_cast<T_container&>(container1), 
        const_cast<T_container&>(container2),
        const_cast<T_container&>(container3),
        const_cast<T_container&>(container4),
        const_cast<T_container&>(container5),
        const_cast<T_container&>(container6),
        const_cast<T_container&>(container7),
        const_cast<T_container&>(container8),
        const_cast<T_container&>(container9));
}

template<typename T_container>
CartesianProduct<TinyVector<int,11>,T_container,11>
indexSet(const T_container& container0, const T_container& container1,
    const T_container& container2, const T_container& container3,
    const T_container& container4, const T_container& container5,
    const T_container& container6, const T_container& container7,
    const T_container& container8, const T_container& container9,
    const T_container& container10)
{
    return CartesianProduct<TinyVector<int,11>,T_container,11>(
        const_cast<T_container&>(container0), 
        const_cast<T_container&>(container1), 
        const_cast<T_container&>(container2),
        const_cast<T_container&>(container3),
        const_cast<T_container&>(container4),
        const_cast<T_container&>(container5),
        const_cast<T_container&>(container6),
        const_cast<T_container&>(container7),
        const_cast<T_container&>(container8),
        const_cast<T_container&>(container9),
        const_cast<T_container&>(container10));
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

template<typename T1, typename T2, typename T3=int, typename T4=int,
         typename T5=int, typename T6=int, typename T7=int, typename T8=int,
         typename T9=int, typename T10=int, typename T11=int>
struct cp_findContainerType {
    typedef T1 T_container;
};

template<typename T2, typename T3, typename T4, typename T5, typename T6,
         typename T7, typename T8, typename T9, typename T10, typename T11>
struct cp_findContainerType<int,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11> {
    typedef _bz_typename
        cp_findContainerType<T2,T3,T4,T5,T6,T7,T8,T9,T10,T11>::T_container
        T_container;
};


// The cp_traits class handles promotion of singleton integers to
// containers.  It takes two template parameters:
//    T = argument type
//    T2 = container type
// If T is an integer, then a container of type T2 is created and the
// integer is inserted.  This container is returned.
// Otherwise, T is assumed to be the same type as T2, and the original
// container is returned.

template<typename T, typename T2>
struct cp_traits {
    typedef T T_container;

    static const T_container& make(const T& x)
    { return x; }
};

template<typename T2>
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

template<typename T1, typename T2>
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

template<typename T1, typename T2, typename T3>
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

template<typename T1, typename T2, typename T3, typename T4>
CartesianProduct<TinyVector<int,4>, _bz_typename
    cp_findContainerType<T1,T2,T3,T4>::T_container, 4>
indexSet(const T1& c1, const T2& c2, const T3& c3, const T4& c4)
{
    typedef _bz_typename cp_findContainerType<T1,T2,T3,T4>::T_container
        T_container;

    return CartesianProduct<TinyVector<int,4>, T_container, 4>(
          cp_traits<T1,T_container>::make(c1),
          cp_traits<T2,T_container>::make(c2),
          cp_traits<T3,T_container>::make(c3),
          cp_traits<T4,T_container>::make(c4));
}

template<typename T1, typename T2, typename T3, typename T4, typename T5>
CartesianProduct<TinyVector<int,5>, _bz_typename
    cp_findContainerType<T1,T2,T3,T4,T5>::T_container, 5>
indexSet(const T1& c1, const T2& c2, const T3& c3, const T4& c4, const T5& c5)
{
    typedef _bz_typename cp_findContainerType<T1,T2,T3,T4,T5>::T_container
        T_container;

    return CartesianProduct<TinyVector<int,5>, T_container, 5>(
          cp_traits<T1,T_container>::make(c1),
          cp_traits<T2,T_container>::make(c2),
          cp_traits<T3,T_container>::make(c3),
          cp_traits<T4,T_container>::make(c4),
          cp_traits<T5,T_container>::make(c5));
}

template<typename T1, typename T2, typename T3, typename T4, typename T5,
         typename T6>
CartesianProduct<TinyVector<int,6>, _bz_typename
    cp_findContainerType<T1,T2,T3,T4,T5,T6>::T_container, 6>
indexSet(const T1& c1, const T2& c2, const T3& c3, const T4& c4, const T5& c5,
    const T6& c6)
{
    typedef _bz_typename cp_findContainerType<T1,T2,T3,T4,T5,T6>::T_container
        T_container;

    return CartesianProduct<TinyVector<int,6>, T_container, 6>(
          cp_traits<T1,T_container>::make(c1),
          cp_traits<T2,T_container>::make(c2),
          cp_traits<T3,T_container>::make(c3),
          cp_traits<T4,T_container>::make(c4),
          cp_traits<T5,T_container>::make(c5),
          cp_traits<T6,T_container>::make(c6));
}

template<typename T1, typename T2, typename T3, typename T4, typename T5,
         typename T6, typename T7>
CartesianProduct<TinyVector<int,7>, _bz_typename
    cp_findContainerType<T1,T2,T3,T4,T5,T6,T7>::T_container, 7>
indexSet(const T1& c1, const T2& c2, const T3& c3, const T4& c4, const T5& c5,
    const T6& c6, const T7& c7)
{
    typedef _bz_typename
        cp_findContainerType<T1,T2,T3,T4,T5,T6,T7>::T_container
        T_container;

    return CartesianProduct<TinyVector<int,7>, T_container, 7>(
          cp_traits<T1,T_container>::make(c1),
          cp_traits<T2,T_container>::make(c2),
          cp_traits<T3,T_container>::make(c3),
          cp_traits<T4,T_container>::make(c4),
          cp_traits<T5,T_container>::make(c5),
          cp_traits<T6,T_container>::make(c6),
          cp_traits<T7,T_container>::make(c7));
}

template<typename T1, typename T2, typename T3, typename T4, typename T5,
         typename T6, typename T7, typename T8>
CartesianProduct<TinyVector<int,8>, _bz_typename
    cp_findContainerType<T1,T2,T3,T4,T5,T6,T7,T8>::T_container, 8>
indexSet(const T1& c1, const T2& c2, const T3& c3, const T4& c4, const T5& c5,
    const T6& c6, const T7& c7, const T8& c8)
{
    typedef _bz_typename
        cp_findContainerType<T1,T2,T3,T4,T5,T6,T7,T8>::T_container
        T_container;

    return CartesianProduct<TinyVector<int,8>, T_container, 8>(
          cp_traits<T1,T_container>::make(c1),
          cp_traits<T2,T_container>::make(c2),
          cp_traits<T3,T_container>::make(c3),
          cp_traits<T4,T_container>::make(c4),
          cp_traits<T5,T_container>::make(c5),
          cp_traits<T6,T_container>::make(c6),
          cp_traits<T7,T_container>::make(c7),
          cp_traits<T8,T_container>::make(c8));
}

template<typename T1, typename T2, typename T3, typename T4, typename T5,
         typename T6, typename T7, typename T8, typename T9>
CartesianProduct<TinyVector<int,9>, _bz_typename
    cp_findContainerType<T1,T2,T3,T4,T5,T6,T7,T8,T9>::T_container, 9>
indexSet(const T1& c1, const T2& c2, const T3& c3, const T4& c4, const T5& c5,
    const T6& c6, const T7& c7, const T8& c8, const T9& c9)
{
    typedef _bz_typename
        cp_findContainerType<T1,T2,T3,T4,T5,T6,T7,T8,T9>::T_container
        T_container;

    return CartesianProduct<TinyVector<int,9>, T_container, 9>(
          cp_traits<T1,T_container>::make(c1),
          cp_traits<T2,T_container>::make(c2),
          cp_traits<T3,T_container>::make(c3),
          cp_traits<T4,T_container>::make(c4),
          cp_traits<T5,T_container>::make(c5),
          cp_traits<T6,T_container>::make(c6),
          cp_traits<T7,T_container>::make(c7),
          cp_traits<T8,T_container>::make(c8),
          cp_traits<T9,T_container>::make(c9));
}

template<typename T1, typename T2, typename T3, typename T4, typename T5,
         typename T6, typename T7, typename T8, typename T9, typename T10>
CartesianProduct<TinyVector<int,10>, _bz_typename
    cp_findContainerType<T1,T2,T3,T4,T5,T6,T7,T8,T9,T10>::T_container, 10>
indexSet(const T1& c1, const T2& c2, const T3& c3, const T4& c4, const T5& c5,
    const T6& c6, const T7& c7, const T8& c8, const T9& c9, const T10& c10)
{
    typedef _bz_typename
        cp_findContainerType<T1,T2,T3,T4,T5,T6,T7,T8,T9,T10>::T_container
        T_container;

    return CartesianProduct<TinyVector<int,10>, T_container, 10>(
          cp_traits<T1,T_container>::make(c1),
          cp_traits<T2,T_container>::make(c2),
          cp_traits<T3,T_container>::make(c3),
          cp_traits<T4,T_container>::make(c4),
          cp_traits<T5,T_container>::make(c5),
          cp_traits<T6,T_container>::make(c6),
          cp_traits<T7,T_container>::make(c7),
          cp_traits<T8,T_container>::make(c8),
          cp_traits<T9,T_container>::make(c9),
          cp_traits<T10,T_container>::make(c10));
}

template<typename T1, typename T2, typename T3, typename T4, typename T5,
         typename T6, typename T7, typename T8, typename T9, typename T10,
         typename T11>
CartesianProduct<TinyVector<int,11>, _bz_typename
    cp_findContainerType<T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11>::T_container, 11>
indexSet(const T1& c1, const T2& c2, const T3& c3, const T4& c4, const T5& c5,
    const T6& c6, const T7& c7, const T8& c8, const T9& c9, const T10& c10,
    const T11& c11)
{
    typedef _bz_typename
        cp_findContainerType<T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11>::T_container
        T_container;

    return CartesianProduct<TinyVector<int,11>, T_container, 11>(
          cp_traits<T1,T_container>::make(c1),
          cp_traits<T2,T_container>::make(c2),
          cp_traits<T3,T_container>::make(c3),
          cp_traits<T4,T_container>::make(c4),
          cp_traits<T5,T_container>::make(c5),
          cp_traits<T6,T_container>::make(c6),
          cp_traits<T7,T_container>::make(c7),
          cp_traits<T8,T_container>::make(c8),
          cp_traits<T9,T_container>::make(c9),
          cp_traits<T10,T_container>::make(c10),
          cp_traits<T11,T_container>::make(c11));
}

BZ_NAMESPACE_END

#endif // BZ_ARRAY_INDIRECT_H
