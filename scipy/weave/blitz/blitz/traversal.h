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
 ***************************************************************************/

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

