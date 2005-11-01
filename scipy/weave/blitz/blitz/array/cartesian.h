/***************************************************************************
 * blitz/array/cartesian.h  Cartesian product of indirection containers
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
#ifndef BZ_ARRAY_CARTESIAN_H
#define BZ_ARRAY_CARTESIAN_H

BZ_NAMESPACE(blitz)

/*
 * CartesianProduct<T_tuple,T_container> is an adaptor which represents
 * the cartesian product of several containers.  
 */

// forward declaration of iterator
template<typename T_tuple, typename T_container, int N_containers>
class CartesianProductIterator;

struct _cp_end_tag { };

template<typename T_tuple, typename T_container, int N_containers>
class CartesianProduct {
public:
    typedef T_tuple value_type;
    typedef T_tuple& reference;
    typedef const T_tuple& const_reference;
    typedef CartesianProductIterator<T_tuple,T_container,N_containers> iterator;
    typedef int difference_type;
    typedef int size_type;

    iterator begin()
    { return iterator(*this); }

    iterator end()
    { return iterator(_cp_end_tag()); }

    CartesianProduct(const T_container& container0, 
        const T_container& container1)
    { 
        BZPRECONDITION(N_containers == 2);
        containers_[0] = &container0;
        containers_[1] = &container1;
    }

    CartesianProduct(const T_container& container0, 
        const T_container& container1,
        const T_container& container2)
    { 
        BZPRECONDITION(N_containers == 3);
        containers_[0] = &container0;
        containers_[1] = &container1;
        containers_[2] = &container2;
    }

    CartesianProduct(const T_container& container0, 
        const T_container& container1,
        const T_container& container2,
        const T_container& container3)
    { 
        BZPRECONDITION(N_containers == 4);
        containers_[0] = &container0;
        containers_[1] = &container1;
        containers_[2] = &container2;
        containers_[3] = &container3;
    }

    CartesianProduct(const T_container& container0, 
        const T_container& container1,
        const T_container& container2,
        const T_container& container3,
        const T_container& container4)
    { 
        BZPRECONDITION(N_containers == 5);
        containers_[0] = &container0;
        containers_[1] = &container1;
        containers_[2] = &container2;
        containers_[3] = &container3;
        containers_[4] = &container4;
    }

    CartesianProduct(const T_container& container0, 
        const T_container& container1,
        const T_container& container2,
        const T_container& container3,
        const T_container& container4,
        const T_container& container5)
    { 
        BZPRECONDITION(N_containers == 6);
        containers_[0] = &container0;
        containers_[1] = &container1;
        containers_[2] = &container2;
        containers_[3] = &container3;
        containers_[4] = &container4;
        containers_[5] = &container5;
    }

    CartesianProduct(const T_container& container0, 
        const T_container& container1,
        const T_container& container2,
        const T_container& container3,
        const T_container& container4,
        const T_container& container5,
        const T_container& container6)
    { 
        BZPRECONDITION(N_containers == 7);
        containers_[0] = &container0;
        containers_[1] = &container1;
        containers_[2] = &container2;
        containers_[3] = &container3;
        containers_[4] = &container4;
        containers_[5] = &container5;
        containers_[6] = &container6;
    }

    CartesianProduct(const T_container& container0, 
        const T_container& container1,
        const T_container& container2,
        const T_container& container3,
        const T_container& container4,
        const T_container& container5,
        const T_container& container6,
        const T_container& container7)
    { 
        BZPRECONDITION(N_containers == 8);
        containers_[0] = &container0;
        containers_[1] = &container1;
        containers_[2] = &container2;
        containers_[3] = &container3;
        containers_[4] = &container4;
        containers_[5] = &container5;
        containers_[6] = &container6;
        containers_[7] = &container7;
    }

    CartesianProduct(const T_container& container0, 
        const T_container& container1,
        const T_container& container2,
        const T_container& container3,
        const T_container& container4,
        const T_container& container5,
        const T_container& container6,
        const T_container& container7,
        const T_container& container8)
    { 
        BZPRECONDITION(N_containers == 9);
        containers_[0] = &container0;
        containers_[1] = &container1;
        containers_[2] = &container2;
        containers_[3] = &container3;
        containers_[4] = &container4;
        containers_[5] = &container5;
        containers_[6] = &container6;
        containers_[7] = &container7;
        containers_[8] = &container8;
    }

    CartesianProduct(const T_container& container0, 
        const T_container& container1,
        const T_container& container2,
        const T_container& container3,
        const T_container& container4,
        const T_container& container5,
        const T_container& container6,
        const T_container& container7,
        const T_container& container8,
        const T_container& container9)
    { 
        BZPRECONDITION(N_containers == 10);
        containers_[0] = &container0;
        containers_[1] = &container1;
        containers_[2] = &container2;
        containers_[3] = &container3;
        containers_[4] = &container4;
        containers_[5] = &container5;
        containers_[6] = &container6;
        containers_[7] = &container7;
        containers_[8] = &container8;
        containers_[9] = &container9;
    }

    CartesianProduct(const T_container& container0, 
        const T_container& container1,
        const T_container& container2,
        const T_container& container3,
        const T_container& container4,
        const T_container& container5,
        const T_container& container6,
        const T_container& container7,
        const T_container& container8,
        const T_container& container9,
        const T_container& container10)
    { 
        BZPRECONDITION(N_containers == 11);
        containers_[0] = &container0;
        containers_[1] = &container1;
        containers_[2] = &container2;
        containers_[3] = &container3;
        containers_[4] = &container4;
        containers_[5] = &container5;
        containers_[6] = &container6;
        containers_[7] = &container7;
        containers_[8] = &container8;
        containers_[9] = &container9;
        containers_[10] = &container10;
    }

    const T_container& operator[](int i)
    { return *(containers_[i]); }

    void debugDump();

protected:
    const T_container* containers_[N_containers]; 
};

template<typename T_tuple, typename T_container, int N_containers>
void CartesianProduct<T_tuple,T_container,N_containers>::debugDump()
{
    cout << "Dump of CartesianProduct<..,..," << N_containers << ">" << endl;
    for (int i=0; i < N_containers; ++i)
    {
        cout << "Container " << (i+1) << ": ";
        _bz_typename T_container::const_iterator iter = containers_[i]->begin(),
            end = containers_[i]->end();
        for (; iter != end; ++iter)
            cout << (*iter) << '\t'; 
    }
}

template<typename T_tuple, typename T_container, int N_containers>
class CartesianProductIterator {
public:
    typedef _bz_typename T_container::const_iterator citerator;
    typedef CartesianProductIterator<T_tuple,T_container,N_containers> iterator;
    typedef CartesianProduct<T_tuple,T_container,N_containers> T_cp;

    CartesianProductIterator(T_cp& container)
    {
        for (int i=0; i < N_containers; ++i)
        {
            firstiters_[i] = container[i].begin();
            iters_[i] = firstiters_[i];
            enditers_[i] = container[i].end();
            tuple_[i] = *iters_[i];
        }

        endflag_ = false;
    }

    void operator++();

    CartesianProductIterator(_cp_end_tag)
    {
        endflag_ = true;
    }

    bool operator==(const iterator& x) const
    {
        return (endflag_ == x.endflag_);
    }

    bool operator!=(const iterator& x) const
    {   
        return endflag_ != x.endflag_;
    }

    const T_tuple& operator*() const
    { return tuple_; }

protected:
    citerator iters_[N_containers];
    citerator firstiters_[N_containers];
    citerator enditers_[N_containers];
    T_tuple   tuple_;
    bool      endflag_;
};

template<typename T_tuple, typename T_container, int N_containers>
void CartesianProductIterator<T_tuple, T_container, 
    N_containers>::operator++()
{
    // Usual stack-style increment
    const int Nminus1 = N_containers - 1;

    int i = Nminus1;

    // Short-circuit for most common case
    // (just increment the last iterator)

    if((++iters_[i]) != enditers_[i])
    {
        tuple_[i] = *iters_[i];
        return;
    }

    // Less common cases

    for (--i; i >= 0; --i)
    {
        ++iters_[i];
        if (iters_[i] != enditers_[i])
            break;
    }

    if (i == -1)
    {
        endflag_ = true;
        return;
    }

    tuple_[i] = *iters_[i];

    for (++i; i < N_containers; ++i)  
    {
        iters_[i] = firstiters_[i];
        tuple_[i] = *iters_[i];
    }
}

BZ_NAMESPACE_END

#endif // BZ_ARRAY_CARTESIAN_H

