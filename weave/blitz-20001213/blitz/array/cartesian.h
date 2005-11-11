/***************************************************************************
 * blitz/array/cartesian.h  Cartesian product of indirection containers
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

#ifndef BZ_ARRAY_CARTESIAN_H
#define BZ_ARRAY_CARTESIAN_H

BZ_NAMESPACE(blitz)

/*
 * CartesianProduct<T_tuple,T_container> is an adaptor which represents
 * the cartesian product of several containers.  
 */

// forward declaration of iterator
template<class T_tuple, class T_container, int N_containers>
class CartesianProductIterator;

struct _cp_end_tag { };

template<class T_tuple, class T_container, int N_containers>
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

    const T_container& operator[](int i)
    { return *(containers_[i]); }

    void debugDump();

protected:
    const T_container* containers_[N_containers]; 
};

template<class T_tuple, class T_container, int N_containers>
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

template<class T_tuple, class T_container, int N_containers>
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

        endflag_ = _bz_false;
    }

    void operator++();

    CartesianProductIterator(_cp_end_tag)
    {
        endflag_ = _bz_true;
    }

    _bz_bool operator==(const iterator& x) const
    {
        return (endflag_ == x.endflag_);
    }

    _bz_bool operator!=(const iterator& x) const
    {   
        return endflag_ != x.endflag_;
    }

    const T_tuple& operator*() const
    { return tuple_; }

protected:
    citerator iters_[N_containers];
    citerator firstiters_[N_containers];
    citerator enditers_[N_containers];
    T_tuple tuple_;
    _bz_bool endflag_;
};

template<class T_tuple, class T_container, int N_containers>
void CartesianProductIterator<T_tuple, T_container, 
    N_containers>::operator++()
{
    // NEEDS_WORK: put in short-circuit for most common case
    // (just increment the last iterator)

    // Usual stack-style increment
    const int Nminus1 = N_containers - 1;

    int i = Nminus1;

    for (; i >= 0; --i)
    {
        ++iters_[i];
        if (iters_[i] != enditers_[i])
            break;
    }

    if (i == -1)
    {
        endflag_ = _bz_true;
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

