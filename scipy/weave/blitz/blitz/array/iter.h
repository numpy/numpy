// -*- C++ -*-
/***************************************************************************
 * blitz/array/iter.h  Basic iterator for arrays.
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
#ifndef BZ_ARRAY_H
 #error <blitz/array/iter.h> must be included via <blitz/array.h>
#endif

#ifndef BZ_ARRAY_ITER_H
#define BZ_ARRAY_ITER_H

#ifdef BZ_HAVE_STL
#include <iterator>
#endif

BZ_NAMESPACE(blitz)

// helper class ConstPointerStack
template<typename P_numtype, int N_rank>
class ConstPointerStack {
public:
    typedef P_numtype                T_numtype;

    void operator=(const ConstPointerStack<P_numtype,N_rank>& rhs) 
    {
        for (int i=0; i<N_rank; ++i)
            stack_[i] = rhs.stack_[i];
    }

    const T_numtype*& operator[](int position)
    {
        return stack_[position];
    }
      
private:
    const T_numtype *                stack_[N_rank];
};


template<typename T, int N>
class ConstArrayIterator {
public:
    ConstArrayIterator() : data_(0) { }

    ConstArrayIterator(const Array<T,N>& array)
    {
        // Making internal copies of these avoids keeping
        // a pointer to the array and doing indirection.
        strides_ = array.stride();
        lbound_ = array.lbound();
        extent_ = array.extent();
        order_ = array.ordering();
        data_ = const_cast<T*>(array.dataFirst());

        maxRank_ = order_(0);
        stride_ = strides_(maxRank_);

        for (int i=0; i < N; ++i)
        {
            stack_[i] = data_;
            last_[i] = data_ + extent_(order_(i)) * strides_(order_(i));
        }

        pos_ = lbound_;
    }

    bool operator==(const ConstArrayIterator<T,N>& x) const 
    { 
        return data_ == x.data_; 
    }
    
    bool operator!=(const ConstArrayIterator<T,N>& x) const 
    { 
        return data_ != x.data_; 
    }
 
    const T& operator*() const
    {
        BZPRECHECK(data_ != 0, "Attempted to dereference invalid iterator "
             << "(empty array or past end of array)");
        return *data_;
    }

    const T* restrict operator->() const
    {
        BZPRECHECK(data_ != 0, "Attempted to dereference invalid iterator "
             << "(empty array or past end of array)");
        return data_;
    }

    ConstArrayIterator<T,N>& operator++();

    ConstArrayIterator<T,N> operator++(int)
    {
        ConstArrayIterator<T,N> tmp = *this;
        ++(*this); 
        return tmp;
    }

    // get the current position of the Array iterator in index space
    const TinyVector<int,N>& position() const
    { 
        BZPRECHECK(data_ != 0, "Array<T,N>::iterator::position() called on"
             << " invalid iterator");
        return pos_; 
    }
   
private:
    TinyVector<int,N> strides_, lbound_, extent_, order_;
    ConstPointerStack<T,N> stack_;
    ConstPointerStack<T,N> last_;
    int stride_;
    int maxRank_;

protected:
    TinyVector<int,N> pos_;
    T * restrict data_;
};


template<typename T, int N>
class ArrayIterator : public ConstArrayIterator<T,N> {
private:
    typedef ConstArrayIterator<T,N> T_base;
    using T_base::data_;

public:
    ArrayIterator() { }

    ArrayIterator(Array<T,N>& x) : T_base(x) { }

    T& operator*() const
    {
        BZPRECHECK(data_ != 0, "Attempted to dereference invalid iterator "
             << "(empty array or past end of array)");
        return *data_;
    }

    T* restrict operator->() const
    {
        BZPRECHECK(data_ != 0, "Attempted to dereference invalid iterator "
             << "(empty array or past end of array)");
        return data_;
    }

    ArrayIterator<T,N>& operator++()
    {
        T_base::operator++();
        return *this;
    }

    ArrayIterator<T,N> operator++(int)
    {
        ArrayIterator<T,N> tmp = *this;
        ++(*this); 
        return tmp;
    }
};


template<typename T, int N>
ConstArrayIterator<T,N>& ConstArrayIterator<T,N>::operator++()
{
    BZPRECHECK(data_ != 0, "Attempted to iterate past the end of an array.");

    data_ += stride_;

    if (data_ != last_[0])
    {
        // We hit this case almost all the time.
        ++pos_[maxRank_];
        return *this;
    }

    // We've hit the end of a row/column/whatever.  Need to
    // increment one of the loops over another dimension.

    int j = 1;
    for (; j < N; ++j)
    {
        int r = order_(j);
        data_ = const_cast<T*>(stack_[j]);
        data_ += strides_[r];
        ++pos_(r);

        if (data_ != last_[j])
            break;
    }

    // All done?
    if (j == N)
    {
        // Setting data_ to 0 indicates the end of the array has
        // been reached, and will match the end iterator.
        data_ = 0;
        return *this;
    }

    stack_[j] = data_;

    // Now reset all the last pointers
    for (--j; j >= 0; --j)
    {
        int r2 = order_(j);
        stack_[j] = data_;
        last_[j] = data_ + extent_(r2) * strides_(r2);
        pos_(r2) = lbound_(r2);
    }

    return *this;
}


BZ_NAMESPACE_END


#ifdef BZ_HAVE_STL
// support for std::iterator_traits
BZ_NAMESPACE(std)

template <typename T, int N>
struct iterator_traits< BZ_BLITZ_SCOPE(ConstArrayIterator)<T,N> > 
{
    typedef forward_iterator_tag               iterator_category;
    typedef T                                  value_type;
    typedef ptrdiff_t                          difference_type;
    typedef const T*                           pointer;
    typedef const T&                           reference;
};

template <typename T, int N>
struct iterator_traits< BZ_BLITZ_SCOPE(ArrayIterator)<T,N> > 
{
    typedef forward_iterator_tag               iterator_category;
    typedef T                                  value_type;
    typedef ptrdiff_t                          difference_type;
    typedef T*                                 pointer;
    typedef T&                                 reference;
};

BZ_NAMESPACE_END

#endif // BZ_HAVE_STL

#endif // BZ_ARRAY_ITER_H

