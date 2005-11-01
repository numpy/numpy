/***************************************************************************
 * blitz/matdiag.h      Declarations for Diagonal matrices 
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

#ifndef BZ_MATDIAG_H
#define BZ_MATDIAG_H

#ifndef BZ_MSTRUCT_H
 #error <blitz/matdiag.h> must be included via <blitz/mstruct.h>
#endif

BZ_NAMESPACE(blitz)

// Diagonal matrix
// [ 0 . . . ]
// [ . 1 . . ]
// [ . . 2 . ]
// [ . . . 3 ]

class DiagonalIterator {
public:
    DiagonalIterator(const unsigned rows,const unsigned cols) {
        BZPRECONDITION(rows==cols);
        size_ = rows;
        i_ = 0;
    }

    operator bool() const { return i_ < size_; }

    void operator++() { ++i_; }

    unsigned row()    const { return i_; }
    unsigned col()    const { return i_; }
    unsigned offset() const { return i_; }

protected:
    unsigned i_, size_;
};

class Diagonal : public MatrixStructure {
public:
    typedef DiagonalIterator T_iterator;

    Diagonal(): size_(0) { }

    Diagonal(const unsigned size): size_(size) { }

    Diagonal(const unsigned rows,const unsigned cols): size_(rows) {
        BZPRECONDITION(rows == cols);
    }

    unsigned columns() const { return size_; }

    unsigned coordToOffset(const unsigned i,const unsigned j) const
    {
        BZPRECONDITION(inRange(i,j));
        BZPRECONDITION(i == j);
        return i;
    }

    unsigned firstInRow(const unsigned i) const { return i; }

    template<typename T_numtype>
    T_numtype get(const T_numtype * restrict data,const unsigned i,const unsigned j) const
    {
        BZPRECONDITION(inRange(i,j));
        return (i==j) ? data[coordToOffset(i,j)] : ZeroElement<T_numtype>::zero();
    }

    template<typename T_numtype>
    T_numtype& get(T_numtype * restrict data,const unsigned i,const unsigned j) {
        BZPRECONDITION(inRange(i,j));
        return (i==j) ? data[coordToOffset(i,j)] : ZeroElement<T_numtype>::zero();
    }

    unsigned lastInRow(const unsigned i)  const { return i; }
    unsigned firstInCol(const unsigned j) const { return j; }
    unsigned lastInCol(const unsigned j)  const { return j; }

    bool inRange(const unsigned i,const unsigned j) const {
        return (i < size_) && (j < size_);
    }

    unsigned numElements() const { return size_; }
    unsigned rows()        const { return size_; }

    void resize(const unsigned size) { size_ = size; }

    void resize(const unsigned rows,const unsigned cols) {
        BZPRECONDITION(rows == cols);
        size_  = rows;
    }

private:
    unsigned size_;
};

BZ_NAMESPACE_END

#endif // BZ_MATSYMM_H
