/***************************************************************************
 * blitz/array/iter.h     Declaration of FastArrayIterator<P_numtype,N_rank>
 *
 * $Id$
 *
 * Copyright (C) 1997-1999 Todd Veldhuizen <tveldhui@oonumerics.org>
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
 * Revision 1.1  2002/01/03 19:50:36  eric
 * renaming compiler to weave
 *
 * Revision 1.1  2001/04/27 17:22:04  ej
 * first attempt to include needed pieces of blitz
 *
 * Revision 1.1.1.1  2000/06/19 12:26:15  tveldhui
 * Imported sources
 *
 * Revision 1.2  1998/03/14 00:04:47  tveldhui
 * 0.2-alpha-05
 *
 * Revision 1.1  1997/07/16 14:51:20  tveldhui
 * Update: Alpha release 0.2 (Arrays)
 *
 */

#ifndef BZ_ARRAY_FASTITER_H
#define BZ_ARRAY_FASTITER_H

#ifdef BZ_HAVE_STD
 #include <strstream>
#else
 #include <strstream.h>
#endif

BZ_NAMESPACE(blitz)

#ifndef BZ_ARRAY_H
 #error <blitz/array/iter.h> must be included via <blitz/array.h>
#endif

template<class P_numtype, int N_rank>
class FastArrayIterator {
public:
    typedef P_numtype                T_numtype;
    typedef Array<T_numtype, N_rank> T_array;
    typedef FastArrayIterator<P_numtype, N_rank> T_iterator;
    typedef const T_array& T_ctorArg1;
    typedef int            T_ctorArg2;    // dummy

    enum { numArrayOperands = 1, numIndexPlaceholders = 0,
        rank = N_rank };

    // NB: this ctor does NOT preserve stack and stride
    // parameters.  This is for speed purposes.
    FastArrayIterator(const FastArrayIterator<P_numtype, N_rank>& x)
        : data_(x.data_), array_(x.array_)
    { }

    void operator=(const FastArrayIterator<P_numtype, N_rank>& x)
    {
        array_ = x.array_;
        data_ = x.data_;
        stack_ = x.stack_;
        stride_ = x.stride_;
    }

    FastArrayIterator(const T_array& array)
        : array_(array)
    {
        data_   = array.data();
    }

    ~FastArrayIterator()
    { }

#ifdef BZ_ARRAY_EXPR_PASS_INDEX_BY_VALUE
    T_numtype operator()(TinyVector<int, N_rank> i)
    { return array_(i); }
#else
    T_numtype operator()(const TinyVector<int, N_rank>& i)
    { return array_(i); }
#endif

    int ascending(int rank)
    {
        if (rank < N_rank)
            return array_.isRankStoredAscending(rank);
        else
            return INT_MIN;   // tiny(int());
    }

    int ordering(int rank)
    {
        if (rank < N_rank)
            return array_.ordering(rank);
        else
            return INT_MIN;   // tiny(int());
    }

    int lbound(int rank)
    { 
        if (rank < N_rank)
            return array_.lbound(rank); 
        else
            return INT_MIN;   // tiny(int());
    }

    int ubound(int rank)
    { 
        if (rank < N_rank)
            return array_.ubound(rank); 
        else
            return INT_MAX;   // huge(int());
    }

    T_numtype operator*()
    { return *data_; }

    T_numtype operator[](int i)
    { return data_[i * stride_]; }

    T_numtype fastRead(int i)
    { return data_[i]; }

    int suggestStride(int rank) const
    { return array_.stride(rank); }

    _bz_bool isStride(int rank, int stride) const
    { return array_.stride(rank) == stride; }

    void push(int position)
    {
        stack_[position] = data_;
    }
  
    void pop(int position)
    { 
        data_ = stack_[position];
    }

    void advance()
    {
        data_ += stride_;
    }

    void advance(int n)
    {
        data_ += n * stride_;
    }

    void loadStride(int rank)
    {
        stride_ = array_.stride(rank);
    }

    const T_numtype * _bz_restrict data() const
    { return data_; }

    void _bz_setData(const T_numtype* ptr)
    { data_ = ptr; }

    int stride() const
    { return stride_; }

    _bz_bool isUnitStride(int rank) const
    { return array_.stride(rank) == 1; }

    void advanceUnitStride()
    { ++data_; }

    _bz_bool canCollapse(int outerLoopRank, int innerLoopRank) const
    { return array_.canCollapse(outerLoopRank, innerLoopRank); }

    void prettyPrint(string& str, prettyPrintFormat& format) const
    {
        if (format.tersePrintingSelected())
            str += format.nextArrayOperandSymbol();
        else if (format.dumpArrayShapesMode())
        {
            ostrstream ostr;
            ostr << array_.shape();
            str += ostr.str();
        }
        else {
            str += "Array<";
            str += BZ_DEBUG_TEMPLATE_AS_STRING_LITERAL(T_numtype);
            str += ",";

            char tmpBuf[10];
            sprintf(tmpBuf, "%d", N_rank);

            str += tmpBuf;
            str += ">";
        }
    }

    template<class T_shape>
    _bz_bool shapeCheck(const T_shape& shape)
    { return areShapesConformable(shape, array_.length()); }


    // Experimental
    T_numtype& operator()(int i)
    {
        return (T_numtype&)data_[i*array_.stride(0)];
    }

    // Experimental
    T_numtype& operator()(int i, int j)
    {
        return (T_numtype&)data_[i*array_.stride(0) + j*array_.stride(1)];
    }

    // Experimental
    T_numtype& operator()(int i, int j, int k)
    {
        return (T_numtype&)data_[i*array_.stride(0) + j*array_.stride(1)
          + k*array_.stride(2)];
    }

    // Experimental

    void moveTo(int i, int j)
    {
        data_ = &const_cast<T_array&>(array_)(i,j);
    }

    void moveTo(int i, int j, int k)
    {
        data_ = &const_cast<T_array&>(array_)(i,j,k);
    }

    void moveTo(const TinyVector<int,N_rank>& i)
    {
        data_ = &const_cast<T_array&>(array_)(i);
    }

    // Experimental
    void operator=(T_numtype x)
    {   *const_cast<T_numtype* _bz_restrict>(data_) = x; }

    // Experimental
    template<class T_value>
    void operator=(T_value x)
    {   *const_cast<T_numtype* _bz_restrict>(data_) = x; }

    // Experimental
    template<class T_value>
    void operator+=(T_value x)
    { *const_cast<T_numtype* _bz_restrict>(data_) += x; }

    // NEEDS_WORK: other operators

    // Experimental
    operator T_numtype() const
    { return *data_; }

    // Experimental
    T_numtype shift(int offset, int dim)
    {
        return data_[offset*array_.stride(dim)];
    }

    // Experimental
    T_numtype shift(int offset1, int dim1, int offset2, int dim2)
    {
        return data_[offset1*array_.stride(dim1) 
            + offset2*array_.stride(dim2)];
    }

private:
    const T_numtype * _bz_restrict          data_;
    const T_array&                          array_;
    const T_numtype *                       stack_[N_rank];
    int                                     stride_;
};

BZ_NAMESPACE_END

#endif // BZ_ARRAY_FASTITER_H
