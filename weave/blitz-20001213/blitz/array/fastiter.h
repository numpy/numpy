/***************************************************************************
 * blitz/array/iter.h     Declaration of FastArrayIterator<P_numtype,N_rank>
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
 * Revision 1.4  2002/06/28 01:39:47  jcumming
 * Changed order of ctor initializers to match order of member data declarations,
 * eliminating warning from gcc compiler.
 *
 * Revision 1.3  2002/03/06 17:45:07  patricg
 *
 * for BZ_HAVE_STD only
 * #include <strstream> replaced by #include <sstream>
 * ostrstream ostr replaced by ostringstream ostr
 *
 * Revision 1.2  2001/01/24 20:22:50  tveldhui
 * Updated copyright date in headers.
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
 #include <sstream>
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
#ifdef BZ_HAVE_STD
						ostringstream ostr;
#else
            ostrstream ostr;
#endif
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
