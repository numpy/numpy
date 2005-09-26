/***************************************************************************
 * blitz/matgen.h       Declarations for RowMajor and ColumnMajor matrices
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
 * Revision 1.2  2001/01/24 20:22:49  tveldhui
 * Updated copyright date in headers.
 *
 * Revision 1.1.1.1  2000/06/19 12:26:11  tveldhui
 * Imported sources
 *
 * Revision 1.3  1998/03/14 00:04:47  tveldhui
 * 0.2-alpha-05
 *
 * Revision 1.2  1997/01/24 14:42:00  tveldhui
 * Periodic RCS update
 *
 * Revision 1.1  1997/01/13 22:19:58  tveldhui
 * Periodic RCS update
 *
 *
 */

#ifndef BZ_MATGEN_H
#define BZ_MATGEN_H

#ifndef BZ_MSTRUCT_H
 #error <blitz/matgen.h> must be included via <blitz/mstruct.h>
#endif // BZ_MSTRUCT_H

BZ_NAMESPACE(blitz)

class GeneralMatrix : public AsymmetricMatrix {

public:
    GeneralMatrix()
    { }

    GeneralMatrix(unsigned rows, unsigned cols)
        : AsymmetricMatrix(rows, cols)
    {
    }

    unsigned firstInRow(unsigned i) const
    { return 0; }

    unsigned lastInRow(unsigned i) const
    { return cols_ - 1; }

    unsigned firstInCol(unsigned j) const
    { return 0; }

    unsigned lastInCol(unsigned j) const
    { return rows_ - 1; }

    unsigned numElements() const
    { return rows_ * cols_; }
};

class GeneralIterator {
public:
    GeneralIterator(unsigned rows, unsigned cols)
    {
        rows_ = rows;
        cols_ = cols;
        i_ = 0;
        j_ = 0;
        offset_ = 0;
        good_ = true;
    }

    unsigned offset() const
    { return offset_; }

    operator _bz_bool() const
    { return good_; }
   
    unsigned row() const
    { return i_; }

    unsigned col() const
    { return j_; }
 
protected:
    unsigned rows_, cols_;
    unsigned offset_;
    unsigned i_, j_;
    _bz_bool good_;
};

class RowMajorIterator : public GeneralIterator {
public:
    RowMajorIterator(unsigned rows, unsigned cols)
        : GeneralIterator(rows, cols)
    { }

    void operator++()
    {
        ++offset_;
        ++j_;
        if (j_ == cols_)
        {
            j_ = 0;
            ++i_;
            if (i_ == rows_)
                good_ = false;
        }
    }
};

class RowMajor : public GeneralMatrix {

public:
    typedef RowMajorIterator T_iterator;

    RowMajor()
    { }

    RowMajor(unsigned rows, unsigned cols)
        : GeneralMatrix(rows, cols)
    { }

    unsigned coordToOffset(unsigned i, unsigned j) const
    {
        return i*cols_+j;
    }

    template<class T_numtype>
    T_numtype get(const T_numtype * _bz_restrict data,
        unsigned i, unsigned j) const
    {
        BZPRECONDITION(inRange(i,j));
        return data[coordToOffset(i,j)];
    }

    template<class T_numtype>
    T_numtype& get(T_numtype * _bz_restrict data, unsigned i, unsigned j)
    {
        BZPRECONDITION(inRange(i,j));
        return data[coordToOffset(i,j)];
    }
};

class ColumnMajorIterator : public GeneralIterator {
public:
    ColumnMajorIterator(unsigned rows, unsigned cols)
        : GeneralIterator(rows, cols)
    {
    }

    void operator++()
    {
        ++offset_;
        ++i_;
        if (i_ == rows_)
        {
            i_ = 0;
            ++j_;
            if (j_ == cols_)
                good_ = false;
        }
    }
};

class ColumnMajor : public GeneralMatrix {

public:
    ColumnMajor()
    { }

    ColumnMajor(unsigned rows, unsigned cols)
        : GeneralMatrix(rows, cols)
    { }

    unsigned coordToOffset(unsigned i, unsigned j) const
    {
        return j*rows_ + i;
    }

    template<class T_numtype>
    T_numtype get(const T_numtype * _bz_restrict data,
        unsigned i, unsigned j) const
    {
        BZPRECONDITION(inRange(i,j));
        return data[coordToOffset(i,j)];
    }

    template<class T_numtype>
    T_numtype& get(T_numtype * _bz_restrict data, unsigned i, unsigned j)
    {
        BZPRECONDITION(inRange(i,j));
        return data[coordToOffset(i,j)];
    }
};

BZ_NAMESPACE_END

#endif // BZ_MATGEN_H

