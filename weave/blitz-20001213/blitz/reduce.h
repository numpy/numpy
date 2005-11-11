/***************************************************************************
 * blitz/reduce.h        Reduction operators: sum, mean, min, max,
 *                       minIndex, maxIndex, product, count, any, all
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
 * Revision 1.2  2001/01/24 20:22:50  tveldhui
 * Updated copyright date in headers.
 *
 * Revision 1.1.1.1  2000/06/19 12:26:12  tveldhui
 * Imported sources
 *
 * Revision 1.2  1998/03/14 00:04:47  tveldhui
 * 0.2-alpha-05
 *
 * Revision 1.1  1997/07/16 14:51:20  tveldhui
 * Update: Alpha release 0.2 (Arrays)
 *
 */

#ifndef BZ_REDUCE_H
#define BZ_REDUCE_H

#ifndef BZ_BLITZ_H
 #include <blitz/blitz.h>
#endif

#ifndef BZ_NUMTRAIT_H
 #include <blitz/numtrait.h>
#endif

#ifndef BZ_NUMINQUIRE_H
 #include <blitz/numinquire.h>
#endif

BZ_NAMESPACE(blitz)

template<class P_sourcetype, class P_resulttype = BZ_SUMTYPE(P_sourcetype)>
class ReduceSum {

public:
    typedef P_sourcetype T_sourcetype;
    typedef P_resulttype T_resulttype;
    typedef T_resulttype T_numtype;

    enum { needIndex = 0, canProvideInitialValue = 1 };

    ReduceSum()
    { reset(); }

    ReduceSum(T_resulttype initialValue)
    { sum_ = initialValue; }

    bool operator()(T_sourcetype x)
    { 
        sum_ += x; 
        return true;
    }

    bool operator()(T_sourcetype x, int)
    { 
        sum_ += x; 
        return true;
    }

    T_resulttype result(int)
    { return sum_; }

    void reset()
    { sum_ = zero(T_resulttype()); }

    void reset(T_resulttype initialValue)
    { sum_ = initialValue; }
 
    static const char* name()
    { return "sum"; }
 
protected:
    T_resulttype sum_;
};

template<class P_sourcetype, class P_resulttype = BZ_FLOATTYPE(P_sourcetype)>
class ReduceMean {

public:
    typedef P_sourcetype T_sourcetype;
    typedef P_resulttype T_resulttype;
    typedef T_resulttype T_numtype;

    enum { needIndex = 0, canProvideInitialValue = 0 };

    ReduceMean()
    { reset(); }

    ReduceMean(T_resulttype)
    { 
        BZPRECHECK(0, "Provided an initial value for ReduceMean");
        reset();
    }

    bool operator()(T_sourcetype x)
    { 
        sum_ += x; 
        return true;
    }

    bool operator()(T_sourcetype x, int)
    { 
        sum_ += x; 
        return true;
    }

    T_resulttype result(int count)
    { return sum_ / count; }

    void reset()
    { sum_ = zero(T_resulttype()); }

    void reset(T_resulttype)
    { 
        BZPRECHECK(0, "Provided an initial value for ReduceMean");
        reset();
    }

    static const char* name() 
    { return "mean"; }

protected:
    T_resulttype sum_;
};

template<class P_sourcetype>
class ReduceMin {

public:
    typedef P_sourcetype T_sourcetype;
    typedef P_sourcetype T_resulttype;
    typedef T_resulttype T_numtype;

    enum { needIndex = 0, canProvideInitialValue = 1 };

    ReduceMin()
    { reset(); }

    ReduceMin(T_resulttype min)
    {
        min_ = min;
    }

    bool operator()(T_sourcetype x)
    { 
        if (x < min_)
            min_ = x;
        return true;
    }

    bool operator()(T_sourcetype x, int)
    {
        if (x < min_)
            min_ = x;
        return true;
    }

    T_resulttype result(int)
    { return min_; }

    void reset()
    { min_ = huge(P_sourcetype()); }

    void reset(T_resulttype initialValue)
    { min_ = initialValue; }

    static const char* name()
    { return "min"; }

protected:
    T_resulttype min_;
};

template<class P_sourcetype>
class ReduceMax {

public:
    typedef P_sourcetype T_sourcetype;
    typedef P_sourcetype T_resulttype;
    typedef T_resulttype T_numtype;

    enum { needIndex = 0, canProvideInitialValue = 1 };

    ReduceMax()
    { reset(); }

    ReduceMax(T_resulttype max)
    {
        max_ = max;
    }

    bool operator()(T_sourcetype x)
    {
        if (x > max_)
            max_ = x;
        return true;
    }

    bool operator()(T_sourcetype x, int)
    {
        if (x > max_)
            max_ = x;
        return true;
    }

    T_resulttype result(int)
    { return max_; }

    void reset()
    { max_ = neghuge(P_sourcetype()); }

    void reset(T_resulttype initialValue)
    { max_ = initialValue; }

    static const char* name()
    { return "max"; }

protected:
    T_resulttype max_;
};

template<class P_sourcetype>
class ReduceMinIndex {

public:
    typedef P_sourcetype T_sourcetype;
    typedef int          T_resulttype;
    typedef T_resulttype T_numtype;

    enum { needIndex = 1, canProvideInitialValue = 0 };

    ReduceMinIndex()
    { reset(); }

    ReduceMinIndex(T_resulttype min)
    {
        reset(min);
    }

    bool operator()(T_sourcetype x)
    {
        BZPRECONDITION(0);
        return false;
    }

    bool operator()(T_sourcetype x, int index)
    {
        if (x < min_)
        {
            min_ = x;
            index_ = index;
        }
        return true;
    }

    T_resulttype result(int)
    { return index_; }

    void reset()
    { 
        min_ = huge(T_sourcetype());
        index_ = tiny(int());        
    }

    void reset(T_resulttype)
    { 
        BZPRECHECK(0, "Provided initial value for ReduceMinIndex");
        reset();
    }

    static const char* name()
    { return "minIndex"; }

protected:
    T_sourcetype min_;
    int index_;
};

template<class P_sourcetype, int N>
class ReduceMinIndexVector {

public:
    typedef P_sourcetype T_sourcetype;
    typedef TinyVector<int,N> T_resulttype;
    typedef T_resulttype T_numtype;

    enum { canProvideInitialValue = 0 };

    ReduceMinIndexVector()
    { reset(); }

    ReduceMinIndexVector(T_resulttype min)
    {
        reset(min);
    }

    bool operator()(T_sourcetype x)
    {
        BZPRECONDITION(0);
        return false;
    }

    bool operator()(T_sourcetype, int)
    {
        BZPRECONDITION(0);
        return false;
    }
   
    bool operator()(T_sourcetype x, const TinyVector<int,N>& index)
    {
        if (x < min_)
        {
            min_ = x;
            index_ = index;
        }
        return true;
    }

    T_resulttype result(int)
    { return index_; }

    void reset()
    {
        min_ = huge(T_sourcetype());
        index_ = tiny(int());
    }

    void reset(T_resulttype)
    {
        BZPRECHECK(0, "Provided initial value for ReduceMinIndex");
        reset();
    }

    static const char* name()
    { return "minIndex"; }

protected:
    T_sourcetype min_;
    TinyVector<int,N> index_;
};

template<class P_sourcetype>
class ReduceMaxIndex {

public:
    typedef P_sourcetype T_sourcetype;
    typedef int          T_resulttype;
    typedef T_resulttype T_numtype;

    enum { needIndex = 1, canProvideInitialValue = 0 };

    ReduceMaxIndex()
    { reset(); }

    ReduceMaxIndex(T_resulttype max)
    {
        reset(max);
    }

    bool operator()(T_sourcetype x)
    {
        BZPRECONDITION(0);
        return false;
    }

    bool operator()(T_sourcetype x, int index)
    {
        if (x > max_)
        {
            max_ = x;
            index_ = index;
        }
        return true;
    }

    T_resulttype result(int)
    { return index_; }

    void reset()
    {
        max_ = neghuge(T_sourcetype());
        index_ = tiny(int());
    }

    void reset(T_resulttype)
    {
        BZPRECHECK(0, "Provided initial value for ReduceMaxIndex");
        reset();
    }

    static const char* name()
    { return "maxIndex"; }

protected:
    T_sourcetype max_;
    int index_;
};

template<class P_sourcetype, int N_rank>
class ReduceMaxIndexVector {

public:
    typedef P_sourcetype T_sourcetype;
    typedef TinyVector<int,N_rank> T_resulttype;
    typedef T_resulttype T_numtype;

    enum { canProvideInitialValue = 0 };

    ReduceMaxIndexVector()
    { reset(); }

    ReduceMaxIndexVector(T_resulttype max)
    {
        reset(max);
    }

    bool operator()(T_sourcetype x)
    {
        BZPRECONDITION(0);
        return false;
    }

    bool operator()(T_sourcetype x, const TinyVector<int,N_rank>& index)
    {
        if (x > max_)
        {
            max_ = x;
            index_ = index;
        }
        return true;
    }

    T_resulttype result(int)
    { return index_; }

    void reset()
    {
        max_ = neghuge(T_sourcetype());
        index_ = tiny(int());
    }

    void reset(T_resulttype)
    {
        BZPRECHECK(0, "Provided initial value for ReduceMaxIndex");
        reset();
    }

    static const char* name()
    { return "maxIndex"; }

protected:
    T_sourcetype max_;
    TinyVector<int,N_rank> index_;
};

template<class P_sourcetype>
class ReduceFirst {

public:
    typedef P_sourcetype T_sourcetype;
    typedef int          T_resulttype;
    typedef T_resulttype T_numtype;

    enum { needIndex = 1, canProvideInitialValue = 0 };

    ReduceFirst()
    { reset(); }

    ReduceFirst(T_resulttype)
    {
        BZPRECONDITION(0);
    }

    bool operator()(T_sourcetype x)
    {
        BZPRECONDITION(0);
        return false;
    }

    bool operator()(T_sourcetype x, int index)
    {
        if (x)
        {
            index_ = index;
            return false;
        }
        else
            return true;
    }

    T_resulttype result(int)
    { return index_; }

    void reset()
    {
        index_ = tiny(int());
    }

    void reset(T_resulttype)
    {
        BZPRECHECK(0, "Provided initial value for ReduceFirst");
        reset();
    }

    static const char* name()
    { return "first"; }

protected:
    int index_;
};

template<class P_sourcetype>
class ReduceLast {

public:
    typedef P_sourcetype T_sourcetype;
    typedef int          T_resulttype;
    typedef T_resulttype T_numtype;

    enum { needIndex = 1, canProvideInitialValue = 0 };

    ReduceLast()
    { reset(); }

    ReduceLast(T_resulttype)
    {
        BZPRECONDITION(0);
    }

    bool operator()(T_sourcetype x)
    {
        BZPRECONDITION(0);
        return false;
    }

    bool operator()(T_sourcetype x, int index)
    {
        if (x)
        {
            index_ = index;
            return true;
        }
        else
            return true;
    }

    T_resulttype result(int)
    { return index_; }

    void reset()
    {
        index_ = huge(int());
    }

    void reset(T_resulttype)
    {
        BZPRECHECK(0, "Provided initial value for ReduceFirst");
        reset();
    }

    static const char* name()
    { return "last"; }

protected:
    int index_;
};

template<class P_sourcetype, class P_resulttype = BZ_SUMTYPE(P_sourcetype)>
class ReduceProduct {

public:
    typedef P_sourcetype T_sourcetype;
    typedef P_resulttype T_resulttype;
    typedef T_resulttype T_numtype;

    enum { needIndex = 0, canProvideInitialValue = 1 };

    ReduceProduct()
    { product_ = one(T_resulttype()); }

    ReduceProduct(T_resulttype initialValue)
    { product_ = initialValue; }

    bool operator()(T_sourcetype x)
    { 
        product_ *= x; 
        return true;
    }

    bool operator()(T_sourcetype x, int)
    { 
        product_ *= x; 
        return true;
    }

    T_resulttype result(int)
    { return product_; }

    void reset()
    { product_ = one(T_resulttype()); }

    void reset(T_resulttype initialValue)
    { product_ = initialValue; }

    static const char* name()
    { return "product"; }

protected:
    T_resulttype product_;
};

template<class P_sourcetype>
class ReduceCount {

public:
    typedef P_sourcetype T_sourcetype;
    typedef int          T_resulttype;
    typedef T_resulttype T_numtype;

    enum { needIndex = 0, canProvideInitialValue = 1 };

    ReduceCount()
    { reset(); }

    ReduceCount(T_resulttype count)
    {
        count_ = count;
    }

    bool operator()(T_sourcetype x)
    {
        if (x)
            ++count_;
        return true;
    }

    bool operator()(T_sourcetype x, int)
    {
        if (x)
            ++count_;
        return true;
    }

    T_resulttype result(int)
    { return count_; }

    void reset()
    { count_ = zero(T_resulttype()); }

    void reset(T_resulttype initialValue)
    { count_ = initialValue; }

    static const char* name()
    { return "count"; }

protected:
    T_resulttype count_;
};

template<class P_sourcetype>
class ReduceAny {

public:
    typedef P_sourcetype T_sourcetype;
    typedef _bz_bool     T_resulttype;
    typedef T_resulttype T_numtype;

    enum { needIndex = 0, canProvideInitialValue = 0 };

    ReduceAny()
    { reset(); }

    ReduceAny(T_resulttype initialValue)
    {
        reset(initialValue);
    }

    bool operator()(T_sourcetype x)
    {
        if (x)
        {
            any_ = _bz_true;
            return false;
        }

        return true;
    }

    bool operator()(T_sourcetype x, int)
    {
        if (x)
        {
            any_ = _bz_true;
            return false;
        }

        return true;
    }

    T_resulttype result(int)
    { return any_; }

    void reset()
    { any_ = _bz_false; }

    void reset(T_resulttype)
    { 
        BZPRECHECK(0, "Provided initial value for ReduceAny");
        reset();
    }

    static const char* name()
    { return "any"; }

protected:
    T_resulttype any_;
};

template<class P_sourcetype>
class ReduceAll {

public:
    typedef P_sourcetype T_sourcetype;
    typedef _bz_bool     T_resulttype;
    typedef T_resulttype T_numtype;

    enum { needIndex = 0, canProvideInitialValue = 0 };

    ReduceAll()
    { reset(); }

    ReduceAll(T_resulttype initialValue)
    {
        reset(initialValue);
    }

    bool operator()(T_sourcetype x)
    {
        if (!_bz_bool(x))
        {
            all_ = _bz_false;
            return false;
        }
        else
            return true;
    }

    bool operator()(T_sourcetype x, int)
    {
        if (!_bz_bool(x))
        {
            all_ = _bz_false;
            return false;
        }
        else
            return true;
    }

    T_resulttype result(int)
    { return all_; }

    void reset()
    { all_ = _bz_true; }

    void reset(T_resulttype)
    {
        BZPRECHECK(0, "Provided initial value for ReduceAll");
        reset();
    }

    static const char* name()
    { return "all"; }

protected:
    T_resulttype all_;
}; 

BZ_NAMESPACE_END

#endif // BZ_REDUCE_H
