/***************************************************************************
 * blitz/reduce.h        Reduction operators: sum, mean, min, max,
 *                       minIndex, maxIndex, product, count, any, all
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
 * Revision 1.1  2002/01/03 19:50:34  eric
 * renaming compiler to weave
 *
 * Revision 1.1  2001/04/27 17:22:04  ej
 * first attempt to include needed pieces of blitz
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
