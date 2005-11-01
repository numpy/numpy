// -*- C++ -*-
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
 ***************************************************************************/

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

template<typename P_sourcetype, typename P_resulttype = BZ_SUMTYPE(P_sourcetype)>
class ReduceSum {

public:
    typedef P_sourcetype T_sourcetype;
    typedef P_resulttype T_resulttype;
    typedef T_resulttype T_numtype;

    static const bool needIndex = false, canProvideInitialValue = true;

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

template<typename P_sourcetype, typename P_resulttype = BZ_FLOATTYPE(P_sourcetype)>
class ReduceMean {

public:
    typedef P_sourcetype T_sourcetype;
    typedef P_resulttype T_resulttype;
    typedef T_resulttype T_numtype;

    static const bool needIndex = false, canProvideInitialValue = false;

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

template<typename P_sourcetype>
class ReduceMin {

public:
    typedef P_sourcetype T_sourcetype;
    typedef P_sourcetype T_resulttype;
    typedef T_resulttype T_numtype;

    static const bool needIndex = false, canProvideInitialValue = false;

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

template<typename P_sourcetype>
class ReduceMax {

public:
    typedef P_sourcetype T_sourcetype;
    typedef P_sourcetype T_resulttype;
    typedef T_resulttype T_numtype;

    static const bool needIndex = false, canProvideInitialValue = true;

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

template<typename P_sourcetype>
class ReduceMinIndex {

public:
    typedef P_sourcetype T_sourcetype;
    typedef int          T_resulttype;
    typedef T_resulttype T_numtype;

    static const bool needIndex = true, canProvideInitialValue = false;

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

template<typename P_sourcetype, int N>
class ReduceMinIndexVector {

public:
    typedef P_sourcetype T_sourcetype;
    typedef TinyVector<int,N> T_resulttype;
    typedef T_resulttype T_numtype;

    static const bool canProvideInitialValue = false;

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

template<typename P_sourcetype>
class ReduceMaxIndex {

public:
    typedef P_sourcetype T_sourcetype;
    typedef int          T_resulttype;
    typedef T_resulttype T_numtype;

    static const bool needIndex = true, canProvideInitialValue = false;

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

template<typename P_sourcetype, int N_rank>
class ReduceMaxIndexVector {

public:
    typedef P_sourcetype T_sourcetype;
    typedef TinyVector<int,N_rank> T_resulttype;
    typedef T_resulttype T_numtype;

    static const bool canProvideInitialValue = false;

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

template<typename P_sourcetype>
class ReduceFirst {

public:
    typedef P_sourcetype T_sourcetype;
    typedef int          T_resulttype;
    typedef T_resulttype T_numtype;

    static const bool needIndex = true, canProvideInitialValue = false;

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

template<typename P_sourcetype>
class ReduceLast {

public:
    typedef P_sourcetype T_sourcetype;
    typedef int          T_resulttype;
    typedef T_resulttype T_numtype;

    static const bool needIndex = true, canProvideInitialValue = false;

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

template<typename P_sourcetype, typename P_resulttype = BZ_SUMTYPE(P_sourcetype)>
class ReduceProduct {

public:
    typedef P_sourcetype T_sourcetype;
    typedef P_resulttype T_resulttype;
    typedef T_resulttype T_numtype;

    static const bool needIndex = false, canProvideInitialValue = true;

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

template<typename P_sourcetype>
class ReduceCount {

public:
    typedef P_sourcetype T_sourcetype;
    typedef int          T_resulttype;
    typedef T_resulttype T_numtype;

    static const bool needIndex = false, canProvideInitialValue = true;

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

template<typename P_sourcetype>
class ReduceAny {

public:
    typedef P_sourcetype T_sourcetype;
    typedef bool     T_resulttype;
    typedef T_resulttype T_numtype;

    static const bool needIndex = false, canProvideInitialValue = false;

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
            any_ = true;
            return false;
        }

        return true;
    }

    bool operator()(T_sourcetype x, int)
    {
        if (x)
        {
            any_ = true;
            return false;
        }

        return true;
    }

    T_resulttype result(int)
    { return any_; }

    void reset()
    { any_ = false; }

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

template<typename P_sourcetype>
class ReduceAll {

public:
    typedef P_sourcetype T_sourcetype;
    typedef bool     T_resulttype;
    typedef T_resulttype T_numtype;

    static const bool needIndex = false, canProvideInitialValue = false;

    ReduceAll()
    { reset(); }

    ReduceAll(T_resulttype initialValue)
    {
        reset(initialValue);
    }

    bool operator()(T_sourcetype x)
    {
        if (!bool(x))
        {
            all_ = false;
            return false;
        }
        else
            return true;
    }

    bool operator()(T_sourcetype x, int)
    {
        if (!bool(x))
        {
            all_ = false;
            return false;
        }
        else
            return true;
    }

    T_resulttype result(int)
    { return all_; }

    void reset()
    { all_ = true; }

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
