/***************************************************************************
 * blitz/bench.h      Benchmark classes
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
 * Revision 1.1.1.1  2000/06/19 12:26:08  tveldhui
 * Imported sources
 *
 * Revision 1.4  1998/03/14 00:04:47  tveldhui
 * 0.2-alpha-05
 *
 * Revision 1.3  1997/07/16 14:51:20  tveldhui
 * Update: Alpha release 0.2 (Arrays)
 *
 * Revision 1.2  1997/01/24 14:30:34  tveldhui
 * Prior to rewrite of Bench class; in this version, Bench contains
 * each benchmark implementation.
 *
 */

#ifndef BZ_BENCH_H
#define BZ_BENCH_H

#ifndef BZ_MATRIX_H
 #include <blitz/matrix.h>
#endif

#ifndef BZ_TIMER_H
 #include <blitz/timer.h>
#endif

#include <math.h>

BZ_NAMESPACE(blitz)

// Forward declaration
template<class P_parameter = unsigned>
class BenchmarkImplementation;


// Declaration of class Benchmark<T>
// The template parameter T is the parameter type which is varied in
// the benchmark.  Typically T will be an unsigned, and will represent
// the length of a vector, size of an array, etc.

template<class P_parameter = unsigned>
class Benchmark {

public:
    typedef P_parameter T_parameter;

    Benchmark(unsigned numImplementations);

    ~Benchmark();

    void addImplementation(BenchmarkImplementation<T_parameter>* 
        implementation);

    void run(ostream& log = cout);

    double getMflops(unsigned implementation, unsigned setting) const;

    double getRate(unsigned implementation, unsigned setting) const;

    void saveMatlabGraph(const char* filename) const;

public:
    // Virtual functions

    virtual const char* description() const
    { return ""; }

    virtual const char* parameterDescription() const
    { return "Vector length"; }

    virtual unsigned numParameterSettings() const
    { return 19; }

    virtual T_parameter getParameterSetting(unsigned i) const
    { return ::pow(10.0, (i+1)/4.0); }

    virtual long getIterationSetting(unsigned i) const
    { return 1000000L / getParameterSetting(i); }

private:
    Benchmark(const Benchmark<P_parameter>&) { }
    void operator=(const Benchmark<P_parameter>&) { }

    enum { uninitialized, initialized, running, done } state_;

    unsigned numImplementations_;
    unsigned numStoredImplementations_;

    BenchmarkImplementation<T_parameter>** implementations_;

    Matrix<double,RowMajor> rates_;       // Iterations per second array
    Matrix<double,RowMajor> Mflops_;
};

template<class P_parameter>
class BenchmarkImplementation {

public:
    typedef P_parameter T_parameter;

    virtual void initialize(P_parameter parameter) { }

    virtual void done() { }

    virtual const char* implementationName() const
    { return ""; }

    virtual void run(long iterations) = 0;

    virtual void runOverhead(long iterations) 
    { 
        for (long i=0; i < iterations; ++i)
        {
        }
    };

    virtual void tickle() { }

    virtual long flopsPerIteration() const
    { return 0; }
};

BZ_NAMESPACE_END

#include <blitz/bench.cc>  

#endif // BZ_BENCH_H
