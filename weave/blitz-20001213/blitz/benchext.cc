/*
 * $Id$
 *
 * Copyright (C) 1997 Todd Veldhuizen <tveldhui@oonumerics.org>
 * All rights reserved.  Please see <blitz/blitz.h> for terms and
 * conditions of use.
 *
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
 * Revision 1.3  2002/06/28 05:05:58  jcumming
 * Changed loop variable j to unsigned to eliminate signed/unsigned comparisons.
 *
 * Revision 1.2  2001/01/26 18:30:50  tveldhui
 * More source code reorganization to reduce compile times.
 *
 * Revision 1.1.1.1  2000/06/19 12:26:11  tveldhui
 * Imported sources
 *
 * Revision 1.2  1998/03/14 00:04:47  tveldhui
 * 0.2-alpha-05
 *
 * Revision 1.1  1997/07/16 14:51:20  tveldhui
 * Update: Alpha release 0.2 (Arrays)
 *
 */

#ifndef BZ_BENCHEXT_CC
#define BZ_BENCHEXT_CC

#ifndef BZ_BENCHEXT_H
 #error <blitz/benchext.cc> must be included via <blitz/benchext.h>
#endif

#include <blitz/vector-et.h>

#ifdef BZ_HAVE_STD
 #include <fstream>
#else
 #include <fstream.h>
#endif

BZ_NAMESPACE(blitz)

template<class P_parameter>
BenchmarkExt<P_parameter>::BenchmarkExt(const char* name, 
    int numImplementations)
{
    BZPRECONDITION(numImplementations > 0);

    description_ = name;
    numImplementations_ = numImplementations;

    implementationDescriptions_.resize(numImplementations);
    parameterDescription_ = "Vector length";
    rateDescription_ = "Mflops/s";

    // Set up default parameters and iterations
    setNumParameters(19);

    // NEEDS_WORK: once pow(X,Y) is supported, can just say
    // parameters_ = pow(10.0, Range(1,20)/4.0);

    for (int i=0; i < numParameters_; ++i)
        parameters_[i] = (P_parameter)::pow(10.0, (i+1)/4.0);

    iterations_ = 5.0e+5 / parameters_;
    flopsPerIteration_ = parameters_;

    // Set up initial state
    state_ = initializing;
    implementationNumber_ = 0;
}

template<class P_parameter>
BenchmarkExt<P_parameter>::~BenchmarkExt()
{
}

template<class P_parameter>
void BenchmarkExt<P_parameter>::setNumParameters(int numParameters)
{
    BZPRECONDITION(state_ == initializing);

    numParameters_ = numParameters;

    parameters_.resize(numParameters_);
    iterations_.resize(numParameters_);
    flopsPerIteration_.resize(numParameters_);

    // Set up timer and Mflops array
    times_.resize(numImplementations_, numParameters_);
}

template<class P_parameter>
void BenchmarkExt<P_parameter>::setParameterVector(Vector<P_parameter> parms)
{
    BZPRECONDITION(state_ == initializing);
    BZPRECONDITION(parms.length() == parameters_.length());

    // NEEDS_WORK: should use operator=(), once that problem
    // gets sorted out.
    // parameters_ = parms;
    for (int i=0; i < parameters_.length(); ++i)
        parameters_[i] = parms(i);
}

template<class P_parameter>
void BenchmarkExt<P_parameter>::setParameterDescription(const char* string)
{
    parameterDescription_ = string;
}

template<class P_parameter>
void BenchmarkExt<P_parameter>::setIterations(Vector<long> iters)
{
    BZPRECONDITION(state_ == initializing);

    // NEEDS_WORK: should use operator=(), once that problem
    // gets sorted out.
    // iterations_ = iters;

    for (int i=0; i < iterations_.length(); ++i)
        iterations_[i] = iters(i);
}

template<class P_parameter>
void BenchmarkExt<P_parameter>::setFlopsPerIteration(Vector<double> 
    flopsPerIteration)
{
    BZPRECONDITION(flopsPerIteration_.length() == flopsPerIteration.length());

    // NEEDS_WORK: should use operator=(), once that problem
    // gets sorted out.
    // flopsPerIteration_ = flopsPerIteration;

    for (int i=0; i < flopsPerIteration_.length(); ++i)
        flopsPerIteration_[i] = flopsPerIteration[i];
}

template<class P_parameter>
void BenchmarkExt<P_parameter>::setRateDescription(const char* string)
{
    rateDescription_ = string;
}

template<class P_parameter>
void BenchmarkExt<P_parameter>::beginBenchmarking()
{
    BZPRECONDITION(state_ == initializing);
    state_ = benchmarking;
}

template<class P_parameter>
void BenchmarkExt<P_parameter>::beginImplementation(const char* description)
{
    BZPRECONDITION(implementationNumber_ < numImplementations_);
    BZPRECONDITION(state_ == benchmarking);

    implementationDescriptions_[implementationNumber_] = description;

    state_ = benchmarkingImplementation;
    parameterNumber_ = 0;
}

template<class P_parameter>
_bz_bool BenchmarkExt<P_parameter>::doneImplementationBenchmark() const
{
    BZPRECONDITION(state_ == benchmarkingImplementation);
    return parameterNumber_ == numParameters_;
}

template<class P_parameter>
P_parameter BenchmarkExt<P_parameter>::getParameter() const
{
    BZPRECONDITION(state_ == benchmarkingImplementation);
    BZPRECONDITION(parameterNumber_ >= 0);
    BZPRECONDITION(parameterNumber_ < numParameters_);

    return parameters_[parameterNumber_];
}

template<class P_parameter>
long BenchmarkExt<P_parameter>::getIterations() const
{
    BZPRECONDITION(state_ == benchmarkingImplementation);
    BZPRECONDITION(parameterNumber_ >= 0);
    BZPRECONDITION(parameterNumber_ < numParameters_);

    return iterations_[parameterNumber_];
}

template<class P_parameter>
inline void BenchmarkExt<P_parameter>::start()
{
    BZPRECONDITION(state_ == benchmarkingImplementation);
    BZPRECONDITION(parameterNumber_ < numParameters_);
    state_ = running;
    timer_.start();
}

template<class P_parameter>
inline void BenchmarkExt<P_parameter>::stop()
{
    timer_.stop();
    BZPRECONDITION(state_ == running);
    state_ = benchmarkingImplementation;
    
    times_(implementationNumber_, parameterNumber_) = timer_.elapsedSeconds();

    ++parameterNumber_;
}

template<class P_parameter>
inline void BenchmarkExt<P_parameter>::startOverhead()
{
    BZPRECONDITION(state_ == benchmarkingImplementation);
    BZPRECONDITION(parameterNumber_ > 0);
    BZPRECONDITION(parameterNumber_ <= numParameters_);
    state_ = runningOverhead;
    overheadTimer_.start();
}

template<class P_parameter>
inline void BenchmarkExt<P_parameter>::stopOverhead()
{
    BZPRECONDITION(state_ == runningOverhead);
    overheadTimer_.stop();
    times_(implementationNumber_, parameterNumber_-1) -= 
        overheadTimer_.elapsedSeconds();

    state_ = benchmarkingImplementation;
}

template<class P_parameter>
void BenchmarkExt<P_parameter>::endImplementation()
{
    BZPRECONDITION(state_ == benchmarkingImplementation);
    BZPRECONDITION(parameterNumber_ == numParameters_);

    ++implementationNumber_;

    state_ = benchmarking;
}

template<class P_parameter>
void BenchmarkExt<P_parameter>::endBenchmarking()
{
    BZPRECONDITION(state_ == benchmarking);
    BZPRECONDITION(implementationNumber_ == numImplementations_);
    
    state_ = done;
}

template<class P_parameter>
double BenchmarkExt<P_parameter>::getMflops(unsigned implementation,
    unsigned parameterNum) const
{
    BZPRECONDITION(state_ == done);
    BZPRECONDITION(implementation < numImplementations_);
    BZPRECONDITION(parameterNum < numParameters_);
    return iterations_(parameterNum) * flopsPerIteration_(parameterNum)
        / times_(implementation, parameterNum) / 1.0e+6;
}

template<class P_parameter>
void BenchmarkExt<P_parameter>::saveMatlabGraph(const char* filename) const
{
    BZPRECONDITION(state_ == done);

    ofstream ofs(filename);
     
    assert(ofs.good());

    ofs << "% This matlab file generated automatically by class Benchmark"
        << endl << "% of the Blitz++ class library." << endl << endl;

    ofs.setf(ios::scientific);

    // This will be a lot simpler once Matlab-style output formatting
    // of vectors & matrices is finished.

    // ofs << "parm = " << parameters_ << ";" << endl << endl;

    ofs << "parm = [ ";
    int i;
    for (i=0; i < numParameters_; ++i)
        ofs << setprecision(12) << double(parameters_[i]) << " ";
    ofs << "]; " << endl << endl;

    ofs << "Mf = [ ";
    for (i=0; i < numParameters_; ++i)
    {
        for (unsigned j=0; j < numImplementations_; ++j)
        {
            ofs << setprecision(12) << getMflops(j,i) << " ";
        }
        if (i != numParameters_ - 1)
            ofs << ";" << endl;
    }
    ofs << "] ;" << endl << endl;

    ofs << "semilogx(parm,Mf), title('" << description_ << "'), " << endl
        << "    xlabel('" << parameterDescription_ << "'), "
        << "ylabel('" << rateDescription_ << "')" << endl
        << "legend(";
    
    for (unsigned j=0; j < numImplementations_; ++j)
    {
        ofs << "'" << implementationDescriptions_(j) << "'";
        if (j != numImplementations_ - 1)
            ofs << ", ";
    } 

    ofs << ")" << endl;
}

BZ_NAMESPACE_END

#endif // BZ_BENCHEXT_CC
