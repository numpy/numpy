/*
 * Copyright (C) 1997 Todd Veldhuizen <tveldhui@oonumerics.org>
 * All rights reserved.  Please see <blitz/blitz.h> for terms and
 * conditions of use.
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

template<typename P_parameter>
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

    for (unsigned i=0; i < numParameters_; ++i)
        parameters_[i] = (P_parameter)::pow(10.0, (i+1)/4.0);

    iterations_ = 5.0e+5 / parameters_;
    flopsPerIteration_ = parameters_;

    // Set up initial state
    state_ = initializing;
    implementationNumber_ = 0;
}

template<typename P_parameter>
BenchmarkExt<P_parameter>::~BenchmarkExt()
{
}

template<typename P_parameter>
void BenchmarkExt<P_parameter>::setNumParameters(int numParameters)
{
    //BZPRECONDITION(state_ == initializing);

    numParameters_ = numParameters;

    parameters_.resize(numParameters_);
    iterations_.resize(numParameters_);
    flopsPerIteration_.resize(numParameters_);

    // Set up timer and Mflops array
    times_.resize(numImplementations_, numParameters_);
}

template<typename P_parameter>
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

template<typename P_parameter>
void BenchmarkExt<P_parameter>::setParameterDescription(const char* string)
{
    parameterDescription_ = string;
}

template<typename P_parameter>
void BenchmarkExt<P_parameter>::setIterations(Vector<long> iters)
{
    BZPRECONDITION(state_ == initializing);

    // NEEDS_WORK: should use operator=(), once that problem
    // gets sorted out.
    // iterations_ = iters;

    for (int i=0; i < iterations_.length(); ++i)
        iterations_[i] = iters(i);
}

template<typename P_parameter>
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

template<typename P_parameter>
void BenchmarkExt<P_parameter>::setRateDescription(const char* string)
{
    rateDescription_ = string;
}

template<typename P_parameter>
void BenchmarkExt<P_parameter>::beginBenchmarking()
{
    BZPRECONDITION(state_ == initializing);
    state_ = benchmarking;
}

template<typename P_parameter>
void BenchmarkExt<P_parameter>::beginImplementation(const char* description)
{
    BZPRECONDITION(implementationNumber_ < numImplementations_);
    BZPRECONDITION(state_ == benchmarking);

    implementationDescriptions_[implementationNumber_] = description;

    state_ = benchmarkingImplementation;
    parameterNumber_ = 0;
}

template<typename P_parameter>
bool BenchmarkExt<P_parameter>::doneImplementationBenchmark() const
{
    BZPRECONDITION(state_ == benchmarkingImplementation);
    return parameterNumber_ == numParameters_;
}

template<typename P_parameter>
P_parameter BenchmarkExt<P_parameter>::getParameter() const
{
    BZPRECONDITION(state_ == benchmarkingImplementation);
    BZPRECONDITION(parameterNumber_ < numParameters_);

    return parameters_[parameterNumber_];
}

template<typename P_parameter>
long BenchmarkExt<P_parameter>::getIterations() const
{
    BZPRECONDITION(state_ == benchmarkingImplementation);
    BZPRECONDITION(parameterNumber_ < numParameters_);

    return iterations_[parameterNumber_];
}

template<typename P_parameter>
inline void BenchmarkExt<P_parameter>::start()
{
    BZPRECONDITION(state_ == benchmarkingImplementation);
    BZPRECONDITION(parameterNumber_ < numParameters_);
    state_ = running;
    timer_.start();
}

template<typename P_parameter>
inline void BenchmarkExt<P_parameter>::stop()
{
    timer_.stop();
    BZPRECONDITION(state_ == running);
    state_ = benchmarkingImplementation;
    
    times_(implementationNumber_, parameterNumber_) = timer_.elapsedSeconds();

    ++parameterNumber_;
}

template<typename P_parameter>
inline void BenchmarkExt<P_parameter>::startOverhead()
{
    BZPRECONDITION(state_ == benchmarkingImplementation);
    BZPRECONDITION(parameterNumber_ > 0);
    BZPRECONDITION(parameterNumber_ <= numParameters_);
    state_ = runningOverhead;
    overheadTimer_.start();
}

template<typename P_parameter>
inline void BenchmarkExt<P_parameter>::stopOverhead()
{
    BZPRECONDITION(state_ == runningOverhead);
    overheadTimer_.stop();
    times_(implementationNumber_, parameterNumber_-1) -= 
        overheadTimer_.elapsedSeconds();

    state_ = benchmarkingImplementation;
}

template<typename P_parameter>
void BenchmarkExt<P_parameter>::endImplementation()
{
    BZPRECONDITION(state_ == benchmarkingImplementation);
    BZPRECONDITION(parameterNumber_ == numParameters_);

    ++implementationNumber_;

    state_ = benchmarking;
}

template<typename P_parameter>
void BenchmarkExt<P_parameter>::endBenchmarking()
{
    BZPRECONDITION(state_ == benchmarking);
    BZPRECONDITION(implementationNumber_ == numImplementations_);
    
    state_ = done;
}

template<typename P_parameter>
double BenchmarkExt<P_parameter>::getMflops(unsigned implementation,
    unsigned parameterNum) const
{
    BZPRECONDITION(state_ == done);
    BZPRECONDITION(implementation < numImplementations_);
    BZPRECONDITION(parameterNum < numParameters_);
    return iterations_(parameterNum) * flopsPerIteration_(parameterNum)
        / times_(implementation, parameterNum) / 1.0e+6;
}

template<typename P_parameter>
void BenchmarkExt<P_parameter>::saveMatlabGraph(const char* filename, const char* graphType) const
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
    unsigned i;
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

    ofs << graphType << "(parm,Mf), title('" << description_ << "'), " << endl
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
