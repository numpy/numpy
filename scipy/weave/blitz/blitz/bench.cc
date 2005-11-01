/*
 * Copyright (C) 1997 Todd Veldhuizen <tveldhui@oonumerics.org>
 * All rights reserved.  Please see <blitz/blitz.h> for terms and
 * conditions of use.
 *
 */

#ifndef BZ_BENCH_CC
#define BZ_BENCH_CC

#ifndef BZ_BENCH_H
 #error <blitz/bench.cc> must be included via <blitz/bench.h>
#endif

#ifdef BZ_HAVE_STD
 #include <fstream>
#else
 #include <fstream.h>
#endif

BZ_NAMESPACE(blitz)

template<typename P_parameter>
Benchmark<P_parameter>::Benchmark(unsigned numImplementations)
{
    state_ = uninitialized;
    numImplementations_ = numImplementations;
    numStoredImplementations_ = 0;
    implementations_ = new BenchmarkImplementation<P_parameter>* [numImplementations_];
    rates_.resize(numImplementations, numParameterSettings());
    Mflops_.resize(numImplementations, numParameterSettings());
}

template<typename P_parameter>
Benchmark<P_parameter>::~Benchmark()
{
    delete [] implementations_;
}

template<typename P_parameter>
void Benchmark<P_parameter>::addImplementation(
    BenchmarkImplementation<P_parameter> * implementation)
{
    BZPRECONDITION(state_ == uninitialized);
    BZPRECONDITION(numStoredImplementations_ < numImplementations_);

    implementations_[numStoredImplementations_++] = implementation;

    if (numStoredImplementations_ == numImplementations_)
        state_ = initialized;
}

template<typename P_parameter>
void Benchmark<P_parameter>::run(ostream& log)
{
    BZPRECONDITION(state_ == initialized);
    state_ = running;

    Timer t;

    for (unsigned j=0; j < numImplementations_; ++j)
    {
        for (unsigned i=0; i < numParameterSettings(); ++i)
        {
            log  << setw(20) << implementations_[j]->implementationName()
                 << " " << setw(8) << getParameterSetting(i) << "  ";
            log.flush();

            implementations_[j]->initialize(getParameterSetting(i));
            implementations_[j]->tickle();

            unsigned long iterations = getIterationSetting(i);

            t.start();
            implementations_[j]->run(iterations);
            t.stop();
            double tm = t.elapsedSeconds();

            t.start();
            implementations_[j]->runOverhead(iterations);
            t.stop();
            double tmOverhead = t.elapsedSeconds();

            rates_(j,i) = iterations / (tm - tmOverhead);
            Mflops_(j,i) = rates_(j,i) 
                * implementations_[j]->flopsPerIteration() / 1.0e+6;

            log << setw(10) << (rates_(j,i)/1.0e+6) << " Mops/s ";

            if (implementations_[j]->flopsPerIteration() != 0)
            {
                log << "[" << setw(7) << Mflops_(j,i) << " Mflops]";
            }

            log << endl;
            log.flush();

            implementations_[j]->done();
        }
    }

    state_ = done;
}

template<typename P_parameter>
double Benchmark<P_parameter>::getMflops(unsigned implementation, 
    unsigned setting) const
{
    BZPRECONDITION(state_ == done);
    BZPRECONDITION(implementation < numImplementations_);
    BZPRECONDITION(setting < numParameterSettings());

    return Mflops_(implementation, setting);
}

template<typename P_parameter>
double Benchmark<P_parameter>::getRate(unsigned implementation,  
    unsigned setting) const
{
    BZPRECONDITION(state_ == done);
    BZPRECONDITION(implementation < numImplementations_);
    BZPRECONDITION(setting < numParameterSettings());

    return rates_(implementation, setting);
}

template<typename P_parameter>
void Benchmark<P_parameter>::saveMatlabGraph(const char* filename) const
{
    BZPRECONDITION(state_ == done);

    ofstream ofs(filename);
     
    assert(ofs.good());

    ofs << "% This matlab file generated automatically by class Benchmark"
        << endl << "% of the Blitz++ class library." << endl << endl;

    ofs.setf(ios::scientific);

    ofs << "parm = [ ";
    int i;
    for (i=0; i < numParameterSettings(); ++i)
        ofs << setprecision(12) << double(getParameterSetting(i)) << " ";
    ofs << "]; " << endl << endl;

    ofs << "Mf = [ ";
    for (i=0; i < numParameterSettings(); ++i)
    {
        for (int j=0; j < numImplementations_; ++j)
        {
            ofs << setprecision(12) << getMflops(j,i) << " ";
        }
        if (i != numParameterSettings()-1)
            ofs << ";" << endl;
    }
    ofs << "] ;" << endl << endl;

    ofs << "semilogx(parm,Mf), title('" << description() << "'), " << endl
        << "    xlabel('" << parameterDescription() << "'), "
        << "ylabel('Mflops')" << endl
        << "legend(";
    
    for (int j=0; j < numImplementations_; ++j)
    {
        ofs << "'" << implementations_[j]->implementationName()
            << "'";
        if (j != numImplementations_ - 1)
            ofs << ", ";
    } 

    ofs << ")" << endl;
}

BZ_NAMESPACE_END

#endif // BZ_BENCH_CC
