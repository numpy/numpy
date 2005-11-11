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
 * Revision 1.1.1.1  2000/06/19 12:26:08  tveldhui
 * Imported sources
 *
 * Revision 1.5  1998/03/14 00:04:47  tveldhui
 * 0.2-alpha-05
 *
 * Revision 1.4  1997/07/16 14:51:20  tveldhui
 * Update: Alpha release 0.2 (Arrays)
 *
 * Revision 1.3  1997/01/24 14:42:00  tveldhui
 * Periodic RCS update
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

template<class P_parameter>
Benchmark<P_parameter>::Benchmark(unsigned numImplementations)
{
    state_ = uninitialized;
    numImplementations_ = numImplementations;
    numStoredImplementations_ = 0;
    implementations_ = new BenchmarkImplementation<P_parameter>* [numImplementations_];
    rates_.resize(numImplementations, numParameterSettings());
    Mflops_.resize(numImplementations, numParameterSettings());
}

template<class P_parameter>
Benchmark<P_parameter>::~Benchmark()
{
    delete [] implementations_;
}

template<class P_parameter>
void Benchmark<P_parameter>::addImplementation(
    BenchmarkImplementation<P_parameter> * implementation)
{
    BZPRECONDITION(state_ == uninitialized);
    BZPRECONDITION(numStoredImplementations_ < numImplementations_);

    implementations_[numStoredImplementations_++] = implementation;

    if (numStoredImplementations_ == numImplementations_)
        state_ = initialized;
}

template<class P_parameter>
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

template<class P_parameter>
double Benchmark<P_parameter>::getMflops(unsigned implementation, 
    unsigned setting) const
{
    BZPRECONDITION(state_ == done);
    BZPRECONDITION(implementation < numImplementations_);
    BZPRECONDITION(setting < numParameterSettings());

    return Mflops_(implementation, setting);
}

template<class P_parameter>
double Benchmark<P_parameter>::getRate(unsigned implementation,  
    unsigned setting) const
{
    BZPRECONDITION(state_ == done);
    BZPRECONDITION(implementation < numImplementations_);
    BZPRECONDITION(setting < numParameterSettings());

    return rates_(implementation, setting);
}

template<class P_parameter>
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
