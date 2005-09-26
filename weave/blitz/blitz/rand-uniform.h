/***************************************************************************
 * blitz/rand-uniform.h    Uniform class, which provides uniformly
 *                         distributed random numbers.
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
 *
 * This random number generator is based on the LAPACK auxilliary
 * routine DLARAN by Jack Dongarra.  It's a multiplicative congruential 
 * generator with modulus 2^48 and multiplier 33952834046453. 
 *
 * See also: G. S. Fishman, Multiplicative congruential random number
 * generators with modulus 2^b: an exhaustive analysis for b=32 and
 * a partial analysis for b=48, Math. Comp. 189, pp 331-344, 1990.
 * 
 * This routine requires 32-bit integers.
 *
 * The generated number lies in the open interval (low,high).  i.e. low and
 * high themselves will never be generated.
 *
 ***************************************************************************
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
 * Revision 1.2  2001/01/24 20:22:50  tveldhui
 * Updated copyright date in headers.
 *
 * Revision 1.1.1.1  2000/06/19 12:26:09  tveldhui
 * Imported sources
 *
 * Revision 1.4  1998/03/14 00:04:47  tveldhui
 * 0.2-alpha-05
 *
 * Revision 1.3  1997/07/16 14:51:20  tveldhui
 * Update: Alpha release 0.2 (Arrays)
 *
 * Revision 1.2  1997/01/24 14:42:00  tveldhui
 * Periodic RCS update
 *
 */

#ifndef BZ_RAND_UNIFORM_H
#define BZ_RAND_UNIFORM_H

#ifndef BZ_RANDOM_H
 #include <blitz/random.h>
#endif

BZ_NAMESPACE(blitz)

class Uniform {

public:
    typedef double T_numtype;

    Uniform(double low = 0.0, double high = 1.0, double = 0.0)
        : low_(low), length_(high-low)
    { 
        BZPRECONDITION(sizeof(int) >= 4);   // Need 32 bit integers!

        seed[0] = 24;       // All seeds in the range [0,4095]
        seed[1] = 711;
        seed[2] = 3;
        seed[3] = 3721;     // The last seed must be odd
    }

    void randomize() 
    { 
        BZ_NOT_IMPLEMENTED();            // NEEDS_WORK

        BZPOSTCONDITION(seed[3] % 2 == 1);
    }
  
    // I'm trying to avoid having a compiled 
    // portion of the library, so this is inline until I
    // figure out a better way to do this or I change my mind.
    // -- TV
    // NEEDS_WORK
    double random()
    { 
        BZPRECONDITION(seed[3] % 2 == 1);

        int it0, it1, it2, it3;
        it3 = seed[3] * 2549;
        it2 = it3 / 4096;
        it3 -= it2 << 12;
        it2 += seed[2] * 2549 + seed[3] * 2508;
        it1 = it2 / 4096;
        it2 -= it1 << 12;
        it1 += seed[1] * 2549 + seed[2] * 2508 + seed[3] * 322;
        it0 = it1 / 4096;
        it1 -= it0 << 12;
        it0 += seed[0] * 2549 + seed[1] * 2508 + seed[2] * 322 + seed[3] * 494;
        it0 %= 4096;
        seed[0] = it0;
        seed[1] = it1;
        seed[2] = it2;
        seed[3] = it3;
      
        const double z = 1 / 4096.;
        return low_ + length_ * (it0 + (it1 + (it2 + it3 * z) * z) * z) * z;
    } 

    operator double() 
    { return random(); }

private:
    double low_, length_;

    int seed[4];
};

BZ_NAMESPACE_END

#endif // BZ_RAND_UNIFORM_H

