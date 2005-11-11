/***************************************************************************
 * blitz/rand-tt800.h      Matsumoto and Kurita's TT800 uniform random 
 *                         number generator.
 *
 * $Id$
 *
 ***************************************************************************
 *
 * The class TT800 encapsulates Makoto Matsumoto and Yoshiharu Kurita's 
 * TT800 twisted generalized feedback shift register (TGFSR) random number 
 * generator.  The generator has period 2^800 - 1.
 *
 * Contact: M. Matsumoto <matumoto@math.keio.ac.jp>
 *
 * See: M. Matsumoto and Y. Kurita, Twisted GFSR Generators II, 
 *      ACM Transactions on Modelling and Computer Simulation,
 *      Vol. 4, No. 3, 1994, pages 254-266.
 *
 * (c) 1994 Association for Computing Machinery.  
 *
 * Distributed with consent of the authors.
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
 * Revision 1.1.1.1  2000/06/19 12:26:12  tveldhui
 * Imported sources
 *
 * Revision 1.2  1997/07/16 14:51:20  tveldhui
 * Update: Alpha release 0.2 (Arrays)
 *
 * Revision 1.1  1997/02/28 13:39:51  tveldhui
 * Initial revision
 *
 */

#ifndef BZ_RAND_TT800_H
#define BZ_RAND_TT800_H

#ifndef BZ_RANDOM_H
 #include <blitz/random.h>
#endif

BZ_NAMESPACE(blitz)

class TT800 {

public:
    typedef double T_numtype;

    TT800(double low = 0.0, double high = 1.0, double = 0.0)
        : low_(low), length_(high-low)
    { 
        // Initial 25 seeds
        x[0] = 0x95f24dab; x[1] = 0x0b685215; x[2] = 0xe76ccae7;
        x[3] = 0xaf3ec239; x[4] = 0x715fad23; x[5] = 0x24a590ad;
        x[6] = 0x69e4b5ef; x[7] = 0xbf456141; x[8] = 0x96bc1b7b;
        x[9] = 0xa7bdf825; x[10] = 0xc1de75b7; x[11] = 0x8858a9c9;
        x[12] = 0x2da87693; x[13] = 0xb657f9dd; x[14] = 0xffdc8a9f;
        x[15] = 0x8121da71; x[16] = 0x8b823ecb; x[17] = 0x885d05f5;
        x[18] = 0x4e20cd47; x[19] = 0x5a9ad5d9; x[20] = 0x512c0c03;
        x[21] = 0xea857ccd; x[22] = 0x4cc1d30f; x[23] = 0x8891a8a1;
        x[24] = 0xa6b7aadb;

        // Magic vector 'a', don't change
        mag01[0] = 0;
        mag01[1] = 0x8ebfd028;

        k = 0;

        // f = 1/(2^32);

        f = (1.0 / 65536) / 65536;
    }

    void randomize() 
    { 
        BZ_NOT_IMPLEMENTED();            // NEEDS_WORK
    }
 
    unsigned long randomUint32()
    {
        if (k==N)
            generate();

        unsigned long y = x[k];
        y ^= (y << 7) & 0x2b5b2500; /* s and b, magic vectors */
        y ^= (y << 15) & 0xdb8b0000; /* t and c, magic vectors */
        y &= 0xffffffff; /* you may delete this line if word size = 32 */

        // the following line was added by Makoto Matsumoto in the 1996 version
        // to improve lower bit's corellation.
        // Delete this line to use the code published in 1994.

        y ^= (y >> 16); /* added to the 1994 version */
        k++;
    }
 
    double random()
    { 
        unsigned long y1 = randomUint32();
        unsigned long y2 = randomUint32();

        return low_ + length_ * ((y1 * f) * y2 * f);
    } 

protected:
    void generate()
    {
        /* generate N words at one time */
        int kk;
        for (kk=0;kk<N-M;kk++) {
            x[kk] = x[kk+M] ^ (x[kk] >> 1) ^ mag01[x[kk] % 2];
        }
        for (; kk<N;kk++) {
            x[kk] = x[kk+(M-N)] ^ (x[kk] >> 1) ^ mag01[x[kk] % 2];
        }
        k=0;
    }

private:
    enum { N = 25, M = 7 };

    double low_, length_;
    double f;
    int k;
    unsigned long x[N];
    unsigned long mag01[2];
};

BZ_NAMESPACE_END

#endif // BZ_RAND_TT800_H

