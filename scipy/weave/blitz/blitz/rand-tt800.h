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
 ***************************************************************************/

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

