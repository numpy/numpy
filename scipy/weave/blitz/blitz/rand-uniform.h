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
 ***************************************************************************/

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

