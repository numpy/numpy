/* A C-program for MT19937: Integer version (1998/4/6)            */
/*  genrand() generates one pseudorandom unsigned integer (32bit) */
/* which is uniformly distributed among 0 to 2^32-1  for each     */
/* call. sgenrand(seed) set initial values to the working area    */
/* of 624 words. Before genrand(), sgenrand(seed) must be         */
/* called once. (seed is any 32-bit integer except for 0).        */
/*   Coded by Takuji Nishimura, considering the suggestions by    */
/* Topher Cooper and Marc Rieffel in July-Aug. 1997.              */

/* This library is free software; you can redistribute it and/or   */
/* modify it under the terms of the GNU Library General Public     */
/* License as published by the Free Software Foundation; either    */
/* version 2 of the License, or (at your option) any later         */
/* version.                                                        */
/* This library is distributed in the hope that it will be useful, */
/* but WITHOUT ANY WARRANTY; without even the implied warranty of  */
/* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.            */
/* See the GNU Library General Public License for more details.    */
/* You should have received a copy of the GNU Library General      */
/* Public License along with this library; if not, write to the    */
/* Free Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA   */ 
/* 02111-1307  USA                                                 */

/* Copyright (C) 1997 Makoto Matsumoto and Takuji Nishimura.       */
/* When you use this, send an email to: matumoto@math.keio.ac.jp   */
/* with an appropriate reference to your work.                     */

/* REFERENCE                                                       */
/* M. Matsumoto and T. Nishimura,                                  */
/* "Mersenne Twister: A 623-Dimensionally Equidistributed Uniform  */
/* Pseudo-Random Number Generator",                                */
/* ACM Transactions on Modeling and Computer Simulation,           */
/* Vol. 8, No. 1, January 1998, pp 3--30.                          */

// See http://www.math.keio.ac.jp/~matumoto/emt.html

// 1999-01-25 adapted to STL-like idiom
// allan@stokes.ca (Allan Stokes) www.stokes.ca

#ifndef BZ_RAND_MT
#define BZ_RAND_MT

#ifndef BZ_BLITZ_H
 #include <blitz/blitz.h>
#endif

#include <vector>

BZ_NAMESPACE(blitz)

// decomposition issues:
//   machine representation of integer types
//   output buffer option verses inline post-conditioning

class MersenneTwister
{
private:
  typedef unsigned int twist_int; // must be at least 32 bits
                                  // larger might be faster
  typedef vector<twist_int> State;
  typedef State::iterator Iter;

  struct BitMixer {
    enum { K = 0x9908b0df };
    BitMixer() : s0(0) {}
    inline friend twist_int low_mask (twist_int s1) {
      return (s1&1u) ? K : 0u;
    }
    inline twist_int high_mask (twist_int s1) const {
      return ((s0&0x80000000)|(s1&0x7fffffff))>>1;
    }
    inline twist_int operator() (twist_int s1) {
      twist_int r = high_mask(s1) ^ low_mask(s1);
      s0 = s1;
      return r;
    }
    twist_int s0;
  };

enum { N = 624, PF = 397, reference_seed = 4357 }; 
  
public: 
  MersenneTwister () {} // S empty will trigger auto-seed

  void seed (twist_int seed = reference_seed)
  {
    if (!S.size()) S.resize (N);
    enum { Knuth_A = 69069 }; 
    twist_int x = seed & 0xFFFFFFFF;
    Iter s = &S[0];
    twist_int mask = (seed == reference_seed) ? 0 : 0xFFFFFFFF;
    for (int j = 0; j < N; ++j) {
      // adding j here avoids the risk of all zeros 
      // we suppress this term in "compatibility" mode  
      *s++ = (x + (mask & j)) & 0xFFFFFFFF; 
      x *= Knuth_A;
    }
  }

  void reload (void)
  {
    if (!S.size()) seed (); // auto-seed detection

    Iter p0 = &S[0];
    Iter pM = p0 + PF;
    BitMixer twist;
    twist (S[0]); // prime the pump
    for (Iter pf_end = &S[N-PF]; p0 != pf_end; ++p0, ++pM)
      *p0 = *pM ^ twist (p0[1]);
    pM = S.begin();
    for (Iter s_end = &S[N-1]; p0 != s_end; ++p0, ++pM)
      *p0 = *pM ^ twist (p0[1]);
    *p0 = *pM ^ twist (S[0]);

    I = &S[0];
  }

  inline twist_int random (void)
  {
    if (I >= S.end()) reload();
    twist_int y = *I++;
    y ^= (y >> 11);
    y ^= (y <<  7) & 0x9D2C5680;
    y ^= (y << 15) & 0xEFC60000;
    y ^= (y >> 18);
    return y;
  }

private:
  State   S;
  Iter    I;
};


// This version returns a double in the range [0,1).

class MersenneTwisterDouble {

public:
  MersenneTwisterDouble()
  {
      // f = 1/(2^32);
      f = (1.0 / 65536) / 65536;
  }

  void randomize(unsigned int seed)
  {
      gen_.seed(seed);
  }

  double random()
  {
      unsigned long y1 = gen_.random();
      unsigned long y2 = gen_.random();

      return ((y1 * f) * y2 * f);
  }

private:
  MersenneTwister gen_;
  double f;
};

BZ_NAMESPACE_END

#endif // BZ_RAND_MT
