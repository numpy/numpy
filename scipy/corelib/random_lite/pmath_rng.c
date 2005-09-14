/* pmath_rng.c
 *
 * This set of routines reproduces the Cray RANF family of random
 * number routines bit-for-bit.  Most of the routines here were
 * written by Jim Rathkopf, CP-Division, LLNL, and modified by
 * William C. Biester, B-Division. LLNL.  They were subsequently
 * modified for use in PMATH by Fred N. Fritsch, LC, LLNL.
 *
 * This version contains only the generator proper (equivalent to PM_RANF8
 * in the official PMATH version) and makes the seed and multiplier
 * directly accessible as arrays of doubles.  This was modified for use by
 * the Basis and Python module ranf.c, which contains its own seed and
 * multiplier management routines.  (FNF -- 9/4/96)
 */

/* -----------------------------------------------------------------------

Copyright (c) 1993,1994,1995,1996.
The Regents of the University of California.  All rights reserved.

This work was produced at the University of California, Lawrence
Livermore National Laboratory (UC LLNL) under contract no. W-7405-ENG-48
between the U.S. Department of Energy (DOE) and The Regents of the
University of California (University) for the operation of UC LLNL.
Copyright is reserved to the University for purposes of controlled
dissemination, commercialization through formal licensing, or other
disposition under terms of Contract 48; DOE policies, regulations and
orders; and U.S. statutes.

                         DISCLAIMER
 
This document was prepared as an account of work sponsored by an agency
of the United States Government.  Neither the United States Government
nor the University of California nor any of their employees, makes any
warranty, express or implied, or assumes any liability or responsibility
for the accuracy, completeness, or usefulness of any information,
apparatus, product, or process disclosed, or represents that its use
would not infringe privately-owned rights.  Reference herein to any
specific commercial products, process, or service by trade name,
trademark, manufacturer, or otherwise, does not necessarily constitute
or imply its endorsement, recommendation, or favoring by the United
States Government or the University of California.  The views and
opinions of authors expressed herein do not necessarily state or reflect
those of the United States Government or the University of California,
and shall not be used for advertising or product endorsement purposes.
*/

/* -----------------------------------------------------------------------
 *
 * Service routines defined:
 *    PM_16to24 - Convert 3 16-bit shorts to 2 24-bit doubles
 *    PM_24to16 - Convert 2 24-bit doubles to 3 16-bit shorts
 *    PM_GSeed  - Get the seed as two 24-bit doubles
 *    PM_SSeed  - Set the seed from two 24-bit doubles (unsafe)
 *    PM_GMult  - Get the multiplier as two 24-bit doubles
 *    PM_SMult  - Set the multiplier from two 24-bit doubles (unsafe)
 *
 * User-callable routines defined:
 *    PM_RANF   - The generator itself
 *
 * Currently responsible person:
 *    Fred N. Fritsch
 *    Computer Applications Organization
 *    LCPD, ICF Group
 *    Lawrence Livermore National Laboratory
 *    fnf@llnl.gov
 *
 * Modifications:
 * (See individual SLATEC-format prologues for detailed modification records.)
 *    03-09-93  Integrated Jim's routines into my library system.  (WCB)
 *    10-21-93  Modified to provide RANF8 for PMATH.  (FNF)
 *    10-22-93  Added code for RNFCNT.  (FNF)
 *    11-09-93  Added code for RNMSET.  (FNF)
 *    12-15-93  Modified to process bytes of seeds and multipliers in
 *              in the reverse order, to agree with the way they come
 *              from the Cray.  (FNF)
 *    03-15-94  Changed general pm_binds.h to pm_rnfset.h.  (FNF)
 *    04-25-94  Added SLATEC-format prologues. (FNF)
 *    09-26-95  Simplified code and improved internal documentation.
 *              This included removing unnecessary internal procedures
 *              setseed and seed48. (FNF)
 *    09-27-95  Simplified further by eliminating internal procedure
 *              rnset and no longer used variable Multiplier.  Also,
 *              moved PM_RANF8 so that it appears first. (FNF)
 *    09-28-95  Modified behavior of RNMSET to agree with MATHLIB. (FNF)
 *    10-02-95  Eliminated unions via calls to CV16TO64 and CV64TO16.
 *              Most conditional compilations are now in the rand48_*
 *              procedures. (FNF)
 *    10-05-95  Corrected "overflow" problem to make code work on Cray.
 *              Removed some unnecessary code in the process. (FNF)
 *    10-09-95  Removed conditional compiles.  (No longer needed after
 *              change of 10-02-95.) (FNF)
 *    08-27-96  Removed code for PM_RNSSET, PM_RNMSET, PM_RNSGET, PM_RNFCNT.
 *              Changed name PM_RANF8 to PM_RANF.  Added simple interface
 *              routines to get/set seed and multiplier.  Removed dependency
 *              on CV16TO64 and CV64TO16.  Eliminated Fortran bindings.  (FNF)
 *    09-04-96  Improved descriptions of the get/set routines.  (FNF)
 *    09-05-96  Added definitions of internal procedures PM_16to24 and
 *              PM_24to16 (see ranf.h).  (FNF)
 * -----------------------------------------------------------------------
 */



   /* Get includes, typedefs, and protypes for ranf.c and pmath_rng.c */
#include "ranf.h"

#define BASE24        16777216.0  /* 2^24 */

#define TWO8               256.0  /* 2^8 */
#define TWO16            65536.0  /* 2^16 */
#define TWO48  281474976710656.0  /* 2^48 */

/* ------------------------------------------------------
   The following static global variables retain the state
   of the random number generator between calls.
   ------------------------------------------------------ */
/* The default values to reproduce Cray values */
static double dseed[2] = {16555217.0, 9732691.0} ;
static double dmult[2] = {15184245.0, 2651554.0} ;
static double dadder = 0.0;    /* This is a relic from drand48. */


/* --------------------------------------
      Define in-line mod function.
   -------------------------------------- */
#define MODF(r,m)  (r - (floor(r/m))*m)


/* -----------------------------------------
      Define PMATH RNG internal procedures.
   -----------------------------------------
*/

/* PM_16to24 -- Take a number stored in 3 16-bit unsigned shorts and move it
 *              to 2 doubles each containing 24 bits of data.
 * 
 * This is a local function.
 *
 * Calling sequence:
 *   u16 x16[3];
 *   double x24[2];
 *   PM_16to24 (x16, x24);
 *
 *   x16 :   the 3 16-bit unsigned shorts.
 *   x24 :=  the 2 doubles of 24 bits returned.
 *
 * Return value:
 *   none.
 *
 * Note:
 *   This is the same as PMATH internal procedure rand48_16to24 except
 *   that the order of the entries in x16 have been reversed and its
 *   type declaration has been changed to u16.  It is intended for use
 *   by user-callable routines that set seeds or multipliers for users.
 */
void PM_16to24(u16 x16[3], double x24[2])
{
   double t0, t1, t2, t1u, t1l;

   t0 = (double) x16[0];
   t1 = (double) x16[1];
   t2 = (double) x16[2];

   t1u = floor(t1/TWO8);
   t1l = t1 -t1u*TWO8;
   x24[0] = t0 + t1l*TWO16;
   x24[1] = t1u + t2*TWO8;

#ifdef RAN_DEBUG
fprintf(stderr,"PM_16to24: x16 = %04x %04x %04x\n",x16[2],x16[1],x16[0]);
fprintf(stderr,"PM_16to24: x24 = %.1f %.1f\n",x24[1],x24[0]);
#endif
   return;
}


/* PM_24to16 -- Take a number stored in 2 doubles each containing 24 bits
 *              of data and move it to 3 16-bit unsigned shorts. 
 *
 * Calling sequence:
 *   double x24[2];
 *   u16 x16[3];
 *   PM_24to16 (x24, x16);
 *
 *   x24 :   the 2 doubles of 24 bits.
 *   x16 :=  the 3 16-bit unsigned shorts returned.
 *
 * Return value:
 *   none.
 *
 * Note:
 *   This is the same as PMATH internal procedure rand48_24to16 except
 *   that the order of the entries in x16 have been reversed and its
 *   type declaration has been changed to u16.  It is intended for use
 *   by user-callable routines that get seeds or multipliers for users.
 */
void PM_24to16(double x24[2], u16 x16[3])
{
   double t0u, t0l, t1u, t1l;

   t0u = floor(x24[0]/TWO16);
   t0l = x24[0] - t0u*TWO16;
   x16[0] = (u16)t0l;

   t1u = floor(x24[1]/TWO8);
   x16[2] = (u16)t1u;
   t1l = x24[1] - t1u*TWO8;
   x16[1] = (u16)(t0u + t1l*TWO8);

#ifdef RAN_DEBUG
fprintf(stderr,"PM_24to16: x24 = %.1f %.1f\n",x24[1],x24[0]);
fprintf(stderr,"PM_24to16: x16 = %04x %04x %04x\n",x16[2],x16[1],x16[0]);
#endif
   return;
}

/*   End of internal procedure definitions.
   -----------------------------------------
*/



/* --------------------------------------
      Define service routines.
   --------------------------------------
*/

/* PM_GSeed -- Get the seed as 2 doubles each containing 24 bits of data.
 * 
 * Calling sequence:
 *   double myseed[2];
 *   PM_GSeed(myseed);
 *
 *   myseed :=  the 2 doubles of 24 bits returned.
 *
 * Return value:
 *   none.
 */
void PM_GSeed(double seedout[2])
{
   seedout[0] = dseed[0];
   seedout[1] = dseed[1];

#ifdef RAN_DEBUG
   fprintf(stderr,"PM_GSeed: seed = %.1f %.1f\n", dseed[1], dseed[0]);
#endif
   return;
}

/* PM_SSeed -- Set the seed from 2 doubles each containing 24 bits of data.
 * 
 * Calling sequence:
 *   double myseed[2];
 *   PM_SSeed(myseed);
 *
 *   myseed :  the 2 doubles of 24 bits to be used as the new seed.
 *
 * Return value:
 *   none.
 *
 * Caution:
 *   This routine does not check for valid seeds!  It is intended for
 *   use in a safe interface such as Setranf (in ranf.c).
 *   The elements of the myseed array are the coefficients in the base
 *   2**24 representation of the odd 48-bit integer
 *         seed =  myseed[1]*2**24 + myseed[0],
 *   so myseed must satisfy:
 *     (1) myseed[0] and myseed[1] are integers, stored in doubles;
 *     (2) 0 < myseed[0] < 2**24, 0 <= myseed[1] < 2**24;
 *     (3) myseed[0] is odd.
 */
void PM_SSeed(double seedin[2])
{
   dseed[0] = seedin[0];
   dseed[1] = seedin[1];

#ifdef RAN_DEBUG
   fprintf(stderr,"PM_SSeed: seed = %.1f %.1f\n", dseed[1], dseed[0]);
#endif
   return;
}


/* PM_GMult -- Get the multiplier as 2 doubles each containing 24 bits 
 *             of data.
 * 
 * Calling sequence:
 *   double mymult[2];
 *   PM_GMult(mymult);
 *
 *   mymult :=  the 2 doubles of 24 bits returned.
 *
 * Return value:
 *   none.
 */
void PM_GMult(double multout[2])
{
   multout[0] = dmult[0];
   multout[1] = dmult[1];

#ifdef RAN_DEBUG
   fprintf(stderr,"PM_GMult: mult = %.1f %.1f\n", dmult[1], dmult[0]);
#endif
   return;
}

/* PM_SMult -- Set the multiplier from 2 doubles each containing 24 bits 
 *             of data.
 * 
 * Calling sequence:
 *   double mymult[2];
 *   PM_SMult(mymult);
 *
 *   mymult :  the 2 doubles of 24 bits to be used as the new multiplier.
 *
 * Return value:
 *   none.
 *
 * Caution:
 *   This routine does not check for valid multipliers!  It is intended for
 *   use in a safe interface such as Setmult (in ranf.c).
 *   The elements of the mymult array are the coefficients in the base
 *   2**24 representation of the odd 48-bit integer
 *         mult =  mymult[1]*2**24 + mymult[0].
 *   Since we require  1 < mult < 2**46, mymult must satisfy:
 *     (1) mymult[0] and mymult[1] are integers, stored in doubles;
 *     (2) 1 < mymult[0] < 2**24, 0 <= mymult[1] < 2**22;
 *     (3) mymult[0] is odd.
 */
void PM_SMult(double multin[2])
{
   dmult[0] = multin[0];
   dmult[1] = multin[1];

#ifdef RAN_DEBUG
   fprintf(stderr,"PM_SMult: mult = %.1f %.1f\n", dmult[1], dmult[0]);
#endif
   return;
}



/* --------------------------------------
      Define the generator itself.
   --------------------------------------
*/

/*DECK PM_RANF
C***BEGIN PROLOGUE  PM_RANF
C***PURPOSE  Uniform random-number generator.
C            The pseudorandom numbers generated by C function PM_RANF
C            are uniformly distributed in the open interval (0,1).
C***LIBRARY   PMATH
C***CATEGORY  L6A21
C***TYPE      REAL*8 (PM_RANF-8)
C***KEYWORDS  RANDOM NUMBER GENERATION, UNIFORM DISTRIBUTION
C***AUTHOR  Rathkopf, Jim, (LLNL/CP-Division)
C           Fritsch, Fred N., (LLNL/LC/MSS)
C***DESCRIPTION
C     (Portable version of Cray MATHLIB routine RANF.)
C     (Equivalent to Fortran-callable RANF8.)
C *Usage:
C        double r;
C        r = PM_RANF();
C
C *Function Return Values:
C     r        A random number between 0 and 1.
C
C *Description:
C     PM_RANF generates pseudorandom numbers lying strictly between 0
C     and 1. Each call to PM_RANF produces a different value, until the
C     sequence cycles after 2**46 calls.
C
C     PM_RANF is a linear congruential pseudorandom-number generator.
C     The default starting seed is
C                SEED = 4510112377116321(oct) = 948253fc9cd1(hex).
C     The multiplier is 1207264271730565(oct) = 2875a2e7b175(hex).
C
C *See Also:
C     This is the same generator used by the Fortran generator family
C     SRANF/DRANF/RANF8.
C     The starting seed for PM_RANF may be set via PM_SSeed.
C     The current PM_RANF seed may be obtained from PM_GSeed.
C     The current PM_RANF multiplier may be obtained from PM_GMult.
C     The PM_RANF multiplier may be set via PM_SMult (changing the
C     multiplier is not recommended).
C
C***ROUTINES CALLED  (NONE)
C***REVISION HISTORY  (YYMMDD)
C   930308  DATE WRITTEN
C           (Date from Biester's math_rnf.c.)
C   931021  Changed name from B_RANF to PM_RANF8.  (FNF)
C   940425  Added SLATEC-format prologue.  (FNF)
C   950922  Re-ordered code to eliminate old EMULRAND routine. (FNF)
C   951005  Eliminated unnecessary upper part computations.
C           Added fmod invocations in t2_48 computation to avoid
C           generating numbers too large for Cray floating point. (FNF)
C   951020  Replaced fmod with macro MODF to avoid library calls. (FNF)
C   951025  Removed MODF from first term of t2_48. (FNF)
C   960827  Changed name from PM_RANF8 to PM_RANF, eliminated reference
C           counter, and removed Fortran binding.  (FNF)
C***END PROLOGUE  PM_RANF
C
C*Internal notes:
C
C    This routine generates pseudo-random numbers through a 48-bit
C    linear congruential algorithm.  This emulates the drand48 library
C    of random number generators available on many, but not all,
C    UNIX machines.
C
C Algorithm:
C    x(n+1) = (a*x(n) + c)  mod m
C
C   where the defaults for the standard UNIX rand48 are:
C
C                   double name    hexdecimal          decimal
C    x: seed       -dseed[0],[1]  1234abcd330e      20017429951246
C    a: multiplier -dmult[0],[1]     5deece66d         25214903917
C    c: adder      -dadder                   b                  11
C    m: 2**48                    1000000000000     281474976710656
C
C     24-bit defaults (decimal) (lower bits listed first)
C    x: dseed[0],[1] = 13447950.0, 1193131.0
C    a: dmult[0],[1] = 15525485.0, 1502.0
C    c: dadder       = 11.0
C
C   The Cray defaults used in this code are:
C
C                   double name    hexdecimal           octal
C    x: seed       -dseed[0],[1]  948253fc9cd1    4510112377116321
C    a: multiplier -dmult[0],[1]  2875a2e7b175    1207264271730565
C    c: adder      -dadder                   0                   0
C    m: 2**48                    1000000000000   10000000000000000
C
C     24-bit defaults (decimal) (lower bits listed first)
C    x: dseed[0],[1] = 16555217.0, 9732691.0
C    a: dmult[0],[1] = 15184245.0, 2651554.0
C    c: dadder       = 0.0
C
C Return value:
C   double random number such that 0.0< d <1.0
C
C**End
 */
double PM_RANF (void)
{
   double t1_48, t2_48, t1_24[2], t2_24;
   double d;

   /* perform 48-bit arithmetic using two part data */
   t1_48 = dmult[0]*dseed[0] + dadder;
   t2_48 = dmult[1]*dseed[0] + MODF(dmult[0]*dseed[1],BASE24);
       /* First term safe, since dmult[1] < 2**22 for default mult */

   t1_24[1] = floor(t1_48/BASE24);       /* upper part */
   t1_24[0] = t1_48 - t1_24[1]*BASE24;   /* lower part */

   t2_24 = MODF(t2_48, BASE24);      /* lower part */

   t2_48 = t2_24 + t1_24[1];

   t2_24 = MODF(t2_48, BASE24);      /* discard anything over 2**24 */

   d = (dseed[0] + dseed[1]*BASE24)/TWO48;

   dseed[0] = t1_24[0];
   dseed[1] = t2_24;

   return (d);
}


