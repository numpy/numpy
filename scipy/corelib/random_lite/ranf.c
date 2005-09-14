/* ranf.c

  The routines below implement an interface based on the drand48 family of
  random number routines, but with seed handling and default starting seed
  compatible with the CRI MATHLIB random number generator (RNG) family "ranf".
*/

/* -----------------------------------------------------------------------
 *
 * User-callable routines defined:
 *    Seedranf - Set seed from 32-bit integer
 *    Mixranf  - Set seed, with options; return seed
 *    Getranf  - Get 48-bit seed in integer array
 *    Setranf  - Set seed from 48-bit integer
 *    Getmult  - Get 48-bit multiplier in integer array
 *    Setmult  - Set multiplier from 48-bit integer
 *    Ranf     - The generator itself
 *
 * Currently responsible person:
 *    Fred N. Fritsch
 *    Computer Applications Organization
 *    LCPD, ICF Group
 *    Lawrence Livermore National Laboratory
 *    fnf@llnl.gov
 *
 * Revision history:
 *   (yymmdd)
 *    91????  DATE WRITTEN
 *            This started with the ranf.c that was checked out of the
 *            Basis repository in August 1996.  It was written by Lee
 *            Busby to provide an interface for Basis to the drand48
 *            family of generators that is compatible with the CRI
 *            MATHLIB ranf.  Following are the relevant cvs log entries.
 *    950511  Added back seedranf, setranf, getranf, mixranf from older
 *            Basis version. (LB)
 *    950530  Changed type of u32 from long to int, which is currently ok
 *            on all our supported platforms. (LB)
 *    960823  Revised to use the portable PMATH generator instead. (FNF)
 *    960903  Added new routines Getmult and Setmult. (FNF)
 *    960904  Changed declarations for 48-bit quantities from pointers to
 *            two-element arrays, to be consistent with actual usage, and
 *            brought internal documentation up to LDOC standard. (FNF)
 *    960905  Moved definitions of internal procedures PM_16to24 and
 *            PM_24to16 to pmath_rng.c (see ranf.h).  (FNF)
 *    960911  Corrected some problems with return values from Mixranf.
 *            Also improved calling sequence descriptions for Seedranf and
 *            Mixranf. (FNF)
 *    960916  Eliminated state variable ranf_started.  Since the PMATH
 *            family has the Cray default starting values built in, this
 *            is no longer needed. (FNF)
 *    961011  Modifed Setranf and Setmult to work OK on Cray's when given
 *            values saved on a 32-bit workstation. (FNF)
 *    961212  Corrected dimension error in Getmult.  (FNF per E. Brooks)
 *    
 * -----------------------------------------------------------------------
 */

#ifdef __cplusplus
extern "C" {
#endif

   /* Get includes, typedefs, and protypes for ranf.c and pmath_rng.c */
#include "ranf.h"


/* Seedranf - Reset the RNG, given a new (32-bit) seed value.
 *
 * Calling sequence:
 *   u32 s;
 *   Seedranf(&s);
 *
 *   s :  (pointer to) the new (32-bit) seed value.
 *
 * Return value:
 *   none.
 *
 * Note:
 *   The upper 16 bits of the 48-bit seed are set to zero and then
 *   Setranf is called with the result.  This is provided primarily
 *   as a user convenience.
 */
void Seedranf(u32 *s)
{
    u32 s48[2];

    s48[0] = *s;
    s48[1] = 0x0;
#ifdef RAN_DEBUG
fprintf(stderr,"Seedranf set 48-bit seed  %08x %08x\n",s48[1],s48[0]);
#endif
    (void)Setranf(s48);
}


/* Mixranf - Reset the RNG with a new 48-bit seed, with options.
 *
 * Calling sequence:
 *   int s;
 *   u32 s48[2];
 *   Mixranf(&s,s48);
 *
 *   s :     the value used to determine the new seed, as follows:
 *             s < 0  ==>  Use the default initial seed value.
 *             s = 0  ==>  Set a "random" value for the seed from the
 *                         system clock.
 *             s > 0  ==>  Set seed directly (32 bits only).
 *   s48 :=  the new 48-bit seed, returned in an array of u32's.
 *           (The upper 16 bits of s48[1] will be zero.)
 *
 * Return value:
 *   none.
 *
 * Note:
 *   In case of a positive s, all of s48[1] will be zero.  In order to set
 *   a seed with more than 32 bits, use Setranf directly.
 *
 * Portability:
 *   The structs timeval and timezone and routine gettimeofday are required
 *   when s=0.  These are generally available in <sys/time.h>.
 */
void Mixranf(int *s,u32 s48[2])
{
    if(*s < 0){ /* Set default initial value */
	s48[0] = s48[1] = 0;
        Setranf(s48);
        Getranf(s48);  /* Return default seed */

    }else if(*s == 0){ /* Use the clock */
	int i;
#ifdef __MWERKS__
/* use this for the Macintosh */
	time_t theTime;
	UnsignedWide	tv_usec;

	(void)time(&theTime);
	Microseconds(&tv_usec); /* the microsecs since start up */
	s48[0] = (u32)theTime;
	s48[1] = (u32)tv_usec.lo;
#else
#if defined(_WIN32)
    // suggestions welcome for something better!
    // one of these is from start of job, the other time of day
	time_t long_time;
	clock_t clock_time;
	time(&long_time);
	clock_time = clock();
	s48[0] = (u32)long_time;
	s48[1] = (u32)clock_time;
#else
	struct timeval tv;
	struct timezone tz;
#if !defined(__sgi)
	int gettimeofday(struct timeval *, struct timezone *);
#endif

	(void)gettimeofday(&tv,&tz);
	s48[0] = (u32)tv.tv_sec;
	s48[1] = (u32)tv.tv_usec;
#endif /* !_WIN32 */
#endif /* !__MWERKS__ */
	Setranf(s48);
	for(i=0;i<10;i++) (void)Ranf();  /* Discard first 10 numbers */
        Getranf(s48);  /* Return seed after these 10 calls */
    }else{ /* s > 0, so set seed directly */
        s48[0] = (u32)*s;
        s48[1] = 0x0;
        Setranf(s48);
        Getranf(s48);  /* Return (possibly) corrected value of seed */
    }
#ifdef RAN_DEBUG
fprintf(stderr,"Mixranf set 48-bit seed  %08x %08x\n",s48[1],s48[0]);
#endif
}

 
/* Getranf - Get the 48-bit seed value.
 *
 * Calling sequence:
 *   u32 s48[2];
 *   Getranf(s48);
 *
 *   s48 :=  the current 48-bit seed, stored in an array of u32's.
 *           (The upper 16 bits of s48[1] will be zero.)
 *
 * Return value:
 *   none.
 */
void Getranf(u32 s48[2])
{
    u16 p[3];
    double pm_seed[2];

    /* Get the PMATH seed in an array of doubles. */
    PM_GSeed(pm_seed);

    /* Convert it to an array of u16's */
    PM_24to16(pm_seed, p);

/* Now put all the bits in the right place.  The picture below shows
   s48[0], the rightmost 16 bits of s48[1], and all 16 bits of p[0],
   p[1], and p[2] as aligned after assignment.

  ---------------------------------------------------------------------
  |        s48[1]        |                    s48[0]                  |
  ---------------------------------------------------------------------
  15                    0|31                  16|15                   0
  ---------------------------------------------------------------------
  |         p[2]         |         p[1]         |        p[0]         |
  ---------------------------------------------------------------------

*/

    s48[0] = p[1];
    s48[0] = (s48[0] << 16) + p[0];
    s48[1] = p[2];
#ifdef RAN_DEBUG
fprintf(stderr,"Getranf returns %08x %08x\n",s48[1],s48[0]);
#endif
}


/* Setranf - Reset the RNG seed from a 48-bit integer value.
 *
 * Calling sequence:
 *   u32 s48[2];
 *   Setranf(s48);
 *
 *   s48 :  the new 48-bit seed, contained in an array of u32's.
 *          If s48 is all zeros, set the default starting value,
 *          948253fc9cd1 (hex).
 *
 * Return value:
 *   none.
 *
 * Note:
 *   A 1 is masked into the lowest bit position to make sure the seed
 *   is odd.  (The upper 16 bits of s48[1] will be ignored if nonzero.)
 */
void Setranf(u32 s48[2])
{
    u16 p[3];
    double pm_seed[2];

#ifdef RAN_DEBUG
fprintf(stderr,"Setranf called with s48 = %08x %08x\n",s48[1],s48[0]);
#endif

    if(s48[0] == 0 && s48[1] == 0){ /* Set default starting value */
	s48[0] = 0x53fc9cd1;
	s48[1] = 0x9482;
    }

    /* Store seed as 3 u16's. */
    p[0] = (s48[0] & 0xffff) | 0x1; /* Force an odd number */
    p[1] = (s48[0] >> 16) & 0xffff; /* Protect against sign extension on Cray*/
    p[2] = s48[1];

    /* Convert to PMATH seed. */
    PM_16to24(p, pm_seed);

    /* Now reset the seed. */
    PM_SSeed(pm_seed);

#ifdef RAN_DEBUG
fprintf(stderr,"Leaving Setranf\n");
#endif
}

 
/* Getmult - Get the current 48-bit multiplier.
 *
 * Calling sequence:
 *   u32 m48[2];
 *   Getmult(m48);
 *
 *   m48 :=  the current 48-bit multiplier, stored in an array of u32's.
 *           (The upper 16 bits of m48[1] will be zero.)
 *
 * Return value:
 *   none.
 */
void Getmult(u32 m48[2])
{
    u16 p[3];
    double pm_mult[2];

    /* Get the PMATH multiplier in an array of doubles. */
    PM_GMult(pm_mult);

    /* Convert it to an array of u16's */
    PM_24to16(pm_mult, p);

/* Now put all the bits in the right place.  (See Getranf for picture.) */

    m48[0] = p[1];
    m48[0] = (m48[0] << 16) + p[0];
    m48[1] = p[2];
#ifdef RAN_DEBUG
fprintf(stderr,"Getmult returns %08x %08x\n",m48[1],m48[0]);
#endif
}


/* Setmult - Reset the RNG multiplier from a 48-bit integer value.
 *
 * Calling sequence:
 *   u32 m48[2];
 *   Setmult(m48);
 *
 *   m48 :  the new 48-bit multiplier, contained in an array of u32's.
 *          If m48 is all zeros, set the default starting value,
 *          2875a2e7b175 (hex).
 *
 * Return value:
 *   none.
 *
 * Note:
 *   A 1 is masked into the lowest bit position to make sure the value
 *   is odd.  (The upper 18 bits of m48[1] will be ignored if nonzero.)
 */
void Setmult(u32 *m48)
{
    u16 p[3];
    double pm_mult[2];

#ifdef RAN_DEBUG
fprintf(stderr,"Setmult called with m48 = %08x %08x\n",m48[1],m48[0]);
#endif
    if(m48[0] == 0 && m48[1] == 0){ /* Set default starting value */
	m48[0] = 0xa2e7b175;
	m48[1] = 0x2875;
    }

    /* Store multiplier as 3 u16's. */
    p[0] = (m48[0] & 0xffff) | 0x1; /* Force an odd number */
    p[1] = (m48[0] >> 16) & 0xffff; /* Protect against sign extension on Cray*/
    p[2] = m48[1] & 0x3fff;       /* Force mult < 2**26 */

    /* Convert to PMATH multiplier. */
    PM_16to24(p, pm_mult);

    /* Now reset the multiplier. */
    PM_SMult(pm_mult);

#ifdef RAN_DEBUG
fprintf(stderr,"Leaving Setmult\n");
#endif
}


/* Ranf - Uniform random number generator.
 *
 * Calling sequence:
 *   f64 value;
 *   value = Ranf();
 *
 * Return value:
 *   value :=  the next number in the ranf sequence.
 *
 * Note:
 *   The Ranf function is just a shell around PM_RANF.  Since the
 *   PMATH generator has the Cray default starting values built in,
 *   no initialization is needed.
 */
f64 Ranf(void)
{
    return(PM_RANF());
}


#ifdef __cplusplus
}
#endif

