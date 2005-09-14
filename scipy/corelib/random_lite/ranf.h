#include <stdlib.h>
#include <math.h>
#ifdef RAN_DEBUG
#include <stdio.h>
#endif

#ifdef __MWERKS__
/*#include <utime.h>*/
#include <time.h>
#include <Timer.h>
#else
#if defined(_WIN32)
#include <time.h>
#else
#include <sys/time.h>
#endif
#endif

typedef unsigned int u32;
typedef unsigned short int u16;
typedef double f64;

/*  Prototypes for routines defined in ranf.c  */
void Seedranf(u32 *s);            /* Set seed from 32-bit integer */
void Mixranf(int *s, u32 s48[2]); /* Set seed, with options; return seed */
void Getranf(u32 s48[2]);         /* Get 48-bit seed in integer array */
void Setranf(u32 s48[2]);         /* Set seed from 48-bit integer */
void Getmult(u32 m48[2]);         /* Get 48-bit multiplier in integer array */
void Setmult(u32 m48[2]);         /* Set multiplier from 48-bit integer */
f64  Ranf(void);                  /* The generator itself */

/*  Prototypes for routines defined in pmath_rng.c  */
void PM_16to24(u16 x16[3], double x24[2]);
                              /* Convert 3 16-bit shorts to 2 24-bit doubles */
void PM_24to16(double x24[2], u16 x16[3]);
                              /* Convert 2 24-bit doubles to 3 16-bit shorts */
void PM_GSeed(double seedout[2]);  /* Get the current seed */
void PM_SSeed(double seedin[2]);   /* Reset the seed (unsafe) */
void PM_GMult(double multout[2]);  /* Get the current multiplier */
void PM_SMult(double multin[2]);   /* Reset the multiplier (unsafe) */
f64  PM_RANF(void);                /* The generator itself */
