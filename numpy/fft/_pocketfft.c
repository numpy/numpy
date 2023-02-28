/*
 * This file is part of pocketfft.
 * Licensed under a 3-clause BSD style license - see LICENSE.md
 */

/*
 *  Main implementation file.
 *
 *  Copyright (C) 2004-2018 Max-Planck-Society
 *  \author Martin Reinecke
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "numpy/arrayobject.h"

#include "npy_config.h"

#include <math.h>
#include <string.h>
#include <stdlib.h>

#define RALLOC(type,num) \
  (assert(num != 0), ((type *)malloc((num)*sizeof(type))))
#define DEALLOC(ptr) \
  do { free(ptr); (ptr)=NULL; } while(0)

#define SWAP(a,b,type) \
  do { type tmp_=(a); (a)=(b); (b)=tmp_; } while(0)

#ifdef __GNUC__
#define NOINLINE __attribute__((noinline))
#define WARN_UNUSED_RESULT __attribute__ ((warn_unused_result))
#else
#define NOINLINE
#define WARN_UNUSED_RESULT
#endif

struct cfft_plan_i;
typedef struct cfft_plan_i * cfft_plan;
struct rfft_plan_i;
typedef struct rfft_plan_i * rfft_plan;

// adapted from https://stackoverflow.com/questions/42792939/
// CAUTION: this function only works for arguments in the range [-0.25; 0.25]!
static void my_sincosm1pi (double a, double *restrict res)
  {
  double s = a * a;
  /* Approximate cos(pi*x)-1 for x in [-0.25,0.25] */
  double r =     -1.0369917389758117e-4;
  r = fma (r, s,  1.9294935641298806e-3);
  r = fma (r, s, -2.5806887942825395e-2);
  r = fma (r, s,  2.3533063028328211e-1);
  r = fma (r, s, -1.3352627688538006e+0);
  r = fma (r, s,  4.0587121264167623e+0);
  r = fma (r, s, -4.9348022005446790e+0);
  double c = r*s;
  /* Approximate sin(pi*x) for x in [-0.25,0.25] */
  r =             4.6151442520157035e-4;
  r = fma (r, s, -7.3700183130883555e-3);
  r = fma (r, s,  8.2145868949323936e-2);
  r = fma (r, s, -5.9926452893214921e-1);
  r = fma (r, s,  2.5501640398732688e+0);
  r = fma (r, s, -5.1677127800499516e+0);
  s = s * a;
  r = r * s;
  s = fma (a, 3.1415926535897931e+0, r);
  res[0] = c;
  res[1] = s;
  }

NOINLINE static void calc_first_octant(size_t den, double * restrict res)
  {
  size_t n = (den+4)>>3;
  if (n==0) return;
  res[0]=1.; res[1]=0.;
  if (n==1) return;
  size_t l1=(size_t)sqrt(n);
  for (size_t i=1; i<l1; ++i)
    my_sincosm1pi((2.*i)/den,&res[2*i]);
  size_t start=l1;
  while(start<n)
    {
    double cs[2];
    my_sincosm1pi((2.*start)/den,cs);
    res[2*start] = cs[0]+1.;
    res[2*start+1] = cs[1];
    size_t end = l1;
    if (start+end>n) end = n-start;
    for (size_t i=1; i<end; ++i)
      {
      double csx[2]={res[2*i], res[2*i+1]};
      res[2*(start+i)] = ((cs[0]*csx[0] - cs[1]*csx[1] + cs[0]) + csx[0]) + 1.;
      res[2*(start+i)+1] = (cs[0]*csx[1] + cs[1]*csx[0]) + cs[1] + csx[1];
      }
    start += l1;
    }
  for (size_t i=1; i<l1; ++i)
    res[2*i] += 1.;
  }

NOINLINE static void calc_first_quadrant(size_t n, double * restrict res)
  {
  double * restrict p = res+n;
  calc_first_octant(n<<1, p);
  size_t ndone=(n+2)>>2;
  size_t i=0, idx1=0, idx2=2*ndone-2;
  for (; i+1<ndone; i+=2, idx1+=2, idx2-=2)
    {
    res[idx1]   = p[2*i];
    res[idx1+1] = p[2*i+1];
    res[idx2]   = p[2*i+3];
    res[idx2+1] = p[2*i+2];
    }
  if (i!=ndone)
    {
    res[idx1  ] = p[2*i];
    res[idx1+1] = p[2*i+1];
    }
  }

NOINLINE static void calc_first_half(size_t n, double * restrict res)
  {
  int ndone=(n+1)>>1;
  double * p = res+n-1;
  calc_first_octant(n<<2, p);
  int i4=0, in=n, i=0;
  for (; i4<=in-i4; ++i, i4+=4) // octant 0
    {
    res[2*i] = p[2*i4]; res[2*i+1] = p[2*i4+1];
    }
  for (; i4-in <= 0; ++i, i4+=4) // octant 1
    {
    int xm = in-i4;
    res[2*i] = p[2*xm+1]; res[2*i+1] = p[2*xm];
    }
  for (; i4<=3*in-i4; ++i, i4+=4) // octant 2
    {
    int xm = i4-in;
    res[2*i] = -p[2*xm+1]; res[2*i+1] = p[2*xm];
    }
  for (; i<ndone; ++i, i4+=4) // octant 3
    {
    int xm = 2*in-i4;
    res[2*i] = -p[2*xm]; res[2*i+1] = p[2*xm+1];
    }
  }

NOINLINE static void fill_first_quadrant(size_t n, double * restrict res)
  {
  const double hsqt2 = 0.707106781186547524400844362104849;
  size_t quart = n>>2;
  if ((n&7)==0)
    res[quart] = res[quart+1] = hsqt2;
  for (size_t i=2, j=2*quart-2; i<quart; i+=2, j-=2)
    {
    res[j  ] = res[i+1];
    res[j+1] = res[i  ];
    }
  }

NOINLINE static void fill_first_half(size_t n, double * restrict res)
  {
  size_t half = n>>1;
  if ((n&3)==0)
    for (size_t i=0; i<half; i+=2)
      {
      res[i+half]   = -res[i+1];
      res[i+half+1] =  res[i  ];
      }
  else
    for (size_t i=2, j=2*half-2; i<half; i+=2, j-=2)
      {
      res[j  ] = -res[i  ];
      res[j+1] =  res[i+1];
      }
  }

NOINLINE static void fill_second_half(size_t n, double * restrict res)
  {
  if ((n&1)==0)
    for (size_t i=0; i<n; ++i)
      res[i+n] = -res[i];
  else
    for (size_t i=2, j=2*n-2; i<n; i+=2, j-=2)
      {
      res[j  ] =  res[i  ];
      res[j+1] = -res[i+1];
      }
  }

NOINLINE static void sincos_2pibyn_half(size_t n, double * restrict res)
  {
  if ((n&3)==0)
    {
    calc_first_octant(n, res);
    fill_first_quadrant(n, res);
    fill_first_half(n, res);
    }
  else if ((n&1)==0)
    {
    calc_first_quadrant(n, res);
    fill_first_half(n, res);
    }
  else
    calc_first_half(n, res);
  }

NOINLINE static void sincos_2pibyn(size_t n, double * restrict res)
  {
  sincos_2pibyn_half(n, res);
  fill_second_half(n, res);
  }

NOINLINE static size_t largest_prime_factor (size_t n)
  {
  size_t res=1;
  size_t tmp;
  while (((tmp=(n>>1))<<1)==n)
    { res=2; n=tmp; }

  size_t limit=(size_t)sqrt(n+0.01);
  for (size_t x=3; x<=limit; x+=2)
  while (((tmp=(n/x))*x)==n)
    {
    res=x;
    n=tmp;
    limit=(size_t)sqrt(n+0.01);
    }
  if (n>1) res=n;

  return res;
  }

NOINLINE static double cost_guess (size_t n)
  {
  const double lfp=1.1; // penalty for non-hardcoded larger factors
  size_t ni=n;
  double result=0.;
  size_t tmp;
  while (((tmp=(n>>1))<<1)==n)
    { result+=2; n=tmp; }

  size_t limit=(size_t)sqrt(n+0.01);
  for (size_t x=3; x<=limit; x+=2)
  while ((tmp=(n/x))*x==n)
    {
    result+= (x<=5) ? x : lfp*x; // penalize larger prime factors
    n=tmp;
    limit=(size_t)sqrt(n+0.01);
    }
  if (n>1) result+=(n<=5) ? n : lfp*n;

  return result*ni;
  }

/* returns the smallest composite of 2, 3, 5, 7 and 11 which is >= n */
NOINLINE static size_t good_size(size_t n)
  {
  if (n<=6) return n;

  size_t bestfac=2*n;
  for (size_t f2=1; f2<bestfac; f2*=2)
    for (size_t f23=f2; f23<bestfac; f23*=3)
      for (size_t f235=f23; f235<bestfac; f235*=5)
        for (size_t f2357=f235; f2357<bestfac; f2357*=7)
          for (size_t f235711=f2357; f235711<bestfac; f235711*=11)
            if (f235711>=n) bestfac=f235711;
  return bestfac;
  }

typedef struct cmplx {
  double r,i;
} cmplx;

#define NFCT 25
typedef struct cfftp_fctdata
  {
  size_t fct;
  cmplx *tw, *tws;
  } cfftp_fctdata;

typedef struct cfftp_plan_i
  {
  size_t length, nfct;
  cmplx *mem;
  cfftp_fctdata fct[NFCT];
  } cfftp_plan_i;
typedef struct cfftp_plan_i * cfftp_plan;

#define PMC(a,b,c,d) { a.r=c.r+d.r; a.i=c.i+d.i; b.r=c.r-d.r; b.i=c.i-d.i; }
#define ADDC(a,b,c) { a.r=b.r+c.r; a.i=b.i+c.i; }
#define SCALEC(a,b) { a.r*=b; a.i*=b; }
#define ROT90(a) { double tmp_=a.r; a.r=-a.i; a.i=tmp_; }
#define ROTM90(a) { double tmp_=-a.r; a.r=a.i; a.i=tmp_; }
#define CH(a,b,c) ch[(a)+ido*((b)+l1*(c))]
#define CC(a,b,c) cc[(a)+ido*((b)+cdim*(c))]
#define WA(x,i) wa[(i)-1+(x)*(ido-1)]
/* a = b*c */
#define A_EQ_B_MUL_C(a,b,c) { a.r=b.r*c.r-b.i*c.i; a.i=b.r*c.i+b.i*c.r; }
/* a = conj(b)*c*/
#define A_EQ_CB_MUL_C(a,b,c) { a.r=b.r*c.r+b.i*c.i; a.i=b.r*c.i-b.i*c.r; }

#define PMSIGNC(a,b,c,d) { a.r=c.r+sign*d.r; a.i=c.i+sign*d.i; b.r=c.r-sign*d.r; b.i=c.i-sign*d.i; }
/* a = b*c */
#define MULPMSIGNC(a,b,c) { a.r=b.r*c.r-sign*b.i*c.i; a.i=b.r*c.i+sign*b.i*c.r; }
/* a *= b */
#define MULPMSIGNCEQ(a,b) { double xtmp=a.r; a.r=b.r*a.r-sign*b.i*a.i; a.i=b.r*a.i+sign*b.i*xtmp; }

NOINLINE static void pass2b (size_t ido, size_t l1, const cmplx * restrict cc,
  cmplx * restrict ch, const cmplx * restrict wa)
  {
  const size_t cdim=2;

  if (ido==1)
    for (size_t k=0; k<l1; ++k)
      PMC (CH(0,k,0),CH(0,k,1),CC(0,0,k),CC(0,1,k))
  else
    for (size_t k=0; k<l1; ++k)
      {
      PMC (CH(0,k,0),CH(0,k,1),CC(0,0,k),CC(0,1,k))
      for (size_t i=1; i<ido; ++i)
        {
        cmplx t;
        PMC (CH(i,k,0),t,CC(i,0,k),CC(i,1,k))
        A_EQ_B_MUL_C (CH(i,k,1),WA(0,i),t)
        }
      }
  }

NOINLINE static void pass2f (size_t ido, size_t l1, const cmplx * restrict cc,
  cmplx * restrict ch, const cmplx * restrict wa)
  {
  const size_t cdim=2;

  if (ido==1)
    for (size_t k=0; k<l1; ++k)
      PMC (CH(0,k,0),CH(0,k,1),CC(0,0,k),CC(0,1,k))
  else
    for (size_t k=0; k<l1; ++k)
      {
      PMC (CH(0,k,0),CH(0,k,1),CC(0,0,k),CC(0,1,k))
      for (size_t i=1; i<ido; ++i)
        {
        cmplx t;
        PMC (CH(i,k,0),t,CC(i,0,k),CC(i,1,k))
        A_EQ_CB_MUL_C (CH(i,k,1),WA(0,i),t)
        }
      }
  }

#define PREP3(idx) \
        cmplx t0 = CC(idx,0,k), t1, t2; \
        PMC (t1,t2,CC(idx,1,k),CC(idx,2,k)) \
        CH(idx,k,0).r=t0.r+t1.r; \
        CH(idx,k,0).i=t0.i+t1.i;
#define PARTSTEP3a(u1,u2,twr,twi) \
        { \
        cmplx ca,cb; \
        ca.r=t0.r+twr*t1.r; \
        ca.i=t0.i+twr*t1.i; \
        cb.i=twi*t2.r; \
        cb.r=-(twi*t2.i); \
        PMC(CH(0,k,u1),CH(0,k,u2),ca,cb) \
        }

#define PARTSTEP3b(u1,u2,twr,twi) \
        { \
        cmplx ca,cb,da,db; \
        ca.r=t0.r+twr*t1.r; \
        ca.i=t0.i+twr*t1.i; \
        cb.i=twi*t2.r; \
        cb.r=-(twi*t2.i); \
        PMC(da,db,ca,cb) \
        A_EQ_B_MUL_C (CH(i,k,u1),WA(u1-1,i),da) \
        A_EQ_B_MUL_C (CH(i,k,u2),WA(u2-1,i),db) \
        }
NOINLINE static void pass3b (size_t ido, size_t l1, const cmplx * restrict cc,
  cmplx * restrict ch, const cmplx * restrict wa)
  {
  const size_t cdim=3;
  const double tw1r=-0.5, tw1i= 0.86602540378443864676;

  if (ido==1)
    for (size_t k=0; k<l1; ++k)
      {
      PREP3(0)
      PARTSTEP3a(1,2,tw1r,tw1i)
      }
  else
    for (size_t k=0; k<l1; ++k)
      {
      {
      PREP3(0)
      PARTSTEP3a(1,2,tw1r,tw1i)
      }
      for (size_t i=1; i<ido; ++i)
        {
        PREP3(i)
        PARTSTEP3b(1,2,tw1r,tw1i)
        }
      }
  }
#define PARTSTEP3f(u1,u2,twr,twi) \
        { \
        cmplx ca,cb,da,db; \
        ca.r=t0.r+twr*t1.r; \
        ca.i=t0.i+twr*t1.i; \
        cb.i=twi*t2.r; \
        cb.r=-(twi*t2.i); \
        PMC(da,db,ca,cb) \
        A_EQ_CB_MUL_C (CH(i,k,u1),WA(u1-1,i),da) \
        A_EQ_CB_MUL_C (CH(i,k,u2),WA(u2-1,i),db) \
        }
NOINLINE static void pass3f (size_t ido, size_t l1, const cmplx * restrict cc,
  cmplx * restrict ch, const cmplx * restrict wa)
  {
  const size_t cdim=3;
  const double tw1r=-0.5, tw1i= -0.86602540378443864676;

  if (ido==1)
    for (size_t k=0; k<l1; ++k)
      {
      PREP3(0)
      PARTSTEP3a(1,2,tw1r,tw1i)
      }
  else
    for (size_t k=0; k<l1; ++k)
      {
      {
      PREP3(0)
      PARTSTEP3a(1,2,tw1r,tw1i)
      }
      for (size_t i=1; i<ido; ++i)
        {
        PREP3(i)
        PARTSTEP3f(1,2,tw1r,tw1i)
        }
      }
  }

NOINLINE static void pass4b (size_t ido, size_t l1, const cmplx * restrict cc,
  cmplx * restrict ch, const cmplx * restrict wa)
  {
  const size_t cdim=4;

  if (ido==1)
    for (size_t k=0; k<l1; ++k)
      {
      cmplx t1, t2, t3, t4;
      PMC(t2,t1,CC(0,0,k),CC(0,2,k))
      PMC(t3,t4,CC(0,1,k),CC(0,3,k))
      ROT90(t4)
      PMC(CH(0,k,0),CH(0,k,2),t2,t3)
      PMC(CH(0,k,1),CH(0,k,3),t1,t4)
      }
  else
    for (size_t k=0; k<l1; ++k)
      {
      {
      cmplx t1, t2, t3, t4;
      PMC(t2,t1,CC(0,0,k),CC(0,2,k))
      PMC(t3,t4,CC(0,1,k),CC(0,3,k))
      ROT90(t4)
      PMC(CH(0,k,0),CH(0,k,2),t2,t3)
      PMC(CH(0,k,1),CH(0,k,3),t1,t4)
      }
      for (size_t i=1; i<ido; ++i)
        {
        cmplx c2, c3, c4, t1, t2, t3, t4;
        cmplx cc0=CC(i,0,k), cc1=CC(i,1,k),cc2=CC(i,2,k),cc3=CC(i,3,k);
        PMC(t2,t1,cc0,cc2)
        PMC(t3,t4,cc1,cc3)
        ROT90(t4)
        cmplx wa0=WA(0,i), wa1=WA(1,i),wa2=WA(2,i);
        PMC(CH(i,k,0),c3,t2,t3)
        PMC(c2,c4,t1,t4)
        A_EQ_B_MUL_C (CH(i,k,1),wa0,c2)
        A_EQ_B_MUL_C (CH(i,k,2),wa1,c3)
        A_EQ_B_MUL_C (CH(i,k,3),wa2,c4)
        }
      }
  }
NOINLINE static void pass4f (size_t ido, size_t l1, const cmplx * restrict cc,
  cmplx * restrict ch, const cmplx * restrict wa)
  {
  const size_t cdim=4;

  if (ido==1)
    for (size_t k=0; k<l1; ++k)
      {
      cmplx t1, t2, t3, t4;
      PMC(t2,t1,CC(0,0,k),CC(0,2,k))
      PMC(t3,t4,CC(0,1,k),CC(0,3,k))
      ROTM90(t4)
      PMC(CH(0,k,0),CH(0,k,2),t2,t3)
      PMC(CH(0,k,1),CH(0,k,3),t1,t4)
      }
  else
    for (size_t k=0; k<l1; ++k)
      {
      {
      cmplx t1, t2, t3, t4;
      PMC(t2,t1,CC(0,0,k),CC(0,2,k))
      PMC(t3,t4,CC(0,1,k),CC(0,3,k))
      ROTM90(t4)
      PMC(CH(0,k,0),CH(0,k,2),t2,t3)
      PMC (CH(0,k,1),CH(0,k,3),t1,t4)
      }
      for (size_t i=1; i<ido; ++i)
        {
        cmplx c2, c3, c4, t1, t2, t3, t4;
        cmplx cc0=CC(i,0,k), cc1=CC(i,1,k),cc2=CC(i,2,k),cc3=CC(i,3,k);
        PMC(t2,t1,cc0,cc2)
        PMC(t3,t4,cc1,cc3)
        ROTM90(t4)
        cmplx wa0=WA(0,i), wa1=WA(1,i),wa2=WA(2,i);
        PMC(CH(i,k,0),c3,t2,t3)
        PMC(c2,c4,t1,t4)
        A_EQ_CB_MUL_C (CH(i,k,1),wa0,c2)
        A_EQ_CB_MUL_C (CH(i,k,2),wa1,c3)
        A_EQ_CB_MUL_C (CH(i,k,3),wa2,c4)
        }
      }
  }

#define PREP5(idx) \
        cmplx t0 = CC(idx,0,k), t1, t2, t3, t4; \
        PMC (t1,t4,CC(idx,1,k),CC(idx,4,k)) \
        PMC (t2,t3,CC(idx,2,k),CC(idx,3,k)) \
        CH(idx,k,0).r=t0.r+t1.r+t2.r; \
        CH(idx,k,0).i=t0.i+t1.i+t2.i;

#define PARTSTEP5a(u1,u2,twar,twbr,twai,twbi) \
        { \
        cmplx ca,cb; \
        ca.r=t0.r+twar*t1.r+twbr*t2.r; \
        ca.i=t0.i+twar*t1.i+twbr*t2.i; \
        cb.i=twai*t4.r twbi*t3.r; \
        cb.r=-(twai*t4.i twbi*t3.i); \
        PMC(CH(0,k,u1),CH(0,k,u2),ca,cb) \
        }

#define PARTSTEP5b(u1,u2,twar,twbr,twai,twbi) \
        { \
        cmplx ca,cb,da,db; \
        ca.r=t0.r+twar*t1.r+twbr*t2.r; \
        ca.i=t0.i+twar*t1.i+twbr*t2.i; \
        cb.i=twai*t4.r twbi*t3.r; \
        cb.r=-(twai*t4.i twbi*t3.i); \
        PMC(da,db,ca,cb) \
        A_EQ_B_MUL_C (CH(i,k,u1),WA(u1-1,i),da) \
        A_EQ_B_MUL_C (CH(i,k,u2),WA(u2-1,i),db) \
        }
NOINLINE static void pass5b (size_t ido, size_t l1, const cmplx * restrict cc,
  cmplx * restrict ch, const cmplx * restrict wa)
  {
  const size_t cdim=5;
  const double tw1r= 0.3090169943749474241,
               tw1i= 0.95105651629515357212,
               tw2r= -0.8090169943749474241,
               tw2i= 0.58778525229247312917;

  if (ido==1)
    for (size_t k=0; k<l1; ++k)
      {
      PREP5(0)
      PARTSTEP5a(1,4,tw1r,tw2r,+tw1i,+tw2i)
      PARTSTEP5a(2,3,tw2r,tw1r,+tw2i,-tw1i)
      }
  else
    for (size_t k=0; k<l1; ++k)
      {
      {
      PREP5(0)
      PARTSTEP5a(1,4,tw1r,tw2r,+tw1i,+tw2i)
      PARTSTEP5a(2,3,tw2r,tw1r,+tw2i,-tw1i)
      }
      for (size_t i=1; i<ido; ++i)
        {
        PREP5(i)
        PARTSTEP5b(1,4,tw1r,tw2r,+tw1i,+tw2i)
        PARTSTEP5b(2,3,tw2r,tw1r,+tw2i,-tw1i)
        }
      }
  }
#define PARTSTEP5f(u1,u2,twar,twbr,twai,twbi) \
        { \
        cmplx ca,cb,da,db; \
        ca.r=t0.r+twar*t1.r+twbr*t2.r; \
        ca.i=t0.i+twar*t1.i+twbr*t2.i; \
        cb.i=twai*t4.r twbi*t3.r; \
        cb.r=-(twai*t4.i twbi*t3.i); \
        PMC(da,db,ca,cb) \
        A_EQ_CB_MUL_C (CH(i,k,u1),WA(u1-1,i),da) \
        A_EQ_CB_MUL_C (CH(i,k,u2),WA(u2-1,i),db) \
        }
NOINLINE static void pass5f (size_t ido, size_t l1, const cmplx * restrict cc,
  cmplx * restrict ch, const cmplx * restrict wa)
  {
  const size_t cdim=5;
  const double tw1r= 0.3090169943749474241,
               tw1i= -0.95105651629515357212,
               tw2r= -0.8090169943749474241,
               tw2i= -0.58778525229247312917;

  if (ido==1)
    for (size_t k=0; k<l1; ++k)
      {
      PREP5(0)
      PARTSTEP5a(1,4,tw1r,tw2r,+tw1i,+tw2i)
      PARTSTEP5a(2,3,tw2r,tw1r,+tw2i,-tw1i)
      }
  else
    for (size_t k=0; k<l1; ++k)
      {
      {
      PREP5(0)
      PARTSTEP5a(1,4,tw1r,tw2r,+tw1i,+tw2i)
      PARTSTEP5a(2,3,tw2r,tw1r,+tw2i,-tw1i)
      }
      for (size_t i=1; i<ido; ++i)
        {
        PREP5(i)
        PARTSTEP5f(1,4,tw1r,tw2r,+tw1i,+tw2i)
        PARTSTEP5f(2,3,tw2r,tw1r,+tw2i,-tw1i)
        }
      }
  }

#define PREP7(idx) \
        cmplx t1 = CC(idx,0,k), t2, t3, t4, t5, t6, t7; \
        PMC (t2,t7,CC(idx,1,k),CC(idx,6,k)) \
        PMC (t3,t6,CC(idx,2,k),CC(idx,5,k)) \
        PMC (t4,t5,CC(idx,3,k),CC(idx,4,k)) \
        CH(idx,k,0).r=t1.r+t2.r+t3.r+t4.r; \
        CH(idx,k,0).i=t1.i+t2.i+t3.i+t4.i;

#define PARTSTEP7a0(u1,u2,x1,x2,x3,y1,y2,y3,out1,out2) \
        { \
        cmplx ca,cb; \
        ca.r=t1.r+x1*t2.r+x2*t3.r+x3*t4.r; \
        ca.i=t1.i+x1*t2.i+x2*t3.i+x3*t4.i; \
        cb.i=y1*t7.r y2*t6.r y3*t5.r; \
        cb.r=-(y1*t7.i y2*t6.i y3*t5.i); \
        PMC(out1,out2,ca,cb) \
        }
#define PARTSTEP7a(u1,u2,x1,x2,x3,y1,y2,y3) \
        PARTSTEP7a0(u1,u2,x1,x2,x3,y1,y2,y3,CH(0,k,u1),CH(0,k,u2))
#define PARTSTEP7(u1,u2,x1,x2,x3,y1,y2,y3) \
        { \
        cmplx da,db; \
        PARTSTEP7a0(u1,u2,x1,x2,x3,y1,y2,y3,da,db) \
        MULPMSIGNC (CH(i,k,u1),WA(u1-1,i),da) \
        MULPMSIGNC (CH(i,k,u2),WA(u2-1,i),db) \
        }

NOINLINE static void pass7(size_t ido, size_t l1, const cmplx * restrict cc,
  cmplx * restrict ch, const cmplx * restrict wa, const int sign)
  {
  const size_t cdim=7;
  const double tw1r= 0.623489801858733530525,
               tw1i= sign * 0.7818314824680298087084,
               tw2r= -0.222520933956314404289,
               tw2i= sign * 0.9749279121818236070181,
               tw3r= -0.9009688679024191262361,
               tw3i= sign * 0.4338837391175581204758;

  if (ido==1)
    for (size_t k=0; k<l1; ++k)
      {
      PREP7(0)
      PARTSTEP7a(1,6,tw1r,tw2r,tw3r,+tw1i,+tw2i,+tw3i)
      PARTSTEP7a(2,5,tw2r,tw3r,tw1r,+tw2i,-tw3i,-tw1i)
      PARTSTEP7a(3,4,tw3r,tw1r,tw2r,+tw3i,-tw1i,+tw2i)
      }
  else
    for (size_t k=0; k<l1; ++k)
      {
      {
      PREP7(0)
      PARTSTEP7a(1,6,tw1r,tw2r,tw3r,+tw1i,+tw2i,+tw3i)
      PARTSTEP7a(2,5,tw2r,tw3r,tw1r,+tw2i,-tw3i,-tw1i)
      PARTSTEP7a(3,4,tw3r,tw1r,tw2r,+tw3i,-tw1i,+tw2i)
      }
      for (size_t i=1; i<ido; ++i)
        {
        PREP7(i)
        PARTSTEP7(1,6,tw1r,tw2r,tw3r,+tw1i,+tw2i,+tw3i)
        PARTSTEP7(2,5,tw2r,tw3r,tw1r,+tw2i,-tw3i,-tw1i)
        PARTSTEP7(3,4,tw3r,tw1r,tw2r,+tw3i,-tw1i,+tw2i)
        }
      }
  }

#define PREP11(idx) \
        cmplx t1 = CC(idx,0,k), t2, t3, t4, t5, t6, t7, t8, t9, t10, t11; \
        PMC (t2,t11,CC(idx,1,k),CC(idx,10,k)) \
        PMC (t3,t10,CC(idx,2,k),CC(idx, 9,k)) \
        PMC (t4,t9 ,CC(idx,3,k),CC(idx, 8,k)) \
        PMC (t5,t8 ,CC(idx,4,k),CC(idx, 7,k)) \
        PMC (t6,t7 ,CC(idx,5,k),CC(idx, 6,k)) \
        CH(idx,k,0).r=t1.r+t2.r+t3.r+t4.r+t5.r+t6.r; \
        CH(idx,k,0).i=t1.i+t2.i+t3.i+t4.i+t5.i+t6.i;

#define PARTSTEP11a0(u1,u2,x1,x2,x3,x4,x5,y1,y2,y3,y4,y5,out1,out2) \
        { \
        cmplx ca,cb; \
        ca.r=t1.r+x1*t2.r+x2*t3.r+x3*t4.r+x4*t5.r+x5*t6.r; \
        ca.i=t1.i+x1*t2.i+x2*t3.i+x3*t4.i+x4*t5.i+x5*t6.i; \
        cb.i=y1*t11.r y2*t10.r y3*t9.r y4*t8.r y5*t7.r; \
        cb.r=-(y1*t11.i y2*t10.i y3*t9.i y4*t8.i y5*t7.i ); \
        PMC(out1,out2,ca,cb) \
        }
#define PARTSTEP11a(u1,u2,x1,x2,x3,x4,x5,y1,y2,y3,y4,y5) \
        PARTSTEP11a0(u1,u2,x1,x2,x3,x4,x5,y1,y2,y3,y4,y5,CH(0,k,u1),CH(0,k,u2))
#define PARTSTEP11(u1,u2,x1,x2,x3,x4,x5,y1,y2,y3,y4,y5) \
        { \
        cmplx da,db; \
        PARTSTEP11a0(u1,u2,x1,x2,x3,x4,x5,y1,y2,y3,y4,y5,da,db) \
        MULPMSIGNC (CH(i,k,u1),WA(u1-1,i),da) \
        MULPMSIGNC (CH(i,k,u2),WA(u2-1,i),db) \
        }

NOINLINE static void pass11 (size_t ido, size_t l1, const cmplx * restrict cc,
  cmplx * restrict ch, const cmplx * restrict wa, const int sign)
  {
  const size_t cdim=11;
  const double tw1r =        0.8412535328311811688618,
               tw1i = sign * 0.5406408174555975821076,
               tw2r =        0.4154150130018864255293,
               tw2i = sign * 0.9096319953545183714117,
               tw3r =       -0.1423148382732851404438,
               tw3i = sign * 0.9898214418809327323761,
               tw4r =       -0.6548607339452850640569,
               tw4i = sign * 0.755749574354258283774,
               tw5r =       -0.9594929736144973898904,
               tw5i = sign * 0.2817325568414296977114;

  if (ido==1)
    for (size_t k=0; k<l1; ++k)
      {
      PREP11(0)
      PARTSTEP11a(1,10,tw1r,tw2r,tw3r,tw4r,tw5r,+tw1i,+tw2i,+tw3i,+tw4i,+tw5i)
      PARTSTEP11a(2, 9,tw2r,tw4r,tw5r,tw3r,tw1r,+tw2i,+tw4i,-tw5i,-tw3i,-tw1i)
      PARTSTEP11a(3, 8,tw3r,tw5r,tw2r,tw1r,tw4r,+tw3i,-tw5i,-tw2i,+tw1i,+tw4i)
      PARTSTEP11a(4, 7,tw4r,tw3r,tw1r,tw5r,tw2r,+tw4i,-tw3i,+tw1i,+tw5i,-tw2i)
      PARTSTEP11a(5, 6,tw5r,tw1r,tw4r,tw2r,tw3r,+tw5i,-tw1i,+tw4i,-tw2i,+tw3i)
      }
  else
    for (size_t k=0; k<l1; ++k)
      {
      {
      PREP11(0)
      PARTSTEP11a(1,10,tw1r,tw2r,tw3r,tw4r,tw5r,+tw1i,+tw2i,+tw3i,+tw4i,+tw5i)
      PARTSTEP11a(2, 9,tw2r,tw4r,tw5r,tw3r,tw1r,+tw2i,+tw4i,-tw5i,-tw3i,-tw1i)
      PARTSTEP11a(3, 8,tw3r,tw5r,tw2r,tw1r,tw4r,+tw3i,-tw5i,-tw2i,+tw1i,+tw4i)
      PARTSTEP11a(4, 7,tw4r,tw3r,tw1r,tw5r,tw2r,+tw4i,-tw3i,+tw1i,+tw5i,-tw2i)
      PARTSTEP11a(5, 6,tw5r,tw1r,tw4r,tw2r,tw3r,+tw5i,-tw1i,+tw4i,-tw2i,+tw3i)
      }
      for (size_t i=1; i<ido; ++i)
        {
        PREP11(i)
        PARTSTEP11(1,10,tw1r,tw2r,tw3r,tw4r,tw5r,+tw1i,+tw2i,+tw3i,+tw4i,+tw5i)
        PARTSTEP11(2, 9,tw2r,tw4r,tw5r,tw3r,tw1r,+tw2i,+tw4i,-tw5i,-tw3i,-tw1i)
        PARTSTEP11(3, 8,tw3r,tw5r,tw2r,tw1r,tw4r,+tw3i,-tw5i,-tw2i,+tw1i,+tw4i)
        PARTSTEP11(4, 7,tw4r,tw3r,tw1r,tw5r,tw2r,+tw4i,-tw3i,+tw1i,+tw5i,-tw2i)
        PARTSTEP11(5, 6,tw5r,tw1r,tw4r,tw2r,tw3r,+tw5i,-tw1i,+tw4i,-tw2i,+tw3i)
        }
      }
  }

#define CX(a,b,c) cc[(a)+ido*((b)+l1*(c))]
#define CX2(a,b) cc[(a)+idl1*(b)]
#define CH2(a,b) ch[(a)+idl1*(b)]

NOINLINE static int passg (size_t ido, size_t ip, size_t l1,
  cmplx * restrict cc, cmplx * restrict ch, const cmplx * restrict wa,
  const cmplx * restrict csarr, const int sign)
  {
  const size_t cdim=ip;
  size_t ipph = (ip+1)/2;
  size_t idl1 = ido*l1;

  cmplx * restrict wal=RALLOC(cmplx,ip);
  if (!wal) return -1;
  wal[0]=(cmplx){1.,0.};
  for (size_t i=1; i<ip; ++i)
    wal[i]=(cmplx){csarr[i].r,sign*csarr[i].i};

  for (size_t k=0; k<l1; ++k)
    for (size_t i=0; i<ido; ++i)
      CH(i,k,0) = CC(i,0,k);
  for (size_t j=1, jc=ip-1; j<ipph; ++j, --jc)
    for (size_t k=0; k<l1; ++k)
      for (size_t i=0; i<ido; ++i)
        PMC(CH(i,k,j),CH(i,k,jc),CC(i,j,k),CC(i,jc,k))
  for (size_t k=0; k<l1; ++k)
    for (size_t i=0; i<ido; ++i)
      {
      cmplx tmp = CH(i,k,0);
      for (size_t j=1; j<ipph; ++j)
        ADDC(tmp,tmp,CH(i,k,j))
      CX(i,k,0) = tmp;
      }
  for (size_t l=1, lc=ip-1; l<ipph; ++l, --lc)
    {
    // j=0
    for (size_t ik=0; ik<idl1; ++ik)
      {
      CX2(ik,l).r = CH2(ik,0).r+wal[l].r*CH2(ik,1).r+wal[2*l].r*CH2(ik,2).r;
      CX2(ik,l).i = CH2(ik,0).i+wal[l].r*CH2(ik,1).i+wal[2*l].r*CH2(ik,2).i;
      CX2(ik,lc).r=-wal[l].i*CH2(ik,ip-1).i-wal[2*l].i*CH2(ik,ip-2).i;
      CX2(ik,lc).i=wal[l].i*CH2(ik,ip-1).r+wal[2*l].i*CH2(ik,ip-2).r;
      }

    size_t iwal=2*l;
    size_t j=3, jc=ip-3;
    for (; j<ipph-1; j+=2, jc-=2)
      {
      iwal+=l; if (iwal>ip) iwal-=ip;
      cmplx xwal=wal[iwal];
      iwal+=l; if (iwal>ip) iwal-=ip;
      cmplx xwal2=wal[iwal];
      for (size_t ik=0; ik<idl1; ++ik)
        {
        CX2(ik,l).r += CH2(ik,j).r*xwal.r+CH2(ik,j+1).r*xwal2.r;
        CX2(ik,l).i += CH2(ik,j).i*xwal.r+CH2(ik,j+1).i*xwal2.r;
        CX2(ik,lc).r -= CH2(ik,jc).i*xwal.i+CH2(ik,jc-1).i*xwal2.i;
        CX2(ik,lc).i += CH2(ik,jc).r*xwal.i+CH2(ik,jc-1).r*xwal2.i;
        }
      }
    for (; j<ipph; ++j, --jc)
      {
      iwal+=l; if (iwal>ip) iwal-=ip;
      cmplx xwal=wal[iwal];
      for (size_t ik=0; ik<idl1; ++ik)
        {
        CX2(ik,l).r += CH2(ik,j).r*xwal.r;
        CX2(ik,l).i += CH2(ik,j).i*xwal.r;
        CX2(ik,lc).r -= CH2(ik,jc).i*xwal.i;
        CX2(ik,lc).i += CH2(ik,jc).r*xwal.i;
        }
      }
    }
  DEALLOC(wal);

  // shuffling and twiddling
  if (ido==1)
    for (size_t j=1, jc=ip-1; j<ipph; ++j, --jc)
      for (size_t ik=0; ik<idl1; ++ik)
        {
        cmplx t1=CX2(ik,j), t2=CX2(ik,jc);
        PMC(CX2(ik,j),CX2(ik,jc),t1,t2)
        }
  else
    {
    for (size_t j=1, jc=ip-1; j<ipph; ++j,--jc)
      for (size_t k=0; k<l1; ++k)
        {
        cmplx t1=CX(0,k,j), t2=CX(0,k,jc);
        PMC(CX(0,k,j),CX(0,k,jc),t1,t2)
        for (size_t i=1; i<ido; ++i)
          {
          cmplx x1, x2;
          PMC(x1,x2,CX(i,k,j),CX(i,k,jc))
          size_t idij=(j-1)*(ido-1)+i-1;
          MULPMSIGNC (CX(i,k,j),wa[idij],x1)
          idij=(jc-1)*(ido-1)+i-1;
          MULPMSIGNC (CX(i,k,jc),wa[idij],x2)
          }
        }
    }
  return 0;
  }

#undef CH2
#undef CX2
#undef CX

NOINLINE WARN_UNUSED_RESULT static int pass_all(cfftp_plan plan, cmplx c[], double fct,
  const int sign)
  {
  if (plan->length==1) return 0;
  size_t len=plan->length;
  size_t l1=1, nf=plan->nfct;
  cmplx *ch = RALLOC(cmplx, len);
  if (!ch) return -1;
  cmplx *p1=c, *p2=ch;

  for(size_t k1=0; k1<nf; k1++)
    {
    size_t ip=plan->fct[k1].fct;
    size_t l2=ip*l1;
    size_t ido = len/l2;
    if     (ip==4)
      sign>0 ? pass4b (ido, l1, p1, p2, plan->fct[k1].tw)
             : pass4f (ido, l1, p1, p2, plan->fct[k1].tw);
    else if(ip==2)
      sign>0 ? pass2b (ido, l1, p1, p2, plan->fct[k1].tw)
             : pass2f (ido, l1, p1, p2, plan->fct[k1].tw);
    else if(ip==3)
      sign>0 ? pass3b (ido, l1, p1, p2, plan->fct[k1].tw)
             : pass3f (ido, l1, p1, p2, plan->fct[k1].tw);
    else if(ip==5)
      sign>0 ? pass5b (ido, l1, p1, p2, plan->fct[k1].tw)
             : pass5f (ido, l1, p1, p2, plan->fct[k1].tw);
    else if(ip==7)  pass7 (ido, l1, p1, p2, plan->fct[k1].tw, sign);
    else if(ip==11) pass11(ido, l1, p1, p2, plan->fct[k1].tw, sign);
    else
      {
      if (passg(ido, ip, l1, p1, p2, plan->fct[k1].tw, plan->fct[k1].tws, sign))
        { DEALLOC(ch); return -1; }
      SWAP(p1,p2,cmplx *);
      }
    SWAP(p1,p2,cmplx *);
    l1=l2;
    }
  if (p1!=c)
    {
    if (fct!=1.)
      for (size_t i=0; i<len; ++i)
        {
        c[i].r = ch[i].r*fct;
        c[i].i = ch[i].i*fct;
        }
    else
      memcpy (c,p1,len*sizeof(cmplx));
    }
  else
    if (fct!=1.)
      for (size_t i=0; i<len; ++i)
        {
        c[i].r *= fct;
        c[i].i *= fct;
        }
  DEALLOC(ch);
  return 0;
  }

#undef PMSIGNC
#undef A_EQ_B_MUL_C
#undef A_EQ_CB_MUL_C
#undef MULPMSIGNC
#undef MULPMSIGNCEQ

#undef WA
#undef CC
#undef CH
#undef ROT90
#undef SCALEC
#undef ADDC
#undef PMC

NOINLINE WARN_UNUSED_RESULT
static int cfftp_forward(cfftp_plan plan, double c[], double fct)
  { return pass_all(plan,(cmplx *)c, fct, -1); }

NOINLINE WARN_UNUSED_RESULT
static int cfftp_backward(cfftp_plan plan, double c[], double fct)
  { return pass_all(plan,(cmplx *)c, fct, 1); }

NOINLINE WARN_UNUSED_RESULT
static int cfftp_factorize (cfftp_plan plan)
  {
  size_t length=plan->length;
  size_t nfct=0;
  while ((length%4)==0)
    { if (nfct>=NFCT) return -1; plan->fct[nfct++].fct=4; length>>=2; }
  if ((length%2)==0)
    {
    length>>=1;
    // factor 2 should be at the front of the factor list
    if (nfct>=NFCT) return -1;
    plan->fct[nfct++].fct=2;
    SWAP(plan->fct[0].fct, plan->fct[nfct-1].fct,size_t);
    }
  size_t maxl=(size_t)(sqrt((double)length))+1;
  for (size_t divisor=3; (length>1)&&(divisor<maxl); divisor+=2)
    if ((length%divisor)==0)
      {
      while ((length%divisor)==0)
        {
        if (nfct>=NFCT) return -1;
        plan->fct[nfct++].fct=divisor;
        length/=divisor;
        }
      maxl=(size_t)(sqrt((double)length))+1;
      }
  if (length>1) plan->fct[nfct++].fct=length;
  plan->nfct=nfct;
  return 0;
  }

NOINLINE static size_t cfftp_twsize (cfftp_plan plan)
  {
  size_t twsize=0, l1=1;
  for (size_t k=0; k<plan->nfct; ++k)
    {
    size_t ip=plan->fct[k].fct, ido= plan->length/(l1*ip);
    twsize+=(ip-1)*(ido-1);
    if (ip>11)
      twsize+=ip;
    l1*=ip;
    }
  return twsize;
  }

NOINLINE WARN_UNUSED_RESULT static int cfftp_comp_twiddle (cfftp_plan plan)
  {
  size_t length=plan->length;
  double *twid = RALLOC(double, 2*length);
  if (!twid) return -1;
  sincos_2pibyn(length, twid);
  size_t l1=1;
  size_t memofs=0;
  for (size_t k=0; k<plan->nfct; ++k)
    {
    size_t ip=plan->fct[k].fct, ido= length/(l1*ip);
    plan->fct[k].tw=plan->mem+memofs;
    memofs+=(ip-1)*(ido-1);
    for (size_t j=1; j<ip; ++j)
      for (size_t i=1; i<ido; ++i)
        {
        plan->fct[k].tw[(j-1)*(ido-1)+i-1].r = twid[2*j*l1*i];
        plan->fct[k].tw[(j-1)*(ido-1)+i-1].i = twid[2*j*l1*i+1];
        }
    if (ip>11)
      {
      plan->fct[k].tws=plan->mem+memofs;
      memofs+=ip;
      for (size_t j=0; j<ip; ++j)
        {
        plan->fct[k].tws[j].r = twid[2*j*l1*ido];
        plan->fct[k].tws[j].i = twid[2*j*l1*ido+1];
        }
      }
    l1*=ip;
    }
  DEALLOC(twid);
  return 0;
  }

static cfftp_plan make_cfftp_plan (size_t length)
  {
  if (length==0) return NULL;
  cfftp_plan plan = RALLOC(cfftp_plan_i,1);
  if (!plan) return NULL;
  plan->length=length;
  plan->nfct=0;
  for (size_t i=0; i<NFCT; ++i)
    plan->fct[i]=(cfftp_fctdata){0,0,0};
  plan->mem=0;
  if (length==1) return plan;
  if (cfftp_factorize(plan)!=0) { DEALLOC(plan); return NULL; }
  size_t tws=cfftp_twsize(plan);
  if (tws != 0) {
    plan->mem=RALLOC(cmplx,tws);
    if (!plan->mem) { DEALLOC(plan); return NULL; }
  }
  if (cfftp_comp_twiddle(plan)!=0)
    { DEALLOC(plan->mem); DEALLOC(plan); return NULL; }
  return plan;
  }

static void destroy_cfftp_plan (cfftp_plan plan)
  {
  DEALLOC(plan->mem);
  DEALLOC(plan);
  }

typedef struct rfftp_fctdata
  {
  size_t fct;
  double *tw, *tws;
  } rfftp_fctdata;

typedef struct rfftp_plan_i
  {
  size_t length, nfct;
  double *mem;
  rfftp_fctdata fct[NFCT];
  } rfftp_plan_i;
typedef struct rfftp_plan_i * rfftp_plan;

#define WA(x,i) wa[(i)+(x)*(ido-1)]
#define PM(a,b,c,d) { a=c+d; b=c-d; }
/* (a+ib) = conj(c+id) * (e+if) */
#define MULPM(a,b,c,d,e,f) { a=c*e+d*f; b=c*f-d*e; }

#define CC(a,b,c) cc[(a)+ido*((b)+l1*(c))]
#define CH(a,b,c) ch[(a)+ido*((b)+cdim*(c))]

NOINLINE static void radf2 (size_t ido, size_t l1, const double * restrict cc,
  double * restrict ch, const double * restrict wa)
  {
  const size_t cdim=2;

  for (size_t k=0; k<l1; k++)
    PM (CH(0,0,k),CH(ido-1,1,k),CC(0,k,0),CC(0,k,1))
  if ((ido&1)==0)
    for (size_t k=0; k<l1; k++)
      {
      CH(    0,1,k) = -CC(ido-1,k,1);
      CH(ido-1,0,k) =  CC(ido-1,k,0);
      }
  if (ido<=2) return;
  for (size_t k=0; k<l1; k++)
    for (size_t i=2; i<ido; i+=2)
      {
      size_t ic=ido-i;
      double tr2, ti2;
      MULPM (tr2,ti2,WA(0,i-2),WA(0,i-1),CC(i-1,k,1),CC(i,k,1))
      PM (CH(i-1,0,k),CH(ic-1,1,k),CC(i-1,k,0),tr2)
      PM (CH(i  ,0,k),CH(ic  ,1,k),ti2,CC(i  ,k,0))
      }
  }

NOINLINE static void radf3(size_t ido, size_t l1, const double * restrict cc,
  double * restrict ch, const double * restrict wa)
  {
  const size_t cdim=3;
  static const double taur=-0.5, taui=0.86602540378443864676;

  for (size_t k=0; k<l1; k++)
    {
    double cr2=CC(0,k,1)+CC(0,k,2);
    CH(0,0,k) = CC(0,k,0)+cr2;
    CH(0,2,k) = taui*(CC(0,k,2)-CC(0,k,1));
    CH(ido-1,1,k) = CC(0,k,0)+taur*cr2;
    }
  if (ido==1) return;
  for (size_t k=0; k<l1; k++)
    for (size_t i=2; i<ido; i+=2)
      {
      size_t ic=ido-i;
      double di2, di3, dr2, dr3;
      MULPM (dr2,di2,WA(0,i-2),WA(0,i-1),CC(i-1,k,1),CC(i,k,1)) // d2=conj(WA0)*CC1
      MULPM (dr3,di3,WA(1,i-2),WA(1,i-1),CC(i-1,k,2),CC(i,k,2)) // d3=conj(WA1)*CC2
      double cr2=dr2+dr3; // c add
      double ci2=di2+di3;
      CH(i-1,0,k) = CC(i-1,k,0)+cr2; // c add
      CH(i  ,0,k) = CC(i  ,k,0)+ci2;
      double tr2 = CC(i-1,k,0)+taur*cr2; // c add
      double ti2 = CC(i  ,k,0)+taur*ci2;
      double tr3 = taui*(di2-di3);  // t3 = taui*i*(d3-d2)?
      double ti3 = taui*(dr3-dr2);
      PM(CH(i-1,2,k),CH(ic-1,1,k),tr2,tr3) // PM(i) = t2+t3
      PM(CH(i  ,2,k),CH(ic  ,1,k),ti3,ti2) // PM(ic) = conj(t2-t3)
      }
  }

NOINLINE static void radf4(size_t ido, size_t l1, const double * restrict cc,
  double * restrict ch, const double * restrict wa)
  {
  const size_t cdim=4;
  static const double hsqt2=0.70710678118654752440;

  for (size_t k=0; k<l1; k++)
    {
    double tr1,tr2;
    PM (tr1,CH(0,2,k),CC(0,k,3),CC(0,k,1))
    PM (tr2,CH(ido-1,1,k),CC(0,k,0),CC(0,k,2))
    PM (CH(0,0,k),CH(ido-1,3,k),tr2,tr1)
    }
  if ((ido&1)==0)
    for (size_t k=0; k<l1; k++)
      {
      double ti1=-hsqt2*(CC(ido-1,k,1)+CC(ido-1,k,3));
      double tr1= hsqt2*(CC(ido-1,k,1)-CC(ido-1,k,3));
      PM (CH(ido-1,0,k),CH(ido-1,2,k),CC(ido-1,k,0),tr1)
      PM (CH(    0,3,k),CH(    0,1,k),ti1,CC(ido-1,k,2))
      }
  if (ido<=2) return;
  for (size_t k=0; k<l1; k++)
    for (size_t i=2; i<ido; i+=2)
      {
      size_t ic=ido-i;
      double ci2, ci3, ci4, cr2, cr3, cr4, ti1, ti2, ti3, ti4, tr1, tr2, tr3, tr4;
      MULPM(cr2,ci2,WA(0,i-2),WA(0,i-1),CC(i-1,k,1),CC(i,k,1))
      MULPM(cr3,ci3,WA(1,i-2),WA(1,i-1),CC(i-1,k,2),CC(i,k,2))
      MULPM(cr4,ci4,WA(2,i-2),WA(2,i-1),CC(i-1,k,3),CC(i,k,3))
      PM(tr1,tr4,cr4,cr2)
      PM(ti1,ti4,ci2,ci4)
      PM(tr2,tr3,CC(i-1,k,0),cr3)
      PM(ti2,ti3,CC(i  ,k,0),ci3)
      PM(CH(i-1,0,k),CH(ic-1,3,k),tr2,tr1)
      PM(CH(i  ,0,k),CH(ic  ,3,k),ti1,ti2)
      PM(CH(i-1,2,k),CH(ic-1,1,k),tr3,ti4)
      PM(CH(i  ,2,k),CH(ic  ,1,k),tr4,ti3)
      }
  }

NOINLINE static void radf5(size_t ido, size_t l1, const double * restrict cc,
  double * restrict ch, const double * restrict wa)
  {
  const size_t cdim=5;
  static const double tr11= 0.3090169943749474241, ti11=0.95105651629515357212,
                      tr12=-0.8090169943749474241, ti12=0.58778525229247312917;

  for (size_t k=0; k<l1; k++)
    {
    double cr2, cr3, ci4, ci5;
    PM (cr2,ci5,CC(0,k,4),CC(0,k,1))
    PM (cr3,ci4,CC(0,k,3),CC(0,k,2))
    CH(0,0,k)=CC(0,k,0)+cr2+cr3;
    CH(ido-1,1,k)=CC(0,k,0)+tr11*cr2+tr12*cr3;
    CH(0,2,k)=ti11*ci5+ti12*ci4;
    CH(ido-1,3,k)=CC(0,k,0)+tr12*cr2+tr11*cr3;
    CH(0,4,k)=ti12*ci5-ti11*ci4;
    }
  if (ido==1) return;
  for (size_t k=0; k<l1;++k)
    for (size_t i=2; i<ido; i+=2)
      {
      double ci2, di2, ci4, ci5, di3, di4, di5, ci3, cr2, cr3, dr2, dr3,
         dr4, dr5, cr5, cr4, ti2, ti3, ti5, ti4, tr2, tr3, tr4, tr5;
      size_t ic=ido-i;
      MULPM (dr2,di2,WA(0,i-2),WA(0,i-1),CC(i-1,k,1),CC(i,k,1))
      MULPM (dr3,di3,WA(1,i-2),WA(1,i-1),CC(i-1,k,2),CC(i,k,2))
      MULPM (dr4,di4,WA(2,i-2),WA(2,i-1),CC(i-1,k,3),CC(i,k,3))
      MULPM (dr5,di5,WA(3,i-2),WA(3,i-1),CC(i-1,k,4),CC(i,k,4))
      PM(cr2,ci5,dr5,dr2)
      PM(ci2,cr5,di2,di5)
      PM(cr3,ci4,dr4,dr3)
      PM(ci3,cr4,di3,di4)
      CH(i-1,0,k)=CC(i-1,k,0)+cr2+cr3;
      CH(i  ,0,k)=CC(i  ,k,0)+ci2+ci3;
      tr2=CC(i-1,k,0)+tr11*cr2+tr12*cr3;
      ti2=CC(i  ,k,0)+tr11*ci2+tr12*ci3;
      tr3=CC(i-1,k,0)+tr12*cr2+tr11*cr3;
      ti3=CC(i  ,k,0)+tr12*ci2+tr11*ci3;
      MULPM(tr5,tr4,cr5,cr4,ti11,ti12)
      MULPM(ti5,ti4,ci5,ci4,ti11,ti12)
      PM(CH(i-1,2,k),CH(ic-1,1,k),tr2,tr5)
      PM(CH(i  ,2,k),CH(ic  ,1,k),ti5,ti2)
      PM(CH(i-1,4,k),CH(ic-1,3,k),tr3,tr4)
      PM(CH(i  ,4,k),CH(ic  ,3,k),ti4,ti3)
      }
  }

#undef CC
#undef CH
#define C1(a,b,c) cc[(a)+ido*((b)+l1*(c))]
#define C2(a,b) cc[(a)+idl1*(b)]
#define CH2(a,b) ch[(a)+idl1*(b)]
#define CC(a,b,c) cc[(a)+ido*((b)+cdim*(c))]
#define CH(a,b,c) ch[(a)+ido*((b)+l1*(c))]
NOINLINE static void radfg(size_t ido, size_t ip, size_t l1,
  double * restrict cc, double * restrict ch, const double * restrict wa,
  const double * restrict csarr)
  {
  const size_t cdim=ip;
  size_t ipph=(ip+1)/2;
  size_t idl1 = ido*l1;

  if (ido>1)
    {
    for (size_t j=1, jc=ip-1; j<ipph; ++j,--jc)              // 114
      {
      size_t is=(j-1)*(ido-1),
             is2=(jc-1)*(ido-1);
      for (size_t k=0; k<l1; ++k)                            // 113
        {
        size_t idij=is;
        size_t idij2=is2;
        for (size_t i=1; i<=ido-2; i+=2)                      // 112
          {
          double t1=C1(i,k,j ), t2=C1(i+1,k,j ),
                 t3=C1(i,k,jc), t4=C1(i+1,k,jc);
          double x1=wa[idij]*t1 + wa[idij+1]*t2,
                 x2=wa[idij]*t2 - wa[idij+1]*t1,
                 x3=wa[idij2]*t3 + wa[idij2+1]*t4,
                 x4=wa[idij2]*t4 - wa[idij2+1]*t3;
          C1(i  ,k,j ) = x1+x3;
          C1(i  ,k,jc) = x2-x4;
          C1(i+1,k,j ) = x2+x4;
          C1(i+1,k,jc) = x3-x1;
          idij+=2;
          idij2+=2;
          }
        }
      }
    }

  for (size_t j=1, jc=ip-1; j<ipph; ++j,--jc)                // 123
    for (size_t k=0; k<l1; ++k)                              // 122
      {
      double t1=C1(0,k,j), t2=C1(0,k,jc);
      C1(0,k,j ) = t1+t2;
      C1(0,k,jc) = t2-t1;
      }

//everything in C
//memset(ch,0,ip*l1*ido*sizeof(double));

  for (size_t l=1,lc=ip-1; l<ipph; ++l,--lc)                 // 127
    {
    for (size_t ik=0; ik<idl1; ++ik)                         // 124
      {
      CH2(ik,l ) = C2(ik,0)+csarr[2*l]*C2(ik,1)+csarr[4*l]*C2(ik,2);
      CH2(ik,lc) = csarr[2*l+1]*C2(ik,ip-1)+csarr[4*l+1]*C2(ik,ip-2);
      }
    size_t iang = 2*l;
    size_t j=3, jc=ip-3;
    for (; j<ipph-3; j+=4,jc-=4)              // 126
      {
      iang+=l; if (iang>=ip) iang-=ip;
      double ar1=csarr[2*iang], ai1=csarr[2*iang+1];
      iang+=l; if (iang>=ip) iang-=ip;
      double ar2=csarr[2*iang], ai2=csarr[2*iang+1];
      iang+=l; if (iang>=ip) iang-=ip;
      double ar3=csarr[2*iang], ai3=csarr[2*iang+1];
      iang+=l; if (iang>=ip) iang-=ip;
      double ar4=csarr[2*iang], ai4=csarr[2*iang+1];
      for (size_t ik=0; ik<idl1; ++ik)                       // 125
        {
        CH2(ik,l ) += ar1*C2(ik,j )+ar2*C2(ik,j +1)
                     +ar3*C2(ik,j +2)+ar4*C2(ik,j +3);
        CH2(ik,lc) += ai1*C2(ik,jc)+ai2*C2(ik,jc-1)
                     +ai3*C2(ik,jc-2)+ai4*C2(ik,jc-3);
        }
      }
    for (; j<ipph-1; j+=2,jc-=2)              // 126
      {
      iang+=l; if (iang>=ip) iang-=ip;
      double ar1=csarr[2*iang], ai1=csarr[2*iang+1];
      iang+=l; if (iang>=ip) iang-=ip;
      double ar2=csarr[2*iang], ai2=csarr[2*iang+1];
      for (size_t ik=0; ik<idl1; ++ik)                       // 125
        {
        CH2(ik,l ) += ar1*C2(ik,j )+ar2*C2(ik,j +1);
        CH2(ik,lc) += ai1*C2(ik,jc)+ai2*C2(ik,jc-1);
        }
      }
    for (; j<ipph; ++j,--jc)              // 126
      {
      iang+=l; if (iang>=ip) iang-=ip;
      double ar=csarr[2*iang], ai=csarr[2*iang+1];
      for (size_t ik=0; ik<idl1; ++ik)                       // 125
        {
        CH2(ik,l ) += ar*C2(ik,j );
        CH2(ik,lc) += ai*C2(ik,jc);
        }
      }
    }
  for (size_t ik=0; ik<idl1; ++ik)                         // 101
    CH2(ik,0) = C2(ik,0);
  for (size_t j=1; j<ipph; ++j)                              // 129
    for (size_t ik=0; ik<idl1; ++ik)                         // 128
      CH2(ik,0) += C2(ik,j);

// everything in CH at this point!
//memset(cc,0,ip*l1*ido*sizeof(double));

  for (size_t k=0; k<l1; ++k)                                // 131
    for (size_t i=0; i<ido; ++i)                             // 130
      CC(i,0,k) = CH(i,k,0);

  for (size_t j=1, jc=ip-1; j<ipph; ++j,--jc)                // 137
    {
    size_t j2=2*j-1;
    for (size_t k=0; k<l1; ++k)                              // 136
      {
      CC(ido-1,j2,k) = CH(0,k,j);
      CC(0,j2+1,k) = CH(0,k,jc);
      }
    }

  if (ido==1) return;

  for (size_t j=1, jc=ip-1; j<ipph; ++j,--jc)                // 140
    {
    size_t j2=2*j-1;
    for(size_t k=0; k<l1; ++k)                               // 139
      for(size_t i=1, ic=ido-i-2; i<=ido-2; i+=2, ic-=2)      // 138
        {
        CC(i   ,j2+1,k) = CH(i  ,k,j )+CH(i  ,k,jc);
        CC(ic  ,j2  ,k) = CH(i  ,k,j )-CH(i  ,k,jc);
        CC(i+1 ,j2+1,k) = CH(i+1,k,j )+CH(i+1,k,jc);
        CC(ic+1,j2  ,k) = CH(i+1,k,jc)-CH(i+1,k,j );
        }
    }
  }
#undef C1
#undef C2
#undef CH2

#undef CH
#undef CC
#define CH(a,b,c) ch[(a)+ido*((b)+l1*(c))]
#define CC(a,b,c) cc[(a)+ido*((b)+cdim*(c))]

NOINLINE static void radb2(size_t ido, size_t l1, const double * restrict cc,
  double * restrict ch, const double * restrict wa)
  {
  const size_t cdim=2;

  for (size_t k=0; k<l1; k++)
    PM (CH(0,k,0),CH(0,k,1),CC(0,0,k),CC(ido-1,1,k))
  if ((ido&1)==0)
    for (size_t k=0; k<l1; k++)
      {
      CH(ido-1,k,0) = 2.*CC(ido-1,0,k);
      CH(ido-1,k,1) =-2.*CC(0    ,1,k);
      }
  if (ido<=2) return;
  for (size_t k=0; k<l1;++k)
    for (size_t i=2; i<ido; i+=2)
      {
      size_t ic=ido-i;
      double ti2, tr2;
      PM (CH(i-1,k,0),tr2,CC(i-1,0,k),CC(ic-1,1,k))
      PM (ti2,CH(i  ,k,0),CC(i  ,0,k),CC(ic  ,1,k))
      MULPM (CH(i,k,1),CH(i-1,k,1),WA(0,i-2),WA(0,i-1),ti2,tr2)
      }
  }

NOINLINE static void radb3(size_t ido, size_t l1, const double * restrict cc,
  double * restrict ch, const double * restrict wa)
  {
  const size_t cdim=3;
  static const double taur=-0.5, taui=0.86602540378443864676;

  for (size_t k=0; k<l1; k++)
    {
    double tr2=2.*CC(ido-1,1,k);
    double cr2=CC(0,0,k)+taur*tr2;
    CH(0,k,0)=CC(0,0,k)+tr2;
    double ci3=2.*taui*CC(0,2,k);
    PM (CH(0,k,2),CH(0,k,1),cr2,ci3);
    }
  if (ido==1) return;
  for (size_t k=0; k<l1; k++)
    for (size_t i=2; i<ido; i+=2)
      {
      size_t ic=ido-i;
      double tr2=CC(i-1,2,k)+CC(ic-1,1,k); // t2=CC(I) + conj(CC(ic))
      double ti2=CC(i  ,2,k)-CC(ic  ,1,k);
      double cr2=CC(i-1,0,k)+taur*tr2;     // c2=CC +taur*t2
      double ci2=CC(i  ,0,k)+taur*ti2;
      CH(i-1,k,0)=CC(i-1,0,k)+tr2;         // CH=CC+t2
      CH(i  ,k,0)=CC(i  ,0,k)+ti2;
      double cr3=taui*(CC(i-1,2,k)-CC(ic-1,1,k));// c3=taui*(CC(i)-conj(CC(ic)))
      double ci3=taui*(CC(i  ,2,k)+CC(ic  ,1,k));
      double di2, di3, dr2, dr3;
      PM(dr3,dr2,cr2,ci3) // d2= (cr2-ci3, ci2+cr3) = c2+i*c3
      PM(di2,di3,ci2,cr3) // d3= (cr2+ci3, ci2-cr3) = c2-i*c3
      MULPM(CH(i,k,1),CH(i-1,k,1),WA(0,i-2),WA(0,i-1),di2,dr2) // ch = WA*d2
      MULPM(CH(i,k,2),CH(i-1,k,2),WA(1,i-2),WA(1,i-1),di3,dr3)
      }
  }

NOINLINE static void radb4(size_t ido, size_t l1, const double * restrict cc,
  double * restrict ch, const double * restrict wa)
  {
  const size_t cdim=4;
  static const double sqrt2=1.41421356237309504880;

  for (size_t k=0; k<l1; k++)
    {
    double tr1, tr2;
    PM (tr2,tr1,CC(0,0,k),CC(ido-1,3,k))
    double tr3=2.*CC(ido-1,1,k);
    double tr4=2.*CC(0,2,k);
    PM (CH(0,k,0),CH(0,k,2),tr2,tr3)
    PM (CH(0,k,3),CH(0,k,1),tr1,tr4)
    }
  if ((ido&1)==0)
    for (size_t k=0; k<l1; k++)
      {
      double tr1,tr2,ti1,ti2;
      PM (ti1,ti2,CC(0    ,3,k),CC(0    ,1,k))
      PM (tr2,tr1,CC(ido-1,0,k),CC(ido-1,2,k))
      CH(ido-1,k,0)=tr2+tr2;
      CH(ido-1,k,1)=sqrt2*(tr1-ti1);
      CH(ido-1,k,2)=ti2+ti2;
      CH(ido-1,k,3)=-sqrt2*(tr1+ti1);
      }
  if (ido<=2) return;
  for (size_t k=0; k<l1;++k)
    for (size_t i=2; i<ido; i+=2)
      {
      double ci2, ci3, ci4, cr2, cr3, cr4, ti1, ti2, ti3, ti4, tr1, tr2, tr3, tr4;
      size_t ic=ido-i;
      PM (tr2,tr1,CC(i-1,0,k),CC(ic-1,3,k))
      PM (ti1,ti2,CC(i  ,0,k),CC(ic  ,3,k))
      PM (tr4,ti3,CC(i  ,2,k),CC(ic  ,1,k))
      PM (tr3,ti4,CC(i-1,2,k),CC(ic-1,1,k))
      PM (CH(i-1,k,0),cr3,tr2,tr3)
      PM (CH(i  ,k,0),ci3,ti2,ti3)
      PM (cr4,cr2,tr1,tr4)
      PM (ci2,ci4,ti1,ti4)
      MULPM (CH(i,k,1),CH(i-1,k,1),WA(0,i-2),WA(0,i-1),ci2,cr2)
      MULPM (CH(i,k,2),CH(i-1,k,2),WA(1,i-2),WA(1,i-1),ci3,cr3)
      MULPM (CH(i,k,3),CH(i-1,k,3),WA(2,i-2),WA(2,i-1),ci4,cr4)
      }
  }

NOINLINE static void radb5(size_t ido, size_t l1, const double * restrict cc,
  double * restrict ch, const double * restrict wa)
  {
  const size_t cdim=5;
  static const double tr11= 0.3090169943749474241, ti11=0.95105651629515357212,
                      tr12=-0.8090169943749474241, ti12=0.58778525229247312917;

  for (size_t k=0; k<l1; k++)
    {
    double ti5=CC(0,2,k)+CC(0,2,k);
    double ti4=CC(0,4,k)+CC(0,4,k);
    double tr2=CC(ido-1,1,k)+CC(ido-1,1,k);
    double tr3=CC(ido-1,3,k)+CC(ido-1,3,k);
    CH(0,k,0)=CC(0,0,k)+tr2+tr3;
    double cr2=CC(0,0,k)+tr11*tr2+tr12*tr3;
    double cr3=CC(0,0,k)+tr12*tr2+tr11*tr3;
    double ci4, ci5;
    MULPM(ci5,ci4,ti5,ti4,ti11,ti12)
    PM(CH(0,k,4),CH(0,k,1),cr2,ci5)
    PM(CH(0,k,3),CH(0,k,2),cr3,ci4)
    }
  if (ido==1) return;
  for (size_t k=0; k<l1;++k)
    for (size_t i=2; i<ido; i+=2)
      {
      size_t ic=ido-i;
      double tr2, tr3, tr4, tr5, ti2, ti3, ti4, ti5;
      PM(tr2,tr5,CC(i-1,2,k),CC(ic-1,1,k))
      PM(ti5,ti2,CC(i  ,2,k),CC(ic  ,1,k))
      PM(tr3,tr4,CC(i-1,4,k),CC(ic-1,3,k))
      PM(ti4,ti3,CC(i  ,4,k),CC(ic  ,3,k))
      CH(i-1,k,0)=CC(i-1,0,k)+tr2+tr3;
      CH(i  ,k,0)=CC(i  ,0,k)+ti2+ti3;
      double cr2=CC(i-1,0,k)+tr11*tr2+tr12*tr3;
      double ci2=CC(i  ,0,k)+tr11*ti2+tr12*ti3;
      double cr3=CC(i-1,0,k)+tr12*tr2+tr11*tr3;
      double ci3=CC(i  ,0,k)+tr12*ti2+tr11*ti3;
      double ci4, ci5, cr5, cr4;
      MULPM(cr5,cr4,tr5,tr4,ti11,ti12)
      MULPM(ci5,ci4,ti5,ti4,ti11,ti12)
      double dr2, dr3, dr4, dr5, di2, di3, di4, di5;
      PM(dr4,dr3,cr3,ci4)
      PM(di3,di4,ci3,cr4)
      PM(dr5,dr2,cr2,ci5)
      PM(di2,di5,ci2,cr5)
      MULPM(CH(i,k,1),CH(i-1,k,1),WA(0,i-2),WA(0,i-1),di2,dr2)
      MULPM(CH(i,k,2),CH(i-1,k,2),WA(1,i-2),WA(1,i-1),di3,dr3)
      MULPM(CH(i,k,3),CH(i-1,k,3),WA(2,i-2),WA(2,i-1),di4,dr4)
      MULPM(CH(i,k,4),CH(i-1,k,4),WA(3,i-2),WA(3,i-1),di5,dr5)
      }
  }

#undef CC
#undef CH
#define CC(a,b,c) cc[(a)+ido*((b)+cdim*(c))]
#define CH(a,b,c) ch[(a)+ido*((b)+l1*(c))]
#define C1(a,b,c) cc[(a)+ido*((b)+l1*(c))]
#define C2(a,b) cc[(a)+idl1*(b)]
#define CH2(a,b) ch[(a)+idl1*(b)]

NOINLINE static void radbg(size_t ido, size_t ip, size_t l1,
  double * restrict cc, double * restrict ch, const double * restrict wa,
  const double * restrict csarr)
  {
  const size_t cdim=ip;
  size_t ipph=(ip+1)/ 2;
  size_t idl1 = ido*l1;

  for (size_t k=0; k<l1; ++k)        // 102
    for (size_t i=0; i<ido; ++i)     // 101
      CH(i,k,0) = CC(i,0,k);
  for (size_t j=1, jc=ip-1; j<ipph; ++j, --jc)   // 108
    {
    size_t j2=2*j-1;
    for (size_t k=0; k<l1; ++k)
      {
      CH(0,k,j ) = 2*CC(ido-1,j2,k);
      CH(0,k,jc) = 2*CC(0,j2+1,k);
      }
    }

  if (ido!=1)
    {
    for (size_t j=1, jc=ip-1; j<ipph; ++j,--jc)   // 111
      {
      size_t j2=2*j-1;
      for (size_t k=0; k<l1; ++k)
        for (size_t i=1, ic=ido-i-2; i<=ido-2; i+=2, ic-=2)      // 109
          {
          CH(i  ,k,j ) = CC(i  ,j2+1,k)+CC(ic  ,j2,k);
          CH(i  ,k,jc) = CC(i  ,j2+1,k)-CC(ic  ,j2,k);
          CH(i+1,k,j ) = CC(i+1,j2+1,k)-CC(ic+1,j2,k);
          CH(i+1,k,jc) = CC(i+1,j2+1,k)+CC(ic+1,j2,k);
          }
      }
    }
  for (size_t l=1,lc=ip-1; l<ipph; ++l,--lc)
    {
    for (size_t ik=0; ik<idl1; ++ik)
      {
      C2(ik,l ) = CH2(ik,0)+csarr[2*l]*CH2(ik,1)+csarr[4*l]*CH2(ik,2);
      C2(ik,lc) = csarr[2*l+1]*CH2(ik,ip-1)+csarr[4*l+1]*CH2(ik,ip-2);
      }
    size_t iang=2*l;
    size_t j=3,jc=ip-3;
    for(; j<ipph-3; j+=4,jc-=4)
      {
      iang+=l; if(iang>ip) iang-=ip;
      double ar1=csarr[2*iang], ai1=csarr[2*iang+1];
      iang+=l; if(iang>ip) iang-=ip;
      double ar2=csarr[2*iang], ai2=csarr[2*iang+1];
      iang+=l; if(iang>ip) iang-=ip;
      double ar3=csarr[2*iang], ai3=csarr[2*iang+1];
      iang+=l; if(iang>ip) iang-=ip;
      double ar4=csarr[2*iang], ai4=csarr[2*iang+1];
      for (size_t ik=0; ik<idl1; ++ik)
        {
        C2(ik,l ) += ar1*CH2(ik,j )+ar2*CH2(ik,j +1)
                    +ar3*CH2(ik,j +2)+ar4*CH2(ik,j +3);
        C2(ik,lc) += ai1*CH2(ik,jc)+ai2*CH2(ik,jc-1)
                    +ai3*CH2(ik,jc-2)+ai4*CH2(ik,jc-3);
        }
      }
    for(; j<ipph-1; j+=2,jc-=2)
      {
      iang+=l; if(iang>ip) iang-=ip;
      double ar1=csarr[2*iang], ai1=csarr[2*iang+1];
      iang+=l; if(iang>ip) iang-=ip;
      double ar2=csarr[2*iang], ai2=csarr[2*iang+1];
      for (size_t ik=0; ik<idl1; ++ik)
        {
        C2(ik,l ) += ar1*CH2(ik,j )+ar2*CH2(ik,j +1);
        C2(ik,lc) += ai1*CH2(ik,jc)+ai2*CH2(ik,jc-1);
        }
      }
    for(; j<ipph; ++j,--jc)
      {
      iang+=l; if(iang>ip) iang-=ip;
      double war=csarr[2*iang], wai=csarr[2*iang+1];
      for (size_t ik=0; ik<idl1; ++ik)
        {
        C2(ik,l ) += war*CH2(ik,j );
        C2(ik,lc) += wai*CH2(ik,jc);
        }
      }
    }
  for (size_t j=1; j<ipph; ++j)
    for (size_t ik=0; ik<idl1; ++ik)
      CH2(ik,0) += CH2(ik,j);
  for (size_t j=1, jc=ip-1; j<ipph; ++j,--jc)   // 124
    for (size_t k=0; k<l1; ++k)
      {
      CH(0,k,j ) = C1(0,k,j)-C1(0,k,jc);
      CH(0,k,jc) = C1(0,k,j)+C1(0,k,jc);
      }

  if (ido==1) return;

  for (size_t j=1, jc=ip-1; j<ipph; ++j, --jc)  // 127
    for (size_t k=0; k<l1; ++k)
      for (size_t i=1; i<=ido-2; i+=2)
        {
        CH(i  ,k,j ) = C1(i  ,k,j)-C1(i+1,k,jc);
        CH(i  ,k,jc) = C1(i  ,k,j)+C1(i+1,k,jc);
        CH(i+1,k,j ) = C1(i+1,k,j)+C1(i  ,k,jc);
        CH(i+1,k,jc) = C1(i+1,k,j)-C1(i  ,k,jc);
        }

// All in CH

  for (size_t j=1; j<ip; ++j)
    {
    size_t is = (j-1)*(ido-1);
    for (size_t k=0; k<l1; ++k)
      {
      size_t idij = is;
      for (size_t i=1; i<=ido-2; i+=2)
        {
        double t1=CH(i,k,j), t2=CH(i+1,k,j);
        CH(i  ,k,j) = wa[idij]*t1-wa[idij+1]*t2;
        CH(i+1,k,j) = wa[idij]*t2+wa[idij+1]*t1;
        idij+=2;
        }
      }
    }
  }
#undef C1
#undef C2
#undef CH2

#undef CC
#undef CH
#undef PM
#undef MULPM
#undef WA

static void copy_and_norm(double *c, double *p1, size_t n, double fct)
  {
  if (p1!=c)
    {
    if (fct!=1.)
      for (size_t i=0; i<n; ++i)
        c[i] = fct*p1[i];
    else
      memcpy (c,p1,n*sizeof(double));
    }
  else
    if (fct!=1.)
      for (size_t i=0; i<n; ++i)
        c[i] *= fct;
  }

WARN_UNUSED_RESULT
static int rfftp_forward(rfftp_plan plan, double c[], double fct)
  {
  if (plan->length==1) return 0;
  size_t n=plan->length;
  size_t l1=n, nf=plan->nfct;
  double *ch = RALLOC(double, n);
  if (!ch) return -1;
  double *p1=c, *p2=ch;

  for(size_t k1=0; k1<nf;++k1)
    {
    size_t k=nf-k1-1;
    size_t ip=plan->fct[k].fct;
    size_t ido=n / l1;
    l1 /= ip;
    if(ip==4)
      radf4(ido, l1, p1, p2, plan->fct[k].tw);
    else if(ip==2)
      radf2(ido, l1, p1, p2, plan->fct[k].tw);
    else if(ip==3)
      radf3(ido, l1, p1, p2, plan->fct[k].tw);
    else if(ip==5)
      radf5(ido, l1, p1, p2, plan->fct[k].tw);
    else
      {
      radfg(ido, ip, l1, p1, p2, plan->fct[k].tw, plan->fct[k].tws);
      SWAP (p1,p2,double *);
      }
    SWAP (p1,p2,double *);
    }
  copy_and_norm(c,p1,n,fct);
  DEALLOC(ch);
  return 0;
  }

WARN_UNUSED_RESULT
static int rfftp_backward(rfftp_plan plan, double c[], double fct)
  {
  if (plan->length==1) return 0;
  size_t n=plan->length;
  size_t l1=1, nf=plan->nfct;
  double *ch = RALLOC(double, n);
  if (!ch) return -1;
  double *p1=c, *p2=ch;

  for(size_t k=0; k<nf; k++)
    {
    size_t ip = plan->fct[k].fct,
           ido= n/(ip*l1);
    if(ip==4)
      radb4(ido, l1, p1, p2, plan->fct[k].tw);
    else if(ip==2)
      radb2(ido, l1, p1, p2, plan->fct[k].tw);
    else if(ip==3)
      radb3(ido, l1, p1, p2, plan->fct[k].tw);
    else if(ip==5)
      radb5(ido, l1, p1, p2, plan->fct[k].tw);
    else
      radbg(ido, ip, l1, p1, p2, plan->fct[k].tw, plan->fct[k].tws);
    SWAP (p1,p2,double *);
    l1*=ip;
    }
  copy_and_norm(c,p1,n,fct);
  DEALLOC(ch);
  return 0;
  }

WARN_UNUSED_RESULT
static int rfftp_factorize (rfftp_plan plan)
  {
  size_t length=plan->length;
  size_t nfct=0;
  while ((length%4)==0)
    { if (nfct>=NFCT) return -1; plan->fct[nfct++].fct=4; length>>=2; }
  if ((length%2)==0)
    {
    length>>=1;
    // factor 2 should be at the front of the factor list
    if (nfct>=NFCT) return -1;
    plan->fct[nfct++].fct=2;
    SWAP(plan->fct[0].fct, plan->fct[nfct-1].fct,size_t);
    }
  size_t maxl=(size_t)(sqrt((double)length))+1;
  for (size_t divisor=3; (length>1)&&(divisor<maxl); divisor+=2)
    if ((length%divisor)==0)
      {
      while ((length%divisor)==0)
        {
        if (nfct>=NFCT) return -1;
        plan->fct[nfct++].fct=divisor;
        length/=divisor;
        }
      maxl=(size_t)(sqrt((double)length))+1;
      }
  if (length>1) plan->fct[nfct++].fct=length;
  plan->nfct=nfct;
  return 0;
  }

static size_t rfftp_twsize(rfftp_plan plan)
  {
  size_t twsize=0, l1=1;
  for (size_t k=0; k<plan->nfct; ++k)
    {
    size_t ip=plan->fct[k].fct, ido= plan->length/(l1*ip);
    twsize+=(ip-1)*(ido-1);
    if (ip>5) twsize+=2*ip;
    l1*=ip;
    }
  return twsize;
  }

WARN_UNUSED_RESULT NOINLINE static int rfftp_comp_twiddle (rfftp_plan plan)
  {
  size_t length=plan->length;
  double *twid = RALLOC(double, 2*length);
  if (!twid) return -1;
  sincos_2pibyn_half(length, twid);
  size_t l1=1;
  double *ptr=plan->mem;
  for (size_t k=0; k<plan->nfct; ++k)
    {
    size_t ip=plan->fct[k].fct, ido=length/(l1*ip);
    if (k<plan->nfct-1) // last factor doesn't need twiddles
      {
      plan->fct[k].tw=ptr; ptr+=(ip-1)*(ido-1);
      for (size_t j=1; j<ip; ++j)
        for (size_t i=1; i<=(ido-1)/2; ++i)
          {
          plan->fct[k].tw[(j-1)*(ido-1)+2*i-2] = twid[2*j*l1*i];
          plan->fct[k].tw[(j-1)*(ido-1)+2*i-1] = twid[2*j*l1*i+1];
          }
      }
    if (ip>5) // special factors required by *g functions
      {
      plan->fct[k].tws=ptr; ptr+=2*ip;
      plan->fct[k].tws[0] = 1.;
      plan->fct[k].tws[1] = 0.;
      for (size_t i=1; i<=(ip>>1); ++i)
        {
        plan->fct[k].tws[2*i  ] = twid[2*i*(length/ip)];
        plan->fct[k].tws[2*i+1] = twid[2*i*(length/ip)+1];
        plan->fct[k].tws[2*(ip-i)  ] = twid[2*i*(length/ip)];
        plan->fct[k].tws[2*(ip-i)+1] = -twid[2*i*(length/ip)+1];
        }
      }
    l1*=ip;
    }
  DEALLOC(twid);
  return 0;
  }

NOINLINE static rfftp_plan make_rfftp_plan (size_t length)
  {
  if (length==0) return NULL;
  rfftp_plan plan = RALLOC(rfftp_plan_i,1);
  if (!plan) return NULL;
  plan->length=length;
  plan->nfct=0;
  plan->mem=NULL;
  for (size_t i=0; i<NFCT; ++i)
    plan->fct[i]=(rfftp_fctdata){0,0,0};
  if (length==1) return plan;
  if (rfftp_factorize(plan)!=0) { DEALLOC(plan); return NULL; }
  size_t tws=rfftp_twsize(plan);
  if (tws != 0) {
    plan->mem=RALLOC(double,tws);
    if (!plan->mem) { DEALLOC(plan); return NULL; }
  }
  if (rfftp_comp_twiddle(plan)!=0)
    { DEALLOC(plan->mem); DEALLOC(plan); return NULL; }
  return plan;
  }

NOINLINE static void destroy_rfftp_plan (rfftp_plan plan)
  {
  DEALLOC(plan->mem);
  DEALLOC(plan);
  }

typedef struct fftblue_plan_i
  {
  size_t n, n2;
  cfftp_plan plan;
  double *mem;
  double *bk, *bkf;
  } fftblue_plan_i;
typedef struct fftblue_plan_i * fftblue_plan;

NOINLINE static fftblue_plan make_fftblue_plan (size_t length)
  {
  fftblue_plan plan = RALLOC(fftblue_plan_i,1);
  if (!plan) return NULL;
  plan->n = length;
  plan->n2 = good_size(plan->n*2-1);
  plan->mem = RALLOC(double, 2*plan->n+2*plan->n2);
  if (!plan->mem) { DEALLOC(plan); return NULL; }
  plan->bk  = plan->mem;
  plan->bkf = plan->bk+2*plan->n;

/* initialize b_k */
  double *tmp = RALLOC(double,4*plan->n);
  if (!tmp) { DEALLOC(plan->mem); DEALLOC(plan); return NULL; }
  sincos_2pibyn(2*plan->n,tmp);
  plan->bk[0] = 1;
  plan->bk[1] = 0;

  size_t coeff=0;
  for (size_t m=1; m<plan->n; ++m)
    {
    coeff+=2*m-1;
    if (coeff>=2*plan->n) coeff-=2*plan->n;
    plan->bk[2*m  ] = tmp[2*coeff  ];
    plan->bk[2*m+1] = tmp[2*coeff+1];
    }

  /* initialize the zero-padded, Fourier transformed b_k. Add normalisation. */
  double xn2 = 1./plan->n2;
  plan->bkf[0] = plan->bk[0]*xn2;
  plan->bkf[1] = plan->bk[1]*xn2;
  for (size_t m=2; m<2*plan->n; m+=2)
    {
    plan->bkf[m]   = plan->bkf[2*plan->n2-m]   = plan->bk[m]   *xn2;
    plan->bkf[m+1] = plan->bkf[2*plan->n2-m+1] = plan->bk[m+1] *xn2;
    }
  for (size_t m=2*plan->n;m<=(2*plan->n2-2*plan->n+1);++m)
    plan->bkf[m]=0.;
  plan->plan=make_cfftp_plan(plan->n2);
  if (!plan->plan)
    { DEALLOC(tmp); DEALLOC(plan->mem); DEALLOC(plan); return NULL; }
  if (cfftp_forward(plan->plan,plan->bkf,1.)!=0)
    { DEALLOC(tmp); DEALLOC(plan->mem); DEALLOC(plan); return NULL; }
  DEALLOC(tmp);

  return plan;
  }

NOINLINE static void destroy_fftblue_plan (fftblue_plan plan)
  {
  DEALLOC(plan->mem);
  destroy_cfftp_plan(plan->plan);
  DEALLOC(plan);
  }

NOINLINE WARN_UNUSED_RESULT
static int fftblue_fft(fftblue_plan plan, double c[], int isign, double fct)
  {
  size_t n=plan->n;
  size_t n2=plan->n2;
  double *bk  = plan->bk;
  double *bkf = plan->bkf;
  double *akf = RALLOC(double, 2*n2);
  if (!akf) return -1;

/* initialize a_k and FFT it */
  if (isign>0)
    for (size_t m=0; m<2*n; m+=2)
      {
      akf[m]   = c[m]*bk[m]   - c[m+1]*bk[m+1];
      akf[m+1] = c[m]*bk[m+1] + c[m+1]*bk[m];
      }
  else
    for (size_t m=0; m<2*n; m+=2)
      {
      akf[m]   = c[m]*bk[m]   + c[m+1]*bk[m+1];
      akf[m+1] =-c[m]*bk[m+1] + c[m+1]*bk[m];
      }
  for (size_t m=2*n; m<2*n2; ++m)
    akf[m]=0;

  if (cfftp_forward (plan->plan,akf,fct)!=0)
    { DEALLOC(akf); return -1; }

/* do the convolution */
  if (isign>0)
    for (size_t m=0; m<2*n2; m+=2)
      {
      double im = -akf[m]*bkf[m+1] + akf[m+1]*bkf[m];
      akf[m  ]  =  akf[m]*bkf[m]   + akf[m+1]*bkf[m+1];
      akf[m+1]  = im;
      }
  else
    for (size_t m=0; m<2*n2; m+=2)
      {
      double im = akf[m]*bkf[m+1] + akf[m+1]*bkf[m];
      akf[m  ]  = akf[m]*bkf[m]   - akf[m+1]*bkf[m+1];
      akf[m+1]  = im;
      }

/* inverse FFT */
  if (cfftp_backward (plan->plan,akf,1.)!=0)
    { DEALLOC(akf); return -1; }

/* multiply by b_k */
  if (isign>0)
    for (size_t m=0; m<2*n; m+=2)
      {
      c[m]   = bk[m]  *akf[m] - bk[m+1]*akf[m+1];
      c[m+1] = bk[m+1]*akf[m] + bk[m]  *akf[m+1];
      }
  else
    for (size_t m=0; m<2*n; m+=2)
      {
      c[m]   = bk[m]  *akf[m] + bk[m+1]*akf[m+1];
      c[m+1] =-bk[m+1]*akf[m] + bk[m]  *akf[m+1];
      }
  DEALLOC(akf);
  return 0;
  }

WARN_UNUSED_RESULT
static int cfftblue_backward(fftblue_plan plan, double c[], double fct)
  { return fftblue_fft(plan,c,1,fct); }

WARN_UNUSED_RESULT
static int cfftblue_forward(fftblue_plan plan, double c[], double fct)
  { return fftblue_fft(plan,c,-1,fct); }

WARN_UNUSED_RESULT
static int rfftblue_backward(fftblue_plan plan, double c[], double fct)
  {
  size_t n=plan->n;
  double *tmp = RALLOC(double,2*n);
  if (!tmp) return -1;
  tmp[0]=c[0];
  tmp[1]=0.;
  memcpy (tmp+2,c+1, (n-1)*sizeof(double));
  if ((n&1)==0) tmp[n+1]=0.;
  for (size_t m=2; m<n; m+=2)
    {
    tmp[2*n-m]=tmp[m];
    tmp[2*n-m+1]=-tmp[m+1];
    }
  if (fftblue_fft(plan,tmp,1,fct)!=0)
    { DEALLOC(tmp); return -1; }
  for (size_t m=0; m<n; ++m)
    c[m] = tmp[2*m];
  DEALLOC(tmp);
  return 0;
  }

WARN_UNUSED_RESULT
static int rfftblue_forward(fftblue_plan plan, double c[], double fct)
  {
  size_t n=plan->n;
  double *tmp = RALLOC(double,2*n);
  if (!tmp) return -1;
  for (size_t m=0; m<n; ++m)
    {
    tmp[2*m] = c[m];
    tmp[2*m+1] = 0.;
    }
  if (fftblue_fft(plan,tmp,-1,fct)!=0)
    { DEALLOC(tmp); return -1; }
  c[0] = tmp[0];
  memcpy (c+1, tmp+2, (n-1)*sizeof(double));
  DEALLOC(tmp);
  return 0;
  }

typedef struct cfft_plan_i
  {
  cfftp_plan packplan;
  fftblue_plan blueplan;
  } cfft_plan_i;

static cfft_plan make_cfft_plan (size_t length)
  {
  if (length==0) return NULL;
  cfft_plan plan = RALLOC(cfft_plan_i,1);
  if (!plan) return NULL;
  plan->blueplan=0;
  plan->packplan=0;
  if ((length<50) || (largest_prime_factor(length)<=sqrt(length)))
    {
    plan->packplan=make_cfftp_plan(length);
    if (!plan->packplan) { DEALLOC(plan); return NULL; }
    return plan;
    }
  double comp1 = cost_guess(length);
  double comp2 = 2*cost_guess(good_size(2*length-1));
  comp2*=1.5; /* fudge factor that appears to give good overall performance */
  if (comp2<comp1) // use Bluestein
    {
    plan->blueplan=make_fftblue_plan(length);
    if (!plan->blueplan) { DEALLOC(plan); return NULL; }
    }
  else
    {
    plan->packplan=make_cfftp_plan(length);
    if (!plan->packplan) { DEALLOC(plan); return NULL; }
    }
  return plan;
  }

static void destroy_cfft_plan (cfft_plan plan)
  {
  if (plan->blueplan)
    destroy_fftblue_plan(plan->blueplan);
  if (plan->packplan)
    destroy_cfftp_plan(plan->packplan);
  DEALLOC(plan);
  }

WARN_UNUSED_RESULT static int cfft_backward(cfft_plan plan, double c[], double fct)
  {
  if (plan->packplan)
    return cfftp_backward(plan->packplan,c,fct);
  // if (plan->blueplan)
  return cfftblue_backward(plan->blueplan,c,fct);
  }

WARN_UNUSED_RESULT static int cfft_forward(cfft_plan plan, double c[], double fct)
  {
  if (plan->packplan)
    return cfftp_forward(plan->packplan,c,fct);
  // if (plan->blueplan)
  return cfftblue_forward(plan->blueplan,c,fct);
  }

typedef struct rfft_plan_i
  {
  rfftp_plan packplan;
  fftblue_plan blueplan;
  } rfft_plan_i;

static rfft_plan make_rfft_plan (size_t length)
  {
  if (length==0) return NULL;
  rfft_plan plan = RALLOC(rfft_plan_i,1);
  if (!plan) return NULL;
  plan->blueplan=0;
  plan->packplan=0;
  if ((length<50) || (largest_prime_factor(length)<=sqrt(length)))
    {
    plan->packplan=make_rfftp_plan(length);
    if (!plan->packplan) { DEALLOC(plan); return NULL; }
    return plan;
    }
  double comp1 = 0.5*cost_guess(length);
  double comp2 = 2*cost_guess(good_size(2*length-1));
  comp2*=1.5; /* fudge factor that appears to give good overall performance */
  if (comp2<comp1) // use Bluestein
    {
    plan->blueplan=make_fftblue_plan(length);
    if (!plan->blueplan) { DEALLOC(plan); return NULL; }
    }
  else
    {
    plan->packplan=make_rfftp_plan(length);
    if (!plan->packplan) { DEALLOC(plan); return NULL; }
    }
  return plan;
  }

static void destroy_rfft_plan (rfft_plan plan)
  {
  if (plan->blueplan)
    destroy_fftblue_plan(plan->blueplan);
  if (plan->packplan)
    destroy_rfftp_plan(plan->packplan);
  DEALLOC(plan);
  }

WARN_UNUSED_RESULT static int rfft_backward(rfft_plan plan, double c[], double fct)
  {
  if (plan->packplan)
    return rfftp_backward(plan->packplan,c,fct);
  else // if (plan->blueplan)
    return rfftblue_backward(plan->blueplan,c,fct);
  }

WARN_UNUSED_RESULT static int rfft_forward(rfft_plan plan, double c[], double fct)
  {
  if (plan->packplan)
    return rfftp_forward(plan->packplan,c,fct);
  else // if (plan->blueplan)
    return rfftblue_forward(plan->blueplan,c,fct);
  }

static PyObject *
execute_complex(PyObject *a1, int is_forward, double fct)
{
    PyArrayObject *data = (PyArrayObject *)PyArray_FromAny(a1,
            PyArray_DescrFromType(NPY_CDOUBLE), 1, 0,
            NPY_ARRAY_ENSURECOPY | NPY_ARRAY_DEFAULT |
            NPY_ARRAY_ENSUREARRAY | NPY_ARRAY_FORCECAST,
            NULL);
    if (!data) return NULL;

    int npts = PyArray_DIM(data, PyArray_NDIM(data) - 1);
    cfft_plan plan=NULL;

    int nrepeats = PyArray_SIZE(data)/npts;
    double *dptr = (double *)PyArray_DATA(data);
    int fail=0;
    Py_BEGIN_ALLOW_THREADS;
    plan = make_cfft_plan(npts);
    if (!plan) fail=1;
    if (!fail)
      for (int i = 0; i < nrepeats; i++) {
          int res = is_forward ?
            cfft_forward(plan, dptr, fct) : cfft_backward(plan, dptr, fct);
          if (res!=0) { fail=1; break; }
          dptr += npts*2;
      }
    if (plan) destroy_cfft_plan(plan);
    Py_END_ALLOW_THREADS;
    if (fail) {
      Py_XDECREF(data);
      return PyErr_NoMemory();
    }
    return (PyObject *)data;
}

static PyObject *
execute_real_forward(PyObject *a1, double fct)
{
    rfft_plan plan=NULL;
    int fail = 0;
    npy_intp tdim[NPY_MAXDIMS];

    PyArrayObject *data = (PyArrayObject *)PyArray_FromAny(a1,
            PyArray_DescrFromType(NPY_DOUBLE), 1, 0,
            NPY_ARRAY_DEFAULT | NPY_ARRAY_ENSUREARRAY | NPY_ARRAY_FORCECAST,
            NULL);
    if (!data) return NULL;

    int ndim = PyArray_NDIM(data);
    const npy_intp *odim = PyArray_DIMS(data);
    int npts = odim[ndim - 1];
    for (int d=0; d<ndim-1; ++d)
      tdim[d] = odim[d];
    tdim[ndim-1] = npts/2 + 1;
    PyArrayObject *ret = (PyArrayObject *)PyArray_Empty(ndim,
            tdim, PyArray_DescrFromType(NPY_CDOUBLE), 0);
    if (!ret) fail=1;
    if (!fail) {
      int rstep = PyArray_DIM(ret, PyArray_NDIM(ret) - 1)*2;

      int nrepeats = PyArray_SIZE(data)/npts;
      double *rptr = (double *)PyArray_DATA(ret),
             *dptr = (double *)PyArray_DATA(data);

      Py_BEGIN_ALLOW_THREADS;
      plan = make_rfft_plan(npts);
      if (!plan) fail=1;
      if (!fail)
        for (int i = 0; i < nrepeats; i++) {
            rptr[rstep-1] = 0.0;
            memcpy((char *)(rptr+1), dptr, npts*sizeof(double));
            if (rfft_forward(plan, rptr+1, fct)!=0) {fail=1; break;}
            rptr[0] = rptr[1];
            rptr[1] = 0.0;
            rptr += rstep;
            dptr += npts;
      }
      if (plan) destroy_rfft_plan(plan);
      Py_END_ALLOW_THREADS;
    }
    if (fail) {
      Py_XDECREF(data);
      Py_XDECREF(ret);
      return PyErr_NoMemory();
    }
    Py_DECREF(data);
    return (PyObject *)ret;
}
static PyObject *
execute_real_backward(PyObject *a1, double fct)
{
    rfft_plan plan=NULL;
    PyArrayObject *data = (PyArrayObject *)PyArray_FromAny(a1,
            PyArray_DescrFromType(NPY_CDOUBLE), 1, 0,
            NPY_ARRAY_DEFAULT | NPY_ARRAY_ENSUREARRAY | NPY_ARRAY_FORCECAST,
            NULL);
    if (!data) return NULL;
    int npts = PyArray_DIM(data, PyArray_NDIM(data) - 1);
    PyArrayObject *ret = (PyArrayObject *)PyArray_Empty(PyArray_NDIM(data),
            PyArray_DIMS(data), PyArray_DescrFromType(NPY_DOUBLE), 0);
    int fail = 0;
    if (!ret) fail=1;
    if (!fail) {
      int nrepeats = PyArray_SIZE(ret)/npts;
      double *rptr = (double *)PyArray_DATA(ret),
             *dptr = (double *)PyArray_DATA(data);

      Py_BEGIN_ALLOW_THREADS;
      plan = make_rfft_plan(npts);
      if (!plan) fail=1;
      if (!fail) {
        for (int i = 0; i < nrepeats; i++) {
          memcpy((char *)(rptr + 1), (dptr + 2), (npts - 1)*sizeof(double));
          rptr[0] = dptr[0];
          if (rfft_backward(plan, rptr, fct)!=0) {fail=1; break;}
          rptr += npts;
          dptr += npts*2;
        }
      }
      if (plan) destroy_rfft_plan(plan);
      Py_END_ALLOW_THREADS;
    }
    if (fail) {
      Py_XDECREF(data);
      Py_XDECREF(ret);
      return PyErr_NoMemory();
    }
    Py_DECREF(data);
    return (PyObject *)ret;
}

static PyObject *
execute_real(PyObject *a1, int is_forward, double fct)
{
    return is_forward ? execute_real_forward(a1, fct)
                      : execute_real_backward(a1, fct);
}

static const char execute__doc__[] = "";

static PyObject *
execute(PyObject *NPY_UNUSED(self), PyObject *args)
{
    PyObject *a1;
    int is_real, is_forward;
    double fct;

    if(!PyArg_ParseTuple(args, "Oiid:execute", &a1, &is_real, &is_forward, &fct)) {
        return NULL;
    }

    return is_real ? execute_real(a1, is_forward, fct)
                   : execute_complex(a1, is_forward, fct);
}

/* List of methods defined in the module */

static struct PyMethodDef methods[] = {
    {"execute",   execute,   1, execute__doc__},
    {NULL, NULL, 0, NULL}          /* sentinel */
};

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_pocketfft_internal",
        NULL,
        -1,
        methods,
        NULL,
        NULL,
        NULL,
        NULL
};

/* Initialization function for the module */
PyMODINIT_FUNC PyInit__pocketfft_internal(void)
{
    PyObject *m;
    m = PyModule_Create(&moduledef);
    if (m == NULL) {
        return NULL;
    }

    /* Import the array object */
    import_array();

    /* XXXX Add constants here */

    return m;
}
