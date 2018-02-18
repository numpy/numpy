/*
 * fftpack.c : A set of FFT routines in C.
 * Algorithmically based on Fortran-77 FFTPACK by Paul N. Swarztrauber (Version 4, 1985).
*/
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <Python.h>
#include <math.h>
#include <stdio.h>
#include <numpy/ndarraytypes.h>

#define DOUBLE
#ifdef DOUBLE
#define Treal double
#else
#define Treal float
#endif

#define ref(u,a) u[a]

/* Macros for accurate calculation of the twiddle factors. */
#define TWOPI 6.283185307179586476925286766559005768391
#define cos2pi(m, n) cos((TWOPI * (m)) / (n))
#define sin2pi(m, n) sin((TWOPI * (m)) / (n))

#define MAXFAC 13    /* maximum number of factors in factorization of n */
#define NSPECIAL 4   /* number of factors for which we have special-case routines */

#ifdef __cplusplus
extern "C" {
#endif

static void sincos2pi(int m, int n, Treal* si, Treal* co)
/* Calculates sin(2pi * m/n) and cos(2pi * m/n). It is more accurate
 * than the naive calculation as the fraction m/n is reduced to [0, 1/8) first.
 * Due to the symmetry of sin(x) and cos(x) the values for all x can be
 * determined from the function values of the reduced argument in the first
 * octant.
 */
    {
        int n8, m8, octant;
        n8 = 8 * n;
        m8 = (8 * m) % n8;
        octant = m8 / n;
        m8 = m8 % n;
        switch(octant) {
            case 0:
                *co = cos2pi(m8, n8);
                *si = sin2pi(m8, n8);
                break;
            case 1:
                *co = sin2pi(n-m8, n8);
                *si = cos2pi(n-m8, n8);
                break;
            case 2:
                *co = -sin2pi(m8, n8);
                *si = cos2pi(m8, n8);
                break;
            case 3:
                *co = -cos2pi(n-m8, n8);
                *si = sin2pi(n-m8, n8);
                break;
            case 4:
                *co = -cos2pi(m8, n8);
                *si = -sin2pi(m8, n8);
                break;
            case 5:
                *co = -sin2pi(n-m8, n8);
                *si = -cos2pi(n-m8, n8);
                break;
            case 6:
                *co = sin2pi(m8, n8);
                *si = -cos2pi(m8, n8);
                break;
            case 7:
                *co = cos2pi(n-m8, n8);
                *si = -sin2pi(n-m8, n8);
                break;
        }
    }

/* ----------------------------------------------------------------------
   passf2, passf3, passf4, passf5, passf. Complex FFT passes fwd and bwd.
----------------------------------------------------------------------- */

static void passf2(int ido, int l1, const Treal cc[], Treal ch[], const Treal wa1[], int isign)
  /* isign==+1 for backward transform */
  {
    int i, k, ah, ac;
    Treal ti2, tr2;
    if (ido <= 2) {
      for (k=0; k<l1; k++) {
        ah = k*ido;
        ac = 2*k*ido;
        ch[ah]              = ref(cc,ac) + ref(cc,ac + ido);
        ch[ah + ido*l1]     = ref(cc,ac) - ref(cc,ac + ido);
        ch[ah+1]            = ref(cc,ac+1) + ref(cc,ac + ido + 1);
        ch[ah + ido*l1 + 1] = ref(cc,ac+1) - ref(cc,ac + ido + 1);
      }
    } else {
      for (k=0; k<l1; k++) {
        for (i=0; i<ido-1; i+=2) {
          ah = i + k*ido;
          ac = i + 2*k*ido;
          ch[ah]   = ref(cc,ac) + ref(cc,ac + ido);
          tr2      = ref(cc,ac) - ref(cc,ac + ido);
          ch[ah+1] = ref(cc,ac+1) + ref(cc,ac + 1 + ido);
          ti2      = ref(cc,ac+1) - ref(cc,ac + 1 + ido);
          ch[ah+l1*ido+1] = wa1[i]*ti2 + isign*wa1[i+1]*tr2;
          ch[ah+l1*ido]   = wa1[i]*tr2 - isign*wa1[i+1]*ti2;
        }
      }
    }
  } /* passf2 */


static void passf3(int ido, int l1, const Treal cc[], Treal ch[],
      const Treal wa1[], const Treal wa2[], int isign)
  /* isign==+1 for backward transform */
  {
    static const Treal taur = -0.5;
    static const Treal taui = 0.86602540378443864676;
    int i, k, ac, ah;
    Treal ci2, ci3, di2, di3, cr2, cr3, dr2, dr3, ti2, tr2;
    if (ido == 2) {
      for (k=1; k<=l1; k++) {
        ac = (3*k - 2)*ido;
        tr2 = ref(cc,ac) + ref(cc,ac + ido);
        cr2 = ref(cc,ac - ido) + taur*tr2;
        ah = (k - 1)*ido;
        ch[ah] = ref(cc,ac - ido) + tr2;

        ti2 = ref(cc,ac + 1) + ref(cc,ac + ido + 1);
        ci2 = ref(cc,ac - ido + 1) + taur*ti2;
        ch[ah + 1] = ref(cc,ac - ido + 1) + ti2;

        cr3 = isign*taui*(ref(cc,ac) - ref(cc,ac + ido));
        ci3 = isign*taui*(ref(cc,ac + 1) - ref(cc,ac + ido + 1));
        ch[ah + l1*ido] = cr2 - ci3;
        ch[ah + 2*l1*ido] = cr2 + ci3;
        ch[ah + l1*ido + 1] = ci2 + cr3;
        ch[ah + 2*l1*ido + 1] = ci2 - cr3;
      }
    } else {
      for (k=1; k<=l1; k++) {
        for (i=0; i<ido-1; i+=2) {
          ac = i + (3*k - 2)*ido;
          tr2 = ref(cc,ac) + ref(cc,ac + ido);
          cr2 = ref(cc,ac - ido) + taur*tr2;
          ah = i + (k-1)*ido;
          ch[ah] = ref(cc,ac - ido) + tr2;
          ti2 = ref(cc,ac + 1) + ref(cc,ac + ido + 1);
          ci2 = ref(cc,ac - ido + 1) + taur*ti2;
          ch[ah + 1] = ref(cc,ac - ido + 1) + ti2;
          cr3 = isign*taui*(ref(cc,ac) - ref(cc,ac + ido));
          ci3 = isign*taui*(ref(cc,ac + 1) - ref(cc,ac + ido + 1));
          dr2 = cr2 - ci3;
          dr3 = cr2 + ci3;
          di2 = ci2 + cr3;
          di3 = ci2 - cr3;
          ch[ah + l1*ido + 1] = wa1[i]*di2 + isign*wa1[i+1]*dr2;
          ch[ah + l1*ido] = wa1[i]*dr2 - isign*wa1[i+1]*di2;
          ch[ah + 2*l1*ido + 1] = wa2[i]*di3 + isign*wa2[i+1]*dr3;
          ch[ah + 2*l1*ido] = wa2[i]*dr3 - isign*wa2[i+1]*di3;
        }
      }
    }
  } /* passf3 */


static void passf4(int ido, int l1, const Treal cc[], Treal ch[],
      const Treal wa1[], const Treal wa2[], const Treal wa3[], int isign)
  /* isign == -1 for forward transform and +1 for backward transform */
  {
    int i, k, ac, ah;
    Treal ci2, ci3, ci4, cr2, cr3, cr4, ti1, ti2, ti3, ti4, tr1, tr2, tr3, tr4;
    if (ido == 2) {
      for (k=0; k<l1; k++) {
        ac = 4*k*ido + 1;
        ti1 = ref(cc,ac) - ref(cc,ac + 2*ido);
        ti2 = ref(cc,ac) + ref(cc,ac + 2*ido);
        tr4 = ref(cc,ac + 3*ido) - ref(cc,ac + ido);
        ti3 = ref(cc,ac + ido) + ref(cc,ac + 3*ido);
        tr1 = ref(cc,ac - 1) - ref(cc,ac + 2*ido - 1);
        tr2 = ref(cc,ac - 1) + ref(cc,ac + 2*ido - 1);
        ti4 = ref(cc,ac + ido - 1) - ref(cc,ac + 3*ido - 1);
        tr3 = ref(cc,ac + ido - 1) + ref(cc,ac + 3*ido - 1);
        ah = k*ido;
        ch[ah] = tr2 + tr3;
        ch[ah + 2*l1*ido] = tr2 - tr3;
        ch[ah + 1] = ti2 + ti3;
        ch[ah + 2*l1*ido + 1] = ti2 - ti3;
        ch[ah + l1*ido] = tr1 + isign*tr4;
        ch[ah + 3*l1*ido] = tr1 - isign*tr4;
        ch[ah + l1*ido + 1] = ti1 + isign*ti4;
        ch[ah + 3*l1*ido + 1] = ti1 - isign*ti4;
      }
    } else {
      for (k=0; k<l1; k++) {
        for (i=0; i<ido-1; i+=2) {
          ac = i + 1 + 4*k*ido;
          ti1 = ref(cc,ac) - ref(cc,ac + 2*ido);
          ti2 = ref(cc,ac) + ref(cc,ac + 2*ido);
          ti3 = ref(cc,ac + ido) + ref(cc,ac + 3*ido);
          tr4 = ref(cc,ac + 3*ido) - ref(cc,ac + ido);
          tr1 = ref(cc,ac - 1) - ref(cc,ac + 2*ido - 1);
          tr2 = ref(cc,ac - 1) + ref(cc,ac + 2*ido - 1);
          ti4 = ref(cc,ac + ido - 1) - ref(cc,ac + 3*ido - 1);
          tr3 = ref(cc,ac + ido - 1) + ref(cc,ac + 3*ido - 1);
          ah = i + k*ido;
          ch[ah] = tr2 + tr3;
          cr3 = tr2 - tr3;
          ch[ah + 1] = ti2 + ti3;
          ci3 = ti2 - ti3;
          cr2 = tr1 + isign*tr4;
          cr4 = tr1 - isign*tr4;
          ci2 = ti1 + isign*ti4;
          ci4 = ti1 - isign*ti4;
          ch[ah + l1*ido] = wa1[i]*cr2 - isign*wa1[i + 1]*ci2;
          ch[ah + l1*ido + 1] = wa1[i]*ci2 + isign*wa1[i + 1]*cr2;
          ch[ah + 2*l1*ido] = wa2[i]*cr3 - isign*wa2[i + 1]*ci3;
          ch[ah + 2*l1*ido + 1] = wa2[i]*ci3 + isign*wa2[i + 1]*cr3;
          ch[ah + 3*l1*ido] = wa3[i]*cr4 -isign*wa3[i + 1]*ci4;
          ch[ah + 3*l1*ido + 1] = wa3[i]*ci4 + isign*wa3[i + 1]*cr4;
        }
      }
    }
  } /* passf4 */


static void passf5(int ido, int l1, const Treal cc[], Treal ch[],
      const Treal wa1[], const Treal wa2[], const Treal wa3[], const Treal wa4[], int isign)
  /* isign == -1 for forward transform and +1 for backward transform */
  {
    static const Treal tr11 = 0.3090169943749474241;
    static const Treal ti11 = 0.95105651629515357212;
    static const Treal tr12 = -0.8090169943749474241;
    static const Treal ti12 = 0.58778525229247312917;
    int i, k, ac, ah;
    Treal ci2, ci3, ci4, ci5, di3, di4, di5, di2, cr2, cr3, cr5, cr4, ti2, ti3,
        ti4, ti5, dr3, dr4, dr5, dr2, tr2, tr3, tr4, tr5;
    if (ido == 2) {
      for (k = 1; k <= l1; ++k) {
        ac = (5*k - 4)*ido + 1;
        ti5 = ref(cc,ac) - ref(cc,ac + 3*ido);
        ti2 = ref(cc,ac) + ref(cc,ac + 3*ido);
        ti4 = ref(cc,ac + ido) - ref(cc,ac + 2*ido);
        ti3 = ref(cc,ac + ido) + ref(cc,ac + 2*ido);
        tr5 = ref(cc,ac - 1) - ref(cc,ac + 3*ido - 1);
        tr2 = ref(cc,ac - 1) + ref(cc,ac + 3*ido - 1);
        tr4 = ref(cc,ac + ido - 1) - ref(cc,ac + 2*ido - 1);
        tr3 = ref(cc,ac + ido - 1) + ref(cc,ac + 2*ido - 1);
        ah = (k - 1)*ido;
        ch[ah] = ref(cc,ac - ido - 1) + tr2 + tr3;
        ch[ah + 1] = ref(cc,ac - ido) + ti2 + ti3;
        cr2 = ref(cc,ac - ido - 1) + tr11*tr2 + tr12*tr3;
        ci2 = ref(cc,ac - ido) + tr11*ti2 + tr12*ti3;
        cr3 = ref(cc,ac - ido - 1) + tr12*tr2 + tr11*tr3;
        ci3 = ref(cc,ac - ido) + tr12*ti2 + tr11*ti3;
        cr5 = isign*(ti11*tr5 + ti12*tr4);
        ci5 = isign*(ti11*ti5 + ti12*ti4);
        cr4 = isign*(ti12*tr5 - ti11*tr4);
        ci4 = isign*(ti12*ti5 - ti11*ti4);
        ch[ah + l1*ido] = cr2 - ci5;
        ch[ah + 4*l1*ido] = cr2 + ci5;
        ch[ah + l1*ido + 1] = ci2 + cr5;
        ch[ah + 2*l1*ido + 1] = ci3 + cr4;
        ch[ah + 2*l1*ido] = cr3 - ci4;
        ch[ah + 3*l1*ido] = cr3 + ci4;
        ch[ah + 3*l1*ido + 1] = ci3 - cr4;
        ch[ah + 4*l1*ido + 1] = ci2 - cr5;
      }
    } else {
      for (k=1; k<=l1; k++) {
        for (i=0; i<ido-1; i+=2) {
          ac = i + 1 + (k*5 - 4)*ido;
          ti5 = ref(cc,ac) - ref(cc,ac + 3*ido);
          ti2 = ref(cc,ac) + ref(cc,ac + 3*ido);
          ti4 = ref(cc,ac + ido) - ref(cc,ac + 2*ido);
          ti3 = ref(cc,ac + ido) + ref(cc,ac + 2*ido);
          tr5 = ref(cc,ac - 1) - ref(cc,ac + 3*ido - 1);
          tr2 = ref(cc,ac - 1) + ref(cc,ac + 3*ido - 1);
          tr4 = ref(cc,ac + ido - 1) - ref(cc,ac + 2*ido - 1);
          tr3 = ref(cc,ac + ido - 1) + ref(cc,ac + 2*ido - 1);
          ah = i + (k - 1)*ido;
          ch[ah] = ref(cc,ac - ido - 1) + tr2 + tr3;
          ch[ah + 1] = ref(cc,ac - ido) + ti2 + ti3;
          cr2 = ref(cc,ac - ido - 1) + tr11*tr2 + tr12*tr3;

          ci2 = ref(cc,ac - ido) + tr11*ti2 + tr12*ti3;
          cr3 = ref(cc,ac - ido - 1) + tr12*tr2 + tr11*tr3;

          ci3 = ref(cc,ac - ido) + tr12*ti2 + tr11*ti3;
          cr5 = isign*(ti11*tr5 + ti12*tr4);
          ci5 = isign*(ti11*ti5 + ti12*ti4);
          cr4 = isign*(ti12*tr5 - ti11*tr4);
          ci4 = isign*(ti12*ti5 - ti11*ti4);
          dr3 = cr3 - ci4;
          dr4 = cr3 + ci4;
          di3 = ci3 + cr4;
          di4 = ci3 - cr4;
          dr5 = cr2 + ci5;
          dr2 = cr2 - ci5;
          di5 = ci2 - cr5;
          di2 = ci2 + cr5;
          ch[ah + l1*ido] = wa1[i]*dr2 - isign*wa1[i+1]*di2;
          ch[ah + l1*ido + 1] = wa1[i]*di2 + isign*wa1[i+1]*dr2;
          ch[ah + 2*l1*ido] = wa2[i]*dr3 - isign*wa2[i+1]*di3;
          ch[ah + 2*l1*ido + 1] = wa2[i]*di3 + isign*wa2[i+1]*dr3;
          ch[ah + 3*l1*ido] = wa3[i]*dr4 - isign*wa3[i+1]*di4;
          ch[ah + 3*l1*ido + 1] = wa3[i]*di4 + isign*wa3[i+1]*dr4;
          ch[ah + 4*l1*ido] = wa4[i]*dr5 - isign*wa4[i+1]*di5;
          ch[ah + 4*l1*ido + 1] = wa4[i]*di5 + isign*wa4[i+1]*dr5;
        }
      }
    }
  } /* passf5 */


static void passf(int *nac, int ido, int ip, int l1, int idl1,
      Treal cc[], Treal ch[],
      const Treal wa[], int isign)
  /* isign is -1 for forward transform and +1 for backward transform */
  {
    int idij, idlj, idot, ipph, i, j, k, l, jc, lc, ik, idj, idl, inc,idp;
    Treal wai, war;

    idot = ido / 2;
    /* nt = ip*idl1;*/
    ipph = (ip + 1) / 2;
    idp = ip*ido;
    if (ido >= l1) {
      for (j=1; j<ipph; j++) {
        jc = ip - j;
        for (k=0; k<l1; k++) {
          for (i=0; i<ido; i++) {
            ch[i + (k + j*l1)*ido] =
                ref(cc,i + (j + k*ip)*ido) + ref(cc,i + (jc + k*ip)*ido);
            ch[i + (k + jc*l1)*ido] =
                ref(cc,i + (j + k*ip)*ido) - ref(cc,i + (jc + k*ip)*ido);
          }
        }
      }
      for (k=0; k<l1; k++)
        for (i=0; i<ido; i++)
          ch[i + k*ido] = ref(cc,i + k*ip*ido);
    } else {
      for (j=1; j<ipph; j++) {
        jc = ip - j;
        for (i=0; i<ido; i++) {
          for (k=0; k<l1; k++) {
            ch[i + (k + j*l1)*ido] = ref(cc,i + (j + k*ip)*ido) + ref(cc,i + (jc + k*
                ip)*ido);
            ch[i + (k + jc*l1)*ido] = ref(cc,i + (j + k*ip)*ido) - ref(cc,i + (jc + k*
                ip)*ido);
          }
        }
      }
      for (i=0; i<ido; i++)
        for (k=0; k<l1; k++)
          ch[i + k*ido] = ref(cc,i + k*ip*ido);
    }

    idl = 2 - ido;
    inc = 0;
    for (l=1; l<ipph; l++) {
      lc = ip - l;
      idl += ido;
      for (ik=0; ik<idl1; ik++) {
        cc[ik + l*idl1] = ch[ik] + wa[idl - 2]*ch[ik + idl1];
        cc[ik + lc*idl1] = isign*wa[idl-1]*ch[ik + (ip-1)*idl1];
      }
      idlj = idl;
      inc += ido;
      for (j=2; j<ipph; j++) {
        jc = ip - j;
        idlj += inc;
        if (idlj > idp) idlj -= idp;
        war = wa[idlj - 2];
        wai = wa[idlj-1];
        for (ik=0; ik<idl1; ik++) {
          cc[ik + l*idl1] += war*ch[ik + j*idl1];
          cc[ik + lc*idl1] += isign*wai*ch[ik + jc*idl1];
        }
      }
    }
    for (j=1; j<ipph; j++)
      for (ik=0; ik<idl1; ik++)
        ch[ik] += ch[ik + j*idl1];
    for (j=1; j<ipph; j++) {
      jc = ip - j;
      for (ik=1; ik<idl1; ik+=2) {
        ch[ik - 1 + j*idl1] = cc[ik - 1 + j*idl1] - cc[ik + jc*idl1];
        ch[ik - 1 + jc*idl1] = cc[ik - 1 + j*idl1] + cc[ik + jc*idl1];
        ch[ik + j*idl1] = cc[ik + j*idl1] + cc[ik - 1 + jc*idl1];
        ch[ik + jc*idl1] = cc[ik + j*idl1] - cc[ik - 1 + jc*idl1];
      }
    }
    *nac = 1;
    if (ido == 2) return;
    *nac = 0;
    for (ik=0; ik<idl1; ik++)
      cc[ik] = ch[ik];
    for (j=1; j<ip; j++) {
      for (k=0; k<l1; k++) {
        cc[(k + j*l1)*ido + 0] = ch[(k + j*l1)*ido + 0];
        cc[(k + j*l1)*ido + 1] = ch[(k + j*l1)*ido + 1];
      }
    }
    if (idot <= l1) {
      idij = 0;
      for (j=1; j<ip; j++) {
        idij += 2;
        for (i=3; i<ido; i+=2) {
          idij += 2;
          for (k=0; k<l1; k++) {
            cc[i - 1 + (k + j*l1)*ido] =
                wa[idij - 2]*ch[i - 1 + (k + j*l1)*ido] -
                isign*wa[idij-1]*ch[i + (k + j*l1)*ido];
            cc[i + (k + j*l1)*ido] =
                wa[idij - 2]*ch[i + (k + j*l1)*ido] +
                isign*wa[idij-1]*ch[i - 1 + (k + j*l1)*ido];
          }
        }
      }
    } else {
      idj = 2 - ido;
      for (j=1; j<ip; j++) {
        idj += ido;
        for (k = 0; k < l1; k++) {
          idij = idj;
          for (i=3; i<ido; i+=2) {
            idij += 2;
            cc[i - 1 + (k + j*l1)*ido] =
                wa[idij - 2]*ch[i - 1 + (k + j*l1)*ido] -
                isign*wa[idij-1]*ch[i + (k + j*l1)*ido];
            cc[i + (k + j*l1)*ido] =
                wa[idij - 2]*ch[i + (k + j*l1)*ido] +
                isign*wa[idij-1]*ch[i - 1 + (k + j*l1)*ido];
          }
        }
      }
    }
  } /* passf */


  /* ----------------------------------------------------------------------
radf2,radb2, radf3,radb3, radf4,radb4, radf5,radb5, radfg,radbg.
Treal FFT passes fwd and bwd.
---------------------------------------------------------------------- */

static void radf2(int ido, int l1, const Treal cc[], Treal ch[], const Treal wa1[])
  {
    int i, k, ic;
    Treal ti2, tr2;
    for (k=0; k<l1; k++) {
      ch[2*k*ido] =
          ref(cc,k*ido) + ref(cc,(k + l1)*ido);
      ch[(2*k+1)*ido + ido-1] =
          ref(cc,k*ido) - ref(cc,(k + l1)*ido);
    }
    if (ido < 2) return;
    if (ido != 2) {
      for (k=0; k<l1; k++) {
        for (i=2; i<ido; i+=2) {
          ic = ido - i;
          tr2 = wa1[i - 2]*ref(cc, i-1 + (k + l1)*ido) + wa1[i - 1]*ref(cc, i + (k + l1)*ido);
          ti2 = wa1[i - 2]*ref(cc, i + (k + l1)*ido) - wa1[i - 1]*ref(cc, i-1 + (k + l1)*ido);
          ch[i + 2*k*ido] = ref(cc,i + k*ido) + ti2;
          ch[ic + (2*k+1)*ido] = ti2 - ref(cc,i + k*ido);
          ch[i - 1 + 2*k*ido] = ref(cc,i - 1 + k*ido) + tr2;
          ch[ic - 1 + (2*k+1)*ido] = ref(cc,i - 1 + k*ido) - tr2;
        }
      }
      if (ido % 2 == 1) return;
    }
    for (k=0; k<l1; k++) {
      ch[(2*k+1)*ido] = -ref(cc,ido-1 + (k + l1)*ido);
      ch[ido-1 + 2*k*ido] = ref(cc,ido-1 + k*ido);
    }
  } /* radf2 */


static void radb2(int ido, int l1, const Treal cc[], Treal ch[], const Treal wa1[])
  {
    int i, k, ic;
    Treal ti2, tr2;
    for (k=0; k<l1; k++) {
      ch[k*ido] =
          ref(cc,2*k*ido) + ref(cc,ido-1 + (2*k+1)*ido);
      ch[(k + l1)*ido] =
          ref(cc,2*k*ido) - ref(cc,ido-1 + (2*k+1)*ido);
    }
    if (ido < 2) return;
    if (ido != 2) {
      for (k = 0; k < l1; ++k) {
        for (i = 2; i < ido; i += 2) {
          ic = ido - i;
          ch[i-1 + k*ido] =
              ref(cc,i-1 + 2*k*ido) + ref(cc,ic-1 + (2*k+1)*ido);
          tr2 = ref(cc,i-1 + 2*k*ido) - ref(cc,ic-1 + (2*k+1)*ido);
          ch[i + k*ido] =
              ref(cc,i + 2*k*ido) - ref(cc,ic + (2*k+1)*ido);
          ti2 = ref(cc,i + (2*k)*ido) + ref(cc,ic + (2*k+1)*ido);
          ch[i-1 + (k + l1)*ido] =
              wa1[i - 2]*tr2 - wa1[i - 1]*ti2;
          ch[i + (k + l1)*ido] =
              wa1[i - 2]*ti2 + wa1[i - 1]*tr2;
        }
      }
      if (ido % 2 == 1) return;
    }
    for (k = 0; k < l1; k++) {
      ch[ido-1 + k*ido] = 2*ref(cc,ido-1 + 2*k*ido);
      ch[ido-1 + (k + l1)*ido] = -2*ref(cc,(2*k+1)*ido);
    }
  } /* radb2 */


static void radf3(int ido, int l1, const Treal cc[], Treal ch[],
      const Treal wa1[], const Treal wa2[])
  {
    static const Treal taur = -0.5;
    static const Treal taui = 0.86602540378443864676;
    int i, k, ic;
    Treal ci2, di2, di3, cr2, dr2, dr3, ti2, ti3, tr2, tr3;
    for (k=0; k<l1; k++) {
      cr2 = ref(cc,(k + l1)*ido) + ref(cc,(k + 2*l1)*ido);
      ch[3*k*ido] = ref(cc,k*ido) + cr2;
      ch[(3*k+2)*ido] = taui*(ref(cc,(k + l1*2)*ido) - ref(cc,(k + l1)*ido));
      ch[ido-1 + (3*k + 1)*ido] = ref(cc,k*ido) + taur*cr2;
    }
    if (ido == 1) return;
    for (k=0; k<l1; k++) {
      for (i=2; i<ido; i+=2) {
        ic = ido - i;
        dr2 = wa1[i - 2]*ref(cc,i - 1 + (k + l1)*ido) +
            wa1[i - 1]*ref(cc,i + (k + l1)*ido);
        di2 = wa1[i - 2]*ref(cc,i + (k + l1)*ido) - wa1[i - 1]*ref(cc,i - 1 + (k + l1)*ido);
        dr3 = wa2[i - 2]*ref(cc,i - 1 + (k + l1*2)*ido) + wa2[i - 1]*ref(cc,i + (k + l1*2)*ido);
        di3 = wa2[i - 2]*ref(cc,i + (k + l1*2)*ido) - wa2[i - 1]*ref(cc,i - 1 + (k + l1*2)*ido);
        cr2 = dr2 + dr3;
        ci2 = di2 + di3;
        ch[i - 1 + 3*k*ido] = ref(cc,i - 1 + k*ido) + cr2;
        ch[i + 3*k*ido] = ref(cc,i + k*ido) + ci2;
        tr2 = ref(cc,i - 1 + k*ido) + taur*cr2;
        ti2 = ref(cc,i + k*ido) + taur*ci2;
        tr3 = taui*(di2 - di3);
        ti3 = taui*(dr3 - dr2);
        ch[i - 1 + (3*k + 2)*ido] = tr2 + tr3;
        ch[ic - 1 + (3*k + 1)*ido] = tr2 - tr3;
        ch[i + (3*k + 2)*ido] = ti2 + ti3;
        ch[ic + (3*k + 1)*ido] = ti3 - ti2;
      }
    }
  } /* radf3 */


static void radb3(int ido, int l1, const Treal cc[], Treal ch[],
      const Treal wa1[], const Treal wa2[])
  {
    static const Treal taur = -0.5;
    static const Treal taui = 0.86602540378443864676;
    int i, k, ic;
    Treal ci2, ci3, di2, di3, cr2, cr3, dr2, dr3, ti2, tr2;
    for (k=0; k<l1; k++) {
      tr2 = 2*ref(cc,ido-1 + (3*k + 1)*ido);
      cr2 = ref(cc,3*k*ido) + taur*tr2;
      ch[k*ido] = ref(cc,3*k*ido) + tr2;
      ci3 = 2*taui*ref(cc,(3*k + 2)*ido);
      ch[(k + l1)*ido] = cr2 - ci3;
      ch[(k + 2*l1)*ido] = cr2 + ci3;
    }
    if (ido == 1) return;
    for (k=0; k<l1; k++) {
      for (i=2; i<ido; i+=2) {
        ic = ido - i;
        tr2 = ref(cc,i - 1 + (3*k + 2)*ido) + ref(cc,ic - 1 + (3*k + 1)*ido);
        cr2 = ref(cc,i - 1 + 3*k*ido) + taur*tr2;
        ch[i - 1 + k*ido] = ref(cc,i - 1 + 3*k*ido) + tr2;
        ti2 = ref(cc,i + (3*k + 2)*ido) - ref(cc,ic + (3*k + 1)*ido);
        ci2 = ref(cc,i + 3*k*ido) + taur*ti2;
        ch[i + k*ido] = ref(cc,i + 3*k*ido) + ti2;
        cr3 = taui*(ref(cc,i - 1 + (3*k + 2)*ido) - ref(cc,ic - 1 + (3*k + 1)*ido));
        ci3 = taui*(ref(cc,i + (3*k + 2)*ido) + ref(cc,ic + (3*k + 1)*ido));
        dr2 = cr2 - ci3;
        dr3 = cr2 + ci3;
        di2 = ci2 + cr3;
        di3 = ci2 - cr3;
        ch[i - 1 + (k + l1)*ido] = wa1[i - 2]*dr2 - wa1[i - 1]*di2;
        ch[i + (k + l1)*ido] = wa1[i - 2]*di2 + wa1[i - 1]*dr2;
        ch[i - 1 + (k + 2*l1)*ido] = wa2[i - 2]*dr3 - wa2[i - 1]*di3;
        ch[i + (k + 2*l1)*ido] = wa2[i - 2]*di3 + wa2[i - 1]*dr3;
      }
    }
  } /* radb3 */


static void radf4(int ido, int l1, const Treal cc[], Treal ch[],
      const Treal wa1[], const Treal wa2[], const Treal wa3[])
  {
    static const Treal hsqt2 = 0.70710678118654752440;
    int i, k, ic;
    Treal ci2, ci3, ci4, cr2, cr3, cr4, ti1, ti2, ti3, ti4, tr1, tr2, tr3, tr4;
    for (k=0; k<l1; k++) {
      tr1 = ref(cc,(k + l1)*ido) + ref(cc,(k + 3*l1)*ido);
      tr2 = ref(cc,k*ido) + ref(cc,(k + 2*l1)*ido);
      ch[4*k*ido] = tr1 + tr2;
      ch[ido-1 + (4*k + 3)*ido] = tr2 - tr1;
      ch[ido-1 + (4*k + 1)*ido] = ref(cc,k*ido) - ref(cc,(k + 2*l1)*ido);
      ch[(4*k + 2)*ido] = ref(cc,(k + 3*l1)*ido) - ref(cc,(k + l1)*ido);
    }
    if (ido < 2) return;
    if (ido != 2) {
      for (k=0; k<l1; k++) {
        for (i=2; i<ido; i += 2) {
          ic = ido - i;
          cr2 = wa1[i - 2]*ref(cc,i - 1 + (k + l1)*ido) + wa1[i - 1]*ref(cc,i + (k + l1)*ido);
          ci2 = wa1[i - 2]*ref(cc,i + (k + l1)*ido) - wa1[i - 1]*ref(cc,i - 1 + (k + l1)*ido);
          cr3 = wa2[i - 2]*ref(cc,i - 1 + (k + 2*l1)*ido) + wa2[i - 1]*ref(cc,i + (k + 2*l1)*
              ido);
          ci3 = wa2[i - 2]*ref(cc,i + (k + 2*l1)*ido) - wa2[i - 1]*ref(cc,i - 1 + (k + 2*l1)*
              ido);
          cr4 = wa3[i - 2]*ref(cc,i - 1 + (k + 3*l1)*ido) + wa3[i - 1]*ref(cc,i + (k + 3*l1)*
              ido);
          ci4 = wa3[i - 2]*ref(cc,i + (k + 3*l1)*ido) - wa3[i - 1]*ref(cc,i - 1 + (k + 3*l1)*
              ido);
          tr1 = cr2 + cr4;
          tr4 = cr4 - cr2;
          ti1 = ci2 + ci4;
          ti4 = ci2 - ci4;
          ti2 = ref(cc,i + k*ido) + ci3;
          ti3 = ref(cc,i + k*ido) - ci3;
          tr2 = ref(cc,i - 1 + k*ido) + cr3;
          tr3 = ref(cc,i - 1 + k*ido) - cr3;
          ch[i - 1 + 4*k*ido] = tr1 + tr2;
          ch[ic - 1 + (4*k + 3)*ido] = tr2 - tr1;
          ch[i + 4*k*ido] = ti1 + ti2;
          ch[ic + (4*k + 3)*ido] = ti1 - ti2;
          ch[i - 1 + (4*k + 2)*ido] = ti4 + tr3;
          ch[ic - 1 + (4*k + 1)*ido] = tr3 - ti4;
          ch[i + (4*k + 2)*ido] = tr4 + ti3;
          ch[ic + (4*k + 1)*ido] = tr4 - ti3;
        }
      }
      if (ido % 2 == 1) return;
    }
    for (k=0; k<l1; k++) {
      ti1 = -hsqt2*(ref(cc,ido-1 + (k + l1)*ido) + ref(cc,ido-1 + (k + 3*l1)*ido));
      tr1 = hsqt2*(ref(cc,ido-1 + (k + l1)*ido) - ref(cc,ido-1 + (k + 3*l1)*ido));
      ch[ido-1 + 4*k*ido] = tr1 + ref(cc,ido-1 + k*ido);
      ch[ido-1 + (4*k + 2)*ido] = ref(cc,ido-1 + k*ido) - tr1;
      ch[(4*k + 1)*ido] = ti1 - ref(cc,ido-1 + (k + 2*l1)*ido);
      ch[(4*k + 3)*ido] = ti1 + ref(cc,ido-1 + (k + 2*l1)*ido);
    }
  } /* radf4 */


static void radb4(int ido, int l1, const Treal cc[], Treal ch[],
      const Treal wa1[], const Treal wa2[], const Treal wa3[])
  {
    static const Treal sqrt2 = 1.41421356237309504880;
    int i, k, ic;
    Treal ci2, ci3, ci4, cr2, cr3, cr4, ti1, ti2, ti3, ti4, tr1, tr2, tr3, tr4;
    for (k = 0; k < l1; k++) {
      tr1 = ref(cc,4*k*ido) - ref(cc,ido-1 + (4*k + 3)*ido);
      tr2 = ref(cc,4*k*ido) + ref(cc,ido-1 + (4*k + 3)*ido);
      tr3 = ref(cc,ido-1 + (4*k + 1)*ido) + ref(cc,ido-1 + (4*k + 1)*ido);
      tr4 = ref(cc,(4*k + 2)*ido) + ref(cc,(4*k + 2)*ido);
      ch[k*ido] = tr2 + tr3;
      ch[(k + l1)*ido] = tr1 - tr4;
      ch[(k + 2*l1)*ido] = tr2 - tr3;
      ch[(k + 3*l1)*ido] = tr1 + tr4;
    }
    if (ido < 2) return;
    if (ido != 2) {
      for (k = 0; k < l1; ++k) {
        for (i = 2; i < ido; i += 2) {
          ic = ido - i;
          ti1 = ref(cc,i + 4*k*ido) + ref(cc,ic + (4*k + 3)*ido);
          ti2 = ref(cc,i + 4*k*ido) - ref(cc,ic + (4*k + 3)*ido);
          ti3 = ref(cc,i + (4*k + 2)*ido) - ref(cc,ic + (4*k + 1)*ido);
          tr4 = ref(cc,i + (4*k + 2)*ido) + ref(cc,ic + (4*k + 1)*ido);
          tr1 = ref(cc,i - 1 + 4*k*ido) - ref(cc,ic - 1 + (4*k + 3)*ido);
          tr2 = ref(cc,i - 1 + 4*k*ido) + ref(cc,ic - 1 + (4*k + 3)*ido);
          ti4 = ref(cc,i - 1 + (4*k + 2)*ido) - ref(cc,ic - 1 + (4*k + 1)*ido);
          tr3 = ref(cc,i - 1 + (4*k + 2)*ido) + ref(cc,ic - 1 + (4*k + 1)*ido);
          ch[i - 1 + k*ido] = tr2 + tr3;
          cr3 = tr2 - tr3;
          ch[i + k*ido] = ti2 + ti3;
          ci3 = ti2 - ti3;
          cr2 = tr1 - tr4;
          cr4 = tr1 + tr4;
          ci2 = ti1 + ti4;
          ci4 = ti1 - ti4;
          ch[i - 1 + (k + l1)*ido] = wa1[i - 2]*cr2 - wa1[i - 1]*ci2;
          ch[i + (k + l1)*ido] = wa1[i - 2]*ci2 + wa1[i - 1]*cr2;
          ch[i - 1 + (k + 2*l1)*ido] = wa2[i - 2]*cr3 - wa2[i - 1]*ci3;
          ch[i + (k + 2*l1)*ido] = wa2[i - 2]*ci3 + wa2[i - 1]*cr3;
          ch[i - 1 + (k + 3*l1)*ido] = wa3[i - 2]*cr4 - wa3[i - 1]*ci4;
          ch[i + (k + 3*l1)*ido] = wa3[i - 2]*ci4 + wa3[i - 1]*cr4;
        }
      }
      if (ido % 2 == 1) return;
    }
    for (k = 0; k < l1; k++) {
      ti1 = ref(cc,(4*k + 1)*ido) + ref(cc,(4*k + 3)*ido);
      ti2 = ref(cc,(4*k + 3)*ido) - ref(cc,(4*k + 1)*ido);
      tr1 = ref(cc,ido-1 + 4*k*ido) - ref(cc,ido-1 + (4*k + 2)*ido);
      tr2 = ref(cc,ido-1 + 4*k*ido) + ref(cc,ido-1 + (4*k + 2)*ido);
      ch[ido-1 + k*ido] = tr2 + tr2;
      ch[ido-1 + (k + l1)*ido] = sqrt2*(tr1 - ti1);
      ch[ido-1 + (k + 2*l1)*ido] = ti2 + ti2;
      ch[ido-1 + (k + 3*l1)*ido] = -sqrt2*(tr1 + ti1);
    }
  } /* radb4 */


static void radf5(int ido, int l1, const Treal cc[], Treal ch[],
      const Treal wa1[], const Treal wa2[], const Treal wa3[], const Treal wa4[])
  {
    static const Treal tr11 = 0.3090169943749474241;
    static const Treal ti11 = 0.95105651629515357212;
    static const Treal tr12 = -0.8090169943749474241;
    static const Treal ti12 = 0.58778525229247312917;
    int i, k, ic;
    Treal ci2, di2, ci4, ci5, di3, di4, di5, ci3, cr2, cr3, dr2, dr3, dr4, dr5,
        cr5, cr4, ti2, ti3, ti5, ti4, tr2, tr3, tr4, tr5;
    for (k = 0; k < l1; k++) {
      cr2 = ref(cc,(k + 4*l1)*ido) + ref(cc,(k + l1)*ido);
      ci5 = ref(cc,(k + 4*l1)*ido) - ref(cc,(k + l1)*ido);
      cr3 = ref(cc,(k + 3*l1)*ido) + ref(cc,(k + 2*l1)*ido);
      ci4 = ref(cc,(k + 3*l1)*ido) - ref(cc,(k + 2*l1)*ido);
      ch[5*k*ido] = ref(cc,k*ido) + cr2 + cr3;
      ch[ido-1 + (5*k + 1)*ido] = ref(cc,k*ido) + tr11*cr2 + tr12*cr3;
      ch[(5*k + 2)*ido] = ti11*ci5 + ti12*ci4;
      ch[ido-1 + (5*k + 3)*ido] = ref(cc,k*ido) + tr12*cr2 + tr11*cr3;
      ch[(5*k + 4)*ido] = ti12*ci5 - ti11*ci4;
    }
    if (ido == 1) return;
    for (k = 0; k < l1; ++k) {
      for (i = 2; i < ido; i += 2) {
        ic = ido - i;
        dr2 = wa1[i - 2]*ref(cc,i - 1 + (k + l1)*ido) + wa1[i - 1]*ref(cc,i + (k + l1)*ido);
        di2 = wa1[i - 2]*ref(cc,i + (k + l1)*ido) - wa1[i - 1]*ref(cc,i - 1 + (k + l1)*ido);
        dr3 = wa2[i - 2]*ref(cc,i - 1 + (k + 2*l1)*ido) + wa2[i - 1]*ref(cc,i + (k + 2*l1)*ido);
        di3 = wa2[i - 2]*ref(cc,i + (k + 2*l1)*ido) - wa2[i - 1]*ref(cc,i - 1 + (k + 2*l1)*ido);
        dr4 = wa3[i - 2]*ref(cc,i - 1 + (k + 3*l1)*ido) + wa3[i - 1]*ref(cc,i + (k + 3*l1)*ido);
        di4 = wa3[i - 2]*ref(cc,i + (k + 3*l1)*ido) - wa3[i - 1]*ref(cc,i - 1 + (k + 3*l1)*ido);
        dr5 = wa4[i - 2]*ref(cc,i - 1 + (k + 4*l1)*ido) + wa4[i - 1]*ref(cc,i + (k + 4*l1)*ido);
        di5 = wa4[i - 2]*ref(cc,i + (k + 4*l1)*ido) - wa4[i - 1]*ref(cc,i - 1 + (k + 4*l1)*ido);
        cr2 = dr2 + dr5;
        ci5 = dr5 - dr2;
        cr5 = di2 - di5;
        ci2 = di2 + di5;
        cr3 = dr3 + dr4;
        ci4 = dr4 - dr3;
        cr4 = di3 - di4;
        ci3 = di3 + di4;
        ch[i - 1 + 5*k*ido] = ref(cc,i - 1 + k*ido) + cr2 + cr3;
        ch[i + 5*k*ido] = ref(cc,i + k*ido) + ci2 + ci3;
        tr2 = ref(cc,i - 1 + k*ido) + tr11*cr2 + tr12*cr3;
        ti2 = ref(cc,i + k*ido) + tr11*ci2 + tr12*ci3;
        tr3 = ref(cc,i - 1 + k*ido) + tr12*cr2 + tr11*cr3;
        ti3 = ref(cc,i + k*ido) + tr12*ci2 + tr11*ci3;
        tr5 = ti11*cr5 + ti12*cr4;
        ti5 = ti11*ci5 + ti12*ci4;
        tr4 = ti12*cr5 - ti11*cr4;
        ti4 = ti12*ci5 - ti11*ci4;
        ch[i - 1 + (5*k + 2)*ido] = tr2 + tr5;
        ch[ic - 1 + (5*k + 1)*ido] = tr2 - tr5;
        ch[i + (5*k + 2)*ido] = ti2 + ti5;
        ch[ic + (5*k + 1)*ido] = ti5 - ti2;
        ch[i - 1 + (5*k + 4)*ido] = tr3 + tr4;
        ch[ic - 1 + (5*k + 3)*ido] = tr3 - tr4;
        ch[i + (5*k + 4)*ido] = ti3 + ti4;
        ch[ic + (5*k + 3)*ido] = ti4 - ti3;
      }
    }
  } /* radf5 */


static void radb5(int ido, int l1, const Treal cc[], Treal ch[],
      const Treal wa1[], const Treal wa2[], const Treal wa3[], const Treal wa4[])
  {
    static const Treal tr11 = 0.3090169943749474241;
    static const Treal ti11 = 0.95105651629515357212;
    static const Treal tr12 = -0.8090169943749474241;
    static const Treal ti12 = 0.58778525229247312917;
    int i, k, ic;
    Treal ci2, ci3, ci4, ci5, di3, di4, di5, di2, cr2, cr3, cr5, cr4, ti2, ti3,
        ti4, ti5, dr3, dr4, dr5, dr2, tr2, tr3, tr4, tr5;
    for (k = 0; k < l1; k++) {
      ti5 = 2*ref(cc,(5*k + 2)*ido);
      ti4 = 2*ref(cc,(5*k + 4)*ido);
      tr2 = 2*ref(cc,ido-1 + (5*k + 1)*ido);
      tr3 = 2*ref(cc,ido-1 + (5*k + 3)*ido);
      ch[k*ido] = ref(cc,5*k*ido) + tr2 + tr3;
      cr2 = ref(cc,5*k*ido) + tr11*tr2 + tr12*tr3;
      cr3 = ref(cc,5*k*ido) + tr12*tr2 + tr11*tr3;
      ci5 = ti11*ti5 + ti12*ti4;
      ci4 = ti12*ti5 - ti11*ti4;
      ch[(k + l1)*ido] = cr2 - ci5;
      ch[(k + 2*l1)*ido] = cr3 - ci4;
      ch[(k + 3*l1)*ido] = cr3 + ci4;
      ch[(k + 4*l1)*ido] = cr2 + ci5;
    }
    if (ido == 1) return;
    for (k = 0; k < l1; ++k) {
      for (i = 2; i < ido; i += 2) {
        ic = ido - i;
        ti5 = ref(cc,i + (5*k + 2)*ido) + ref(cc,ic + (5*k + 1)*ido);
        ti2 = ref(cc,i + (5*k + 2)*ido) - ref(cc,ic + (5*k + 1)*ido);
        ti4 = ref(cc,i + (5*k + 4)*ido) + ref(cc,ic + (5*k + 3)*ido);
        ti3 = ref(cc,i + (5*k + 4)*ido) - ref(cc,ic + (5*k + 3)*ido);
        tr5 = ref(cc,i - 1 + (5*k + 2)*ido) - ref(cc,ic - 1 + (5*k + 1)*ido);
        tr2 = ref(cc,i - 1 + (5*k + 2)*ido) + ref(cc,ic - 1 + (5*k + 1)*ido);
        tr4 = ref(cc,i - 1 + (5*k + 4)*ido) - ref(cc,ic - 1 + (5*k + 3)*ido);
        tr3 = ref(cc,i - 1 + (5*k + 4)*ido) + ref(cc,ic - 1 + (5*k + 3)*ido);
        ch[i - 1 + k*ido] = ref(cc,i - 1 + 5*k*ido) + tr2 + tr3;
        ch[i + k*ido] = ref(cc,i + 5*k*ido) + ti2 + ti3;
        cr2 = ref(cc,i - 1 + 5*k*ido) + tr11*tr2 + tr12*tr3;

        ci2 = ref(cc,i + 5*k*ido) + tr11*ti2 + tr12*ti3;
        cr3 = ref(cc,i - 1 + 5*k*ido) + tr12*tr2 + tr11*tr3;

        ci3 = ref(cc,i + 5*k*ido) + tr12*ti2 + tr11*ti3;
        cr5 = ti11*tr5 + ti12*tr4;
        ci5 = ti11*ti5 + ti12*ti4;
        cr4 = ti12*tr5 - ti11*tr4;
        ci4 = ti12*ti5 - ti11*ti4;
        dr3 = cr3 - ci4;
        dr4 = cr3 + ci4;
        di3 = ci3 + cr4;
        di4 = ci3 - cr4;
        dr5 = cr2 + ci5;
        dr2 = cr2 - ci5;
        di5 = ci2 - cr5;
        di2 = ci2 + cr5;
        ch[i - 1 + (k + l1)*ido] = wa1[i - 2]*dr2 - wa1[i - 1]*di2;
        ch[i + (k + l1)*ido] = wa1[i - 2]*di2 + wa1[i - 1]*dr2;
        ch[i - 1 + (k + 2*l1)*ido] = wa2[i - 2]*dr3 - wa2[i - 1]*di3;
        ch[i + (k + 2*l1)*ido] = wa2[i - 2]*di3 + wa2[i - 1]*dr3;
        ch[i - 1 + (k + 3*l1)*ido] = wa3[i - 2]*dr4 - wa3[i - 1]*di4;
        ch[i + (k + 3*l1)*ido] = wa3[i - 2]*di4 + wa3[i - 1]*dr4;
        ch[i - 1 + (k + 4*l1)*ido] = wa4[i - 2]*dr5 - wa4[i - 1]*di5;
        ch[i + (k + 4*l1)*ido] = wa4[i - 2]*di5 + wa4[i - 1]*dr5;
      }
    }
  } /* radb5 */


static void radfg(int ido, int ip, int l1, int idl1,
      Treal cc[], Treal ch[], const Treal wa[])
  {
    int idij, ipph, i, j, k, l, j2, ic, jc, lc, ik, is, nbd;    
    Treal dc2, ai1, ai2, ar1, ar2, ds2, dcp, dsp, ar1h, ar2h;
    sincos2pi(1, ip, &dsp, &dcp);
    ipph = (ip + 1) / 2;
    nbd = (ido - 1) / 2;
    if (ido != 1) {
      for (ik=0; ik<idl1; ik++) ch[ik] = cc[ik];
      for (j=1; j<ip; j++)
        for (k=0; k<l1; k++)
          ch[(k + j*l1)*ido] = cc[(k + j*l1)*ido];
      if (nbd <= l1) {
        is = -ido;
        for (j=1; j<ip; j++) {
          is += ido;
          idij = is-1;
          for (i=2; i<ido; i+=2) {
            idij += 2;
            for (k=0; k<l1; k++) {
              ch[i - 1 + (k + j*l1)*ido] =
                  wa[idij - 1]*cc[i - 1 + (k + j*l1)*ido] + wa[idij]*cc[i + (k + j*l1)*ido];
              ch[i + (k + j*l1)*ido] =
                  wa[idij - 1]*cc[i + (k + j*l1)*ido] - wa[idij]*cc[i - 1 + (k + j*l1)*ido];
            }
          }
        }
      } else {
        is = -ido;
        for (j=1; j<ip; j++) {
          is += ido;
          for (k=0; k<l1; k++) {
            idij = is-1;
            for (i=2; i<ido; i+=2) {
              idij += 2;
              ch[i - 1 + (k + j*l1)*ido] =
                  wa[idij - 1]*cc[i - 1 + (k + j*l1)*ido] + wa[idij]*cc[i + (k + j*l1)*ido];
              ch[i + (k + j*l1)*ido] =
                  wa[idij - 1]*cc[i + (k + j*l1)*ido] - wa[idij]*cc[i - 1 + (k + j*l1)*ido];
            }
          }
        }
      }
      if (nbd >= l1) {
        for (j=1; j<ipph; j++) {
          jc = ip - j;
          for (k=0; k<l1; k++) {
            for (i=2; i<ido; i+=2) {
              cc[i - 1 + (k + j*l1)*ido] = ch[i - 1 + (k + j*l1)*ido] + ch[i - 1 + (k + jc*l1)*ido];
              cc[i - 1 + (k + jc*l1)*ido] = ch[i + (k + j*l1)*ido] - ch[i + (k + jc*l1)*ido];
              cc[i + (k + j*l1)*ido] = ch[i + (k + j*l1)*ido] + ch[i + (k + jc*l1)*ido];
              cc[i + (k + jc*l1)*ido] = ch[i - 1 + (k + jc*l1)*ido] - ch[i - 1 + (k + j*l1)*ido];
            }
          }
        }
      } else {
        for (j=1; j<ipph; j++) {
          jc = ip - j;
          for (i=2; i<ido; i+=2) {
            for (k=0; k<l1; k++) {
              cc[i - 1 + (k + j*l1)*ido] =
                  ch[i - 1 + (k + j*l1)*ido] + ch[i - 1 + (k + jc*l1)*ido];
              cc[i - 1 + (k + jc*l1)*ido] = ch[i + (k + j*l1)*ido] - ch[i + (k + jc*l1)*ido];
              cc[i + (k + j*l1)*ido] = ch[i + (k + j*l1)*ido] + ch[i + (k + jc*l1)*ido];
              cc[i + (k + jc*l1)*ido] = ch[i - 1 + (k + jc*l1)*ido] - ch[i - 1 + (k + j*l1)*ido];
            }
          }
        }
      }
    } else {  /* now ido == 1 */
      for (ik=0; ik<idl1; ik++) cc[ik] = ch[ik];
    }
    for (j=1; j<ipph; j++) {
      jc = ip - j;
      for (k=0; k<l1; k++) {
        cc[(k + j*l1)*ido] = ch[(k + j*l1)*ido] + ch[(k + jc*l1)*ido];
        cc[(k + jc*l1)*ido] = ch[(k + jc*l1)*ido] - ch[(k + j*l1)*ido];
      }
    }

    ar1 = 1;
    ai1 = 0;    
    for (l=1; l<ipph; l++) {
      lc = ip - l;
      ar1h = dcp*ar1 - dsp*ai1;
      ai1 = dcp*ai1 + dsp*ar1;
      ar1 = ar1h;
      for (ik=0; ik<idl1; ik++) {
        ch[ik + l*idl1] = cc[ik] + ar1*cc[ik + idl1];
        ch[ik + lc*idl1] = ai1*cc[ik + (ip-1)*idl1];
      }
      dc2 = ar1;
      ds2 = ai1;
      ar2 = ar1;
      ai2 = ai1;
      for (j=2; j<ipph; j++) {
        jc = ip - j;
        ar2h = dc2*ar2 - ds2*ai2;
        ai2 = dc2*ai2 + ds2*ar2;
        ar2 = ar2h;
        for (ik=0; ik<idl1; ik++) {
          ch[ik + l*idl1] += ar2*cc[ik + j*idl1];
          ch[ik + lc*idl1] += ai2*cc[ik + jc*idl1];
        }
      }
    }
    
    for (j=1; j<ipph; j++)
      for (ik=0; ik<idl1; ik++)
        ch[ik] += cc[ik + j*idl1];

    if (ido >= l1) {
      for (k=0; k<l1; k++) {
        for (i=0; i<ido; i++) {
          ref(cc,i + k*ip*ido) = ch[i + k*ido];
        }
      }
    } else {
      for (i=0; i<ido; i++) {
        for (k=0; k<l1; k++) {
          ref(cc,i + k*ip*ido) = ch[i + k*ido];
        }
      }
    }
    for (j=1; j<ipph; j++) {
      jc = ip - j;
      j2 = 2*j;
      for (k=0; k<l1; k++) {
        ref(cc,ido-1 + (j2 - 1 + k*ip)*ido) =
            ch[(k + j*l1)*ido];
        ref(cc,(j2 + k*ip)*ido) =
            ch[(k + jc*l1)*ido];
      }
    }
    if (ido == 1) return;
    if (nbd >= l1) {
      for (j=1; j<ipph; j++) {
        jc = ip - j;
        j2 = 2*j;
        for (k=0; k<l1; k++) {
          for (i=2; i<ido; i+=2) {
            ic = ido - i;
            ref(cc,i - 1 + (j2 + k*ip)*ido) = ch[i - 1 + (k + j*l1)*ido] + ch[i - 1 + (k + jc*l1)*ido];
            ref(cc,ic - 1 + (j2 - 1 + k*ip)*ido) = ch[i - 1 + (k + j*l1)*ido] - ch[i - 1 + (k + jc*l1)*ido];
            ref(cc,i + (j2 + k*ip)*ido) = ch[i + (k + j*l1)*ido] + ch[i + (k + jc*l1)*ido];
            ref(cc,ic + (j2 - 1 + k*ip)*ido) = ch[i + (k + jc*l1)*ido] - ch[i + (k + j*l1)*ido];
          }
        }
      }
    } else {
      for (j=1; j<ipph; j++) {
        jc = ip - j;
        j2 = 2*j;
        for (i=2; i<ido; i+=2) {
          ic = ido - i;
          for (k=0; k<l1; k++) {
            ref(cc,i - 1 + (j2 + k*ip)*ido) = ch[i - 1 + (k + j*l1)*ido] + ch[i - 1 + (k + jc*l1)*ido];
            ref(cc,ic - 1 + (j2 - 1 + k*ip)*ido) = ch[i - 1 + (k + j*l1)*ido] - ch[i - 1 + (k + jc*l1)*ido];
            ref(cc,i + (j2 + k*ip)*ido) = ch[i + (k + j*l1)*ido] + ch[i + (k + jc*l1)*ido];
            ref(cc,ic + (j2 - 1 + k*ip)*ido) = ch[i + (k + jc*l1)*ido] - ch[i + (k + j*l1)*ido];
          }
        }
      }
    }
  } /* radfg */


static void radbg(int ido, int ip, int l1, int idl1,
      Treal cc[], Treal ch[], const Treal wa[])
  {
    int idij, ipph, i, j, k, l, j2, ic, jc, lc, ik, is;
    Treal dc2, ai1, ai2, ar1, ar2, ds2;
    int nbd;
    Treal dcp, dsp, ar1h, ar2h;
    sincos2pi(1, ip, &dsp, &dcp);
    nbd = (ido - 1) / 2;
    ipph = (ip + 1) / 2;
    if (ido >= l1) {
      for (k=0; k<l1; k++) {
        for (i=0; i<ido; i++) {
          ch[i + k*ido] = ref(cc,i + k*ip*ido);
        }
      }
    } else {
      for (i=0; i<ido; i++) {
        for (k=0; k<l1; k++) {
          ch[i + k*ido] = ref(cc,i + k*ip*ido);
        }
      }
    }
    for (j=1; j<ipph; j++) {
      jc = ip - j;
      j2 = 2*j;
      for (k=0; k<l1; k++) {
        ch[(k + j*l1)*ido] = ref(cc,ido-1 + (j2 - 1 + k*ip)*ido) + ref(cc,ido-1 + (j2 - 1 + k*ip)*
            ido);
        ch[(k + jc*l1)*ido] = ref(cc,(j2 + k*ip)*ido) + ref(cc,(j2 + k*ip)*ido);
      }
    }

    if (ido != 1) {
      if (nbd >= l1) {
        for (j=1; j<ipph; j++) {
          jc = ip - j;
          for (k=0; k<l1; k++) {
            for (i=2; i<ido; i+=2) {
              ic = ido - i;
              ch[i - 1 + (k + j*l1)*ido] = ref(cc,i - 1 + (2*j + k*ip)*ido) + ref(cc,
                  ic - 1 + (2*j - 1 + k*ip)*ido);
              ch[i - 1 + (k + jc*l1)*ido] = ref(cc,i - 1 + (2*j + k*ip)*ido) -
                  ref(cc,ic - 1 + (2*j - 1 + k*ip)*ido);
              ch[i + (k + j*l1)*ido] = ref(cc,i + (2*j + k*ip)*ido) - ref(cc,ic
                  + (2*j - 1 + k*ip)*ido);
              ch[i + (k + jc*l1)*ido] = ref(cc,i + (2*j + k*ip)*ido) + ref(cc,ic
                  + (2*j - 1 + k*ip)*ido);
            }
          }
        }
      } else {
        for (j=1; j<ipph; j++) {
          jc = ip - j;
          for (i=2; i<ido; i+=2) {
            ic = ido - i;
            for (k=0; k<l1; k++) {
              ch[i - 1 + (k + j*l1)*ido] = ref(cc,i - 1 + (2*j + k*ip)*ido) + ref(cc,
                  ic - 1 + (2*j - 1 + k*ip)*ido);
              ch[i - 1 + (k + jc*l1)*ido] = ref(cc,i - 1 + (2*j + k*ip)*ido) -
                  ref(cc,ic - 1 + (2*j - 1 + k*ip)*ido);
              ch[i + (k + j*l1)*ido] = ref(cc,i + (2*j + k*ip)*ido) - ref(cc,ic
                  + (2*j - 1 + k*ip)*ido);
              ch[i + (k + jc*l1)*ido] = ref(cc,i + (2*j + k*ip)*ido) + ref(cc,ic
                  + (2*j - 1 + k*ip)*ido);
            }
          }
        }
      }
    }

    ar1 = 1;
    ai1 = 0;
    for (l=1; l<ipph; l++) {
      lc = ip - l;
      ar1h = dcp*ar1 - dsp*ai1;
      ai1 = dcp*ai1 + dsp*ar1;
      ar1 = ar1h;
      for (ik=0; ik<idl1; ik++) {
        cc[ik + l*idl1] = ch[ik] + ar1*ch[ik + idl1];
        cc[ik + lc*idl1] = ai1*ch[ik + (ip-1)*idl1];
      }
      dc2 = ar1;
      ds2 = ai1;
      ar2 = ar1;
      ai2 = ai1;
      for (j=2; j<ipph; j++) {
        jc = ip - j;
        ar2h = dc2*ar2 - ds2*ai2;
        ai2 = dc2*ai2 + ds2*ar2;
        ar2 = ar2h;
        for (ik=0; ik<idl1; ik++) {
          cc[ik + l*idl1] += ar2*ch[ik + j*idl1];
          cc[ik + lc*idl1] += ai2*ch[ik + jc*idl1];
        }
      }
    }
    for (j=1; j<ipph; j++) {
      for (ik=0; ik<idl1; ik++) {
        ch[ik] += ch[ik + j*idl1];
      }
    }
    for (j=1; j<ipph; j++) {
      jc = ip - j;
      for (k=0; k<l1; k++) {
        ch[(k + j*l1)*ido] = cc[(k + j*l1)*ido] - cc[(k + jc*l1)*ido];
        ch[(k + jc*l1)*ido] = cc[(k + j*l1)*ido] + cc[(k + jc*l1)*ido];
      }
    }

    if (ido == 1) return;
    if (nbd >= l1) {
      for (j=1; j<ipph; j++) {
        jc = ip - j;
        for (k=0; k<l1; k++) {
          for (i=2; i<ido; i+=2) {
            ch[i - 1 + (k + j*l1)*ido] = cc[i - 1 + (k + j*l1)*ido] - cc[i + (k + jc*l1)*ido];
            ch[i - 1 + (k + jc*l1)*ido] = cc[i - 1 + (k + j*l1)*ido] + cc[i + (k + jc*l1)*ido];
            ch[i + (k + j*l1)*ido] = cc[i + (k + j*l1)*ido] + cc[i - 1 + (k + jc*l1)*ido];
            ch[i + (k + jc*l1)*ido] = cc[i + (k + j*l1)*ido] - cc[i - 1 + (k + jc*l1)*ido];
          }
        }
      }
    } else {
      for (j=1; j<ipph; j++) {
        jc = ip - j;
        for (i=2; i<ido; i+=2) {
          for (k=0; k<l1; k++) {
            ch[i - 1 + (k + j*l1)*ido] = cc[i - 1 + (k + j*l1)*ido] - cc[i + (k + jc*l1)*ido];
            ch[i - 1 + (k + jc*l1)*ido] = cc[i - 1 + (k + j *l1)*ido] + cc[i + (k + jc*l1)*ido];
            ch[i + (k + j*l1)*ido] = cc[i + (k + j*l1)*ido] + cc[i - 1 + (k + jc*l1)*ido];
            ch[i + (k + jc*l1)*ido] = cc[i + (k + j*l1)*ido] - cc[i - 1 + (k + jc*l1)*ido];
          }
        }
      }
    }
    for (ik=0; ik<idl1; ik++) cc[ik] = ch[ik];
    for (j=1; j<ip; j++)
      for (k=0; k<l1; k++)
        cc[(k + j*l1)*ido] = ch[(k + j*l1)*ido];
    if (nbd <= l1) {
      is = -ido;
      for (j=1; j<ip; j++) {
        is += ido;
        idij = is-1;
        for (i=2; i<ido; i+=2) {
          idij += 2;
          for (k=0; k<l1; k++) {
            cc[i - 1 + (k + j*l1)*ido] = wa[idij - 1]*ch[i - 1 + (k + j*l1)*ido] - wa[idij]*
                ch[i + (k + j*l1)*ido];
            cc[i + (k + j*l1)*ido] = wa[idij - 1]*ch[i + (k + j*l1)*ido] + wa[idij]*ch[i - 1 + (k + j*l1)*ido];
          }
        }
      }
    } else {
      is = -ido;
      for (j=1; j<ip; j++) {
        is += ido;
        for (k=0; k<l1; k++) {
          idij = is - 1;
          for (i=2; i<ido; i+=2) {
            idij += 2;
            cc[i - 1 + (k + j*l1)*ido] = wa[idij-1]*ch[i - 1 + (k + j*l1)*ido] - wa[idij]*
                ch[i + (k + j*l1)*ido];
            cc[i + (k + j*l1)*ido] = wa[idij-1]*ch[i + (k + j*l1)*ido] + wa[idij]*ch[i - 1 + (k + j*l1)*ido];
          }
        }
      }
    }
  } /* radbg */

  /* ------------------------------------------------------------
cfftf1, npy_cfftf, npy_cfftb, cffti1, npy_cffti. Complex FFTs.
--------------------------------------------------------------- */

static void cfftf1(int n, Treal c[], Treal ch[], const Treal wa[], const int ifac[MAXFAC+2], int isign)
  {
    int idot, i;
    int k1, l1, l2;
    int na, nf, ip, iw, ix2, ix3, ix4, nac, ido, idl1;
    Treal *cinput, *coutput;
    nf = ifac[1];
    na = 0;
    l1 = 1;
    iw = 0;
    for (k1=2; k1<=nf+1; k1++) {
      ip = ifac[k1];
      l2 = ip*l1;
      ido = n / l2;
      idot = ido + ido;
      idl1 = idot*l1;
      if (na) {
        cinput = ch;
        coutput = c;
      } else {
        cinput = c;
        coutput = ch;
      }
      switch (ip) {
      case 4:
        ix2 = iw + idot;
        ix3 = ix2 + idot;
        passf4(idot, l1, cinput, coutput, &wa[iw], &wa[ix2], &wa[ix3], isign);
        na = !na;
        break;
      case 2:
        passf2(idot, l1, cinput, coutput, &wa[iw], isign);
        na = !na;
        break;
      case 3:
        ix2 = iw + idot;
        passf3(idot, l1, cinput, coutput, &wa[iw], &wa[ix2], isign);
        na = !na;
        break;
      case 5:
        ix2 = iw + idot;
        ix3 = ix2 + idot;
        ix4 = ix3 + idot;
        passf5(idot, l1, cinput, coutput, &wa[iw], &wa[ix2], &wa[ix3], &wa[ix4], isign);
        na = !na;
        break;
      default:
        passf(&nac, idot, ip, l1, idl1, cinput, coutput, &wa[iw], isign);
        if (nac != 0) na = !na;
      }
      l1 = l2;
      iw += (ip - 1)*idot;
    }
    if (na == 0) return;
    for (i=0; i<2*n; i++) c[i] = ch[i];
  } /* cfftf1 */


NPY_VISIBILITY_HIDDEN void npy_cfftf(int n, Treal c[], Treal wsave[])
  {
    int iw1, iw2;
    if (n == 1) return;
    iw1 = 2*n;
    iw2 = iw1 + 2*n;
    cfftf1(n, c, wsave, wsave+iw1, (int*)(wsave+iw2), -1);
  } /* npy_cfftf */


NPY_VISIBILITY_HIDDEN void npy_cfftb(int n, Treal c[], Treal wsave[])
  {
    int iw1, iw2;
    if (n == 1) return;
    iw1 = 2*n;
    iw2 = iw1 + 2*n;
    cfftf1(n, c, wsave, wsave+iw1, (int*)(wsave+iw2), +1);
  } /* npy_cfftb */


static void factorize(int n, int ifac[MAXFAC+2], const int ntryh[NSPECIAL])
  /* Factorize n in factors in ntryh and rest. On exit,
ifac[0] contains n and ifac[1] contains number of factors,
the factors start from ifac[2]. */
  {
    int ntry=3, i, j=0, ib, nf=0, nl=n, nq, nr;
startloop:
    if (j < NSPECIAL)
      ntry = ntryh[j];
    else
      ntry+= 2;
    j++;
    do {
      nq = nl / ntry;
      nr = nl - ntry*nq;
      if (nr != 0) goto startloop;
      nf++;
      ifac[nf + 1] = ntry;
      nl = nq;
      if (ntry == 2 && nf != 1) {
        for (i=2; i<=nf; i++) {
          ib = nf - i + 2;
          ifac[ib + 1] = ifac[ib];
        }
        ifac[2] = 2;
      }
    } while (nl != 1);
    ifac[0] = n;
    ifac[1] = nf;
  }


static void cffti1(int n, Treal wa[], int ifac[MAXFAC+2])
  {
    int fi, idot, i, j;
    int i1, k1, l1, l2;
    int ld, ii, nf, ip;
    int ido, ipm;

    static const int ntryh[NSPECIAL] = {
      3,4,2,5    }; /* Do not change the order of these. */

    factorize(n,ifac,ntryh);
    nf = ifac[1];
    i = 1;
    l1 = 1;
    for (k1=1; k1<=nf; k1++) {
      ip = ifac[k1+1];
      ld = 0;
      l2 = l1*ip;
      ido = n / l2;
      idot = ido + ido + 2;
      ipm = ip - 1;
      for (j=1; j<=ipm; j++) {
        i1 = i;
        wa[i-1] = 1;
        wa[i] = 0;
        ld += l1;
        fi = 0;
        for (ii=4; ii<=idot; ii+=2) {
          i+= 2;
          fi+= 1;
          sincos2pi(fi*ld, n, wa+i, wa+i-1);
        }
        if (ip > 5) {
          wa[i1-1] = wa[i-1];
          wa[i1] = wa[i];
        }
      }
      l1 = l2;
    }
  } /* cffti1 */


NPY_VISIBILITY_HIDDEN void npy_cffti(int n, Treal wsave[])
 {
    int iw1, iw2;
    if (n == 1) return;
    iw1 = 2*n;
    iw2 = iw1 + 2*n;
    cffti1(n, wsave+iw1, (int*)(wsave+iw2));
  } /* npy_cffti */

  /* -------------------------------------------------------------------
rfftf1, rfftb1, npy_rfftf, npy_rfftb, rffti1, npy_rffti. Treal FFTs.
---------------------------------------------------------------------- */

static void rfftf1(int n, Treal c[], Treal ch[], const Treal wa[], const int ifac[MAXFAC+2])
  {
    int i;
    int k1, l1, l2, na, kh, nf, ip, iw, ix2, ix3, ix4, ido, idl1;
    Treal *cinput, *coutput;
    nf = ifac[1];
    na = 1;
    l2 = n;
    iw = n-1;
    for (k1 = 1; k1 <= nf; ++k1) {
      kh = nf - k1;
      ip = ifac[kh + 2];
      l1 = l2 / ip;
      ido = n / l2;
      idl1 = ido*l1;
      iw -= (ip - 1)*ido;
      na = !na;
      if (na) {
        cinput = ch;
        coutput = c;
      } else {
        cinput = c;
        coutput = ch;
      }
      switch (ip) {
      case 4:
        ix2 = iw + ido;
        ix3 = ix2 + ido;
        radf4(ido, l1, cinput, coutput, &wa[iw], &wa[ix2], &wa[ix3]);
        break;
      case 2:
        radf2(ido, l1, cinput, coutput, &wa[iw]);
        break;
      case 3:
        ix2 = iw + ido;
        radf3(ido, l1, cinput, coutput, &wa[iw], &wa[ix2]);
        break;
      case 5:
        ix2 = iw + ido;
        ix3 = ix2 + ido;
        ix4 = ix3 + ido;
        radf5(ido, l1, cinput, coutput, &wa[iw], &wa[ix2], &wa[ix3], &wa[ix4]);
        break;
      default:
        if (ido == 1)
          na = !na;
        if (na == 0) {
          radfg(ido, ip, l1, idl1, c, ch, &wa[iw]);
          na = 1;
        } else {
          radfg(ido, ip, l1, idl1, ch, c, &wa[iw]);
          na = 0;
        }
      }
      l2 = l1;
    }
    if (na == 1) return;
    for (i = 0; i < n; i++) c[i] = ch[i];
  } /* rfftf1 */


static void rfftb1(int n, Treal c[], Treal ch[], const Treal wa[], const int ifac[MAXFAC+2])
  {
    int i;
    int k1, l1, l2, na, nf, ip, iw, ix2, ix3, ix4, ido, idl1;
    Treal *cinput, *coutput;
    nf = ifac[1];
    na = 0;
    l1 = 1;
    iw = 0;
    for (k1=1; k1<=nf; k1++) {
      ip = ifac[k1 + 1];
      l2 = ip*l1;
      ido = n / l2;
      idl1 = ido*l1;
      if (na) {
        cinput = ch;
        coutput = c;
      } else {
        cinput = c;
        coutput = ch;
      }
      switch (ip) {
      case 4:
        ix2 = iw + ido;
        ix3 = ix2 + ido;
        radb4(ido, l1, cinput, coutput, &wa[iw], &wa[ix2], &wa[ix3]);
        na = !na;
        break;
      case 2:
        radb2(ido, l1, cinput, coutput, &wa[iw]);
        na = !na;
        break;
      case 3:
        ix2 = iw + ido;
        radb3(ido, l1, cinput, coutput, &wa[iw], &wa[ix2]);
        na = !na;
        break;
      case 5:
        ix2 = iw + ido;
        ix3 = ix2 + ido;
        ix4 = ix3 + ido;
        radb5(ido, l1, cinput, coutput, &wa[iw], &wa[ix2], &wa[ix3], &wa[ix4]);
        na = !na;
        break;
      default:
        radbg(ido, ip, l1, idl1, cinput, coutput, &wa[iw]);
        if (ido == 1) na = !na;
      }
      l1 = l2;
      iw += (ip - 1)*ido;
    }
    if (na == 0) return;
    for (i=0; i<n; i++) c[i] = ch[i];
  } /* rfftb1 */


NPY_VISIBILITY_HIDDEN void npy_rfftf(int n, Treal r[], Treal wsave[])
  {
    if (n == 1) return;
    rfftf1(n, r, wsave, wsave+n, (int*)(wsave+2*n));
  } /* npy_rfftf */


NPY_VISIBILITY_HIDDEN void npy_rfftb(int n, Treal r[], Treal wsave[])
  {
    if (n == 1) return;
    rfftb1(n, r, wsave, wsave+n, (int*)(wsave+2*n));
  } /* npy_rfftb */


static void rffti1(int n, Treal wa[], int ifac[MAXFAC+2])
  {
    int fi, i, j;
    int k1, l1, l2;
    int ld, ii, nf, ip, is;
    int ido, ipm, nfm1;
    static const int ntryh[NSPECIAL] = {
      4,2,3,5    }; /* Do not change the order of these. */
    factorize(n,ifac,ntryh);
    nf = ifac[1];
    is = 0;
    nfm1 = nf - 1;
    l1 = 1;
    if (nfm1 == 0) return;
    for (k1 = 1; k1 <= nfm1; k1++) {
      ip = ifac[k1 + 1];
      ld = 0;
      l2 = l1*ip;
      ido = n / l2;
      ipm = ip - 1;
      for (j = 1; j <= ipm; ++j) {
        ld += l1;
        i = is;
        fi = 0;
        for (ii = 3; ii <= ido; ii += 2) {
          i += 2;
          fi += 1;
          sincos2pi(fi*ld, n, wa+i-1, wa+i-2);
        }
        is += ido;
      }
      l1 = l2;
    }
  } /* rffti1 */


NPY_VISIBILITY_HIDDEN void npy_rffti(int n, Treal wsave[])
  {
    if (n == 1) return;
    rffti1(n, wsave+n, (int*)(wsave+2*n));
  } /* npy_rffti */

#ifdef __cplusplus
}
#endif
