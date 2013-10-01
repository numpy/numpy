#include <fenv.h>
#include <float.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>

#define PYTESTPRECISION
#define PYTESTFUNC

#ifdef FLOAT
#define TYPE float
#define SUFFIX f
#define EPS FLT_EPSILON
#define CLOSE_ATOL 3
#define CLOSE_RTOL 1e-5
#define FMT "%.8e"
#define NPY_PI_2 1.570796326794896619231321691639751442f
#define NPY_PI 3.141592653589793238462643383279502884f
#define NPY_LOG2E 1.442695040888963407359924681001892137f
#define NPY_SQRT2 1.414213562373095048801688724209698079f
#else
#ifdef DOUBLE
#define TYPE double
#define SUFFIX 
#define EPS DBL_EPSILON
#define CLOSE_ATOL 0
#define CLOSE_RTOL 1e-12
#define FMT "%.16e"
#define NPY_PI_2 1.570796326794896619231321691639751442
#define NPY_PI 3.141592653589793238462643383279502884 
#define NPY_LOG2E 1.442695040888963407359924681001892137
#define NPY_SQRT2 1.414213562373095048801688724209698079
#else
#ifdef LONGDOUBLE
#define TYPE long double
#define SUFFIX l
#define EPS 50*LDBL_EPSILON
#define CLOSE_ATOL 0
#define CLOSE_RTOL 1e-12
#define FMT "%.18Le"
#define NPY_PI_2 1.570796326794896619231321691639751442L
#define NPY_PI 3.141592653589793238462643383279502884L 
#define NPY_LOG2E 1.442695040888963407359924681001892137L
#define NPY_SQRT2 1.414213562373095048801688724209698079L
#else
#error "Define FLOAT or DOUBLE or LONGDOUBLE"
#endif
#endif
#endif

const TYPE NZERO =  -1.0 * 0.0; 

#define STRINGIZE_INT(A) #A
#define STRINGIZE(A) STRINGIZE_INT(A)

#define CONCAT(A, B) A ## B
#define ADDSUFFIX_INT(A, B) CONCAT(A, B)
#define ADDSUFFIX(A) ADDSUFFIX_INT(A, SUFFIX)

#define TEST_PRINTF(func, xr, xi, er, ei, rr, ri) \
    printf("%d: " STRINGIZE(func) STRINGIZE(SUFFIX) "(" FMT " + " FMT "j): " \
           "expected: " FMT " + " FMT "j: received: " FMT " + " FMT "j\n", \
           __LINE__, xr, xi, er, ei, rr, ri)

#define TEST_INT(func, xr, xi, er, ei, rtest, itest) \
    do { \
        TYPE dxr = xr; \
        TYPE dxi = xi; \
        TYPE der = er; \
        TYPE dei = ei; \
        TYPE complex x = cpack(dxr, dxi); \
        TYPE complex r = ADDSUFFIX(func)(x); \
        TYPE rr = ADDSUFFIX(creal)(r); \
        TYPE ri = ADDSUFFIX(cimag)(r); \
        if (!(rtest(rr, der) && itest(ri, dei))) { \
            ret = 0; \
            TEST_PRINTF(func, dxr, dxi, der, dei, rr, ri); \
        } \
    } \
    while(0)

#define TEST_EE(func, xr, xi, er, ei) \
    TEST_INT(func, xr, xi, er, ei, isequal, isequal)

#define TEST_EC(func, xr, xi, er, ei) \
    TEST_INT(func, xr, xi, er, ei, isequal, isclose)

#define TEST_CE(func, xr, xi, er, ei) \
    TEST_INT(func, xr, xi, er, ei, isclose, isequal)

#define TEST_CC(func, xr, xi, er, ei) \
    TEST_INT(func, xr, xi, er, ei, isclose, isclose)

#define TEST_UNSPECIFIED2(func, xr, xi, er1, ei1, er2, ei2) \
    do { \
        TYPE dxr = xr; \
        TYPE dxi = xi; \
        TYPE der1 = er1; \
        TYPE dei1 = ei1; \
        TYPE der2 = er2; \
        TYPE dei2 = ei2; \
        TYPE complex x = cpack(dxr, dxi); \
        TYPE complex r = ADDSUFFIX(func)(x); \
        TYPE rr = ADDSUFFIX(creal)(r); \
        TYPE ri = ADDSUFFIX(cimag)(r); \
        if (!((isequal(rr, der1) && isequal(ri, dei1)) || \
              (isequal(rr, der2) && isequal(ri, dei2)))) { \
            ret = 0; \
            TEST_PRINTF(func, dxr, dxi, der1, dei1, rr, ri); \
            printf("or"); \
            TEST_PRINTF(func, dxr, dxi, der2, dei2, rr, ri); \
        } \
    } \
    while(0)

#define TEST_UNSPECIFIED4(func, xr, xi, er1, ei1, er2, ei2, er3, ei3, er4, ei4)\
    do { \
        TYPE dxr = xr; \
        TYPE dxi = xi; \
        TYPE der1 = er1; \
        TYPE dei1 = ei1; \
        TYPE der2 = er2; \
        TYPE dei2 = ei2; \
        TYPE der3 = er3; \
        TYPE dei3 = ei3; \
        TYPE der4 = er4; \
        TYPE dei4 = ei4; \
        TYPE complex x = cpack(dxr, dxi); \
        TYPE complex r = func(x); \
        TYPE rr = ADDSUFFIX(creal)(r); \
        TYPE ri = ADDSUFFIX(cimag)(r); \
        if (!((isequal(rr, der1) && isequal(ri, dei1)) || \
              (isequal(rr, der2) && isequal(ri, dei2)) || \
              (isequal(rr, der3) && isequal(ri, dei3)) || \
              (isequal(rr, der4) && isequal(ri, dei4)))) { \
            ret = 0; \
            TEST_PRINTF(func, dxr, dxi, der1, dei1, rr, ri); \
            printf("or"); \
            TEST_PRINTF(func, dxr, dxi, der2, dei2, rr, ri); \
            printf("or"); \
            TEST_PRINTF(func, dxr, dxi, der3, dei3, rr, ri); \
            printf("or"); \
            TEST_PRINTF(func, dxr, dxi, der4, dei4, rr, ri); \
        } \
    } \
    while(0)

#define TEST_CPOW_INT(xr, xi, yr, yi, er, ei, test) \
    do { \
        TYPE dxr = xr; \
        TYPE dxi = xi; \
        TYPE dyr = yr; \
        TYPE dyi = yi; \
        TYPE der = er; \
        TYPE dei = ei; \
        TYPE complex x = cpack(xr, xi); \
        TYPE complex y = cpack(yr, yi); \
        TYPE complex r = ADDSUFFIX(cpow)(x, y); \
        TYPE rr = ADDSUFFIX(creal)(r); \
        TYPE ri = ADDSUFFIX(cimag)(r); \
        if (!(test(rr, der) && test(ri, dei))) { \
            ret = 0; \
            printf("%d: " STRINGIZE(cpow) STRINGIZE(SUFFIX) "(" FMT " + " FMT \
                   "j, " FMT " + " FMT "j): expected: " FMT " + " FMT \
                   "j: received: " FMT " +  " FMT "j\n", __LINE__, dxr, dxi, \
                   dyr, dyi, der, dei, rr, ri); \
        } \
    } \
    while(0)

#define TEST_CPOW_EE(xr, xi, yr, yi, er, ei) \
    TEST_CPOW_INT(xr, xi, yr, yi, er, ei, isequal)

#define TEST_CPOW_CC(xr, xi, yr, yi, er, ei) \
    TEST_CPOW_INT(xr, xi, yr, yi, er, ei, isclose)

#define TEST_RAISES(func, xr, xi, er, ei, fpe) \
    do { \
        int except; \
        TYPE dxr = xr; \
        TYPE dxi = xi; \
        TYPE der = er; \
        TYPE dei = ei; \
        TYPE complex r; \
        TYPE complex x = cpack(xr, xi); \
        TYPE rr, ri; \
        feclearexcept(FE_ALL_EXCEPT); \
        r = ADDSUFFIX(func)(x); \
        except = fetestexcept(fpe); \
        rr = ADDSUFFIX(creal)(r); \
        ri = ADDSUFFIX(cimag)(r); \
        if (!(except & fpe && isequal(rr, der) && isequal(ri, dei))) { \
            ret = 0; \
            TEST_PRINTF(func, dxr, dxi, der, dei, rr, ri); \
        } \
    } \
    while(0)

#define TEST_RAISES_UNSPECIFIED2(func, xr, xi, er1, ei1, er2, ei2, fpe) \
    do { \
        int except; \
        TYPE dxr = xr; \
        TYPE dxi = xi; \
        TYPE der1 = er1; \
        TYPE dei1 = ei1; \
        TYPE der2 = er2; \
        TYPE dei2 = ei2; \
        TYPE complex r; \
        TYPE complex x = cpack(xr, xi); \
        TYPE rr, ri; \
        feclearexcept(FE_ALL_EXCEPT); \
        r = ADDSUFFIX(func)(x); \
        except = fetestexcept(fpe); \
        rr = ADDSUFFIX(creal)(r); \
        ri = ADDSUFFIX(cimag)(r); \
        if (!(except & fpe && ((isequal(rr, der1) && isequal(ri, dei1)) \
                               || (isequal(rr, der2) && isequal(ri, dei2))))) {\
            ret = 0; \
            TEST_PRINTF(func, dxr, dxi, der1, dei1, rr, ri); \
            printf("or"); \
            TEST_PRINTF(func, dxr, dxi, der2, dei2, rr, ri); \
        } \
    } \
    while(0) 

#define TEST_BRANCH_CUT(func, xr, xi, dxr, dxi, rsign, isign, cksignzero) \
    do { \
        TYPE vxr = xr; \
        TYPE vxi = xi; \
        TYPE vdxr = dxr; \
        TYPE vdxi = dxi; \
        int vrsign = rsign; \
        int visign = isign; \
        int vcksignzero = cksignzero; \
        TYPE complex x = cpack(vxr, vxi); \
        TYPE complex dx = cpack(vdxr, vdxi); \
        int q = check_branch_cut(ADDSUFFIX(func), x, dx, \
                                 vrsign, visign, vcksignzero); \
        if (!q) { \
            ret = 0; \
            printf(STRINGIZE(func) STRINGIZE(SUFFIX) ": branch cut failure: " \
                   "x = " FMT " + " FMT "j, dx = " FMT " + " FMT "j, rsign = %d, " \
                   "isign = %d, check_sign_zero = %d\n", vxr, vxi, \
                   vdxr, vdxi, vrsign, visign, vcksignzero); \
        } \
    } \
    while(0)

#define TEST_LOSS_OF_PRECISION(cfunc, rfunc, real) \
    do { \
        if (!check_loss_of_precision(ADDSUFFIX(cfunc), ADDSUFFIX(rfunc), real, \
                                     STRINGIZE(cfunc) STRINGIZE(SUFFIX))) { \
            ret = 0; \
        } \
        if (!check_near_crossover(ADDSUFFIX(cfunc), \
                                  STRINGIZE(cfunc) STRINGIZE(SUFFIX))) { \
            ret = 0; \
        } \
    } \
    while(0)

TYPE complex cpack(TYPE r, TYPE i)
{
    union {
        TYPE complex z;
        TYPE a[2];
    } z1;
    z1.a[0] = r;
    z1.a[1] = i;
    return z1.z;
}

int isclose(TYPE a, TYPE b)
{
    const TYPE atol = CLOSE_ATOL;
    const TYPE rtol = CLOSE_RTOL;

    if (isfinite(a) && isfinite(b)) {
        return (ADDSUFFIX(fabs)(a - b) <= (atol + rtol*ADDSUFFIX(fabs)(b)));
    }
    return 0;
}

int isequal(TYPE a, TYPE b)
{
    if (isfinite(a) && isfinite(b)) {
        if (a == 0 && b == 0) {
            TYPE signa = ADDSUFFIX(copysign)(1.0, a);
            TYPE signb = ADDSUFFIX(copysign)(1.0, b);
            return signa == signb;
        }
        else {
            return a == b;
        }
    }
    else if (isnan(a) && isnan(b)) {
        return 1;
    }
    else {/* infs */
        return a == b;
    }
}

typedef TYPE complex (*complexfunc)(TYPE complex);
typedef TYPE (*realfunc)(TYPE);

int check_branch_cut(complexfunc cfunc, TYPE complex x0, TYPE complex dx, 
                     int re_sign, int im_sign, int sig_zero_ok)
{
    const TYPE scale = EPS * 1e3;
    const TYPE atol = 1e-4;
    
    TYPE complex shift = dx*scale*ADDSUFFIX(cabs)(x0)/ADDSUFFIX(cabs)(dx);
    TYPE complex y0 = cfunc(x0);
    TYPE complex yp = cfunc(x0 + shift);
    TYPE complex ym = cfunc(x0 - shift);

    TYPE y0r, y0i, ypr, ypi, ymr, ymi;

    y0r = ADDSUFFIX(creal)(y0);
    y0i = ADDSUFFIX(cimag)(y0);
    ypr = ADDSUFFIX(creal)(yp);
    ypi = ADDSUFFIX(cimag)(yp);
    ymr = ADDSUFFIX(creal)(ym);
    ymi = ADDSUFFIX(cimag)(ym);

    if (ADDSUFFIX(fabs)(y0r - ypr) >= atol)
        return 0;
    if (ADDSUFFIX(fabs)(y0i - ypi) >= atol)
        return 0;
    if (ADDSUFFIX(fabs)(y0r - re_sign*ymr) >= atol)
        return 0;
    if (ADDSUFFIX(fabs)(y0i - im_sign*ymi) >= atol)
        return 0;

    if (sig_zero_ok) { 
        if (ADDSUFFIX(creal)(x0) == 0 && ADDSUFFIX(creal)(dx) != 0) {
            x0 = cpack(NZERO, ADDSUFFIX(cimag)(x0));
            ym = cfunc(x0);

            ymr = ADDSUFFIX(creal)(ym);
            ymi = ADDSUFFIX(cimag)(ym);
            if (ADDSUFFIX(fabs)(y0r - re_sign*ymr) >= atol)
                return 0;
            if (ADDSUFFIX(fabs)(y0i - im_sign*ymi) >= atol)
                return 0;
        }
        else if (ADDSUFFIX(cimag)(x0) == 0 && ADDSUFFIX(cimag)(dx) != 0) {
            x0 = cpack(ADDSUFFIX(creal)(x0), NZERO);
            ym = cfunc(x0);

            ymr = ADDSUFFIX(creal)(ym);
            ymi = ADDSUFFIX(cimag)(ym);
            if (ADDSUFFIX(fabs)(y0r - re_sign*ymr) >= atol)
                return 0;
            if (ADDSUFFIX(fabs)(y0i - im_sign*ymi) >= atol)
                return 0;
        } 
    }
    return 1;
}

int check_near_crossover(complexfunc cfunc, const char* fname)
{
    const TYPE x = 1e-3;
    const int rpnt[] = {-1, -1, -1,  0,  0,  1,  1,  1};
    const int ipnt[] = {-1,  0,  1, -1,  1, -1,  0, -1};
    const int npnt = sizeof(rpnt) / sizeof(int);
    const int dr[] = {1, 0, 1};
    const int di[] = {0, 1, 1};
    const int ndr = sizeof(dr) / sizeof(int);
    int k, j;
    int ret = 1;
    TYPE drj, dij, diff;
    TYPE complex zp, zm, czp, czm;

    for (j = 0; j < ndr; j++) {
        drj = 2 * x * dr[j] * EPS;
        dij = 2 * x * di[j] * EPS;
        for (k = 0; k < npnt; k++) {
            zp = cpack(x*rpnt[k] + drj, x*ipnt[k] + dij);
            zm = cpack(x*rpnt[k] - drj, x*ipnt[k] - dij);

            czp = cfunc(zp);
            czm = cfunc(zm);

            diff = ADDSUFFIX(cabs)(czp - czm);
            if ( diff > 2*EPS || czp == czm) {
                printf(fname);
                printf(": Loss of precision: j = %d, k = %d\n", j, k);
                printf("zp = (" FMT " + " FMT "j) -> (" FMT " + " FMT "j)\n", \
                       ADDSUFFIX(creal)(zp), ADDSUFFIX(cimag)(zp),     \
                       ADDSUFFIX(creal)(czp), ADDSUFFIX(cimag)(czp));
                printf("zm = (" FMT " + " FMT "j) -> (" FMT " + " FMT "j)\n", \
                       ADDSUFFIX(creal)(zm), ADDSUFFIX(cimag)(zm),     \
                       ADDSUFFIX(creal)(czm), ADDSUFFIX(cimag)(czm));
                printf("diff = " FMT ", exact match = %d\n", diff, czp == czm);
                ret = 0;
            }
        }
    }
    return ret;
}

int clp_internal(complexfunc cfunc, realfunc rfunc, int real, TYPE x)
{
    TYPE num = rfunc(x);
    TYPE den;
    TYPE complex z;

    if (real == 1) {
        z = cpack(x, 0);
        z = cfunc(z);
        den = ADDSUFFIX(creal)(z);
    }
    else {
        z = cpack(0, x);
        z = cfunc(z);
        den = ADDSUFFIX(cimag)(z);
    }
    return ADDSUFFIX(fabs)(num/den - 1);
}
         
int check_loss_of_precision(complexfunc cfunc, realfunc rfunc, int real,
                            const char* fname)
{
    const int n_series = 200;
    const int n_basic = 10;
    const TYPE rtol = 2*EPS;

    const TYPE xsb = -20;
    const TYPE xse = -3.001;
    const TYPE dxs = (xse - xsb) / n_series;

    const TYPE xbb = -2.999;
    const TYPE xbe = 0;
    const TYPE dxb = (xbe - xbb) / n_basic;

    TYPE x, ratio;
    int k;
    int ret = 1;
    
    for(k = 0; k < n_series; k++) {
        x = ADDSUFFIX(pow)(10.0, xsb + k*dxs);
        ratio = clp_internal(cfunc, rfunc, real, x);
        if (ratio > rtol) {
            printf(fname);
            printf(": Loss of precision vs real:\n");
            printf("x = " FMT "\n", x);
            printf("ratio = " FMT "\n", ratio);
            ret = 0;
        }
    }

    for(k = 0; k < n_basic; k++) {
        x = ADDSUFFIX(pow)(10.0, xbb + k*dxb);
        ratio = clp_internal(cfunc, rfunc, real, x);
        if (ratio > rtol) {
            printf(fname);
            printf(": Loss of precision vs. real:\n");
            printf("x = " FMT "\n", x);
            printf("ratio = " FMT "\n", ratio);
            ret = 0;
        }
    }
    return ret;
}

#ifdef CACOS
int test_cacos()
{
    int ret = 1;
    /* cacos(conj(z)) = conj(cacos(z)) */
    TEST_CE(cacos, 0, 0, NPY_PI_2, NZERO);
    TEST_CE(cacos, 0, NZERO, NPY_PI_2, 0);

    TEST_CE(cacos, NZERO, 0, NPY_PI_2, NZERO);
    TEST_CE(cacos, NZERO, NZERO, NPY_PI_2, 0);

    TEST_CE(cacos, 0, NAN, NPY_PI_2, NAN);
    TEST_CE(cacos, NZERO, NAN, NPY_PI_2, NAN);
    
    TEST_CE(cacos, 2.0, INFINITY, NPY_PI_2, -INFINITY);
    TEST_CE(cacos, 2.0, -INFINITY, NPY_PI_2, INFINITY);

    /* can raise FE_INVALID or not */
    TEST_EE(cacos, 2.0, NAN, NAN, NAN);

    TEST_CE(cacos, -INFINITY, 2.0, NPY_PI, -INFINITY);
    TEST_CE(cacos, -INFINITY, -2.0, NPY_PI, INFINITY);

    TEST_EE(cacos, INFINITY, 2.0, 0, -INFINITY);
    TEST_EE(cacos, INFINITY, -2.0, 0, INFINITY);

    TEST_CE(cacos, -INFINITY, INFINITY, 0.75 * NPY_PI, -INFINITY);
    TEST_CE(cacos, -INFINITY, -INFINITY, 0.75 * NPY_PI, INFINITY);

    TEST_CE(cacos, INFINITY, INFINITY, 0.25 * NPY_PI, -INFINITY);
    TEST_CE(cacos, INFINITY, -INFINITY, 0.25 * NPY_PI, -INFINITY);

    /* sign of imaginary part is unspecified. */
    TEST_UNSPECIFIED2(cacos, INFINITY, NAN, NAN, INFINITY, NAN, -INFINITY);
    TEST_UNSPECIFIED2(cacos, -INFINITY, NAN, NAN, INFINITY, NAN, -INFINITY);

    /* can raise FE_INVALID or not */
    TEST_EE(cacos, NAN, 2.0, NAN, NAN);
    TEST_EE(cacos, NAN, -2.0, NAN, NAN);

    TEST_EE(cacos, NAN, INFINITY, NAN, -INFINITY);
    TEST_EE(cacos, NAN, -INFINITY, NAN, INFINITY);

    TEST_EE(cacos, NAN, NAN, NAN, NAN);

    TEST_BRANCH_CUT(cacos, -2, 0, 0, 1, 1, -1, 1);
    TEST_BRANCH_CUT(cacos, 2, 0, 0, -1, 1, -1, 1);
    TEST_BRANCH_CUT(cacos, 0, -2, 1, 0, 1, 1, 1);
    TEST_BRANCH_CUT(cacos, 0, 2, 1, 0, 1, 1, 1);

    TEST_CC(cacos, 0.5, 0.0, ADDSUFFIX(acos)(0.5), 0.0);
    
    return ret;
}
#endif

#ifdef CASIN
int test_casin()
{
    int ret = 1;

    /* casin(conj(z)) = conj(casin(z)) and casin is odd */
    TEST_EE(casin, 0, 0, 0, 0);
    TEST_EE(casin, 0, NZERO, 0, NZERO);
    TEST_EE(casin, NZERO, 0, NZERO, 0);
    TEST_EE(casin, NZERO, NZERO, NZERO, NZERO);

    TEST_CE(casin, -INFINITY, 2.0, -NPY_PI_2, INFINITY);
    TEST_CE(casin, INFINITY, 2.0, NPY_PI_2, INFINITY);
    TEST_CE(casin, -INFINITY, -2.0, -NPY_PI_2, -INFINITY);
    TEST_CE(casin, INFINITY, -2.0, NPY_PI_2, -INFINITY);

    /* can raise FE_INVALID or not */
    TEST_EE(casin, NAN, -2.0, NAN, NAN);
    TEST_EE(casin, NAN, 2.0, NAN, NAN);

    TEST_EE(casin, -2.0, INFINITY, NZERO, INFINITY);
    TEST_EE(casin, 2.0, INFINITY, 0, INFINITY);
    TEST_EE(casin, -2.0, -INFINITY, NZERO, -INFINITY);
    TEST_EE(casin, 2.0, -INFINITY, 0, -INFINITY);

    TEST_CE(casin, -INFINITY, INFINITY, -0.25*NPY_PI, INFINITY);
    TEST_CE(casin, INFINITY, INFINITY, 0.25*NPY_PI, INFINITY);
    TEST_CE(casin, -INFINITY, -INFINITY, -0.25*NPY_PI, -INFINITY);
    TEST_CE(casin, INFINITY, -INFINITY, 0.25*NPY_PI, -INFINITY);

    TEST_EE(casin, NAN, INFINITY, NAN, INFINITY);
    TEST_EE(casin, NAN, -INFINITY, NAN, -INFINITY);

    TEST_EE(casin, 0, NAN, 0, NAN);
    TEST_EE(casin, NZERO, NAN, NZERO, NAN);
    
    /* can raise FE_INVALID or not */
    TEST_EE(casin, -2.0, NAN, NAN, NAN);
    TEST_EE(casin, 2.0, NAN, NAN, NAN);

    /* sign of real part is unspecified */
    TEST_UNSPECIFIED2(casin, -INFINITY, NAN, NAN, INFINITY, NAN, -INFINITY);
    TEST_UNSPECIFIED2(casin, INFINITY, NAN, NAN, INFINITY, NAN, -INFINITY);

    TEST_EE(casin, NAN, NAN, NAN, NAN);
    
    TEST_LOSS_OF_PRECISION(casin, asin, 0);

    TEST_CC(casin, 1e-5, 1e-5, 9.999999999666666667e-6, 1.0000000000333333333e-5);

    TEST_BRANCH_CUT(casin, -2, 0, 0, 1, 1, -1, 1);
    TEST_BRANCH_CUT(casin, 2, 0, 0, -1, 1, -1, 1);
    TEST_BRANCH_CUT(casin, 0, -2, 1, 0, 1, 1, 1);
    TEST_BRANCH_CUT(casin, 0, 2, 1, 0, 1, 1, 1);

    TEST_CC(casin, 0.5, 0, ADDSUFFIX(asin)(0.5), 0);

    return ret;
}
#endif

#ifdef CATAN
int test_catan()
{
    int ret = 1;
    /* catan(conj(z)) = conj(catan(z)) and catan is odd */
    TEST_EE(catan, 0, 0, 0, 0);
    TEST_EE(catan, 0, NZERO, 0, NZERO);
    TEST_EE(catan, NZERO, 0, NZERO, 0);
    TEST_EE(catan, NZERO, NZERO, NZERO, NZERO);
    
    TEST_EE(catan, NAN, 0, NAN, 0);
    TEST_EE(catan, NAN, NZERO, NAN, NZERO);

    TEST_RAISES(catan, NZERO, 1, NZERO, INFINITY, FE_DIVBYZERO);
    TEST_RAISES(catan, 0, 1, 0, INFINITY, FE_DIVBYZERO);
    TEST_RAISES(catan, NZERO, -1, NZERO, -INFINITY, FE_DIVBYZERO);
    TEST_RAISES(catan, 0, -1, 0, -INFINITY, FE_DIVBYZERO);

    TEST_CE(catan, -INFINITY, 2.0, -NPY_PI_2, 0);
    TEST_CE(catan, INFINITY, 2.0, NPY_PI_2, 0);
    TEST_CE(catan, -INFINITY, -2.0, -NPY_PI_2, NZERO);
    TEST_CE(catan, INFINITY, -2.0, NPY_PI_2, NZERO);

    /* can raise FE_INVALID or not */
    TEST_EE(catan, NAN, -2.0, NAN, NAN);
    TEST_EE(catan, NAN, 2.0, NAN, NAN);

    TEST_CE(catan, -2.0, INFINITY, -NPY_PI_2, 0);
    TEST_CE(catan, 2.0, INFINITY, NPY_PI_2, 0);
    TEST_CE(catan, -2.0, -INFINITY, -NPY_PI_2, NZERO);
    TEST_CE(catan, 2.0, -INFINITY, NPY_PI_2, NZERO);

    TEST_CE(catan, -INFINITY, INFINITY, -NPY_PI_2, 0);
    TEST_CE(catan, INFINITY, INFINITY, NPY_PI_2, 0);
    TEST_CE(catan, -INFINITY, -INFINITY, -NPY_PI_2, NZERO);
    TEST_CE(catan, INFINITY, -INFINITY, NPY_PI_2, NZERO);

    TEST_EE(catan, NAN, INFINITY, NAN, 0);
    TEST_EE(catan, NAN, -INFINITY, NAN, NZERO);

    /* can raise FE_INVALID or not */
    TEST_EE(catan, -2.0, NAN, NAN, NAN);
    TEST_EE(catan, 2.0, NAN, NAN, NAN);

    /* sign of real part is unspecified */
    TEST_UNSPECIFIED2(catan, -INFINITY, NAN, -NPY_PI_2, 0, -NPY_PI_2, NZERO);
    TEST_UNSPECIFIED2(catan, INFINITY, NAN, NPY_PI_2, 0, NPY_PI_2, NZERO);

    TEST_EE(catan, NAN, NAN, NAN, NAN);

    TEST_LOSS_OF_PRECISION(catan, atan, 0);

    TEST_CC(catan, 1e-5, 1e-5, 1.000000000066666666e-5, 9.999999999333333333e-6);

    TEST_BRANCH_CUT(catan, 0, -2, 1, 0, -1, 1, 1);
    TEST_BRANCH_CUT(catan, 0, 2, -1, 0, -1, 1, 1);
    TEST_BRANCH_CUT(catan, -2, 0, 0, 1, 1, 1, 1);
    TEST_BRANCH_CUT(catan, 2, 0, 0, 1, 1, 1, 1);

    TEST_CC(catan, 0.5, 0, ADDSUFFIX(catan)(0.5), 0);

    return ret;
}
#endif

#ifdef CACOSH
int test_cacosh()
{
    int ret = 1;
    /* cacosh(conj(z)) = conj(cacosh(z)) */
    TEST_EC(cacosh, 0, 0, 0, NPY_PI_2);
    TEST_EC(cacosh, 0, NZERO, 0, -NPY_PI_2);

    TEST_EC(cacosh, NZERO, 0, 0, NPY_PI_2);
    TEST_EC(cacosh, NZERO, NZERO, 0, -NPY_PI_2);

    TEST_EC(cacosh, 2.0, INFINITY, INFINITY, NPY_PI_2);
    TEST_EC(cacosh, 2.0, -INFINITY, INFINITY, -NPY_PI_2);

    /* can raise FE_INVALID or not */
    TEST_EE(cacosh, 2.0, NAN, NAN, NAN);

    TEST_EC(cacosh, -INFINITY, 2.0, INFINITY, NPY_PI);
    TEST_EC(cacosh, -INFINITY, -2.0, INFINITY, -NPY_PI);

    TEST_EE(cacosh, INFINITY, 2.0, INFINITY, 0);
    TEST_EE(cacosh, INFINITY, -2.0, INFINITY, NZERO);

    TEST_EC(cacosh, -INFINITY, INFINITY, INFINITY, 0.75*NPY_PI);
    TEST_EC(cacosh, -INFINITY, -INFINITY, INFINITY, -0.75*NPY_PI);
    
    TEST_EC(cacosh, INFINITY, INFINITY, INFINITY, 0.25*NPY_PI);
    TEST_EC(cacosh, INFINITY, -INFINITY, INFINITY, -0.25*NPY_PI);

    TEST_EE(cacosh, INFINITY, NAN, INFINITY, NAN);
    TEST_EE(cacosh, -INFINITY, NAN, INFINITY, NAN);

    /* can raise FE_INVALID or not */
    TEST_EE(cacosh, NAN, 2.0, NAN, NAN);
    TEST_EE(cacosh, NAN, -2.0, NAN, NAN);

    TEST_EE(cacosh, NAN, INFINITY, INFINITY, NAN);
    TEST_EE(cacosh, NAN, -INFINITY, INFINITY, NAN);

    TEST_EE(cacosh, NAN, NAN, NAN, NAN);

    TEST_BRANCH_CUT(cacosh, -1, 0, 0, 1, 1, -1, 1);
    TEST_BRANCH_CUT(cacosh, 0.5, 0, 0, 1, 1, -1, 1);
    TEST_BRANCH_CUT(cacosh, 0, -2, 1, 0, 1, 1, 1);
    TEST_BRANCH_CUT(cacosh, 0, 2, 1, 0, 1, 1, 1);
    TEST_BRANCH_CUT(cacosh, 2, 0, 0, 1, 1, 1, 1);

    TEST_CC(cacosh, 1.5, 0, ADDSUFFIX(acosh)(1.5), 0);
    return ret;
}
#endif

#ifdef CASINH
int test_casinh()
{
    int ret = 1;
    /* casinh(conj(z)) = conj(casinh(z)) and casinh is odd */
    TEST_EE(casinh, 0, 0, 0, 0);
    TEST_EE(casinh, 0, NZERO, 0, NZERO);
    TEST_EE(casinh, NZERO, 0, NZERO, 0);
    TEST_EE(casinh, NZERO, NZERO, NZERO, NZERO);

    TEST_EC(casinh, 2.0, INFINITY, INFINITY, NPY_PI_2);
    TEST_EC(casinh, 2.0, -INFINITY, INFINITY, -NPY_PI_2);
    TEST_EC(casinh, -2.0, INFINITY, -INFINITY, NPY_PI_2);
    TEST_EC(casinh, -2.0, -INFINITY, -INFINITY, -NPY_PI_2);

    /* can raise FE_INVALID or not */
    TEST_EE(casinh, 2.0, NAN, NAN, NAN);
    TEST_EE(casinh, -2.0, NAN, NAN, NAN);

    TEST_EE(casinh, INFINITY, 2.0, INFINITY, 0);
    TEST_EE(casinh, INFINITY, -2.0, INFINITY, NZERO);
    TEST_EE(casinh, -INFINITY, 2.0, -INFINITY, 0);
    TEST_EE(casinh, -INFINITY, -2.0, -INFINITY, NZERO);

    TEST_EC(casinh, INFINITY, INFINITY, INFINITY, 0.25*NPY_PI);
    TEST_EC(casinh, INFINITY, -INFINITY, INFINITY, -0.25*NPY_PI);
    TEST_EC(casinh, -INFINITY, INFINITY, -INFINITY, 0.25*NPY_PI);
    TEST_EC(casinh, -INFINITY, -INFINITY, -INFINITY, -0.25*NPY_PI);

    TEST_EE(casinh, INFINITY, NAN, INFINITY, NAN);
    TEST_EE(casinh, -INFINITY, NAN, -INFINITY, NAN);
    
    TEST_EE(casinh, NAN, 0, NAN, 0);
    TEST_EE(casinh, NAN, NZERO, NAN, NZERO);
    
    /* can raise FE_INVALID or not */
    TEST_EE(casinh, NAN, 2.0, NAN, NAN);
    TEST_EE(casinh, NAN, -2.0, NAN, NAN);

    /* sign of real part is unspecified */
    TEST_UNSPECIFIED2(casinh, NAN, INFINITY, INFINITY, NAN, -INFINITY, NAN);
    TEST_UNSPECIFIED2(casinh, NAN, -INFINITY, INFINITY, NAN, -INFINITY, NAN);

    TEST_EE(casinh, NAN, NAN, NAN, NAN);

    TEST_LOSS_OF_PRECISION(casinh, asinh, 1);

    TEST_CC(casinh, 1e-5, 1e-5, 1.0000000000333333333e-5, 9.999999999666666667e-6);

    TEST_BRANCH_CUT(casinh, 0, -2, -1, 0, -1, 1, 1);
    TEST_BRANCH_CUT(casinh, 0, 2, 1, 0, -1, 1, 1);
    TEST_BRANCH_CUT(casinh, -2, 0, 0, 1, 1, 1, 1);
    TEST_BRANCH_CUT(casinh, 2, 0, 0, 1, 1, 1, 1);
    TEST_BRANCH_CUT(casinh, 0, 0, 1, 0, 1, 1, 1);

    TEST_CC(casinh, 0.5, 0, ADDSUFFIX(asinh)(0.5), 0);

    return ret;
}
#endif

#ifdef CATANH
int test_catanh()
{
    int ret = 1;
    /* catanh(conj(z)) = conj(catanh(z)) and catanh is odd */
    TEST_EE(catanh, 0, 0, 0, 0);
    TEST_EE(catanh, 0, NZERO, 0, NZERO);
    TEST_EE(catanh, NZERO, 0, NZERO, 0);
    TEST_EE(catanh, NZERO, NZERO, NZERO, NZERO);
    
    TEST_EE(catanh, 0, NAN, 0, NAN);
    TEST_EE(catanh, NZERO, NAN, NZERO, NAN);

    TEST_RAISES(catanh, 1, 0, INFINITY, 0, FE_DIVBYZERO);
    TEST_RAISES(catanh, 1, NZERO, INFINITY, NZERO, FE_DIVBYZERO);
    TEST_RAISES(catanh, -1, 0, -INFINITY, 0, FE_DIVBYZERO);
    TEST_RAISES(catanh, -1, NZERO, -INFINITY, NZERO, FE_DIVBYZERO);

    TEST_EC(catanh, 2.0, INFINITY, 0, NPY_PI_2);
    TEST_EC(catanh, 2.0, -INFINITY, 0, -NPY_PI_2);
    TEST_EC(catanh, -2.0, INFINITY, NZERO, NPY_PI_2);
    TEST_EC(catanh, -2.0, -INFINITY, NZERO, -NPY_PI_2);

    /* can raise FE_INVALID or not */
    TEST_EE(catanh, 2.0, NAN, NAN, NAN);
    TEST_EE(catanh, -2.0, NAN, NAN, NAN);

    TEST_EC(catanh, INFINITY, 2.0, 0, NPY_PI_2);
    TEST_EC(catanh, INFINITY, -2.0, 0, -NPY_PI_2);
    TEST_EC(catanh, -INFINITY, 2.0, NZERO, NPY_PI_2);
    TEST_EC(catanh, -INFINITY, -2.0, NZERO, -NPY_PI_2);

    TEST_EC(catanh, INFINITY, INFINITY, 0, NPY_PI_2);
    TEST_EC(catanh, INFINITY, -INFINITY, 0, -NPY_PI_2);
    TEST_EC(catanh, -INFINITY, INFINITY, NZERO, NPY_PI_2);
    TEST_EC(catanh, -INFINITY, -INFINITY, NZERO, -NPY_PI_2);

    TEST_EE(catanh, INFINITY, NAN, 0, NAN);
    TEST_EE(catanh, -INFINITY, NAN, NZERO, NAN);

    /* can raise FE_INVALID or not */
    TEST_EE(catanh, NAN, 2.0, NAN, NAN);
    TEST_EE(catanh, NAN, -2.0, NAN, NAN);

    /* sign of real part is unspecified */
    TEST_UNSPECIFIED2(catanh, NAN, INFINITY, 0, NPY_PI_2, NZERO, NPY_PI_2);
    TEST_UNSPECIFIED2(catanh, NAN, -INFINITY, 0, -NPY_PI_2, NZERO, -NPY_PI_2);

    /* TEST(catanh, NAN, INFINITY, 0, NPY_PI_2); */
    TEST_EE(catanh, NAN, NAN, NAN, NAN);

    TEST_LOSS_OF_PRECISION(catanh, atanh, 1);

    TEST_CC(catanh, 1e-5, 1e-5, 9.999999999333333333e-6, 1.000000000066666666e-5);

    TEST_BRANCH_CUT(catanh, -2, 0, 0, 1, 1, -1, 1);
    TEST_BRANCH_CUT(catanh, 2, 0, 0, -1, 1, -1, 1);
    TEST_BRANCH_CUT(catanh, 0, -2, 1, 0, 1, 1, 1);
    TEST_BRANCH_CUT(catanh, 0, 2, 1, 0, 1, 1, 1);
    TEST_BRANCH_CUT(catanh, 0, 0, 0, 1, 1, 1, 1);

    TEST_CC(catanh, 0.5, 0, ADDSUFFIX(atanh)(0.5), 0);

    return ret;
}
#endif

#ifdef CCOS
int test_ccos()
{
    int ret = 1;
    /* ccos(conj(z)) = conj(ccos(z)) and ccos is even */
    TEST_EE(ccos, NZERO, 0, 1, 0);
    TEST_EE(ccos, 0, 0, 1, NZERO);
    TEST_EE(ccos, NZERO, NZERO, 1, NZERO);
    TEST_EE(ccos, 0, NZERO, 1, 0);

    /* sign of imaginary part is unspecified */
    TEST_RAISES_UNSPECIFIED2(ccos, -INFINITY, 0, NAN, 0, \
                             NAN, NZERO, FE_INVALID);

    /* sign of imaginary part is unspecified */
    TEST_UNSPECIFIED2(ccos, NAN, 0, NAN, 0, NAN, NZERO);
    TEST_UNSPECIFIED2(ccos, NAN, NZERO, NAN, 0, NAN, NZERO);

    TEST_RAISES(ccos, -INFINITY, 2.0, NAN, NAN, FE_INVALID);
    TEST_RAISES(ccos, INFINITY, 2.0, NAN, NAN, FE_INVALID);
    TEST_RAISES(ccos, -INFINITY, -2.0, NAN, NAN, FE_INVALID);
    TEST_RAISES(ccos, INFINITY, -2.0, NAN, NAN, FE_INVALID);

    /* can raise FE_INVALID or not */
    TEST_EE(ccos, NAN, 2.0, NAN, NAN);
    TEST_EE(ccos, NAN, -2.0, NAN, NAN);

    TEST_EE(ccos, NZERO, INFINITY, INFINITY, 0);
    TEST_EE(ccos, 0, INFINITY, INFINITY, NZERO);
    TEST_EE(ccos, NZERO, -INFINITY, INFINITY, NZERO);
    TEST_EE(ccos, 0, -INFINITY, INFINITY, 0);

    TEST_EE(ccos, -1.0, INFINITY, INFINITY, INFINITY);
    TEST_EE(ccos, 1.0, INFINITY, INFINITY, -INFINITY);
    TEST_EE(ccos, -1.0, -INFINITY, INFINITY, -INFINITY);
    TEST_EE(ccos, 1.0, -INFINITY, INFINITY, INFINITY);
    TEST_EE(ccos, -2.0, INFINITY, -INFINITY, INFINITY);
    TEST_EE(ccos, 2.0, INFINITY, -INFINITY, -INFINITY);
    TEST_EE(ccos, -2.0, -INFINITY, -INFINITY, -INFINITY);
    TEST_EE(ccos, 2.0, -INFINITY, -INFINITY, INFINITY);
    TEST_EE(ccos, -4.0, INFINITY, -INFINITY, -INFINITY);
    TEST_EE(ccos, 4.0, INFINITY, -INFINITY, INFINITY);
    TEST_EE(ccos, -4.0, -INFINITY, -INFINITY, INFINITY);
    TEST_EE(ccos, 4.0, -INFINITY, -INFINITY, -INFINITY);
    TEST_EE(ccos, -5.0, INFINITY, INFINITY, -INFINITY);
    TEST_EE(ccos, 5.0, INFINITY, INFINITY, INFINITY);
    TEST_EE(ccos, -5.0, -INFINITY, INFINITY, INFINITY);
    TEST_EE(ccos, 5.0, -INFINITY, INFINITY, -INFINITY);

    /* sign of real part is unspecified */
    TEST_RAISES_UNSPECIFIED2(ccos, -INFINITY, INFINITY, INFINITY, NAN, \
                             -INFINITY, NAN, FE_INVALID);

    TEST_EE(ccos, NAN, INFINITY, INFINITY, NAN);
    TEST_EE(ccos, NAN, -INFINITY, INFINITY, NAN);

    /* sign of imaginary part is unspecified */
    TEST_UNSPECIFIED2(ccos, 0, NAN, NAN, 0, NAN, NZERO);
    TEST_UNSPECIFIED2(ccos, NZERO, NAN, NAN, 0, NAN, NZERO);

    /* can raise FE_INVALID or not */
    TEST_EE(ccos, -2.0, NAN, NAN, NAN);
    TEST_EE(ccos, 2.0, NAN, NAN, NAN);

    TEST_EE(ccos, NAN, NAN, NAN, NAN);

    TEST_CC(ccos, 0.5, 0, ADDSUFFIX(cos)(0.5), 0);
 
    return ret;
}
#endif

#ifdef CSIN
int test_csin()
{
    int ret = 1;
    /* csin(conj(z)) = conj(csin(z)) and csin is odd */
    TEST_EE(csin, 0, 0, 0, 0);
    TEST_EE(csin, 0, NZERO, 0, NZERO);
    TEST_EE(csin, NZERO, 0, NZERO, 0);
    TEST_EE(csin, NZERO, NZERO, NZERO, NZERO);

    /* sign of imaginary part is unspecified */
    TEST_RAISES_UNSPECIFIED2(csin, -INFINITY, 0, NAN, 0, \
                             NAN, NZERO, FE_INVALID);

    /* sign of imaginary part is unspecified */
    TEST_UNSPECIFIED2(csin, NAN, 0, NAN, 0, NAN, NZERO);
    TEST_UNSPECIFIED2(csin, NAN, NZERO, NAN, 0, NAN, NZERO);

    TEST_RAISES(csin, -INFINITY, 2.0, NAN, NAN, FE_INVALID);
    TEST_RAISES(csin, INFINITY, 2.0, NAN, NAN, FE_INVALID);
    TEST_RAISES(csin, -INFINITY, -2.0, NAN, NAN, FE_INVALID);
    TEST_RAISES(csin, INFINITY, -2.0, NAN, NAN, FE_INVALID);

    /* can raise FE_INVALID or not */
    TEST_EE(csin, NAN, 2.0, NAN, NAN);
    TEST_EE(csin, NAN, -2.0, NAN, NAN);

    TEST_EE(csin, NZERO, INFINITY, NZERO, INFINITY);
    TEST_EE(csin, 0, INFINITY, 0, INFINITY);
    TEST_EE(csin, NZERO, -INFINITY, NZERO, -INFINITY);
    TEST_EE(csin, 0, -INFINITY, 0, -INFINITY);

    TEST_EE(csin, -1.0, INFINITY, -INFINITY, INFINITY);
    TEST_EE(csin, 1.0, INFINITY, INFINITY, INFINITY);
    TEST_EE(csin, -1.0, -INFINITY, -INFINITY, -INFINITY);
    TEST_EE(csin, 1.0, -INFINITY, INFINITY, -INFINITY);
    TEST_EE(csin, -2.0, INFINITY, -INFINITY, -INFINITY);
    TEST_EE(csin, 2.0, INFINITY, INFINITY, -INFINITY);
    TEST_EE(csin, -2.0, -INFINITY, -INFINITY, INFINITY);
    TEST_EE(csin, 2.0, -INFINITY, INFINITY, INFINITY);
    TEST_EE(csin, -4.0, INFINITY, INFINITY, -INFINITY);
    TEST_EE(csin, 4.0, INFINITY, -INFINITY, -INFINITY);
    TEST_EE(csin, -4.0, -INFINITY, INFINITY, INFINITY);
    TEST_EE(csin, 4.0, -INFINITY, -INFINITY, INFINITY);
    TEST_EE(csin, -5.0, INFINITY, INFINITY, INFINITY);
    TEST_EE(csin, 5.0, INFINITY, -INFINITY, INFINITY);
    TEST_EE(csin, -5.0, -INFINITY, INFINITY, -INFINITY);
    TEST_EE(csin, 5.0, -INFINITY, -INFINITY, -INFINITY);

    /* sign of imaginary part is unspecified */
    TEST_RAISES_UNSPECIFIED2(csin, -INFINITY, INFINITY, NAN, INFINITY, \
                             NAN, -INFINITY, FE_INVALID);

    /* sign of imaginary part is unspecified */
    TEST_UNSPECIFIED2(csin, NAN, INFINITY, NAN, INFINITY, NAN, -INFINITY);
    TEST_UNSPECIFIED2(csin, NAN, -INFINITY, NAN, INFINITY, NAN, -INFINITY);

    TEST_EE(csin, 0, NAN, 0, NAN);
    TEST_EE(csin, NZERO, NAN, NZERO, NAN);

    /* can raise FE_INVALID or not */
    TEST_EE(csin, -2.0, NAN, NAN, NAN);
    TEST_EE(csin, 2.0, NAN, NAN, NAN);

    TEST_EE(csin, NAN, NAN, NAN, NAN);

    TEST_CC(csin, 0.5, 0, ADDSUFFIX(sin)(0.5), 0);
    
    return ret;
}
#endif

#ifdef CTAN
int test_ctan()
{
    int ret = 1;
    /* ctan(conj(z)) = conj(ctan(z)) and ctan is odd */
    TEST_EE(ctan, 0, 0, 0, 0);
    TEST_EE(ctan, 0, NZERO, 0, NZERO);
    TEST_EE(ctan, NZERO, 0, NZERO, 0);
    TEST_EE(ctan, NZERO, NZERO, NZERO, NZERO);

    TEST_RAISES(ctan, -INFINITY, 2.0, NAN, NAN, FE_INVALID);
    TEST_RAISES(ctan, -INFINITY, -2.0, NAN, NAN, FE_INVALID);

    /* can raise FE_INVALID or not */
    TEST_EE(ctan, NAN, 2.0, NAN, NAN);
    TEST_EE(ctan, NAN, -2.0, NAN, NAN);

    TEST_EE(ctan, -1.0, INFINITY, NZERO, 1.0);
    TEST_EE(ctan, 1.0, INFINITY, 0, 1.0);
    TEST_EE(ctan, -1.0, -INFINITY, NZERO, -1.0);
    TEST_EE(ctan, 1.0, -INFINITY, 0, -1.0);
    TEST_EE(ctan, -2.0, INFINITY, 0, 1);
    TEST_EE(ctan, 2.0, INFINITY, NZERO, 1);
    TEST_EE(ctan, -2.0, -INFINITY, 0, -1);
    TEST_EE(ctan, 2.0, -INFINITY, NZERO, -1);

    /* sign of real part is unspecified */
    TEST_UNSPECIFIED2(ctan, INFINITY, INFINITY, 0, 1, NZERO, 1);
    TEST_UNSPECIFIED2(ctan, -INFINITY, INFINITY, 0, 1, NZERO, 1);
    TEST_UNSPECIFIED2(ctan, INFINITY, -INFINITY, 0, -1, NZERO, -1);
    TEST_UNSPECIFIED2(ctan, -INFINITY, -INFINITY, 0, -1, NZERO, -1);

    /* sign of real part is unspecified */
    TEST_UNSPECIFIED2(ctan, NAN, INFINITY, 0, 1, NZERO, 1);
    TEST_UNSPECIFIED2(ctan, NAN, -INFINITY, 0, -1, NZERO, -1);

    TEST_EE(ctan, 0, NAN, 0, NAN);
    TEST_EE(ctan, NZERO, NAN, NZERO, NAN);

    /* can raise FE_INVALID or not */
    TEST_EE(ctan, 2.0, NAN, NAN, NAN);
    TEST_EE(ctan, -2.0, NAN, NAN, NAN);

    TEST_EE(ctan, NAN, NAN, NAN, NAN);

    TEST_CC(ctan, 0.5, 0, ADDSUFFIX(tan)(0.5), 0);

    TEST_CC(ctan, 0, 1000, 0, 1);
    TEST_CC(ctan, 0, -1000, 0, -1);

    return ret;
}
#endif

#ifdef CCOSH
int test_ccosh()
{
    int ret = 1;
    /* ccosh(conj(z)) = conj(ccosh(z)) and ccosh is even */
    TEST_EE(ccosh, 0, 0, 1, 0);
    TEST_EE(ccosh, 0, NZERO, 1, NZERO);
    TEST_EE(ccosh, NZERO, 0, 1, NZERO);
    TEST_EE(ccosh, NZERO, NZERO, 1, 0);

    /* sign of imaginary part is unspecified */
    TEST_RAISES_UNSPECIFIED2(ccosh, 0, INFINITY, NAN, 0, \
                             NAN, NZERO, FE_INVALID);

    /* sign of imaginary part is unspecified */
    TEST_UNSPECIFIED2(ccosh, 0, NAN, NAN, 0, NAN, NZERO);
    TEST_UNSPECIFIED2(ccosh, NZERO, NAN, NAN, 0, NAN, NZERO);

    TEST_RAISES(ccosh, 2.0, INFINITY, NAN, NAN, FE_INVALID);
    TEST_RAISES(ccosh, 2.0, -INFINITY, NAN, NAN, FE_INVALID);
    TEST_RAISES(ccosh, -2.0, INFINITY, NAN, NAN, FE_INVALID);
    TEST_RAISES(ccosh, -2.0, -INFINITY, NAN, NAN, FE_INVALID);

    /* can raise FE_INVALID or not */
    TEST_EE(ccosh, 2.0, NAN, NAN, NAN);
    TEST_EE(ccosh, -2.0, NAN, NAN, NAN);

    TEST_EE(ccosh, INFINITY, 0, INFINITY, 0);
    TEST_EE(ccosh, INFINITY, NZERO, INFINITY, NZERO);
    TEST_EE(ccosh, -INFINITY, 0, INFINITY, NZERO);
    TEST_EE(ccosh, -INFINITY, NZERO, INFINITY, 0);

    TEST_EE(ccosh, INFINITY, 1.0, INFINITY, INFINITY);
    TEST_EE(ccosh, INFINITY, -1.0, INFINITY, -INFINITY);
    TEST_EE(ccosh, -INFINITY, 1.0, INFINITY, -INFINITY);
    TEST_EE(ccosh, -INFINITY, -1.0, INFINITY, INFINITY);
    TEST_EE(ccosh, INFINITY, 2.0, -INFINITY, INFINITY);
    TEST_EE(ccosh, INFINITY, -2.0, -INFINITY, -INFINITY);
    TEST_EE(ccosh, -INFINITY, 2.0, -INFINITY, -INFINITY);
    TEST_EE(ccosh, -INFINITY, -2.0, -INFINITY, INFINITY);
    TEST_EE(ccosh, INFINITY, 4.0, -INFINITY, -INFINITY);
    TEST_EE(ccosh, INFINITY, -4.0, -INFINITY, INFINITY);
    TEST_EE(ccosh, -INFINITY, 4.0, -INFINITY, INFINITY);
    TEST_EE(ccosh, -INFINITY, -4.0, -INFINITY, -INFINITY);
    TEST_EE(ccosh, INFINITY, 5.0, INFINITY, -INFINITY);
    TEST_EE(ccosh, INFINITY, -5.0, INFINITY, INFINITY);
    TEST_EE(ccosh, -INFINITY, 5.0, INFINITY, INFINITY);
    TEST_EE(ccosh, -INFINITY, -5.0, INFINITY, -INFINITY);

    /* sign of real part is unspecified */
    TEST_RAISES_UNSPECIFIED2(ccosh, INFINITY, INFINITY, INFINITY, NAN, \
                             -INFINITY, NAN, FE_INVALID);

    TEST_EE(ccosh, INFINITY, NAN, INFINITY, NAN);
    TEST_EE(ccosh, -INFINITY, NAN, INFINITY, NAN);

    /* sign of imaginary part is unspecified */
    TEST_UNSPECIFIED2(ccosh, NAN, 0, NAN, 0, NAN, NZERO);
    TEST_UNSPECIFIED2(ccosh, NAN, NZERO, NAN, 0, NAN, NZERO);

    /* can raise FE_INVALID or not */
    TEST_EE(ccosh, NAN, 2.0, NAN, NAN);
    TEST_EE(ccosh, NAN, -2.0, NAN, NAN);

    TEST_EE(ccosh, NAN, NAN, NAN, NAN);

    TEST_CC(ccosh, 0.5, 0, ADDSUFFIX(cosh)(0.5), 0);

    return ret;
}
#endif

#ifdef CSINH
int test_csinh()
{
    int ret = 1;
    /* csinh(conj(z)) = conj(csinh(z)) and csinh is odd */
    TEST_EE(csinh, 0, 0, 0, 0);
    TEST_EE(csinh, 0, NZERO, 0, NZERO);
    TEST_EE(csinh, NZERO, 0, NZERO, 0);
    TEST_EE(csinh, NZERO, NZERO, NZERO, NZERO);

    /* sign of real part is unspecified */
    TEST_RAISES_UNSPECIFIED2(csinh, 0, INFINITY, 0, NAN, \
                             NZERO, NAN, FE_INVALID);

    /* sign of real part is unspecified */
    TEST_UNSPECIFIED2(csinh, 0, NAN, 0, NAN, NZERO, NAN);
    TEST_UNSPECIFIED2(csinh, NZERO, NAN, 0, NAN, NZERO, NAN);

    TEST_RAISES(csinh, 2.0, INFINITY, NAN, NAN, FE_INVALID);
    TEST_RAISES(csinh, 2.0, -INFINITY, NAN, NAN, FE_INVALID);
    TEST_RAISES(csinh, -2.0, INFINITY, NAN, NAN, FE_INVALID);
    TEST_RAISES(csinh, -2.0, -INFINITY, NAN, NAN, FE_INVALID);

    /* can raise FE_INVALID or not */
    TEST_EE(csinh, 2.0, NAN, NAN, NAN);
    TEST_EE(csinh, -2.0, NAN, NAN, NAN);

    TEST_EE(csinh, INFINITY, 0, INFINITY, 0);
    TEST_EE(csinh, INFINITY, NZERO, INFINITY, NZERO);
    TEST_EE(csinh, -INFINITY, 0, -INFINITY, 0);
    TEST_EE(csinh, -INFINITY, NZERO, -INFINITY, NZERO);

    TEST_EE(csinh, INFINITY, 1.0, INFINITY, INFINITY);
    TEST_EE(csinh, INFINITY, -1.0, INFINITY, -INFINITY);
    TEST_EE(csinh, -INFINITY, 1.0, -INFINITY, INFINITY);
    TEST_EE(csinh, -INFINITY, -1.0, -INFINITY, -INFINITY);
    TEST_EE(csinh, INFINITY, 2.0, -INFINITY, INFINITY);
    TEST_EE(csinh, INFINITY, -2.0, -INFINITY, -INFINITY);
    TEST_EE(csinh, -INFINITY, 2.0, INFINITY, INFINITY);
    TEST_EE(csinh, -INFINITY, -2.0, INFINITY, -INFINITY);
    TEST_EE(csinh, INFINITY, 4.0, -INFINITY, -INFINITY);
    TEST_EE(csinh, INFINITY, -4.0, -INFINITY, INFINITY);
    TEST_EE(csinh, -INFINITY, 4.0, INFINITY, -INFINITY);
    TEST_EE(csinh, -INFINITY, -4.0, INFINITY, INFINITY);
    TEST_EE(csinh, INFINITY, 5.0, INFINITY, -INFINITY);
    TEST_EE(csinh, INFINITY, -5.0, INFINITY, INFINITY);
    TEST_EE(csinh, -INFINITY, 5.0, -INFINITY, -INFINITY);
    TEST_EE(csinh, -INFINITY, -5.0, -INFINITY, INFINITY);

    /* sign of real part is unspecified */
    TEST_RAISES_UNSPECIFIED2(csinh, INFINITY, INFINITY, INFINITY, NAN, \
                             -INFINITY, NAN, FE_INVALID);

    /* sign of real part is unspecified */
    TEST_UNSPECIFIED2(csinh, INFINITY, NAN, INFINITY, NAN, -INFINITY, NAN);
    TEST_UNSPECIFIED2(csinh, -INFINITY, NAN, INFINITY, NAN, -INFINITY, NAN);

    TEST_EE(csinh, NAN, 0, NAN, 0);
    TEST_EE(csinh, NAN, NZERO, NAN, NZERO);

    /* can raise FE_INVALID or not */
    TEST_EE(csinh, NAN, 2.0, NAN, NAN);
    TEST_EE(csinh, NAN, -2.0, NAN, NAN);

    TEST_EE(csinh, NAN, NAN, NAN, NAN);

    TEST_CC(csinh, 0.5, 0, ADDSUFFIX(sinh)(0.5), 0);
    
    return ret;
}
#endif

#ifdef CTANH
int test_ctanh()
{
    int ret = 1;
    /* ctanh(conj(z)) = conj(ctanh(z)) and ctanh is odd */
    TEST_EE(ctanh, 0, 0, 0, 0);
    TEST_EE(ctanh, 0, NZERO, 0, NZERO);
    TEST_EE(ctanh, NZERO, 0, NZERO, 0);
    TEST_EE(ctanh, NZERO, NZERO, NZERO, NZERO);

    TEST_RAISES(ctanh, 2.0, INFINITY, NAN, NAN, FE_INVALID);
    TEST_RAISES(ctanh, -2.0, INFINITY, NAN, NAN, FE_INVALID);

    /* can raise FE_INVALID or not */
    TEST_EE(ctanh, 2.0, NAN, NAN, NAN);
    TEST_EE(ctanh, -2.0, NAN, NAN, NAN);

    TEST_EE(ctanh, INFINITY, 1.0, 1.0, 0);
    TEST_EE(ctanh, INFINITY, -1.0, 1.0, NZERO);
    TEST_EE(ctanh, -INFINITY, 1.0, -1.0, 0);
    TEST_EE(ctanh, -INFINITY, -1.0, -1.0, NZERO);
    TEST_EE(ctanh, INFINITY, 2.0, 1.0, NZERO);
    TEST_EE(ctanh, INFINITY, -2.0, 1.0, 0);
    TEST_EE(ctanh, -INFINITY, 2.0, -1.0, NZERO);
    TEST_EE(ctanh, -INFINITY, -2.0, -1.0, 0);

    /* sign of imaginary part is unspecified */
    TEST_UNSPECIFIED2(ctanh, INFINITY, INFINITY, 1, 0, 1, NZERO);
    TEST_UNSPECIFIED2(ctanh, INFINITY, -INFINITY, 1, 0, 1, NZERO);
    TEST_UNSPECIFIED2(ctanh, -INFINITY, INFINITY, -1, 0, -1, NZERO);
    TEST_UNSPECIFIED2(ctanh, -INFINITY, -INFINITY, -1, 0, -1, NZERO);

    /* sign of imaginary part is unspecified */
    TEST_UNSPECIFIED2(ctanh, INFINITY, NAN, 1, 0, 1, NZERO);
    TEST_UNSPECIFIED2(ctanh, -INFINITY, NAN, -1, 0, -1, NZERO);

    TEST_EE(ctanh, NAN, 0, NAN, 0);
    TEST_EE(ctanh, NAN, NZERO, NAN, NZERO);

    /* can raise FE_INVALID or not */
    TEST_EE(ctanh, NAN, 2.0, NAN, NAN);
    TEST_EE(ctanh, NAN, -2.0, NAN, NAN);

    TEST_EE(ctanh, NAN, NAN, NAN, NAN);

    TEST_CC(ctanh, 0.5, 0, ADDSUFFIX(tanh)(0.5), 0);

    TEST_CC(ctanh, 1000, 0, 1, 0);
    TEST_CC(ctanh, -1000, 0, -1, 0);

    return ret;
}
#endif

#ifdef CEXP
int test_cexp()
{
    int ret = 1;
    /* cexp(conj(z)) = conj(cexp(z)) */
    TEST_EE(cexp, 0, 0, 1, 0);
    TEST_EE(cexp, 0, NZERO, 1, NZERO);

    TEST_EE(cexp, NZERO, 0, 1, 0);
    TEST_EE(cexp, NZERO, NZERO, 1, NZERO);

    TEST_RAISES(cexp, 2.0, INFINITY, NAN, NAN, FE_INVALID);
    TEST_RAISES(cexp, 2.0, -INFINITY, NAN, NAN, FE_INVALID);

    /* can raise FE_INVALID  or not */
    TEST_EE(cexp, 42.0, NAN, NAN, NAN);

    TEST_EE(cexp, INFINITY, 0, INFINITY, 0);
    TEST_EE(cexp, INFINITY, NZERO, INFINITY, NZERO);

    TEST_EE(cexp, -INFINITY, 1.0, 0, 0);
    TEST_EE(cexp, -INFINITY, -1.0, 0, NZERO);
    TEST_EE(cexp, -INFINITY, 2.0, NZERO, 0);
    TEST_EE(cexp, -INFINITY, -2.0, NZERO, NZERO);
    TEST_EE(cexp, -INFINITY, 4.0, NZERO, NZERO);
    TEST_EE(cexp, -INFINITY, -4.0, NZERO, 0);
    TEST_EE(cexp, -INFINITY, 5.0, 0, NZERO);
    TEST_EE(cexp, -INFINITY, -5.0, 0, 0);

    TEST_EE(cexp, INFINITY, 1.0, INFINITY, INFINITY);
    TEST_EE(cexp, INFINITY, -1.0, INFINITY, -INFINITY);
    TEST_EE(cexp, INFINITY, 2.0, -INFINITY, INFINITY);
    TEST_EE(cexp, INFINITY, -2.0, -INFINITY, -INFINITY);
    TEST_EE(cexp, INFINITY, 4.0, -INFINITY, -INFINITY);
    TEST_EE(cexp, INFINITY, -4.0, -INFINITY, INFINITY);
    TEST_EE(cexp, INFINITY, 5.0, INFINITY, -INFINITY);
    TEST_EE(cexp, INFINITY, -5.0, INFINITY, INFINITY);

    /* signs of both parts are unspecified */
    TEST_UNSPECIFIED4(cexp, -INFINITY, INFINITY, 0, 0, NZERO, 0, \
                      0, NZERO, NZERO, NZERO);
    TEST_UNSPECIFIED4(cexp, -INFINITY, -INFINITY, 0, 0, NZERO, 0, \
                      0, NZERO, NZERO, NZERO);

    /* sign of real part is unspecifed */
    TEST_RAISES_UNSPECIFIED2(cexp, INFINITY, INFINITY, INFINITY, \
                             NAN, -INFINITY, NAN, FE_INVALID);

    /* signs of both parts are unspecified */
    TEST_UNSPECIFIED4(cexp, -INFINITY, NAN, 0, 0, NZERO, 0, \
                      0, NZERO, NZERO, NZERO);

    /* sign of real part is unspecified */
    TEST_UNSPECIFIED2(cexp, INFINITY, NAN, INFINITY, NAN, -INFINITY, NAN);

    TEST_EE(cexp, NAN, 0, NAN, 0);
    TEST_EE(cexp, NAN, NZERO, NAN, NZERO);
 
    /* can raise FE_INVALID or not */
    TEST_EE(cexp, NAN, 2.0, NAN, NAN);
    TEST_EE(cexp, NAN, -2.0, NAN, NAN);

    TEST_EE(cexp, NAN, NAN, NAN, NAN);

    TEST_CC(cexp, 0.5, 0, ADDSUFFIX(exp)(0.5), 0);

    TEST_CC(cexp, 1, 0, M_E, 0);
    TEST_CC(cexp, 0, 1, ADDSUFFIX(cos)(1), ADDSUFFIX(sin)(1));
    TEST_CC(cexp, 1, 1, M_E*ADDSUFFIX(cos)(1), M_E*ADDSUFFIX(sin)(1));

    return ret;
}
#endif

#ifdef CLOG
int test_clog()
{
    int ret = 1;
    /* clog(conj(z)) = conj(clog(z)) */
    TEST_RAISES(clog, NZERO, 0, -INFINITY, NPY_PI, FE_DIVBYZERO);
    TEST_RAISES(clog, NZERO, NZERO, -INFINITY, -NPY_PI, FE_DIVBYZERO);

    TEST_RAISES(clog, 0, 0, -INFINITY, 0, FE_DIVBYZERO);
    TEST_RAISES(clog, 0, NZERO, -INFINITY, NZERO, FE_DIVBYZERO);

    TEST_EC(clog, 2.0, INFINITY, INFINITY, NPY_PI_2);
    TEST_EC(clog, 2.0, -INFINITY, INFINITY, -NPY_PI_2);

    /* can raise FE_INVALID or not */
    TEST_EE(clog, 2.0, NAN, NAN, NAN);

    TEST_EC(clog, -INFINITY, 2.0, INFINITY, NPY_PI);
    TEST_EC(clog, -INFINITY, -2.0, INFINITY, -NPY_PI);
    
    TEST_EE(clog, INFINITY, 2.0, INFINITY, 0);
    TEST_EE(clog, INFINITY, -2.0, INFINITY, NZERO);

    TEST_EC(clog, -INFINITY, INFINITY, INFINITY, 0.75 * NPY_PI);
    TEST_EC(clog, -INFINITY, -INFINITY, INFINITY, -0.75 * NPY_PI);

    TEST_EC(clog, INFINITY, INFINITY, INFINITY, 0.25 * NPY_PI);
    TEST_EC(clog, INFINITY, -INFINITY, INFINITY, -0.25 * NPY_PI);

    TEST_EE(clog, INFINITY, NAN, INFINITY, NAN);
    TEST_EE(clog, -INFINITY, NAN, INFINITY, NAN);

    /* can raise FE_INVALID or not */
    TEST_EE(clog, NAN, 2.0, NAN, NAN);
    TEST_EE(clog, NAN, -2.0, NAN, NAN);

    TEST_EE(clog, NAN, INFINITY, INFINITY, NAN);
    TEST_EE(clog, NAN, -INFINITY, INFINITY, NAN);

    TEST_EE(clog, NAN, NAN, NAN, NAN);
    
    TEST_BRANCH_CUT(clog, -0.5, 0, 0, 1, 1, -1, 1);

    TEST_CC(clog, 0.5, 0, ADDSUFFIX(log)(0.5), 0);

    TEST_CC(clog, 1, 0, 0, 0);
    TEST_CC(clog, 1,  2, 0.80471895621705014, 1.1071487177940904);

    return ret;
}
#endif

#ifdef CPOW
int test_cpow()
{
    int ret = 1;

    /* there are _no_ annex G values for cpow. */
    /* We can check for branch cuts in here */

    /* tests from test_umath.py: TestPower: test_power_complex */
    TEST_CPOW_CC(1, 2, 0, 0, 1, 0);
    TEST_CPOW_CC(2, 3, 0, 0, 1, 0);
    TEST_CPOW_CC(3, 4, 0, 0, 1, 0);

    TEST_CPOW_CC(1, 2, 1, 0, 1, 2);
    TEST_CPOW_CC(2, 3, 1, 0, 2, 3);
    TEST_CPOW_CC(3, 4, 1, 0, 3, 4);

    TEST_CPOW_CC(1, 2, 2, 0, -3, 4);
    TEST_CPOW_CC(2, 3, 2, 0, -5, 12);
    TEST_CPOW_CC(3, 4, 2, 0, -7, 24);

    TEST_CPOW_CC(1, 2, 3, 0, -11, -2);
    TEST_CPOW_CC(2, 3, 3, 0, -46, 9);
    TEST_CPOW_CC(3, 4, 3, 0, -117, 44);

    TEST_CPOW_CC(1, 2, 4, 0, -7, -24);
    TEST_CPOW_CC(2, 3, 4, 0, -119, -120);
    TEST_CPOW_CC(3, 4, 4, 0, -527, -336);

    TEST_CPOW_CC(1, 2, -1, 0, 1.0/5.0, -2.0/5.0);
    TEST_CPOW_CC(2, 3, -1, 0, 2.0/13.0, -3.0/13.0);
    TEST_CPOW_CC(3, 4, -1, 0, 3.0/25.0, -4.0/25.0);

    TEST_CPOW_CC(1, 2, -2, 0, -3.0/25.0, -4.0/25.0);
    TEST_CPOW_CC(2, 3, -2, 0, -5.0/169.0, -12.0/169.0);
    TEST_CPOW_CC(3, 4, -2, 0, -7.0/625.0, -24.0/625.0);

    TEST_CPOW_CC(1, 2, -3, 0, -11.0/125.0, 2.0/125.0);
    TEST_CPOW_CC(2, 3, -3, 0, -46.0/2197.0, -9.0/2197.0);
    TEST_CPOW_CC(3, 4, -3, 0, -117.0/15625.0, -44.0/15625.0);
    
    TEST_CPOW_CC(1, 2, 0.5, 0, 1.272019649514069, 0.7861513777574233);
    TEST_CPOW_CC(2, 3, 0.5, 0, 1.6741492280355401, 0.895977476129838);
    TEST_CPOW_CC(3, 4, 0.5, 0, 2, 1);

    TEST_CPOW_CC(1, 2, 14, 0, -76443, 16124);
    TEST_CPOW_CC(2, 3, 14, 0, 23161315, 58317492);
    TEST_CPOW_CC(3, 4, 14, 0, 5583548873, 2465133864);

    TEST_CPOW_EE(0, INFINITY, 1, 0, 0, INFINITY);
    TEST_CPOW_EE(0, INFINITY, 2, 0, -INFINITY, NAN);
    TEST_CPOW_EE(0, INFINITY, 3, 0, NAN, NAN);

    TEST_CPOW_EE(1, INFINITY, 1, 0, 1, INFINITY);
    TEST_CPOW_EE(1, INFINITY, 2, 0, -INFINITY, INFINITY);
    TEST_CPOW_EE(1, INFINITY, 3, 0, -INFINITY, NAN);
    
    /* tests from test_umath.py: TestPower: test_power_zero */
    TEST_CPOW_CC(0, 0, 0.33, 0, 0, 0);
    TEST_CPOW_CC(0, 0, 0.5, 0, 0, 0);
    TEST_CPOW_CC(0, 0, 1.0, 0, 0, 0);
    TEST_CPOW_CC(0, 0, 1.5, 0, 0, 0);
    TEST_CPOW_CC(0, 0, 2.0, 0, 0, 0);
    TEST_CPOW_CC(0, 0, 3.0, 0, 0, 0);
    TEST_CPOW_CC(0, 0, 4.0, 0, 0, 0);
    TEST_CPOW_CC(0, 0, 5.0, 0, 0, 0);
    TEST_CPOW_CC(0, 0, 6.6, 0, 0, 0);

    TEST_CPOW_EE(0, 0, 0, 0, 1, 0);
    TEST_CPOW_EE(0, 0, 0, 1, NAN, NAN);

    TEST_CPOW_EE(0, 0, -0.33, 0, NAN, NAN);
    TEST_CPOW_EE(0, 0, -0.5, 0, NAN, NAN);
    TEST_CPOW_EE(0, 0, -1.0, 0, NAN, NAN);
    TEST_CPOW_EE(0, 0, -1.5, 0, NAN, NAN);
    TEST_CPOW_EE(0, 0, -2.0, 0, NAN, NAN);
    TEST_CPOW_EE(0, 0, -3.0, 0, NAN, NAN);
    TEST_CPOW_EE(0, 0, -4.0, 0, NAN, NAN);
    TEST_CPOW_EE(0, 0, -5.0, 0, NAN, NAN);
    TEST_CPOW_EE(0, 0, -6.6, 0, NAN, NAN);
    TEST_CPOW_EE(0, 0, -1, 0.2, NAN, NAN);

    /* tests from test_umath_complex.py: TestCpow: test_simple 
     * --- skip, duplicating existing tests ---
     */

    /* tests from test_umath_complex.py: TestCpow: test_scalar, test_array
     * these tests are equilvent for this level.
     */
    TEST_CPOW_CC(1, 0, 1, 0, 1, 0);
    TEST_CPOW_CC(1, 0, 0, 1, 1, 0);
    TEST_CPOW_CC(1, 0, -0.5, 1.5, 1, 0);
    TEST_CPOW_CC(1, 0, 2, 0, 1, 0);
    TEST_CPOW_CC(1, 0, 3, 0, 1, 0);

    TEST_CPOW_CC(0, 1, 1, 0, 0, 1);
    TEST_CPOW_CC(0, 1, 0, 1, ADDSUFFIX(exp)(-NPY_PI_2), 0);
    TEST_CPOW_CC(0, 1, -0.5, 1.5, 0.067019739708273365, 0.067019739708273365);
    TEST_CPOW_CC(0, 1, 2, 0, -1, 0);
    TEST_CPOW_CC(0, 1, 3, 0, 0, -1);

    TEST_CPOW_CC(2, 0, 1, 0, 2, 0);
    TEST_CPOW_CC(2, 0, 0, 1, ADDSUFFIX(cos)(NPY_LOG2E), ADDSUFFIX(sin)(NPY_LOG2E));
    TEST_CPOW_CC(2, 0, 2, 0, 4, 0);
    TEST_CPOW_CC(2, 0, 3, 0, 8, 0);

    TEST_CPOW_CC(2.5, 0.375, 1, 0, 2.5, 0.375);
    TEST_CPOW_CC(2.5, 0.375, 0, 1, 0.51691507509598866, 0.68939360813851125);
    TEST_CPOW_CC(2.5, 0.375, -0.5, 1.5, 0.12646517347496394, 0.48690593271654437);
    TEST_CPOW_CC(2.5, 0.375, 2, 0, 391.0/64.0, 15.0/8.0);
    TEST_CPOW_CC(2.5, 0.375, 3, 0, 1865.0/128.0, 3573.0/512.0);

    TEST_CPOW_EE(INFINITY, 0, 1, 0, NAN, NAN);
    TEST_CPOW_EE(INFINITY, 0, 0, 1, NAN, NAN);
    TEST_CPOW_EE(INFINITY, 0, -0.5, 1.5, NAN, NAN);
    TEST_CPOW_EE(INFINITY, 0, 2, 0, NAN, NAN);
    TEST_CPOW_EE(INFINITY, 0, 3, 0, NAN, NAN);

    TEST_CPOW_EE(NAN, 0, 1, 0, NAN, NAN);
    TEST_CPOW_EE(NAN, 0, 0, 1, NAN, NAN);
    TEST_CPOW_EE(NAN, 0, -0.5, 1.5, NAN, NAN);
    TEST_CPOW_EE(NAN, 0, 2, 0, NAN, NAN);
    TEST_CPOW_EE(NAN, 0, 3, 0, NAN, NAN);

    return ret;
}
#endif

#ifdef CSQRT
int test_csqrt()
{
    int ret = 1;
    /* csqrt(conj(z)) = conj(csqrt(z)) */
    TEST_EE(csqrt, 0, 0, 0, 0);
    TEST_EE(csqrt, 0, NZERO, 0, NZERO);

    TEST_EE(csqrt, NZERO, 0, 0, 0);
    TEST_EE(csqrt, NZERO, NZERO, 0, NZERO);

    TEST_EE(csqrt, 2.0, INFINITY, INFINITY, INFINITY);
    TEST_EE(csqrt, 2.0, -INFINITY, INFINITY, -INFINITY);

    TEST_EE(csqrt, NAN, INFINITY, INFINITY, INFINITY);
    TEST_EE(csqrt, NAN, -INFINITY, INFINITY, -INFINITY);
    
    TEST_EE(csqrt, INFINITY, INFINITY, INFINITY, INFINITY);
    TEST_EE(csqrt, INFINITY, -INFINITY, INFINITY, -INFINITY);

    TEST_EE(csqrt, -INFINITY, INFINITY, INFINITY, INFINITY);
    TEST_EE(csqrt, -INFINITY, -INFINITY, INFINITY, -INFINITY);

    /* can raise FE_INVALID or not */
    TEST_EE(csqrt, 2.0, NAN, NAN, NAN);

    TEST_EE(csqrt, -INFINITY, 2.0, 0, INFINITY);
    TEST_EE(csqrt, -INFINITY, -2.0, 0, -INFINITY);

    TEST_EE(csqrt, INFINITY, 2.0, INFINITY, 0);
    TEST_EE(csqrt, INFINITY, -2.0, INFINITY, NZERO);
    
    /* sign of imaginary part is unspecified */
    TEST_UNSPECIFIED2(csqrt, -INFINITY, NAN, NAN, INFINITY, NAN, -INFINITY);

    TEST_EE(csqrt, INFINITY, NAN, INFINITY, NAN);

    /* can raise FE_INVALID or not */
    TEST_EE(csqrt, NAN, 2.0, NAN, NAN);
    TEST_EE(csqrt, NAN, -2.0, NAN, NAN);

    TEST_EE(csqrt, NAN, NAN, NAN, NAN);

    TEST_BRANCH_CUT(csqrt, -0.5, 0, 0, 1, 1, -1, 1);

    TEST_CC(csqrt, 0.5, 0, ADDSUFFIX(sqrt)(0.5), 0);

    TEST_CC(csqrt, 1, 0, 1, 0);
    TEST_CC(csqrt, 0, 1, NPY_SQRT2/2.0, NPY_SQRT2/2.0);
    TEST_CC(csqrt, -1, 0, 0, 1);
    TEST_CC(csqrt, 1, 1, 1.0986841134678100, 0.4550898605622273);
    TEST_CC(csqrt, 1, -1, 1.0986841134678100, -0.4550898605622273);

    return ret;
}
#endif

int main(int argc, char** argv)
{
#ifdef CACOS
    return !test_cacos();
#endif
#ifdef CASIN
    return !test_casin();
#endif
#ifdef CATAN
    return !test_catan();
#endif
#ifdef CACOSH
    return !test_cacosh();
#endif
#ifdef CASINH
    return !test_casinh();
#endif
#ifdef CATANH
    return !test_catanh();
#endif
#ifdef CCOS
    return !test_ccos();
#endif
#ifdef CSIN
    return !test_csin();
#endif
#ifdef CTAN
    return !test_ctan();
#endif
#ifdef CCOSH
    return !test_ccosh();
#endif
#ifdef CSINH
    return !test_csinh();
#endif
#ifdef CTANH
    return !test_ctanh();
#endif
#ifdef CEXP
    return !test_cexp();
#endif
#ifdef CLOG
    return !test_clog();
#endif
#ifdef CPOW
    return !test_cpow();
#endif
#ifdef CSQRT
    return !test_csqrt();
#endif
}

