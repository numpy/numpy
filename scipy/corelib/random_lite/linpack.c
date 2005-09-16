#include <math.h>
float sdot(long n,float *sx,long incx,float *sy,long incy)
{
static long i,ix,iy,m,mp1;
static float sdot,stemp;
    stemp = sdot = 0.0;
    if(n <= 0) return sdot;
    if(incx == 1 && incy == 1) goto S20;
    ix = iy = 1;
    if(incx < 0) ix = (-n+1)*incx+1;
    if(incy < 0) iy = (-n+1)*incy+1;
    for(i=1; i<=n; i++) {
        stemp += (*(sx+ix-1)**(sy+iy-1));
        ix += incx;
        iy += incy;
    }
    sdot = stemp;
    return sdot;
S20:
    m = n % 5L;
    if(m == 0) goto S40;
    for(i=0; i<m; i++) stemp += (*(sx+i)**(sy+i));
    if(n < 5) goto S60;
S40:
    mp1 = m+1;
    for(i=mp1; i<=n; i+=5) stemp += (*(sx+i-1)**(sy+i-1)+*(sx+i)**(sy+i)+*(sx+i
      +1)**(sy+i+1)+*(sx+i+2)**(sy+i+2)+*(sx+i+3)**(sy+i+3));
S60:
    sdot = stemp;
    return sdot;
}
void spofa(float *a,long lda,long n,long *info)
/*
     SPOFA FACTORS A REAL SYMMETRIC POSITIVE DEFINITE MATRIX.
     SPOFA IS USUALLY CALLED BY SPOCO, BUT IT CAN BE CALLED
     DIRECTLY WITH A SAVING IN TIME IF  RCOND  IS NOT NEEDED.
     (TIME FOR SPOCO) = (1 + 18/N)*(TIME FOR SPOFA) .
     ON ENTRY
        A       REAL(LDA, N)
                THE SYMMETRIC MATRIX TO BE FACTORED.  ONLY THE
                DIAGONAL AND UPPER TRIANGLE ARE USED.
        LDA     INTEGER
                THE LEADING DIMENSION OF THE ARRAY  A .
        N       INTEGER
                THE ORDER OF THE MATRIX  A .
     ON RETURN
        A       AN UPPER TRIANGULAR MATRIX  R  SO THAT  A = TRANS(R)*R
                WHERE  TRANS(R)  IS THE TRANSPOSE.
                THE STRICT LOWER TRIANGLE IS UNALTERED.
                IF  INFO .NE. 0 , THE FACTORIZATION IS NOT COMPLETE.
        INFO    INTEGER
                = 0  FOR NORMAL RETURN.
                = K  SIGNALS AN ERROR CONDITION.  THE LEADING MINOR
                     OF ORDER  K  IS NOT POSITIVE DEFINITE.
     LINPACK.  THIS VERSION DATED 08/14/78 .
     CLEVE MOLER, UNIVERSITY OF NEW MEXICO, ARGONNE NATIONAL LAB.
     SUBROUTINES AND FUNCTIONS
     BLAS SDOT
     FORTRAN SQRT
     INTERNAL VARIABLES
*/
{
extern float sdot(long n,float *sx,long incx,float *sy,long incy);
static long j,jm1,k;
static float t,s;
/*
     BEGIN BLOCK WITH ...EXITS TO 40
*/
    for(j=1; j<=n; j++) {
        *info = j;
        s = 0.0;
        jm1 = j-1;
        if(jm1 < 1) goto S20;
        for(k=0; k<jm1; k++) {
            t = *(a+k+(j-1)*lda)-sdot(k,(a+k*lda),1L,(a+(j-1)*lda),1L);
            t /=  *(a+k+k*lda);
            *(a+k+(j-1)*lda) = t;
            s += (t*t);
        }
S20:
        s = *(a+j-1+(j-1)*lda)-s;
/*
     ......EXIT
*/
        if(s <= 0.0) goto S40;
        *(a+j-1+(j-1)*lda) = sqrt(s);
    }
    *info = 0;
S40:
    return;
}
