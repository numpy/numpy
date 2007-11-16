#! /usr/bin/env python
# Last Change: Fri Nov 16 04:00 PM 2007 J

# This module should contains useful test code (as strings). They are mainly
# useful for checkers who need to run the tests (to check the mere presence of
# symbols or headers, those are overkill).

# Check whether CBLAS sgemm works
cblas_sgemm = r"""
enum CBLAS_ORDER {CblasRowMajor=101, CblasColMajor=102};
enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113};

void cblas_sgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const float alpha, const float *A,
                 const int lda, const float *B, const int ldb,
                 const float beta, float *C, const int ldc);
int
main (void)
{
    int lda = 3;
    float A[] = {1, 2, 3,
                 4, 5, 6};

    int ldb = 2;
    float B[] = {1, 2, 
	         3, 4,
		 5, 6};

    int ldc = 2;
    float C[] = { 0.00, 0.00,
                 0.00, 0.00 };

    /* Compute C = A B */
    cblas_sgemm (CblasRowMajor, 
                CblasNoTrans, CblasNoTrans, 2, 2, 3,
                1.0, A, lda, B, ldb, 0.0, C, ldc);

    return 0;  
}
"""

# Check whether calling sgemm from C works (FOLLOW FORTRAN CONVENTION !). This
# is useful to test sunperf, for example.
c_sgemm = r"""
/*
 * sunperf, when calling func wo cblas_ prefix, follows Fortran convention for
 * array layout in memory !
 */
int
main (void)
{
    int lda = 2;
    float A[] = {1, 4,
		 2, 5,
		 3, 6};

    int ldb = 3;
    float B[] = {1, 3, 5,
	         2, 4, 6}; 
    int ldc = 2;
    float C[] = { 0.00, 0.00,
                 0.00, 0.00 };

    /* Compute C = A B */
    sgemm('N', 'N', 2, 2, 3,
          1.0, A, lda, B, ldb, 0.0, C, ldc);

    printf("C = {%f, %f; %f, %f}\n", C[0], C[2], C[1], C[3]);
    return 0;  
}
"""

# Code which try sgesv (the exact symbol has to be given by lapack_sgsev % symbol)
lapack_sgesv = r"""
#define our_fancy_func %s

extern int our_fancy_func(int *n, int *nrhs, float a[], int *lda, int ipiv[], 
                  float b[], int *ldb, int *info);

int compare(float A[], float B[], int sz)
{
        int i;

        for(i = 0; i < sz; ++i) {
                if ( (A[i] - B[i] > 0.01) || (A[i] - B[i] < -0.01)) {
                        return -1;
                }
        }
        return 0;
}

int main(void)
{
    int n = 2;
    int nrhs = 2;
    int lda = 2;
    float A[] = { 1, 3, 2, 4};

    int ldb = 2;
    float B[] = { 1, 0, 0, 1};
    float X[] = { -2, 1.5, 1, -0.5};

    int ipov[] = {0, 0};
    int info;

    /* Compute X in A * X = B */
    our_fancy_func(&n, &nrhs, A, &lda, ipov, B, &ldb, &info);

    return compare(B, X, 4);
}
"""

# Simple test of blas (pure F77 program)
blas_sgemm = """
      program dot_main
          real x(2, 2), y(2, 2), z(2, 2)
          real sgemm, res, alpha
          integer n, m, k, incx, incy, i
          external sgemm
          n = 2
          m = 2
          k = 2
          alpha = 1
          
          x(1, 1) = 1
          x(2, 1) = 2
          x(1, 2) = 3
          x(2, 2) = 4
          
          y(1, 1) = 1
          y(2, 1) = -2
          y(1, 2) = -1
          y(2, 2) = 2
          res = sgemm('n', 'n', n, m, k, alpha, x, n, y, n, 0, z, n)
c          z should be ((-5, 5), (-6, 6))
c          print*, 'sgemm = ', z(1, 1), z(1, 2)
c          print*, '        ', z(2, 1), z(2, 2)
      end
"""

# Check whether calling sgemm from C works (FOLLOW FORTRAN CONVENTION !). 
c_sgemm2 = r"""
#include <stdio.h>

int
main (void)
{
    char transa = 'N', transb = 'N';
    int lda = 2;
    int ldb = 3;
    int n = 2, m = 2, k = 3;
    float alpha = 1.0, beta = 0.0;

    float A[] = {1, 4,
		 2, 5,
		 3, 6};

    float B[] = {1, 3, 5,
	         2, 4, 6}; 
    int ldc = 2;
    float C[] = { 0.00, 0.00,
                 0.00, 0.00 };

    /* Compute C = A B */
    %(func)s(&transa, &transb, &n, &m, &k,
          &alpha, A, &lda, B, &ldb, &beta, C, &ldc);

    printf("C = {%%f, %%f; %%f, %%f}\n", C[0], C[2], C[1], C[3]);
    return 0;  
}
"""

if __name__ == '__main__':
    pass
