#! /usr/bin/env python
# Last Change: Thu Oct 25 01:00 PM 2007 J

# This module should contains useful test code (as strings). They are mainly
# useful for checkers who need to run the tests (to check the mere presence of
# symbols or headers, those are overkill).

# Check whether CBLAS sgemm works
cblas_sgemm = r"""

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
