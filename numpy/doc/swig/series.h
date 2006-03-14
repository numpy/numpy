#ifndef SERIES_H
#define SERIES_H

/*** One Dimensional Arrays ***/

/* Examples of functions that take 1D C arrays as input */
short shortSum( short* series, int size);
short shortProd(short* series, int size);

int intSum( int* series, int size);
int intProd(int* series, int size);

long longSum( long* series, int size);
long longProd(long* series, int size);

float floatSum( float* series, int size);
float floatProd(float* series, int size);

double doubleSum( double* series, int size);
double doubleProd(double* series, int size);

/* Examples of functions that manipulate 1D C arrays as in-place */
void intZeros( int* array, int size);
void intOnes(  int* array, int size);
void intNegate(int* array, int size);

void doubleZeros( double* array, int size);
void doubleOnes(  double* array, int size);
void doubleNegate(double* array, int size);

/*** Two Dimensional Arrays ***/

/* Examples of functions that take 2D arrays as input */
int intMax(int* matrix, int rows, int cols);
void intFloor(int* matrix, int rows, int cols, int floor);

double doubleMax(double* matrix, int rows, int cols);
void doubleFloor(double* matrix, int rows, int cols, double floor);

#endif
