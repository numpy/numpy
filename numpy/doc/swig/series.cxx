#include <stdlib.h>
#include <math.h>
#include <iostream>
#include "series.h"

// *** One Dimensional Arrays ***

// Functions that take 1D arrays of type SHORT as input
short shortSum(short* series, int size) {
  short result = 0;
  for (int i=0; i<size; ++i) result += series[i];
  return result;
}

short shortProd(short* series, int size) {
  short result = 1;
  for (int i=0; i<size; ++i) result *= series[i];
  return result;
}

// Functions that take 1D arrays of type INT as input
int intSum(int* series, int size) {
  int result = 0;
  for (int i=0; i<size; ++i) result += series[i];
  return result;
}

int intProd(int* series, int size) {
  int result = 1;
  for (int i=0; i<size; ++i) result *= series[i];
  return result;
}

// Functions that take 1D arrays of type LONG as input
long longSum(long* series, int size) {
  long result = 0;
  for (int i=0; i<size; ++i) result += series[i];
  return result;
}

long longProd(long* series, int size) {
  long result = 1;
  for (int i=0; i<size; ++i) result *= series[i];
  return result;
}

// Functions that take 1D arrays of type FLOAT as input
float floatSum(float* series, int size) {
  float result = 0.0;
  for (int i=0; i<size; ++i) result += series[i];
  return result;
}

float floatProd(float* series, int size) {
  float result = 1.0;
  for (int i=0; i<size; ++i) result *= series[i];
  return result;
}

// Functions that take 1D arrays of type DOUBLE as input
double doubleSum(double* series, int size) {
  double result = 0.0;
  for (int i=0; i<size; ++i) result += series[i];
  return result;
}

double doubleProd(double* series, int size) {
  double result = 1.0;
  for (int i=0; i<size; ++i) result *= series[i];
  return result;
}

// Functions that manipulate 1D arrays of type INT in-place
void intZeros(int* array, int size) {
  for (int i=0; i<size; ++i) array[i] = 0;
}

void intOnes(int* array, int size) {
  for (int i=0; i<size; ++i) array[i] = 1;
}

void intNegate(int* array, int size) {
  for (int i=0; i<size; ++i) array[i] *= -1;
}

// Functions that manipulate 1D arrays of type DOUBLE in-place
void doubleZeros(double* array, int size) {
  for (int i=0; i<size; ++i) array[i] = 0.0;
}

void doubleOnes(double* array, int size) {
  for (int i=0; i<size; ++i) array[i] = 1.0;
}

void doubleNegate(double* array, int size) {
  for (int i=0; i<size; ++i) array[i] *= -1.0;
}

// *** Two Dimensional Arrays ***

// Functions that take 2D arrays of type INT as input
int intMax(int* matrix, int rows, int cols) {
  int i, j, index;
  int result = matrix[0];
  for (j=0; j<cols; ++j) {
    for (i=0; i<rows; ++i) {
      index = j*rows + i;
      if (matrix[index] > result) result = matrix[index];
    }
  }
  return result;
}

void intFloor(int* matrix, int rows, int cols, int floor) {
  int i, j, index;
  for (j=0; j<cols; ++j) {
    for (i=0; i<rows; ++i) {
      index = j*rows + i;
      if (matrix[index] < floor) matrix[index] = 0;
    }
  }
}

// Functions that take 2D arrays of type DOUBLE as input
double doubleMax(double* matrix, int rows, int cols) {
  int i, j, index;
  double result = matrix[0];
  for (j=0; j<cols; ++j) {
    for (i=0; i<rows; ++i) {
      index = j*rows + i;
      if (matrix[index] > result) result = matrix[index];
    }
  }
  return result;
}

void doubleFloor(double* matrix, int rows, int cols, double floor) {
  int i, j, index;
  for (j=0; j<cols; ++j) {
    for (i=0; i<rows; ++i) {
      index = j*rows + i;
      if (matrix[index] < floor) matrix[index] = 0.0;
    }
  }
}
