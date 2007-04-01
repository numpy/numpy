#include <stdlib.h>
#include <math.h>
#include <iostream>
#include "series.h"

// The following macro defines a family of functions with the forms
//
//     TYPE SNAMELength(  TYPE vector[3]);
//     TYPE SNAMEProd(    TYPE * series, int size);
//     TYPE SNAMESum(     int size, TYPE * series);
//     void SNAMEReverse( TYPE array[3]);
//     void SNAMEOnes(    TYPE * array,  int size);
//     void SNAMEZeros(   int size, TYPE * array);
//     void SNAMEEOSplit( TYPE vector[3], TYPE even[3], odd[3]);
//     TYPE SNAMEDet(     TYPE matrix[2][2]);
//     TYPE SNAMEMax(     TYPE * matrix, int rows, int cols);
//     TYPE SNAMEMin(     int rows, int cols, TYPE * matrix);
//     void SNAMEScale(   TYPE matrix[3][3]);
//     void SNAMEFloor(   TYPE * array,  int rows, int cols, TYPE floor);
//     void SNAMECeil(    int rows, int cols, TYPE * array, TYPE ceil);
//     void SNAMELUSplit( TYPE in[3][3], TYPE lower[3][3], TYPE upper[3][3]);
//
// for any specified type TYPE (for example: short, unsigned int, long
// long, etc.) with given short name SNAME (for example: short, uint,
// longLong, etc.).  The macro is then expanded for the given
// TYPE/SNAME pairs.  The resulting functions are for testing numpy
// interfaces, respectively, for:
//
//  * 1D input arrays, hard-coded length
//  * 1D input arrays
//  * 1D input arrays, data last
//  * 1D in-place arrays, hard-coded length
//  * 1D in-place arrays
//  * 1D in-place arrays, data last
//  * 1D argout arrays, hard-coded length
//  * 2D input arrays, hard-coded length
//  * 2D input arrays
//  * 2D input arrays, data last
//  * 2D in-place arrays, hard-coded lengths
//  * 2D in-place arrays
//  * 2D in-place arrays, data last
//  * 2D argout arrays, hard-coded length
//
#define TEST_FUNCS(TYPE, SNAME) \
\
TYPE SNAME ## Length(TYPE vector[3]) {                   \
  double result = 0;                                     \
  for (int i=0; i<3; ++i) result += vector[i]*vector[i]; \
  return (TYPE)sqrt(result);   			         \
}                                                        \
\
TYPE SNAME ## Prod(TYPE * series, int size) {     \
  TYPE result = 1;                                \
  for (int i=0; i<size; ++i) result *= series[i]; \
  return result;                                  \
}                                                 \
\
TYPE SNAME ## Sum(int size, TYPE * series) {      \
  TYPE result = 0;                                \
  for (int i=0; i<size; ++i) result += series[i]; \
  return result;                                  \
}                                                 \
\
void SNAME ## Reverse(TYPE array[3]) { \
  TYPE temp = array[0];		       \
  array[0] = array[2];                 \
  array[2] = temp;                     \
}                                      \
\
void SNAME ## Ones(TYPE * array, int size) { \
  for (int i=0; i<size; ++i) array[i] = 1;   \
}                                            \
\
void SNAME ## Zeros(int size, TYPE * array) { \
  for (int i=0; i<size; ++i) array[i] = 0;    \
}                                             \
\
void SNAME ## EOSplit(TYPE vector[3], TYPE even[3], TYPE odd[3]) { \
  for (int i=0; i<3; ++i) {					   \
    if (i % 2 == 0) {						   \
      even[i] = vector[i];					   \
      odd[ i] = 0;						   \
    } else {							   \
      even[i] = 0;						   \
      odd[ i] = vector[i];					   \
    }								   \
  }								   \
}								   \
\
void SNAME ## Twos(TYPE* twoVec, int size) { \
  for (int i=0; i<size; ++i) twoVec[i] = 2;  \
}					     \
\
void SNAME ## Threes(int size, TYPE* threeVec) { \
  for (int i=0; i<size; ++i) threeVec[i] = 3;	 \
}						 \
\
TYPE SNAME ## Det(TYPE matrix[2][2]) {                          \
  return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]; \
}                                                               \
\
TYPE SNAME ## Max(TYPE * matrix, int rows, int cols) {	  \
  int i, j, index;                                        \
  TYPE result = matrix[0];                                \
  for (j=0; j<cols; ++j) {                                \
    for (i=0; i<rows; ++i) {                              \
      index = j*rows + i;                                 \
      if (matrix[index] > result) result = matrix[index]; \
    }                                                     \
  }                                                       \
  return result;                                          \
}                                                         \
\
TYPE SNAME ## Min(int rows, int cols, TYPE * matrix) {    \
  int i, j, index;                                        \
  TYPE result = matrix[0];                                \
  for (j=0; j<cols; ++j) {                                \
    for (i=0; i<rows; ++i) {                              \
      index = j*rows + i;                                 \
      if (matrix[index] < result) result = matrix[index]; \
    }                                                     \
  }                                                       \
  return result;                                          \
}                                                         \
\
void SNAME ## Scale(TYPE array[3][3], TYPE val) { \
  for (int i=0; i<3; ++i)                         \
    for (int j=0; j<3; ++j)                       \
      array[i][j] *= val;                         \
}                                                 \
\
void SNAME ## Floor(TYPE * array, int rows, int cols, TYPE floor) { \
  int i, j, index;                                                  \
  for (j=0; j<cols; ++j) {                                          \
    for (i=0; i<rows; ++i) {                                        \
      index = j*rows + i;                                           \
      if (array[index] < floor) array[index] = floor;               \
    }                                                               \
  }                                                                 \
}                                                                   \
\
void SNAME ## Ceil(int rows, int cols, TYPE * array, TYPE ceil) { \
  int i, j, index;                                                \
  for (j=0; j<cols; ++j) {                                        \
    for (i=0; i<rows; ++i) {                                      \
      index = j*rows + i;                                         \
      if (array[index] > ceil) array[index] = ceil;               \
    }                                                             \
  }                                                               \
}								  \
\
void SNAME ## LUSplit(TYPE matrix[3][3], TYPE lower[3][3], TYPE upper[3][3]) { \
  for (int i=0; i<3; ++i) {						       \
    for (int j=0; j<3; ++j) {						       \
      if (i >= j) {						 	       \
	lower[i][j] = matrix[i][j];					       \
	upper[i][j] = 0;					 	       \
      } else {							 	       \
	lower[i][j] = 0;					 	       \
	upper[i][j] = matrix[i][j];					       \
      }								 	       \
    }								 	       \
  }								 	       \
}

TEST_FUNCS(signed char       , schar    )
TEST_FUNCS(unsigned char     , uchar    )
TEST_FUNCS(short             , short    )
TEST_FUNCS(unsigned short    , ushort   )
TEST_FUNCS(int               , int      )
TEST_FUNCS(unsigned int      , uint     )
TEST_FUNCS(long              , long     )
TEST_FUNCS(unsigned long     , ulong    )
TEST_FUNCS(long long         , longLong )
TEST_FUNCS(unsigned long long, ulongLong)
TEST_FUNCS(float             , float    )
TEST_FUNCS(double            , double   )
