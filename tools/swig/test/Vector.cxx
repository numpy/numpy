#include <stdlib.h>
#include <math.h>
#include <iostream>
#include "Vector.h"

// The following macro defines a family of functions that work with 1D
// arrays with the forms
//
//     TYPE SNAMELength( TYPE vector[3]);
//     TYPE SNAMEProd(   TYPE * series, int size);
//     TYPE SNAMESum(    int size, TYPE * series);
//     void SNAMEReverse(TYPE array[3]);
//     void SNAMEOnes(   TYPE * array,  int size);
//     void SNAMEZeros(  int size, TYPE * array);
//     void SNAMEEOSplit(TYPE vector[3], TYPE even[3], odd[3]);
//     void SNAMETwos(   TYPE * twoVec, int size);
//     void SNAMEThrees( int size, TYPE * threeVec);
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
//  * 1D argout arrays
//  * 1D argout arrays, data last
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
