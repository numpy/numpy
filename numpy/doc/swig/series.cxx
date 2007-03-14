#include <stdlib.h>
#include <math.h>
#include <iostream>
#include "series.h"

// The following macro defines a family of functions with the forms
//
//     TYPE SNAMEProd( TYPE * series, int size);
//     void SNAMEOnes( TYPE * array,  int size);
//     TYPE SNAMEMax(  TYPE * matrix, int rows, int cols);
//     void SNAMEFloor(TYPE * array,  int rows, int cols, TYPE floor);
//
// for any specified type TYPE (for example: short, unsigned int, long
// long, etc.) with given short name SNAME (for example: short, uint,
// longLong, etc.).  The macro is then expanded for the given
// TYPE/SNAME pairs.  The resulting functions are for testing numpy
// interfaces, respectively, for:
//
//  * 1D input arrays
//  * 1D in-place arrays
//  * 2D input arrays
//  * 2D in-place arrays
//
#define TEST_FUNCS(TYPE, SNAME) \
\
TYPE SNAME ## Prod(TYPE * series, int size) {     \
  TYPE result = 1;                                \
  for (int i=0; i<size; ++i) result *= series[i]; \
  return result;                                  \
}                                                 \
\
void SNAME ## Ones(TYPE * array, int size) { \
  for (int i=0; i<size; ++i) array[i] = 1;   \
}                                            \
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
void SNAME ## Floor(TYPE * array, int rows, int cols, TYPE floor) { \
  int i, j, index;                                                    \
  for (j=0; j<cols; ++j) {                                            \
    for (i=0; i<rows; ++i) {                                          \
      index = j*rows + i;                                             \
      if (array[index] < floor) array[index] = 0;                     \
    }                                                                 \
  }                                                                   \
}

TEST_FUNCS(signed char       , schar     )
TEST_FUNCS(unsigned char     , uchar     )
TEST_FUNCS(short             , short     )
TEST_FUNCS(unsigned short    , ushort    )
TEST_FUNCS(int               , int       )
TEST_FUNCS(unsigned int      , uint      )
TEST_FUNCS(long              , long      )
TEST_FUNCS(unsigned long     , ulong     )
TEST_FUNCS(long long         , longLong  )
TEST_FUNCS(unsigned long long, ulongLong )
TEST_FUNCS(float             , float     )
TEST_FUNCS(double            , double    )
TEST_FUNCS(long double       , longDouble)
