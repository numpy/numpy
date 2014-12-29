#include <stdlib.h>
#include <math.h>
#include <iostream>
#include "SuperTensor.h"

// The following macro defines a family of functions that work with 3D
// arrays with the forms
//
//     TYPE SNAMENorm(   TYPE supertensor[2][2][2][2]);
//     TYPE SNAMEMax(    TYPE * supertensor, int cubes, int slices, int rows, int cols);
//     TYPE SNAMEMin(    int cubes, int slices, int rows, int cols, TYPE * supertensor);
//     void SNAMEScale(  TYPE supertensor[3][3][3][3]);
//     void SNAMEFloor(  TYPE * array,  int cubes, int slices, int rows, int cols, TYPE floor);
//     void SNAMECeil(   int slices, int cubes, int slices, int rows, int cols, TYPE * array, TYPE ceil);
//     void SNAMELUSplit(TYPE in[2][2][2][2], TYPE lower[2][2][2][2], TYPE upper[2][2][2][2]);
//
// for any specified type TYPE (for example: short, unsigned int, long
// long, etc.) with given short name SNAME (for example: short, uint,
// longLong, etc.).  The macro is then expanded for the given
// TYPE/SNAME pairs.  The resulting functions are for testing numpy
// interfaces, respectively, for:
//
//  * 4D input arrays, hard-coded length
//  * 4D input arrays
//  * 4D input arrays, data last
//  * 4D in-place arrays, hard-coded lengths
//  * 4D in-place arrays
//  * 4D in-place arrays, data last
//  * 4D argout arrays, hard-coded length
//
#define TEST_FUNCS(TYPE, SNAME) \
\
TYPE SNAME ## Norm(TYPE supertensor[2][2][2][2]) {	     \
  double result = 0;				     \
  for (int l=0; l<2; ++l)			     \
    for (int k=0; k<2; ++k)			     \
      for (int j=0; j<2; ++j)			     \
        for (int i=0; i<2; ++i)			     \
          result += supertensor[l][k][j][i] * supertensor[l][k][j][i]; \
  return (TYPE)sqrt(result/16);			     \
}						     \
\
TYPE SNAME ## Max(TYPE * supertensor, int cubes, int slices, int rows, int cols) { \
  int i, j, k, l, index;						\
  TYPE result = supertensor[0];					\
  for (l=0; l<cubes; ++l) {					\
    for (k=0; k<slices; ++k) {					\
      for (j=0; j<rows; ++j) {					\
        for (i=0; i<cols; ++i) {					\
          index = l*slices*rows*cols + k*rows*cols + j*cols + i;			\
          if (supertensor[index] > result) result = supertensor[index];	\
        }                           \
      }								\
    }								\
  }								\
  return result;						\
}								\
\
TYPE SNAME ## Min(int cubes, int slices, int rows, int cols, TYPE * supertensor) {	\
  int i, j, k, l, index;						\
  TYPE result = supertensor[0];					\
  for (l=0; l<cubes; ++l) {					\
    for (k=0; k<slices; ++k) {					\
      for (j=0; j<rows; ++j) {					\
        for (i=0; i<cols; ++i) {					\
          index = l*slices*rows*cols + k*rows*cols + j*cols + i;			\
          if (supertensor[index] < result) result = supertensor[index];	\
        }                       \
      }								\
    }								\
  }								\
  return result;						\
}								\
\
void SNAME ## Scale(TYPE array[3][3][3][3], TYPE val) { \
  for (int l=0; l<3; ++l)			     \
    for (int k=0; k<3; ++k)			     \
      for (int j=0; j<3; ++j)			     \
        for (int i=0; i<3; ++i)			     \
          array[l][k][j][i] *= val;			     \
}						     \
\
void SNAME ## Floor(TYPE * array, int cubes, int slices, int rows, int cols, TYPE floor) { \
  int i, j, k, l, index;							     \
  for (l=0; l<cubes; ++l) {			     \
    for (k=0; k<slices; ++k) {						     \
      for (j=0; j<rows; ++j) {						     \
        for (i=0; i<cols; ++i) {						     \
          index = l*slices*rows*cols + k*rows*cols + j*cols + i;			\
          if (array[index] < floor) array[index] = floor;			     \
        }									     \
      }                                         \
    }									     \
  }									     \
}									     \
\
void SNAME ## Ceil(int cubes, int slices, int rows, int cols, TYPE * array, TYPE ceil) { \
  int i, j, k, l, index;							   \
  for (l=0; l<cubes; ++l) {						   \
    for (k=0; k<slices; ++k) {						   \
      for (j=0; j<rows; ++j) {						   \
        for (i=0; i<cols; ++i) {						   \
          index = l*slices*rows*cols + k*rows*cols + j*cols + i;			\
          if (array[index] > ceil) array[index] = ceil;			   \
        }                                   \
      }									   \
    }									   \
  }									   \
}									   \
\
void SNAME ## LUSplit(TYPE supertensor[2][2][2][2], TYPE lower[2][2][2][2], \
		      TYPE upper[2][2][2][2]) {			 \
  int sum;							 \
  for (int l=0; l<2; ++l) {					 \
    for (int k=0; k<2; ++k) {					 \
      for (int j=0; j<2; ++j) {					 \
        for (int i=0; i<2; ++i) {					 \
          sum = i + j + k + l;					 \
          if (sum < 2) {						 \
            lower[l][k][j][i] = supertensor[l][k][j][i];			 \
            upper[l][k][j][i] = 0;					 \
          } else {						 \
            upper[l][k][j][i] = supertensor[l][k][j][i];			 \
            lower[l][k][j][i] = 0;					 \
          }							 \
        }                           \
      }								 \
    }								 \
  }								 \
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

