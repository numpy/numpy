#include <stdlib.h>
#include <math.h>
#include <iostream>
#include "Fortran.h"

#define TEST_FUNCS(TYPE, SNAME) \
\
TYPE SNAME ## SecondElement(TYPE * matrix, int rows, int cols) {	  \
  TYPE result = matrix[1];                                \
  return result;                                          \
}                                                         \

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
