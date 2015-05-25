#include <stdlib.h>
#include <math.h>
#include <iostream>
#include "Flat.h"

// The following macro defines a family of functions that work with 1D
// arrays with the forms
//
//     void SNAMEProcess(TYPE * array,  int size);
//
// for any specified type TYPE (for example: short, unsigned int, long
// long, etc.) with given short name SNAME (for example: short, uint,
// longLong, etc.).  The macro is then expanded for the given
// TYPE/SNAME pairs.  The resulting functions are for testing numpy
// interfaces for:
//
//  * in-place arrays (arbitrary number of dimensions) with a fixed number of elements
//
#define TEST_FUNCS(TYPE, SNAME) \
\
void SNAME ## Process(TYPE * array, int size) {          \
  for (int i=0; i<size; ++i) array[i] += 1;              \
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
