#ifndef VECTOR_H
#define VECTOR_H

// The following macro defines the prototypes for a family of
// functions that work with 1D arrays with the forms
//
//     TYPE SNAMELength( TYPE vector[3]);
//     TYPE SNAMEProd(   TYPE * series, int size);
//     TYPE SNAMESum(    int size, TYPE * series);
//     void SNAMEReverse(TYPE array[3]);
//     void SNAMEOnes(   TYPE * array,  int size);
//     void SNAMEZeros(  int size, TYPE * array);
//     void SNAMEEOSplit(TYPE vector[3], TYPE even[3], TYPE odd[3]);
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
#define TEST_FUNC_PROTOS(TYPE, SNAME) \
\
TYPE SNAME ## Length( TYPE vector[3]); \
TYPE SNAME ## Prod(   TYPE * series, int size); \
TYPE SNAME ## Sum(    int size, TYPE * series); \
void SNAME ## Reverse(TYPE array[3]); \
void SNAME ## Ones(   TYPE * array,  int size); \
void SNAME ## Zeros(  int size, TYPE * array); \
void SNAME ## EOSplit(TYPE vector[3], TYPE even[3], TYPE odd[3]); \
void SNAME ## Twos(   TYPE * twoVec, int size); \
void SNAME ## Threes( int size, TYPE * threeVec); \

TEST_FUNC_PROTOS(signed char       , schar    )
TEST_FUNC_PROTOS(unsigned char     , uchar    )
TEST_FUNC_PROTOS(short             , short    )
TEST_FUNC_PROTOS(unsigned short    , ushort   )
TEST_FUNC_PROTOS(int               , int      )
TEST_FUNC_PROTOS(unsigned int      , uint     )
TEST_FUNC_PROTOS(long              , long     )
TEST_FUNC_PROTOS(unsigned long     , ulong    )
TEST_FUNC_PROTOS(long long         , longLong )
TEST_FUNC_PROTOS(unsigned long long, ulongLong)
TEST_FUNC_PROTOS(float             , float    )
TEST_FUNC_PROTOS(double            , double   )

#endif
