#ifndef SERIES_H
#define SERIES_H

// The following macro defines the prototypes for a family of
// functions with the forms
//
//     TYPE SNAMELength(TYPE vector[3]);
//     TYPE SNAMEProd(  TYPE * series, int size);
//     TYPE SNAMESum(   int size, TYPE * series);
//     void SNAMEOnes(  TYPE * array,  int size);
//     void SNAMEZeros( int size, TYPE * array);
//     TYPE SNAMEDet(   TYPE matrix[2][2]);
//     TYPE SNAMEMax(   TYPE * matrix, int rows, int cols);
//     TYPE SNAMEMin(   int rows, int cols, TYPE * matrix);
//     void SNAMEFloor( TYPE * array,  int rows, int cols, TYPE floor);
//     void SNAMECeil(  int rows, int cols, TYPE * array,  TYPE ceil );
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
//  * 1D in-place arrays
//  * 1D in-place arrays, data last
//  * 2D input arrays, hard-coded lengths
//  * 2D input arrays
//  * 2D input arrays, data last
//  * 2D in-place arrays
//  * 2D in-place arrays, data last
//
#define TEST_FUNC_PROTOS(TYPE, SNAME) \
\
TYPE SNAME ## Length(TYPE vector[3]); \
TYPE SNAME ## Prod( TYPE * series, int size); \
TYPE SNAME ## Sum(  int size, TYPE * series); \
void SNAME ## Ones( TYPE * array,  int size); \
void SNAME ## Zeros(int size, TYPE * array); \
TYPE SNAME ## Det(  TYPE matrix[2][2]); \
TYPE SNAME ## Max(  TYPE * matrix, int rows, int cols); \
TYPE SNAME ## Min(  int rows, int cols, TYPE * matrix); \
void SNAME ## Floor(TYPE * array, int rows, int cols, TYPE floor); \
void SNAME ## Ceil( int rows, int cols, TYPE * array, TYPE ceil );

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
