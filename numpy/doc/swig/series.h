#ifndef SERIES_H
#define SERIES_H

// The following macro defines the prototypes for a family of
// functions with the forms
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
#define TEST_FUNC_PROTOS(TYPE, SNAME) \
\
TYPE SNAME ## Prod( TYPE * series, int size); \
void SNAME ## Ones( TYPE * array,  int size); \
TYPE SNAME ## Max(  TYPE * matrix, int rows, int cols); \
void SNAME ## Floor(TYPE * array,  int rows, int cols, TYPE floor);

TEST_FUNC_PROTOS(signed char       , schar     )
TEST_FUNC_PROTOS(unsigned char     , uchar     )
TEST_FUNC_PROTOS(short             , short     )
TEST_FUNC_PROTOS(unsigned short    , ushort    )
TEST_FUNC_PROTOS(int               , int       )
TEST_FUNC_PROTOS(unsigned int      , uint      )
TEST_FUNC_PROTOS(long              , long      )
TEST_FUNC_PROTOS(unsigned long     , ulong     )
TEST_FUNC_PROTOS(long long         , longLong  )
TEST_FUNC_PROTOS(unsigned long long, ulongLong )
TEST_FUNC_PROTOS(float             , float     )
TEST_FUNC_PROTOS(double            , double    )
TEST_FUNC_PROTOS(long double       , longDouble)

#endif
