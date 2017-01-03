#ifndef FLAT_H
#define FLAT_H

// The following macro defines the prototypes for a family of
// functions that work with arrays with the forms
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
#define TEST_FUNC_PROTOS(TYPE, SNAME) \
\
void SNAME ## Process(TYPE * array, int size); \

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
