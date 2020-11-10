cdef extern from "stdint_fake.h":
    ctypedef signed char             int8_t;
    ctypedef signed short            int16_t;
    ctypedef signed int              int32_t;
    ctypedef signed long long int    int64_t;
    ctypedef signed char             int_least8_t;
    ctypedef signed short            int_least16_t;
    ctypedef signed int              int_least32_t;
    ctypedef signed long long int    int_least64_t;
    ctypedef signed char             int_fast8_t;
    ctypedef signed int              int_fast16_t;
    ctypedef signed int              int_fast32_t;
    ctypedef signed long long int    int_fast64_t;
    ctypedef signed long long int    intmax_t;
    ctypedef unsigned char           uint8_t;
    ctypedef unsigned short          uint16_t;
    ctypedef unsigned int            uint32_t;
    ctypedef unsigned long long int  uint64_t;
    ctypedef unsigned char           uint_least8_t;
    ctypedef unsigned short          uint_least16_t;
    ctypedef unsigned int            uint_least32_t;
    ctypedef unsigned long long int  uint_least64_t;
    ctypedef unsigned char           uint_fast8_t;
    ctypedef unsigned int            uint_fast16_t;
    ctypedef unsigned int            uint_fast32_t;
    ctypedef unsigned long long int  uint_fast64_t;
    ctypedef unsigned long long int  uintmax_t;
    ctypedef signed long long int    intptr_t;
    ctypedef unsigned long long int  uintptr_t;

    cdef int INT8_MIN
    cdef int INT16_MIN
    cdef int INT32_MIN
    cdef int INT64_MIN
    cdef int INTMAX_MIN
    cdef int INT_LEAST8_MIN
    cdef int INT_LEAST16_MIN
    cdef int INT_LEAST32_MIN
    cdef int INT_LEAST64_MIN
    cdef int INT_FAST8_MIN
    cdef int INT_FAST16_MIN
    cdef int INT_FAST32_MIN
    cdef int INT_FAST64_MIN
    cdef int INT8_MAX
    cdef int INT16_MAX
    cdef int INT32_MAX
    cdef int INT64_MAX
    cdef int INTMAX_MAX
    cdef int INT_LEAST8_MAX
    cdef int INT_LEAST16_MAX
    cdef int INT_LEAST32_MAX
    cdef int INT_LEAST64_MAX
    cdef int INT_FAST8_MAX
    cdef int INT_FAST16_MAX
    cdef int INT_FAST32_MAX
    cdef int INT_FAST64_MAX
    cdef int UINT8_MAX
    cdef int UINT16_MAX
    cdef int UINT32_MAX
    cdef int UINT64_MAX
    cdef int UINTMAX_MAX
    cdef int UINT_LEAST8_MAX
    cdef int UINT_LEAST16_MAX
    cdef int UINT_LEAST32_MAX
    cdef int UINT_LEAST64_MAX
    cdef int UINT_FAST8_MAX
    cdef int UINT_FAST16_MAX
    cdef int UINT_FAST32_MAX
    cdef int UINT_FAST64_MAX
    cdef int INTPTR_MIN
    cdef int INTPTR_MAX
    cdef int UINTPTR_MAX
    cdef int INT8_C
    cdef int INT16_C
    cdef int INT32_C
    cdef int INT64_C
    cdef int INTMAX_C
    cdef int UINT8_C
    cdef int UINT16_C
    cdef int UINT32_C
    cdef int UINT64_C
    cdef int UINTMAX_C
    cdef int PTRDIFF_MIN
    cdef int PTRDIFF_MAX
    cdef int SIZE_MAX
    cdef int SIG_ATOMIC_MIN
    cdef int SIG_ATOMIC_MAX