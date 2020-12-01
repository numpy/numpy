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

    cdef int8_t INT8_MIN
    cdef int16_t INT16_MIN
    cdef int32_t INT32_MIN
    cdef int64_t INT64_MIN
    cdef int64_t INTMAX_MIN
    cdef int8_t INT_LEAST8_MIN
    cdef int16_t INT_LEAST16_MIN
    cdef int32_t INT_LEAST32_MIN
    cdef int64_t INT_LEAST64_MIN
    cdef int8_t INT_FAST8_MIN
    cdef int16_t INT_FAST16_MIN
    cdef int32_t INT_FAST32_MIN
    cdef int64_t INT_FAST64_MIN
    cdef int8_t INT8_MAX
    cdef int16_t INT16_MAX
    cdef int32_t INT32_MAX
    cdef int64_t INT64_MAX
    cdef int64_t INTMAX_MAX
    cdef int8_t INT_LEAST8_MAX
    cdef int16_t INT_LEAST16_MAX
    cdef int32_t INT_LEAST32_MAX
    cdef int64_t INT_LEAST64_MAX
    cdef int8_t INT_FAST8_MAX
    cdef int16_t INT_FAST16_MAX
    cdef int32_t INT_FAST32_MAX
    cdef int64_t INT_FAST64_MAX
    cdef uint8_t UINT8_MAX
    cdef uint16_t UINT16_MAX
    cdef uint32_t UINT32_MAX
    cdef uint64_t UINT64_MAX
    cdef uint64_t UINTMAX_MAX
    cdef uint8_t UINT_LEAST8_MAX
    cdef uint16_t UINT_LEAST16_MAX
    cdef uint32_t UINT_LEAST32_MAX
    cdef uint64_t UINT_LEAST64_MAX
    cdef uint8_t UINT_FAST8_MAX
    cdef uint16_t UINT_FAST16_MAX
    cdef uint32_t UINT_FAST32_MAX
    cdef uint64_t UINT_FAST64_MAX
    cdef int64_t INTPTR_MIN
    cdef int64_t INTPTR_MAX
    cdef uint64_t UINTPTR_MAX
    cdef int32_t PTRDIFF_MIN
    cdef int32_t PTRDIFF_MAX
    cdef uint32_t SIZE_MAX
    cdef int32_t SIG_ATOMIC_MIN
    cdef int32_t SIG_ATOMIC_MAX