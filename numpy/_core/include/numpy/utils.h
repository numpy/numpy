#ifndef NUMPY_CORE_INCLUDE_NUMPY_UTILS_H_
#define NUMPY_CORE_INCLUDE_NUMPY_UTILS_H_

#ifndef __COMP_NPY_UNUSED
    #if defined(__GNUC__)
        #define __COMP_NPY_UNUSED __attribute__ ((__unused__))
    #elif defined(__ICC)
        #define __COMP_NPY_UNUSED __attribute__ ((__unused__))
    #elif defined(__clang__)
        #define __COMP_NPY_UNUSED __attribute__ ((unused))
    #else
        #define __COMP_NPY_UNUSED
    #endif
#endif

#if defined(__GNUC__) || defined(__ICC) || defined(__clang__)
    #define NPY_DECL_ALIGNED(x) __attribute__ ((aligned (x)))
#elif defined(_MSC_VER)
    #define NPY_DECL_ALIGNED(x) __declspec(align(x))
#elif defined(__cplusplus)
    #define NPY_DECL_ALIGNED(x) alignas(x)
#else
    #define NPY_DECL_ALIGNED(x) _Alignas(x)
#endif

/*
 * Force the first field of a struct to be 8-byte aligned on Python 3.15+.
 * On older versions, this is a no-op. This is needed to avoid breaking ABI for
 * older abi3 or 3.14t wheels that are built older versions of numpy.
 */
#if PY_VERSION_HEX >= 0x030f0000
    #define _NPY_OPAQUE_FIRST_FIELD NPY_DECL_ALIGNED(8)
#else
    #define _NPY_OPAQUE_FIRST_FIELD
#endif

/* Use this to tag a variable as not used. It will remove unused variable
 * warning on support platforms (see __COM_NPY_UNUSED) and mangle the variable
 * to avoid accidental use */
#define NPY_UNUSED(x) __NPY_UNUSED_TAGGED ## x __COMP_NPY_UNUSED
#define NPY_EXPAND(x) x

#define NPY_STRINGIFY(x) #x
#define NPY_TOSTRING(x) NPY_STRINGIFY(x)

#define NPY_CAT__(a, b) a ## b
#define NPY_CAT_(a, b) NPY_CAT__(a, b)
#define NPY_CAT(a, b) NPY_CAT_(a, b)

#endif  /* NUMPY_CORE_INCLUDE_NUMPY_UTILS_H_ */
