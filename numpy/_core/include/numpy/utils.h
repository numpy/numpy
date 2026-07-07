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

// Older compilers may name `_Alignas` differently; to allow compilation on such
// unsupported platforms, we don't redefine NPY_DECL_ALIGNED if it's already
// defined similar to CPython's _Py_ALIGNED_DEF.
#ifndef NPY_DECL_ALIGNED
#if defined(__GNUC__) || defined(__ICC) || defined(__clang__)
    #define NPY_DECL_ALIGNED(x) __attribute__ ((aligned (x)))
#elif defined(_MSC_VER)
    #define NPY_DECL_ALIGNED(x) __declspec(align(x))
#elif defined(__cplusplus)
    #define NPY_DECL_ALIGNED(x) alignas(x)
#else
    #define NPY_DECL_ALIGNED(x) _Alignas(x)
#endif
#endif
/*
 * Force the first field of a struct to be 8-byte aligned on Python 3.15+ to achieve
 * ABI compatibility of `_fields` structs with and without `PyObject_HEAD`.
 *
 * When we expose (or may want to) expose an object structs contents to the Python
 * stable ABI we define it as:
 *
 *     struct _PyArray_Descr {
 *     #if !defined(Py_TARGET_ABI3T)
 *         PyObject_HEAD;
 *     #endif
 *         _NPY_OPAQUE_FIRST_FIELD T first_field;
 *         ...
 *     } fields;
 *
 * Effectively creating two versions, one `fields_obj` and `fields_no_obj` depending
 * on the build target. These builds must be ABI compatible and for that:
 *
 *     offsetof(fields_obj, first_field) % alignof(fields_no_obj) == 0
 *
 * must hold. If it does not hold, then `fields_obj` (minus the PyObject_HEAD) will
 * be padded differently from `fields_no_obj`.  Concretely, this happened for 32bit
 * free-threaded builds (PyObject_HEAD having a size of 20 bytes) where the uint64
 * flags required padding to achieve 8 byte alignment, but with the `PyObject_HEAD`.
 *
 * The only way to ensure the above holds always is to pad the first field so that
 * it's offset is a multiple of the alignment. The simplest path (e.g. what PEP 697
 * also does) may be to always pad to `alignof(max_align_t)`.
 * We do _not_ do this, because we have to remain ABI compatible to abi3 builds and
 * Python GIL enabled builds have a PyObject_HEAD size that is a multiple of
 * 8/16 (32bit/64bit systems) alignment.
 * So, instead we use 8 bytes. We could choose 16 for 64bit systems but it currently
 * makes no difference (there is a static_assert to notify us if it might matter).
 *
 * Because the ABI3T target was added in Python 3.15 we _can_ freely add padding for
 * Python 3.15+ builds for them. And this is where it matters, because 32bit
 * free-threaded builds have a `sizeof(PyObject) == 20` which gives 4 byte max
 * alignment which breaks the requirement above.
 *
 * As an actual implementation we add the padding by explicitly aligning the first
 * field to 8 bytes.
 * In theory this might guarantee an alignment larger than `alignof(max_align_t)`,
 * we assume that we would notice this first and in practice malloc alignment is >=8.
 *
 * One may be able to expose >8 byte aligned fields, but this requires thoughts on
 * how it affects abi3, abi3t builds and compatibility with older Python/NumPy builds.
 *
 * Note that assertions concerning these assumptions are paired with the
 * `*_GET_ITEM_DATA` function definitions (in `arrayobject.c`).
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
