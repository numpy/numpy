/*
 * Header file for descriptor related definitions.  This file is included
 * in `ndarraytypes.h` for additional structure and is not a standalone header
 * meant for direct inclusion.
 */

#ifndef NUMPY_CORE_INCLUDE_NUMPY_DESCRIPTOR_H_
#define NUMPY_CORE_INCLUDE_NUMPY_DESCRIPTOR_H_


struct _PyArray_Descr;  /* Forward declaration */

/*******************************************
 * Type numbers
 *******************************************/

enum NPY_TYPES {
    NPY_BOOL=0,
    NPY_BYTE, NPY_UBYTE, NPY_SHORT, NPY_USHORT, NPY_INT, NPY_UINT,
    NPY_LONG, NPY_ULONG, NPY_LONGLONG, NPY_ULONGLONG,
    NPY_FLOAT, NPY_DOUBLE, NPY_LONGDOUBLE,
    NPY_CFLOAT, NPY_CDOUBLE, NPY_CLONGDOUBLE,
    NPY_OBJECT=17,
    NPY_STRING, NPY_UNICODE,
    NPY_VOID,
    /*
     * New 1.6 types appended, may be integrated
     * into the above in 2.0.
     */
    NPY_DATETIME, NPY_TIMEDELTA, NPY_HALF,

    NPY_CHAR NPY_ATTR_DEPRECATE("Use NPY_STRING"),

    /*
     * New types added after NumPy 2.0
     */
    NPY_VSTRING,

    /* NPY_NTYPES is version-dependent and defined in npy_2_compat.h */

    /* assign a high value to avoid changing this in the
       future when new dtypes are added */
    NPY_NOTYPE=64,

    NPY_USERDEF=256,  /* leave room for characters */

    /* The number of types not including the new 1.6 types */
    NPY_NTYPES_ABI_COMPATIBLE=21,
};

/* The number of legacy old-style dtypes */
#define NPY_NTYPES_LEGACY 24

/* How many floating point types are there (excluding half) */
#define NPY_NUM_FLOATTYPE 3


/*
 * These characters correspond to the array type and the struct
 * module
 */

enum NPY_TYPECHAR {
    NPY_BOOLLTR = '?',
    NPY_BYTELTR = 'b',
    NPY_UBYTELTR = 'B',
    NPY_SHORTLTR = 'h',
    NPY_USHORTLTR = 'H',
    NPY_INTLTR = 'i',
    NPY_UINTLTR = 'I',
    NPY_LONGLTR = 'l',
    NPY_ULONGLTR = 'L',
    NPY_LONGLONGLTR = 'q',
    NPY_ULONGLONGLTR = 'Q',
    NPY_HALFLTR = 'e',
    NPY_FLOATLTR = 'f',
    NPY_DOUBLELTR = 'd',
    NPY_LONGDOUBLELTR = 'g',
    NPY_CFLOATLTR = 'F',
    NPY_CDOUBLELTR = 'D',
    NPY_CLONGDOUBLELTR = 'G',
    NPY_OBJECTLTR = 'O',
    NPY_STRINGLTR = 'S',
    NPY_DEPRECATED_STRINGLTR2 = 'a',
    NPY_UNICODELTR = 'U',
    NPY_VOIDLTR = 'V',
    NPY_DATETIMELTR = 'M',
    NPY_TIMEDELTALTR = 'm',
    NPY_CHARLTR = 'c',

    /*
     * New non-legacy DTypes
     */
    NPY_VSTRINGLTR = 'T',

    /*
     * Note, we removed `NPY_INTPLTR` due to changing its definition
     * to 'n', rather than 'p'.  On any typical platform this is the
     * same integer.  'n' should be used for the `np.intp` with the same
     * size as `size_t` while 'p' remains pointer sized.
     *
     * 'p', 'P', 'n', and 'N' are valid and defined explicitly
     * in `arraytypes.c.src`.
     */

    /*
     * These are for dtype 'kinds', not dtype 'typecodes'
     * as the above are for.
     */
    NPY_GENBOOLLTR ='b',
    NPY_SIGNEDLTR = 'i',
    NPY_UNSIGNEDLTR = 'u',
    NPY_FLOATINGLTR = 'f',
    NPY_COMPLEXLTR = 'c',
};


/*******************************************
 * PyArray_DTypeMeta related definitions.
 *******************************************/
NPY_NO_EXPORT extern PyTypeObject PyArrayDTypeMeta_Type;

/*
 * While NumPy DTypes would not need to be heap types the plan is to
 * make DTypes available in Python at which point they will be heap types.
 * Since we also wish to add fields to the DType class, this looks like
 * a typical instance definition, but with PyHeapTypeObject instead of
 * only the PyObject_HEAD.
 * This must only be exposed very extremely careful consideration, since
 * it is a fairly complex construct which may be better to allow
 * refactoring of.
 */
#ifdef Py_LIMITED_API
/*
 * When compiling with the limited API we cannot expose the struct.  This
 * means that certain inline functions/macros are not compatible.
 * That is unfortunate, but can hopefully be improved on a case-by-case basis.
 */
typedef PyTypeObject PyArray_DTypeMeta

#else  /* Py_LIMITED_API */

typedef struct {
    PyHeapTypeObject super;
    /*
     * Most DTypes will have a singleton default instance, for the
     * parametric legacy DTypes (bytes, string, void, datetime) this
     * may be a pointer to the *prototype* instance?
     */
    struct _PyArray_Descr *singleton;
    /* Copy of the legacy DTypes type number, usually invalid. */
    int type_num;

    /* The type object of the scalar instances (may be NULL?) */
    PyTypeObject *scalar_type;
    /*
     * DType flags to signal legacy, parametric, or
     * abstract.  But plenty of space for additional information/flags.
     */
    npy_uint64 flags;

    /*
     * Use indirection in order to allow a fixed size for this struct.
     * A stable ABI size makes creating a static DType less painful
     * while also ensuring flexibility for all opaque API (with one
     * indirection due the pointer lookup).
     */
    void *dt_slots;
    void *reserved[3];
} PyArray_DTypeMeta;

#endif  /* Py_LIMITED_API */


/*******************************
 * Legacy Array functions (stored on the dtype)
 *******************************/


/* These must deal with unaligned and swapped data if necessary */
typedef PyObject * (PyArray_GetItemFunc) (void *, void *);
typedef int (PyArray_SetItemFunc)(PyObject *, void *, void *);

typedef void (PyArray_CopySwapNFunc)(void *, npy_intp, void *, npy_intp,
                                     npy_intp, int, void *);

typedef void (PyArray_CopySwapFunc)(void *, void *, int, void *);
typedef npy_bool (PyArray_NonzeroFunc)(void *, void *);


/*
 * These assume aligned and notswapped data -- a buffer will be used
 * before or contiguous data will be obtained
 */

typedef int (PyArray_CompareFunc)(const void *, const void *, void *);
typedef int (PyArray_ArgFunc)(void*, npy_intp, npy_intp*, void *);

typedef void (PyArray_DotFunc)(void *, npy_intp, void *, npy_intp, void *,
                               npy_intp, void *);

typedef void (PyArray_VectorUnaryFunc)(void *, void *, npy_intp, void *,
                                       void *);

/*
 * XXX the ignore argument should be removed next time the API version
 * is bumped. It used to be the separator.
 */
typedef int (PyArray_ScanFunc)(FILE *fp, void *dptr,
                               char *ignore, struct _PyArray_Descr *);
typedef int (PyArray_FromStrFunc)(char *s, void *dptr, char **endptr,
                                  struct _PyArray_Descr *);

typedef int (PyArray_FillFunc)(void *, npy_intp, void *);

typedef int (PyArray_SortFunc)(void *, npy_intp, void *);
typedef int (PyArray_ArgSortFunc)(void *, npy_intp *, npy_intp, void *);

typedef int (PyArray_FillWithScalarFunc)(void *, npy_intp, void *, void *);

typedef int (PyArray_ScalarKindFunc)(void *);

typedef struct {
    /*
     * Functions to cast to most other standard types
     * Can have some NULL entries. The types
     * DATETIME, TIMEDELTA, and HALF go into the castdict
     * even though they are built-in.
     */
    PyArray_VectorUnaryFunc *cast[NPY_NTYPES_ABI_COMPATIBLE];

    /* The next four functions *cannot* be NULL */

    /*
     * Functions to get and set items with standard Python types
     * -- not array scalars
     */
    PyArray_GetItemFunc *getitem;
    PyArray_SetItemFunc *setitem;

    /*
     * Copy and/or swap data.  Memory areas may not overlap
     * Use memmove first if they might
     */
    PyArray_CopySwapNFunc *copyswapn;
    PyArray_CopySwapFunc *copyswap;

    /*
     * Function to compare items
     * Can be NULL
     */
    PyArray_CompareFunc *compare;

    /*
     * Function to select largest
     * Can be NULL
     */
    PyArray_ArgFunc *argmax;

    /*
     * Function to compute dot product
     * Can be NULL
     */
    PyArray_DotFunc *dotfunc;

    /*
     * Function to scan an ASCII file and
     * place a single value plus possible separator
     * Can be NULL
     */
    PyArray_ScanFunc *scanfunc;

    /*
     * Function to read a single value from a string
     * and adjust the pointer; Can be NULL
     */
    PyArray_FromStrFunc *fromstr;

    /*
     * Function to determine if data is zero or not
     * If NULL a default version is
     * used at Registration time.
     */
    PyArray_NonzeroFunc *nonzero;

    /*
     * Used for arange. Should return 0 on success
     * and -1 on failure.
     * Can be NULL.
     */
    PyArray_FillFunc *fill;

    /*
     * Function to fill arrays with scalar values
     * Can be NULL
     */
    PyArray_FillWithScalarFunc *fillwithscalar;

    /*
     * Sorting functions
     * Can be NULL
     */
    PyArray_SortFunc *sort[NPY_NSORTS];
    PyArray_ArgSortFunc *argsort[NPY_NSORTS];

    /*
        * Dictionary of additional casting functions
        * PyArray_VectorUnaryFuncs
        * which can be populated to support casting
        * to other registered types. Can be NULL
        */
    PyObject *castdict;

    /*
     * Functions useful for generalizing
     * the casting rules.
     * Can be NULL;
     */
    PyArray_ScalarKindFunc *scalarkind;
    int **cancastscalarkindto;
    int *cancastto;

    void *_unused1;
    void *_unused2;
    void *_unused3;

    /*
     * Function to select smallest
     * Can be NULL
     */
    PyArray_ArgFunc *argmin;

} PyArray_ArrFuncs;

/* The item must be reference counted when it is inserted or extracted. */
#define NPY_ITEM_REFCOUNT   0x01
/* Same as needing REFCOUNT */
#define NPY_ITEM_HASOBJECT  0x01
/* Convert to list for pickling */
#define NPY_LIST_PICKLE     0x02
/* The item is a POINTER  */
#define NPY_ITEM_IS_POINTER 0x04
/* memory needs to be initialized for this data-type */
#define NPY_NEEDS_INIT      0x08
/* operations need Python C-API so don't give-up thread. */
#define NPY_NEEDS_PYAPI     0x10
/* Use f.getitem when extracting elements of this data-type */
#define NPY_USE_GETITEM     0x20
/* Use f.setitem when setting creating 0-d array from this data-type.*/
#define NPY_USE_SETITEM     0x40
/* A sticky flag specifically for structured arrays */
#define NPY_ALIGNED_STRUCT  0x80

/*
 *These are inherited for global data-type if any data-types in the
 * field have them
 */
#define NPY_FROM_FIELDS    (NPY_NEEDS_INIT | NPY_LIST_PICKLE | \
                            NPY_ITEM_REFCOUNT | NPY_NEEDS_PYAPI)

#define NPY_OBJECT_DTYPE_FLAGS (NPY_LIST_PICKLE | NPY_USE_GETITEM | \
                                NPY_ITEM_IS_POINTER | NPY_ITEM_REFCOUNT | \
                                NPY_NEEDS_INIT | NPY_NEEDS_PYAPI)

#define PyDataType_FLAGCHK(dtype, flag) \
        (((dtype)->flags & (flag)) == (flag))

#define PyDataType_REFCHK(dtype) \
        PyDataType_FLAGCHK(dtype, NPY_ITEM_REFCOUNT)



/*
 * The public, stable PyArray_Descr struct.  When building for 2.x ABI only
 * we can allow accessing the full flags (used internally also).
 * (See below for notes.)
 */
typedef struct _PyArray_Descr {
    PyObject_HEAD
    PyTypeObject *typeobj;
    char kind;
    char type;
    char byteorder;
    char _undefined;
    int type_num;
#if NPY_FEATURE_VERSION >= NPY_NUMPY_2_0_API
    npy_uint64 flags;
    npy_intp elsize;
    npy_intp alignment;
    npy_hash_t hash;
#endif
} PyArray_Descr;

/*
 * This struct is public, but only correct to use on NumPy 2.x with legacy
 * dtypes.
 */
typedef struct {
    PyObject_HEAD
    /*
     * the type object representing an
     * instance of this type -- should not
     * be two type_numbers with the same type
     * object.
     */
    PyTypeObject *typeobj;
    /* kind for this type */
    char kind;
    /* unique-character representing this type */
    char type;
    /*
     * '>' (big), '<' (little), '|' (not-applicable), or '=' (native).
     * Must be supported by all dtypes, only supporting '|' may break e.g.
     * pickling.
     */
    char byteorder;
    /* Empty space, used to keep `type_num` aligned between 1.x and 2.x */
    char _scratch1;
    /* number representing this type */
    int type_num;
    /* flags describing data type */
    npy_uint64 flags;
    /* element size (itemsize) for this type */
    npy_intp elsize;
    /* alignment needed for this type */
    npy_intp alignment;
    /*
     * Cached hash value (-1 if not yet computed).
     */
    npy_hash_t hash;

    PyArray_ArrFuncs *f;
    /*
     * Non-NULL if this type is
     * is an array (C-contiguous)
     * of some other type
     */
    struct _arr_descr *subarray;
    /*
     * The fields dictionary for this type
     * For statically defined descr this
     * is always Py_None
     */
    PyObject *fields;
    /*
     * An ordered tuple of field names or NULL
     * if no fields are defined
     */
    PyObject *names;
    /* Metadata about this dtype */
    PyObject *metadata;
    /*
     * Metadata specific to the C implementation
     * of the particular dtype. This was added
     * for NumPy 1.7.0.
     */
    NpyAuxData *c_metadata;
} _PyArray_LegacyDescr;


/*
 * Umodified PyArray_Descr struct identical to NumPy 1.x.  This struct is
 * used as a prototype for registering a new legacy DType.
 * It is also used to access the fields in user code running on 1.x.
 */
typedef struct {
    PyObject_HEAD
    PyTypeObject *typeobj;
    char kind;
    char type;
    char byteorder;
    char flags;
    int type_num;
    int elsize;
    int alignment;
    struct _arr_descr *subarray;
    PyObject *fields;
    PyObject *names;
    PyArray_ArrFuncs *f;
    PyObject *metadata;
    NpyAuxData *c_metadata;
    npy_hash_t hash;
} PyArray_DescrProto;


typedef struct _arr_descr {
    PyArray_Descr *base;
    PyObject *shape;       /* a tuple */
} PyArray_ArrayDescr;


/****************************************
 * NpyString
 *
 * Types used by the NpyString API.
 ****************************************/

/*
 * A "packed" encoded string. The string data must be accessed by first unpacking the string.
 */
typedef struct npy_packed_static_string npy_packed_static_string;

/*
 * An unpacked read-only view onto the data in a packed string
 */
typedef struct npy_unpacked_static_string {
    size_t size;
    const char *buf;
} npy_static_string;

/*
 * Handles heap allocations for static strings.
 */
typedef struct npy_string_allocator npy_string_allocator;

typedef struct {
    PyArray_Descr base;
    // The object representing a null value
    PyObject *na_object;
    // Flag indicating whether or not to coerce arbitrary objects to strings
    char coerce;
    // Flag indicating the na object is NaN-like
    char has_nan_na;
    // Flag indicating the na object is a string
    char has_string_na;
    // If nonzero, indicates that this instance is owned by an array already
    char array_owned;
    // The string data to use when a default string is needed
    npy_static_string default_string;
    // The name of the missing data object, if any
    npy_static_string na_name;
    // the allocator should only be directly accessed after
    // acquiring the allocator_lock and the lock should
    // be released immediately after the allocator is
    // no longer needed
    npy_string_allocator *allocator;
} PyArray_StringDTypeObject;



/****************************************
 * Datetime dtype specific definitions
 ****************************************/

/* The special not-a-time (NaT) value */
#define NPY_DATETIME_NAT NPY_MIN_INT64

/*
 * Upper bound on the length of a DATETIME ISO 8601 string
 *   YEAR: 21 (64-bit year)
 *   MONTH: 3
 *   DAY: 3
 *   HOURS: 3
 *   MINUTES: 3
 *   SECONDS: 3
 *   ATTOSECONDS: 1 + 3*6
 *   TIMEZONE: 5
 *   NULL TERMINATOR: 1
 */
#define NPY_DATETIME_MAX_ISO8601_STRLEN (21 + 3*5 + 1 + 3*6 + 6 + 1)

/* The FR in the unit names stands for frequency */
typedef enum {
    /* Force signed enum type, must be -1 for code compatibility */
    NPY_FR_ERROR = -1,      /* error or undetermined */

    /* Start of valid units */
    NPY_FR_Y = 0,           /* Years */
    NPY_FR_M = 1,           /* Months */
    NPY_FR_W = 2,           /* Weeks */
    /* Gap where 1.6 NPY_FR_B (value 3) was */
    NPY_FR_D = 4,           /* Days */
    NPY_FR_h = 5,           /* hours */
    NPY_FR_m = 6,           /* minutes */
    NPY_FR_s = 7,           /* seconds */
    NPY_FR_ms = 8,          /* milliseconds */
    NPY_FR_us = 9,          /* microseconds */
    NPY_FR_ns = 10,         /* nanoseconds */
    NPY_FR_ps = 11,         /* picoseconds */
    NPY_FR_fs = 12,         /* femtoseconds */
    NPY_FR_as = 13,         /* attoseconds */
    NPY_FR_GENERIC = 14     /* unbound units, can convert to anything */
} NPY_DATETIMEUNIT;

/*
 * NOTE: With the NPY_FR_B gap for 1.6 ABI compatibility, NPY_DATETIME_NUMUNITS
 * is technically one more than the actual number of units.
 */
#define NPY_DATETIME_NUMUNITS (NPY_FR_GENERIC + 1)
#define NPY_DATETIME_DEFAULTUNIT NPY_FR_GENERIC

/*
 * Business day conventions for mapping invalid business
 * days to valid business days.
 */
typedef enum {
    /* Go forward in time to the following business day. */
    NPY_BUSDAY_FORWARD,
    NPY_BUSDAY_FOLLOWING = NPY_BUSDAY_FORWARD,
    /* Go backward in time to the preceding business day. */
    NPY_BUSDAY_BACKWARD,
    NPY_BUSDAY_PRECEDING = NPY_BUSDAY_BACKWARD,
    /*
     * Go forward in time to the following business day, unless it
     * crosses a month boundary, in which case go backward
     */
    NPY_BUSDAY_MODIFIEDFOLLOWING,
    /*
     * Go backward in time to the preceding business day, unless it
     * crosses a month boundary, in which case go forward.
     */
    NPY_BUSDAY_MODIFIEDPRECEDING,
    /* Produce a NaT for non-business days. */
    NPY_BUSDAY_NAT,
    /* Raise an exception for non-business days. */
    NPY_BUSDAY_RAISE
} NPY_BUSDAY_ROLL;


#endif  /* NUMPY_CORE_INCLUDE_NUMPY_DESCRIPTOR_H_ */