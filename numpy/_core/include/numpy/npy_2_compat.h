/*
 * This header file defines relevant features which:
 * - Require runtime inspection depending on the NumPy version.
 * - May be needed when compiling with an older version of NumPy to allow
 *   a smooth transition.
 *
 * As such, it is shipped with NumPy 2.0, but designed to be vendored in full
 * or parts by downstream projects.
 *
 * It must be included after any other includes.  `import_array()` must have
 * been called in the scope or version dependency will misbehave, even when
 * only `PyUFunc_` API is used.
 *
 * If required complicated defs (with inline functions) should be written as:
 *
 *     #if NPY_FEATURE_VERSION >= NPY_2_0_API_VERSION
 *         Simple definition when NumPy 2.0 API is guaranteed.
 *     #else
 *         static inline definition of a 1.x compatibility shim
 *         #if NPY_ABI_VERSION < 0x02000000
 *            Make 1.x compatibility shim the public API (1.x only branch)
 *         #else
 *             Runtime dispatched version (1.x or 2.x)
 *         #endif
 *     #endif
 *
 * An internal build always passes NPY_FEATURE_VERSION >= NPY_2_0_API_VERSION
 */

#ifndef NUMPY_CORE_INCLUDE_NUMPY_NPY_2_COMPAT_H_
#define NUMPY_CORE_INCLUDE_NUMPY_NPY_2_COMPAT_H_

/*
 * New macros for accessing real and complex part of a complex number can be
 * found in "npy_2_complexcompat.h".
 */


/*
 * This header is meant to be included by downstream directly for 1.x compat.
 * In that case we need to ensure that users first included the full headers
 * and not just `ndarraytypes.h`.
 */

#ifndef NPY_FEATURE_VERSION
  #error "The NumPy 2 compat header requires `import_array()` for which "  \
         "the `ndarraytypes.h` header include is not sufficient.  Please "  \
         "include it after `numpy/ndarrayobject.h` or similar.\n"  \
         "To simplify inclusion, you may use `PyArray_ImportNumPy()` " \
         "which is defined in the compat header and is lightweight (can be)."
#endif

#if NPY_ABI_VERSION < 0x02000000
  /*
   * Define 2.0 feature version as it is needed below to decide whether we
   * compile for both 1.x and 2.x (defining it guarantees 1.x only).
   */
  #define NPY_2_0_API_VERSION 0x00000012
  /*
   * If we are compiling with NumPy 1.x, PyArray_RUNTIME_VERSION so we
   * pretend the `PyArray_RUNTIME_VERSION` is `NPY_FEATURE_VERSION`.
   * This allows downstream to use `PyArray_RUNTIME_VERSION` if they need to.
   */
  #define PyArray_RUNTIME_VERSION NPY_FEATURE_VERSION
  /* Compiling on NumPy 1.x where these are the same: */
  #define PyArray_DescrProto PyArray_Descr
#endif


/*
 * Define a better way to call `_import_array()` to simplify backporting as
 * we now require imports more often (necessary to make ABI flexible).
 */
#ifdef import_array1

static inline int
PyArray_ImportNumPyAPI(void)
{
    if (NPY_UNLIKELY(PyArray_API == NULL)) {
        import_array1(-1);
    }
    return 0;
}

#endif  /* import_array1 */


/*
 * NPY_DEFAULT_INT
 *
 * The default integer has changed, `NPY_DEFAULT_INT` is available at runtime
 * for use as type number, e.g. `PyArray_DescrFromType(NPY_DEFAULT_INT)`.
 *
 * NPY_RAVEL_AXIS
 *
 * This was introduced in NumPy 2.0 to allow indicating that an axis should be
 * raveled in an operation. Before NumPy 2.0, NPY_MAXDIMS was used for this purpose.
 *
 * NPY_MAXDIMS
 *
 * A constant indicating the maximum number dimensions allowed when creating
 * an ndarray.
 *
 * NPY_NTYPES_LEGACY
 *
 * The number of built-in NumPy dtypes.
 */
#if NPY_FEATURE_VERSION >= NPY_2_0_API_VERSION
    #define NPY_DEFAULT_INT NPY_INTP
    #define NPY_RAVEL_AXIS NPY_MIN_INT
    #define NPY_MAXARGS 64

#elif NPY_ABI_VERSION < 0x02000000
    #define NPY_DEFAULT_INT NPY_LONG
    #define NPY_RAVEL_AXIS 32
    #define NPY_MAXARGS 32

    /* Aliases of 2.x names to 1.x only equivalent names */
    #define NPY_NTYPES NPY_NTYPES_LEGACY
    #define PyArray_DescrProto PyArray_Descr
    #define _PyArray_LegacyDescr PyArray_Descr
    /* NumPy 2 definition always works, but add it for 1.x only */
    #define PyDataType_ISLEGACY(dtype) (1)
#else
    #define NPY_DEFAULT_INT  \
        (PyArray_RUNTIME_VERSION >= NPY_2_0_API_VERSION ? NPY_INTP : NPY_LONG)
    #define NPY_RAVEL_AXIS  \
        (PyArray_RUNTIME_VERSION >= NPY_2_0_API_VERSION ? NPY_MIN_INT : 32)
    #define NPY_MAXARGS  \
        (PyArray_RUNTIME_VERSION >= NPY_2_0_API_VERSION ? 64 : 32)
#endif


/*
 * Access inline functions for descriptor fields.  Except for the first
 * few fields, these needed to be moved (elsize, alignment) for
 * additional space.  Or they are descriptor specific and are not generally
 * available anymore (metadata, c_metadata, subarray, names, fields).
 *
 * Most of these are defined via the `DESCR_ACCESSOR` macro helper.
 */
#if NPY_FEATURE_VERSION >= NPY_2_0_API_VERSION || NPY_ABI_VERSION < 0x02000000
    /* Compiling for 1.x or 2.x only, direct field access is OK: */

    static inline void
    PyDataType_SET_ELSIZE(PyArray_Descr *dtype, npy_intp size)
    {
        _PyDataType_GET_ITEM_DATA(dtype)->elsize = size;
    }

    static inline npy_uint64
    PyDataType_FLAGS(const PyArray_Descr *dtype)
    {
    #if NPY_FEATURE_VERSION >= NPY_2_0_API_VERSION
        return _PyDataType_GET_ITEM_DATA(dtype)->flags;
    #else
        return (unsigned char)dtype->flags;  /* Need unsigned cast on 1.x */
    #endif
    }

    #define DESCR_ACCESSOR(FIELD, field, type, legacy_only)    \
        static inline type                                     \
        PyDataType_##FIELD(const PyArray_Descr *dtype) {       \
            if (legacy_only && !PyDataType_ISLEGACY(dtype)) {  \
                return (type)0;                                \
            }                                                  \
            return _PyArray_LegacyDescr_GET_ITEM_DATA((const _PyArray_LegacyDescr *)dtype)->field;     \
        }
#else  /* compiling for both 1.x and 2.x */

    static inline void
    PyDataType_SET_ELSIZE(PyArray_Descr *dtype, npy_intp size)
    {
        if (PyArray_RUNTIME_VERSION >= NPY_2_0_API_VERSION) {
            ((_PyArray_DescrNumPy2 *)dtype)->elsize = size;
        }
        else {
            ((PyArray_DescrProto *)dtype)->elsize = (int)size;
        }
    }

    static inline npy_uint64
    PyDataType_FLAGS(const PyArray_Descr *dtype)
    {
        if (PyArray_RUNTIME_VERSION >= NPY_2_0_API_VERSION) {
            return ((_PyArray_DescrNumPy2 *)dtype)->flags;
        }
        else {
            return (unsigned char)((PyArray_DescrProto *)dtype)->flags;
        }
    }

    /* Cast to LegacyDescr always fine but needed when `legacy_only` */
    #define DESCR_ACCESSOR(FIELD, field, type, legacy_only)        \
        static inline type                                         \
        PyDataType_##FIELD(const PyArray_Descr *dtype) {           \
            if (legacy_only && !PyDataType_ISLEGACY(dtype)) {      \
                return (type)0;                                    \
            }                                                      \
            if (PyArray_RUNTIME_VERSION >= NPY_2_0_API_VERSION) {  \
                return ((_PyArray_LegacyDescr *)dtype)->field;     \
            }                                                      \
            else {                                                 \
                return ((PyArray_DescrProto *)dtype)->field;       \
            }                                                      \
        }
#endif

DESCR_ACCESSOR(ELSIZE, elsize, npy_intp, 0)
DESCR_ACCESSOR(ALIGNMENT, alignment, npy_intp, 0)
DESCR_ACCESSOR(METADATA, metadata, PyObject *, 1)
DESCR_ACCESSOR(SUBARRAY, subarray, PyArray_ArrayDescr *, 1)
DESCR_ACCESSOR(NAMES, names, PyObject *, 1)
DESCR_ACCESSOR(FIELDS, fields, PyObject *, 1)
DESCR_ACCESSOR(C_METADATA, c_metadata, NpyAuxData *, 1)
/* ABI compatible in 1.x and 2.x, but defined together with others */
DESCR_ACCESSOR(TYPE, type, char, 0)
DESCR_ACCESSOR(KIND, kind, char, 0)
DESCR_ACCESSOR(BYTEORDER, byteorder, char, 0)
DESCR_ACCESSOR(TYPEOBJ, typeobj, PyTypeObject *, 0)

#undef DESCR_ACCESSOR


#if !(defined(NPY_INTERNAL_BUILD) && NPY_INTERNAL_BUILD)
#if NPY_FEATURE_VERSION >= NPY_2_0_API_VERSION
    static inline PyArray_ArrFuncs *
    PyDataType_GetArrFuncs(const PyArray_Descr *descr)
    {
        return _PyDataType_GetArrFuncs(descr);
    }
#elif NPY_ABI_VERSION < 0x02000000
    static inline PyArray_ArrFuncs *
    PyDataType_GetArrFuncs(const PyArray_Descr *descr)
    {
        return descr->f;
    }
#else
    static inline PyArray_ArrFuncs *
    PyDataType_GetArrFuncs(const PyArray_Descr *descr)
    {
        if (PyArray_RUNTIME_VERSION >= NPY_2_0_API_VERSION) {
            return _PyDataType_GetArrFuncs(descr);
        }
        else {
            return ((PyArray_DescrProto *)descr)->f;
        }
    }
#endif


/*
 * Backport of `NPY_DT_legacy_descriptor_proto` (and ABI fix for slot IDs).
 * This backport allows dtypes that are currently implemented as legacy
 * (i.e. have a kind, char, a character code, and only the byte-order parameter)
 * to work with only minor changes on NumPy 2.0+ but use any part of the new
 * DType API they want to.
 * This also will allow us to deprecate the weirder parts of it, i.e. cast
 * registration.
 * (Possibly the only remaining change may be poor `dtype=` printing in
 * arrays, which can be worked around.)
 */
/*
 * `NPY_2_4_API_VERSION` and `NPY_2_5_API_VERSION` may not be defined when
 * this header is vendored alongside an older `numpyconfig.h`.  Provide
 * fallback definitions so the rest of the backport can use named constants.
 */
#ifndef NPY_2_4_API_VERSION
#define NPY_2_4_API_VERSION 0x00000015
#endif
#ifndef NPY_2_5_API_VERSION
#define NPY_2_5_API_VERSION 0x00000016
#endif

#if NPY_TARGET_VERSION < NPY_2_5_API_VERSION \
        && NPY_TARGET_VERSION >= NPY_2_0_API_VERSION

#ifndef NPY_DT_legacy_descriptor_proto
#define NPY_DT_legacy_descriptor_proto ((1 << 11) - 1)
#endif

#define _PyArrayInitDTypeMeta_FromSpec \
    (*(int (*)(PyArray_DTypeMeta *, PyArrayDTypeMeta_Spec *))PyArray_API[362])
#undef PyArrayInitDTypeMeta_FromSpec

static inline int PyArrayInitDTypeMeta_FromSpec(
        PyArray_DTypeMeta *DType, PyArrayDTypeMeta_Spec *spec)
{
    PyArray_DescrProto *proto = NULL;
    if (spec->slots[0].slot == NPY_DT_legacy_descriptor_proto) {
        proto = (PyArray_DescrProto *)spec->slots[0].pfunc;
    }

#if NPY_TARGET_VERSION < NPY_2_4_API_VERSION
    /* < NumPy 2.4, slot IDs accidentally changed: translate them. */
    PyType_Slot *slot = spec->slots;
    int bad_offset = (PyArray_RUNTIME_VERSION >= NPY_2_4_API_VERSION)
            ? (1 << 10) : (1 << 11);
    int good_offset = (PyArray_RUNTIME_VERSION >= NPY_2_4_API_VERSION)
            ? (1 << 11) : (1 << 10);
    while (slot->slot != 0 || slot->pfunc != NULL) {
        if (slot->slot >= bad_offset && slot->slot < bad_offset + 30) {
            slot->slot += good_offset - bad_offset;
        }
        slot++;
    }
#endif

    if (proto == NULL || PyArray_RUNTIME_VERSION >= NPY_2_5_API_VERSION) {
        return _PyArrayInitDTypeMeta_FromSpec(DType, spec);
    }

#if defined(Py_LIMITED_API)
    PyErr_SetString(PyExc_RuntimeError,
        "NPY_DT_legacy_descriptor_proto backport not supported in Python limited API");
    return -1;
#else

    /*
     * Step 1: Register old-style with a garbage typeobj so that
     * _PyArray_MapPyTypeToDType does NOT add the auto-DTypeMeta to the
     * pytype-to-DType dict (it bails out on NPY_DT_is_legacy for non-generic
     * types), regardless of whether the real scalar subclasses np.generic.
     */
    PyArray_DescrProto new_proto = *proto;
    new_proto.typeobj = &PyBaseObject_Type;
    int typenum = PyArray_RegisterDataType(&new_proto);
    if (typenum < 0) {
        return -1;
    }

    /*
     * Step 2: Initialise the user's DType with new-style slots and casts.
     * type_num stays at -1 / 0 for now; we fix it in step 3.
     */
    PyArrayDTypeMeta_Spec new_spec = *spec;
    new_spec.slots = &spec->slots[1];  /* skip proto slot */
    if (_PyArrayInitDTypeMeta_FromSpec(DType, &new_spec) < 0) {
        return -1;
    }

    /*
     * Step 3: Steal the singleton descriptor and type_num from the legacy
     * registration.  Point the descriptor's Python type at the user's DType
     * and fix up its typeobj field (which we temporarily set to
     * PyBaseObject_Type in step 1).
     */
    PyArray_Descr *descr = PyArray_DescrFromType(typenum);
    if (descr == NULL) {
        return -1;
    }

    /* Save the auto-DTypeMeta so we can decref it after the swap. */
    PyObject *old_meta = (PyObject *)Py_TYPE(descr);

    DType->type_num = typenum;
    /* PyArray_DescrFromType returns a new reference; transfer ownership. */
    DType->singleton = descr;
    /*
     * Set the legacy flag (bit 0 == _NPY_DT_LEGACY_FLAG) so NumPy uses
     * legacy code paths (copyswap, ArrFuncs, etc.) where the new-style API
     * doesn't cover them yet.
     */
    DType->flags |= 1;

    /* Re-type the descriptor so it belongs to the user's DType class. */
    Py_INCREF(DType);
    Py_SET_TYPE(descr, (PyTypeObject *)(DType));
    Py_DECREF(old_meta);

    /*
     * Fix the descriptor's scalar-type field (it was set to
     * PyBaseObject_Type in step 1 by PyArray_RegisterDataType copying
     * proto->typeobj).
     */
    Py_INCREF(proto->typeobj);
    Py_XDECREF(descr->typeobj);
    descr->typeobj = proto->typeobj;

    /*
     * Initialize legacy ArrFuncs from the descriptor prototype.
     */
    if (proto->f != NULL) {
        PyArray_ArrFuncs *f = _PyDataType_GetArrFuncs(descr);
        /*
         * Preserve ArrFuncs that were explicitly set via the new API slots
         * (step 2), and fill missing ones from the legacy prototype.
         * getitem/setitem always come from the legacy descriptor path.
         */
         if (proto->f->getitem != NULL) {
            f->getitem = proto->f->getitem;
        }
        if (proto->f->setitem != NULL) {
            f->setitem = proto->f->setitem;
        }
#define NPY_PROTO_FILL_IF_NULL(FIELD) \
        if (f->FIELD == NULL) { \
            f->FIELD = proto->f->FIELD; \
        }
        NPY_PROTO_FILL_IF_NULL(copyswap);
        NPY_PROTO_FILL_IF_NULL(copyswapn);
        NPY_PROTO_FILL_IF_NULL(compare);
        NPY_PROTO_FILL_IF_NULL(argmax);
        NPY_PROTO_FILL_IF_NULL(dotfunc);
        NPY_PROTO_FILL_IF_NULL(scanfunc);
        NPY_PROTO_FILL_IF_NULL(fromstr);
        NPY_PROTO_FILL_IF_NULL(nonzero);
        NPY_PROTO_FILL_IF_NULL(fill);
        NPY_PROTO_FILL_IF_NULL(fillwithscalar);
        NPY_PROTO_FILL_IF_NULL(scalarkind);
        NPY_PROTO_FILL_IF_NULL(argmin);
#undef NPY_PROTO_FILL_IF_NULL
        for (int i = 0; i < NPY_NSORTS; i++) {
            f->sort[i] = proto->f->sort[i];
            f->argsort[i] = proto->f->argsort[i];
        }
    }
#endif  /* Py_LIMITED_API */
    return 0;
}
#endif


#endif  /* not internal build */

#endif  /* NUMPY_CORE_INCLUDE_NUMPY_NPY_2_COMPAT_H_ */
