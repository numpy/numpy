#ifndef _NPY_PRIVATE__NA_SINGLETON_H_
#define _NPY_PRIVATE__NA_SINGLETON_H_

/* Direct access to the fields of the NA object is just internal to NumPy. */
typedef struct {
    PyObject_HEAD
    /* NA payload, 0 by default */
    npy_uint8 payload;
    /* NA dtype, NULL by default */
    PyArray_Descr *dtype;
    /* Internal flag, whether this is the singleton numpy.NA or not */
    int is_singleton;
} NpyNA_fields;

NPY_NO_EXPORT NpyNA_fields _Npy_NASingleton;
#define Npy_NA ((PyObject *)&_Npy_NASingleton)

#define NPY_NA_NOPAYLOAD (255)

static NPY_INLINE npy_uint8
NpyNA_CombinePayloads(npy_uint p1, npy_uint p2)
{
    if (p1 == NPY_NA_NOPAYLOAD || p2 == NPY_NA_NOPAYLOAD) {
        return NPY_NA_NOPAYLOAD;
    }
    else {
        return (p1 < p2) ? p1 : p2;
    }
}

/* Combines two NA values together, merging their payloads and dtypes. */
NPY_NO_EXPORT NpyNA *
NpyNA_CombineNA(NpyNA *na1, NpyNA *na2);

/*
 * Combines an NA with an object, raising an error if the object has
 * no extractable NumPy dtype.
 */
NPY_NO_EXPORT NpyNA *
NpyNA_CombineNAWithObject(NpyNA *na, PyObject *obj);

/*
 * Converts an object into an NA if possible.
 *
 * If 'suppress_error' is enabled, doesn't raise an error when something
 * isn't NA.
 */
NPY_NO_EXPORT NpyNA *
NpyNA_FromObject(PyObject *obj, int suppress_error);

/*
 * Converts a dtype reference and mask value into an NA.
 * Doesn't steal the 'dtype' reference. Raises an error
 * if 'maskvalue' represents an exposed mask.
 */
NPY_NO_EXPORT NpyNA *
NpyNA_FromDTypeAndMaskValue(PyArray_Descr *dtype, npy_mask maskvalue,
                                                            int multina);

/*
 * Returns a mask value corresponding to the NA.
 */
NPY_NO_EXPORT npy_mask
NpyNA_AsMaskValue(NpyNA *na);

#endif
