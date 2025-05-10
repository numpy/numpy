#ifndef NUMPY_CORE_SRC_MULTIARRAY_ARRAY_CONVERTER_H_
#define NUMPY_CORE_SRC_MULTIARRAY_ARRAY_CONVERTER_H_


#include "numpy/ndarraytypes.h"
extern NPY_NO_EXPORT PyTypeObject PyArrayArrayConverter_Type;

typedef enum {
    NPY_CH_ALL_SCALARS = 1 << 0,
    NPY_CH_ALL_PYSCALARS = 1 << 1,
} npy_array_converter_flags;


typedef struct {
    PyObject *object;
    PyArrayObject *array;
    PyArray_DTypeMeta *DType;
    PyArray_Descr *descr;
    int scalar_input;
} creation_item;


typedef struct {
    PyObject_VAR_HEAD
    int narrs;
    /* store if all objects are scalars (unless zero objects) */
    npy_array_converter_flags flags;
    /* __array_wrap__ cache */
    PyObject *wrap;
    PyObject *wrap_type;
    creation_item items[];
}  PyArrayArrayConverterObject;


#endif  /* NUMPY_CORE_SRC_MULTIARRAY_ARRAY_CONVERTER_H_ */
