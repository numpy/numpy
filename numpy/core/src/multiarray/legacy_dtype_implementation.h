#ifndef _NPY_LEGACY_DTYPE_IMPLEMENTATION_H
#define _NPY_LEGACY_DTYPE_IMPLEMENTATION_H

NPY_NO_EXPORT npy_bool
PyArray_LegacyCanCastTypeTo(PyArray_Descr *from, PyArray_Descr *to,
        NPY_CASTING casting);

#endif /*_NPY_LEGACY_DTYPE_IMPLEMENTATION_H*/
