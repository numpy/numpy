#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <unordered_map>
#include <vector>
#include <random>
#include <iostream>

#define _MULTIARRAYMODULE
#include "numpy/ndarraytypes.h"

#include "numpy/arrayobject.h"
#include "numpy/npy_common.h"

#include "numpy/npy_2_compat.h"

template<typename T>
npy_intp unique(PyArrayObject *self)
{
    NpyIter* iter;
    NpyIter_IterNextFunc *iternext;
    char** dataptr;
    npy_intp* strideptr,* innersizeptr;
    std::unordered_map<T, char> hashmap;

    iter = NpyIter_New(self, NPY_ITER_READONLY|
                             NPY_ITER_EXTERNAL_LOOP|
                             NPY_ITER_REFS_OK,
                        NPY_KEEPORDER, NPY_NO_CASTING,
                        NULL);
    if (iter == NULL) {
        return -1;
    }

    /*
     * The iternext function gets stored in a local variable
     * so it can be called repeatedly in an efficient manner.
     */
    iternext = NpyIter_GetIterNext(iter, NULL);
    if (iternext == NULL) {
        NpyIter_Deallocate(iter);
        return -1;
    }
    /* The location of the data pointer which the iterator may update */
    dataptr = NpyIter_GetDataPtrArray(iter);
    /* The location of the stride which the iterator may update */
    strideptr = NpyIter_GetInnerStrideArray(iter);
    /* The location of the inner loop size which the iterator may update */
    innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);

    std::cout << "printing values: " << std::endl;
    do {
        /* Get the inner loop data/stride/count values */
        char* data = *dataptr;
        npy_intp stride = *strideptr;
        npy_intp count = *innersizeptr;

        while (count--) {
            std::cout << (T)* data << std::endl;
            hashmap[(T)* data] = 0;
            data += stride;
        }

        /* Increment the iterator to the next inner loop */
    } while(iternext(iter));

    std::vector<T> res;
    std::cout << "unique values :" << std::endl;
    res.reserve(hashmap.size());
    for (auto it = hashmap.begin(); it != hashmap.end(); it++) {
        res.emplace_back(it->first);
        std::cout << it->first << std::endl;
    }

    NpyIter_Deallocate(iter);
    return 0;
}

NPY_NO_EXPORT npy_intp
PyArray_Unique(PyArrayObject *self)
{
    npy_intp itemsize;

    /* Handle zero-sized arrays specially */
    if (PyArray_SIZE(self) == 0) {
        return 0;
    }

    itemsize = PyArray_ITEMSIZE(self);
    std::cout << "Item size: " << itemsize << std::endl;

    if (sizeof(char) == itemsize) {
        unique<char>(self);
    } else if (sizeof(int) == itemsize) {
        unique<int>(self);
    }
    return 0;
}
