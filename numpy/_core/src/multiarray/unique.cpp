#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <ctime>
#include <unordered_map>
#include <map>
#include <random>
#include <iostream>
#include <string>

#define _MULTIARRAYMODULE
#include "numpy/ndarraytypes.h"

#include "numpy/arrayobject.h"
#include "numpy/npy_common.h"

#include "numpy/npy_2_compat.h"


template <typename T>
T *random_data(std::size_t size, std::size_t max, T type)
{
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> rnd(0, max);
    T *res = new T[size];
    for (std::size_t i = 0; i < size; i++)
    {
        res[i] = rnd(rng);
    }
    return res;
}

void process_args(int argc, char *argv[], std::string &alg, std::size_t &size, std::size_t &max)
{
    if (argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << " {hash,rbt} <size> <max>" << std::endl;
        std::exit(1);
    }
    alg = argv[1];
    size = (std::size_t)std::stoi(argv[2]);
    max = (std::size_t)std::stoi(argv[3]);
}

template <typename ContainerType, typename DataType>
std::vector<DataType> _unique(ContainerType &container, DataType *data, std::size_t size)
{
    for (std::size_t i = 0; i < size; i++)
        container[data[i]] = 0;

    std::vector<DataType> res;
    res.reserve(container.size());
    for (auto it = container.begin(); it != container.end(); it++)
        res.emplace_back(it->first);

    return res;
}

template <typename T>
std::vector<T> unique(std::string &alg, T *data, std::size_t size)
{
    if (alg == "hash")
    {
        std::unordered_map<T, char> umap;
        return _unique(umap, data, size);
    }
    else if (alg == "rbt")
    {
        std::map<T, char> map;
        return _unique(map, data, size);
    }
    else
    {
        std::cerr << "Unknown algorithm: " << alg << std::endl;
        std::exit(1);
    }
}

NPY_NO_EXPORT npy_intp
PyArray_Unique(PyArrayObject *self)
{
    /* Nonzero boolean function */
    // PyArray_NonzeroFunc* nonzero = PyDataType_GetArrFuncs(PyArray_DESCR(self))->nonzero;

    NpyIter* iter;
    NpyIter_IterNextFunc *iternext;
    char** dataptr;
    npy_intp nonzero_count;
    npy_intp* strideptr,* innersizeptr;

    /* Handle zero-sized arrays specially */
    if (PyArray_SIZE(self) == 0) {
        return 0;
    }

    /*
     * Create and use an iterator to count the nonzeros.
     *   flag NPY_ITER_READONLY
     *     - The array is never written to.
     *   flag NPY_ITER_EXTERNAL_LOOP
     *     - Inner loop is done outside the iterator for efficiency.
     *   flag NPY_ITER_NPY_ITER_REFS_OK
     *     - Reference types are acceptable.
     *   order NPY_KEEPORDER
     *     - Visit elements in memory order, regardless of strides.
     *       This is good for performance when the specific order
     *       elements are visited is unimportant.
     *   casting NPY_NO_CASTING
     *     - No casting is required for this operation.
     */
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

    sum = 0;
    do {
        /* Get the inner loop data/stride/count values */
        char* data = *dataptr;
        npy_intp stride = *strideptr;
        npy_intp count = *innersizeptr;
        npy_intp size = PyArray_ITEMSIZE(self);
        /* This is a typical inner loop for NPY_ITER_EXTERNAL_LOOP */
        while (count--) {
            if (nonzero(data, self)) {
                ++nonzero_count;
            }
            data += stride;
        }

        /* Increment the iterator to the next inner loop */
    } while(iternext(iter));

    NpyIter_Deallocate(iter);

    return nonzero_count;
}

int main(int argc, char *argv[])
{
    std::size_t size, max;
    std::string alg;
    process_args(argc, argv, alg, size, max);
    double sample = 0;
    double *data = random_data(size, max, sample);
    const clock_t begin_time = clock();
    std::vector<double> unique_values = unique(alg, data, size);
    std::cout << float( clock () - begin_time ) /  CLOCKS_PER_SEC;
    delete data;
}
