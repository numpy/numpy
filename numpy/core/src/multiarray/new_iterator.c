
#define NPY_ITER_FLAGS_HASPERM  0x001
#define NPY_ITER_FLAGS_HASINDEX 0x002
#define NPY_ITER_FLAGS_HASCOORD 0x004

/* Size of all the data before the AXISDATA starts */
#define NIT_SIZEOF_BASEDATA(flags, ndim, niter) ( \
        /* uint32 flags AND uint16 ndim AND uint16 niter */ \
        8 + \
        /* PyArray_Descr* dtypes[niter */ \
        NPY_SIZEOF_INTP*niter)

/* Size of one AXISDATA struct within the iterator */
#define NIT_SIZEOF_AXISDATA(flags, ndim, niter) (( \
        /* intp shape */ \
        1 + \
        /* intp coord */ \
        1 + \
        /* intp indexstride AND intp index (when index is provided) */ \
        ((flags&NPY_ITER_FLAGS_HASINDEX) ? 2 : 0) + \
        /* intp stride[niter] AND char* ptr[niter] */ \
        2*niter + \
        )*NPY_SIZEOF_INTP)

/* Size of the whole iterator */
#define NIT_SIZEOF_ITERATOR(flags, ndim, niter) ( \
        NIT_SIZEOF_BASEDATA(flags, ndim, niter) + \
        NIT_SIZEOF_AXISDATA(flags, ndim, niter)*ndim + \
        NPY_SIZEOF_INTP*ndim + \
        NPY_SIZEOF_INTP*niter)

#define NIT_SIZEOF_BASEDATA_INSTANCE(iter) \
        NIT_SIZEOF_BASEDATA(NIT_FLAGS(iter), NIT_NDIM(iter), NIT_NITER(iter))
#define NIT_SIZEOF_AXISDATA_INSTANCE(iter) \
        NIT_SIZEOF_AXISDATA(NIT_FLAGS(iter), NIT_NDIM(iter), NIT_NITER(iter))
#define NIT_SIZEOF_ITERATOR_INSTANCE(iter) \
        NIT_SIZEOF_ITERATOR(NIT_FLAGS(iter), NIT_NDIM(iter), NIT_NITER(iter))

/* Internal-only iterator data member access. */
#define NIT_FLAGS(iter)    (*((npy_uint32*)iter))
#define NIT_NDIM(iter)     (*((npy_uint16*)iter + 2))
#define NIT_NITER(iter)    (*((npy_uint16*)iter + 3))
#define NIT_DTYPES(iter)   (*((PyArray_Descr**)((char*)iter + 8)))
#define NIT_AXISDATA(iter) ((char*)iter + NIT_SIZEOF_BASEDATA_INSTANCE(iter))
#define NIT_PERM(iter)  ((npy_intp*)( \
        (char*)iter + NIT_SIZEOF_BASEDATA_INSTANCE(iter) + \
        NIT_SIZEOF_AXISDATA_INSTANCE(iter)*NIT_NDIM(iter)))
#define NIT_DESTRUCTDATA(iter) ((PyObject*)( \
        (char*)iter + NIT_SIZEOF_BASEDATA_INSTANCE(iter) + \
        (NIT_SIZEOF_AXISDATA_INSTANCE(iter) + \
         ((NIT_FLAGS(iter)&NPY_ITER_FLAGS_HASPERM) ? NPY_SIZEOF_INTP : 0) * \
        )*NIT_NDIM(iter)))

void* NpyIter_New(PyObject* op, npy_uint32 flags, PyArray_Descr* dtype,
                  int min_depth, int max_depth)
{
}

int NpyIter_Deallocate(void* iter)
{
}

NpyIter_IterNext_Function NpyIter_GetIterNext(void *iter)
{
}

char **NpyIter_GetDataPtrArray(void *iter)
{
}
