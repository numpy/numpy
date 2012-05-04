#include "Python.h"
#include "structmember.h"
#include "numpy/noprefix.h"
#include "numpy/npy_3kcompat.h"
#include "npy_config.h"

static intp
incr_slot_(double x, double *bins, intp lbins)
{
    intp i;

    for ( i = 0; i < lbins; i ++ ) {
        if ( x < bins [i] ) {
            return i;
        }
    }
    return lbins;
}

static intp
decr_slot_(double x, double * bins, intp lbins)
{
    intp i;

    for ( i = lbins - 1; i >= 0; i -- ) {
        if (x < bins [i]) {
            return i + 1;
        }
    }
    return 0;
}

static int
monotonic_(double * a, int lena)
{
    int i;

    if (a [0] <= a [1]) {
        /* possibly monotonic increasing */
        for (i = 1; i < lena - 1; i ++) {
            if (a [i] > a [i + 1]) {
                return 0;
            }
        }
        return 1;
    }
    else {
        /* possibly monotonic decreasing */
        for (i = 1; i < lena - 1; i ++) {
            if (a [i] < a [i + 1]) {
                return 0;
            }
        }
        return -1;
    }
}



/* find the index of the maximum element of an integer array */
static intp
mxx (intp *i , intp len)
{
    intp mx = 0, max = i[0];
    intp j;

    for ( j = 1; j < len; j ++ ) {
        if ( i [j] > max ) {
            max = i [j];
            mx = j;
        }
    }
    return mx;
}

/* find the index of the minimum element of an integer array */
static intp
mnx (intp *i , intp len)
{
    intp mn = 0, min = i [0];
    intp j;

    for ( j = 1; j < len; j ++ )
        if ( i [j] < min )
            {min = i [j];
                mn = j;}
    return mn;
}


/*
 * arr_bincount is registered as bincount.
 *
 * bincount accepts one, two or three arguments. The first is an array of
 * non-negative integers The second, if present, is an array of weights,
 * which must be promotable to double. Call these arguments list and
 * weight. Both must be one-dimensional with len(weight) == len(list). If
 * weight is not present then bincount(list)[i] is the number of occurrences
 * of i in list.  If weight is present then bincount(self,list, weight)[i]
 * is the sum of all weight[j] where list [j] == i.  Self is not used.
 * The third argument, if present, is a minimum length desired for the
 * output array.
 */
static PyObject *
arr_bincount(PyObject *NPY_UNUSED(self), PyObject *args, PyObject *kwds)
{
    PyArray_Descr *type;
    PyObject *list = NULL, *weight=Py_None, *mlength=Py_None;
    PyObject *lst=NULL, *ans=NULL, *wts=NULL;
    intp *numbers, *ians, len , mxi, mni, ans_size, minlength;
    int i;
    double *weights , *dans;
    static char *kwlist[] = {"list", "weights", "minlength", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OO",
                kwlist, &list, &weight, &mlength)) {
            goto fail;
    }

    lst = (PyArrayObject *)PyArray_ContiguousFromAny(list, NPY_INTP, 1, 1);
    if (lst == NULL) {
        goto fail;
    }
    len = PyArray_SIZE(lst);
    type = PyArray_DescrFromType(NPY_INTP);

    /* handle empty list */
    if (len < 1) {
        if (mlength == Py_None) {
            minlength = 0;
        }
        else if (!(minlength = PyArray_PyIntAsIntp(mlength))) {
            goto fail;
        }
        if (!(ans = PyArray_Zeros(1, &minlength, type, 0))){
            goto fail;
        }
        Py_DECREF(lst);
        return ans;
    }

    numbers = (npy_intp *) PyArray_DATA(lst);
    mxi = mxx(numbers, len);
    mni = mnx(numbers, len);
    if (numbers[mni] < 0) {
        PyErr_SetString(PyExc_ValueError,
                "The first argument of bincount must be non-negative");
        goto fail;
    }
    ans_size = numbers [mxi] + 1;
    if (mlength != Py_None) {
        if (!(minlength = PyArray_PyIntAsIntp(mlength))) {
            goto fail;
        }
        if (minlength <= 0) {
            /* superfluous, but may catch incorrect usage */
            PyErr_SetString(PyExc_ValueError,
                    "minlength must be positive");
            goto fail;
        }
        if (ans_size < minlength) {
            ans_size = minlength;
        }
    }
    if (weight == Py_None) {
        if (!(ans = PyArray_Zeros(1, &ans_size, type, 0))) {
            goto fail;
        }
        ians = (intp *)(PyArray_DATA(ans));
        for (i = 0; i < len; i++)
            ians [numbers [i]] += 1;
        Py_DECREF(lst);
    }
    else {
        if (!(wts = PyArray_ContiguousFromAny(weight, PyArray_DOUBLE, 1, 1))) {
            goto fail;
        }
        weights = (double *)PyArray_DATA (wts);
        if (PyArray_SIZE(wts) != len) {
            PyErr_SetString(PyExc_ValueError,
                    "The weights and list don't have the same length.");
            goto fail;
        }
        type = PyArray_DescrFromType(PyArray_DOUBLE);
        if (!(ans = PyArray_Zeros(1, &ans_size, type, 0))) {
            goto fail;
        }
        dans = (double *)PyArray_DATA (ans);
        for (i = 0; i < len; i++) {
            dans[numbers[i]] += weights[i];
        }
        Py_DECREF(lst);
        Py_DECREF(wts);
    }
    return ans;

fail:
    Py_XDECREF(lst);
    Py_XDECREF(wts);
    Py_XDECREF(ans);
    return NULL;
}


/*
 * digitize (x, bins) returns an array of python integers the same
 * length of x. The values i returned are such that bins [i - 1] <= x <
 * bins [i] if bins is monotonically increasing, or bins [i - 1] > x >=
 * bins [i] if bins is monotonically decreasing.  Beyond the bounds of
 * bins, returns either i = 0 or i = len (bins) as appropriate.
 */
static PyObject *
arr_digitize(PyObject *NPY_UNUSED(self), PyObject *args, PyObject *kwds)
{
    /* self is not used */
    PyObject *ox, *obins;
    PyObject *ax = NULL, *abins = NULL, *aret = NULL;
    double *dx, *dbins;
    intp lbins, lx;             /* lengths */
    intp *iret;
    int m, i;
    static char *kwlist[] = {"x", "bins", NULL};
    PyArray_Descr *type;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO", kwlist, &ox, &obins)) {
        goto fail;
    }
    type = PyArray_DescrFromType(PyArray_DOUBLE);
    if (!(ax = PyArray_FromAny(ox, type, 1, 1, CARRAY, NULL))) {
        goto fail;
    }
    Py_INCREF(type);
    if (!(abins = PyArray_FromAny(obins, type, 1, 1, CARRAY, NULL))) {
        goto fail;
    }

    lx = PyArray_SIZE(ax);
    dx = (double *)PyArray_DATA(ax);
    lbins = PyArray_SIZE(abins);
    dbins = (double *)PyArray_DATA(abins);
    if (!(aret = PyArray_SimpleNew(1, &lx, PyArray_INTP))) {
        goto fail;
    }
    iret = (intp *)PyArray_DATA(aret);

    if (lx <= 0 || lbins < 0) {
        PyErr_SetString(PyExc_ValueError,
                "Both x and bins must have non-zero length");
            goto fail;
    }

    if (lbins == 1)  {
        for (i = 0; i < lx; i++) {
            if (dx [i] >= dbins[0]) {
                iret[i] = 1;
            }
            else {
                iret[i] = 0;
            }
        }
    }
    else {
        m = monotonic_ (dbins, lbins);
        if ( m == -1 ) {
            for ( i = 0; i < lx; i ++ ) {
                iret [i] = decr_slot_ ((double)dx[i], dbins, lbins);
            }
        }
        else if ( m == 1 ) {
            for ( i = 0; i < lx; i ++ ) {
                iret [i] = incr_slot_ ((double)dx[i], dbins, lbins);
            }
        }
        else {
            PyErr_SetString(PyExc_ValueError,
                    "The bins must be montonically increasing or decreasing");
            goto fail;
        }
    }

    Py_DECREF(ax);
    Py_DECREF(abins);
    return aret;

fail:
    Py_XDECREF(ax);
    Py_XDECREF(abins);
    Py_XDECREF(aret);
    return NULL;
}



static char arr_insert__doc__[] = "Insert vals sequentially into equivalent 1-d positions indicated by mask.";

/*
 * Returns input array with values inserted sequentially into places
 * indicated by the mask
 */
static PyObject *
arr_insert(PyObject *NPY_UNUSED(self), PyObject *args, PyObject *kwdict)
{
    PyObject *mask = NULL, *vals = NULL;
    PyArrayObject *ainput = NULL, *amask = NULL, *avals = NULL, *tmp = NULL;
    int numvals, totmask, sameshape;
    char *input_data, *mptr, *vptr, *zero = NULL;
    int melsize, delsize, copied, nd;
    intp *instrides, *inshape;
    int mindx, rem_indx, indx, i, k, objarray;

    static char *kwlist[] = {"input", "mask", "vals", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwdict, "O&OO", kwlist,
                PyArray_Converter, &ainput,
                &mask, &vals)) {
        goto fail;
    }

    amask = (PyArrayObject *) PyArray_FROM_OF(mask, CARRAY);
    if (amask == NULL) {
        goto fail;
    }
    /* Cast an object array */
    if (amask->descr->type_num == PyArray_OBJECT) {
        tmp = (PyArrayObject *)PyArray_Cast(amask, PyArray_INTP);
        if (tmp == NULL) {
            goto fail;
        }
        Py_DECREF(amask);
        amask = tmp;
    }

    sameshape = 1;
    if (amask->nd == ainput->nd) {
        for (k = 0; k < amask->nd; k++) {
            if (amask->dimensions[k] != ainput->dimensions[k]) {
                sameshape = 0;
            }
        }
    }
    else {
        /* Test to see if amask is 1d */
        if (amask->nd != 1) {
            sameshape = 0;
        }
        else if ((PyArray_SIZE(ainput)) != PyArray_SIZE(amask)) {
            sameshape = 0;
        }
    }
    if (!sameshape) {
        PyErr_SetString(PyExc_TypeError,
                        "mask array must be 1-d or same shape as input array");
        goto fail;
    }

    avals = (PyArrayObject *)PyArray_FromObject(vals, ainput->descr->type_num, 0, 1);
    if (avals == NULL) {
        goto fail;
    }
    numvals = PyArray_SIZE(avals);
    nd = ainput->nd;
    input_data = ainput->data;
    mptr = amask->data;
    melsize = amask->descr->elsize;
    vptr = avals->data;
    delsize = avals->descr->elsize;
    zero = PyArray_Zero(amask);
    if (zero == NULL) {
        goto fail;
    }
    objarray = (ainput->descr->type_num == PyArray_OBJECT);

    /* Handle zero-dimensional case separately */
    if (nd == 0) {
        if (memcmp(mptr,zero,melsize) != 0) {
            /* Copy value element over to input array */
            memcpy(input_data,vptr,delsize);
            if (objarray) {
                Py_INCREF(*((PyObject **)vptr));
            }
        }
        Py_DECREF(amask);
        Py_DECREF(avals);
        PyDataMem_FREE(zero);
        Py_DECREF(ainput);
        Py_INCREF(Py_None);
        return Py_None;
    }

    /*
     * Walk through mask array, when non-zero is encountered
     * copy next value in the vals array to the input array.
     * If we get through the value array, repeat it as necessary.
     */
    totmask = (int) PyArray_SIZE(amask);
    copied = 0;
    instrides = ainput->strides;
    inshape = ainput->dimensions;
    for (mindx = 0; mindx < totmask; mindx++) {
        if (memcmp(mptr,zero,melsize) != 0) {
            /* compute indx into input array */
            rem_indx = mindx;
            indx = 0;
            for(i = nd - 1; i > 0; --i) {
                indx += (rem_indx % inshape[i]) * instrides[i];
                rem_indx /= inshape[i];
            }
            indx += rem_indx * instrides[0];
            /* fprintf(stderr, "mindx = %d, indx=%d\n", mindx, indx); */
            /* Copy value element over to input array */
            memcpy(input_data+indx,vptr,delsize);
            if (objarray) {
                Py_INCREF(*((PyObject **)vptr));
            }
            vptr += delsize;
            copied += 1;
            /* If we move past value data.  Reset */
            if (copied >= numvals) {
                vptr = avals->data;
            }
        }
        mptr += melsize;
    }

    Py_DECREF(amask);
    Py_DECREF(avals);
    PyDataMem_FREE(zero);
    Py_DECREF(ainput);
    Py_INCREF(Py_None);
    return Py_None;

fail:
    PyDataMem_FREE(zero);
    Py_XDECREF(ainput);
    Py_XDECREF(amask);
    Py_XDECREF(avals);
    return NULL;
}

/** @brief Use bisection on a sorted array to find first entry > key.
 *
 * Use bisection to find an index i s.t. arr[i] <= key < arr[i + 1]. If there is
 * no such i the error returns are:
 *     key < arr[0] -- -1
 *     key == arr[len - 1] -- len - 1
 *     key > arr[len - 1] -- len
 * The array is assumed contiguous and sorted in ascending order.
 *
 * @param key key value.
 * @param arr contiguous sorted array to be searched.
 * @param len length of the array.
 * @return index
 */
static npy_intp
binary_search(double key, double arr [], npy_intp len)
{
    npy_intp imin = 0;
    npy_intp imax = len;

    if (key > arr[len - 1]) {
        return len;
    }
    while (imin < imax) {
        npy_intp imid = imin + ((imax - imin) >> 1);
        if (key >= arr[imid]) {
            imin = imid + 1;
        }
        else {
            imax = imid;
        }
    }
    return imin - 1;
}

static PyObject *
arr_interp(PyObject *NPY_UNUSED(self), PyObject *args, PyObject *kwdict)
{

    PyObject *fp, *xp, *x;
    PyObject *left = NULL, *right = NULL;
    PyArrayObject *afp = NULL, *axp = NULL, *ax = NULL, *af = NULL;
    npy_intp i, lenx, lenxp, indx;
    double lval, rval;
    double *dy, *dx, *dz, *dres, *slopes;

    static char *kwlist[] = {"x", "xp", "fp", "left", "right", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwdict, "OOO|OO", kwlist,
                                     &x, &xp, &fp, &left, &right)) {
        return NULL;
    }

    afp = (NPY_AO*)PyArray_ContiguousFromAny(fp, NPY_DOUBLE, 1, 1);
    if (afp == NULL) {
        return NULL;
    }
    axp = (NPY_AO*)PyArray_ContiguousFromAny(xp, NPY_DOUBLE, 1, 1);
    if (axp == NULL) {
        goto fail;
    }
    ax = (NPY_AO*)PyArray_ContiguousFromAny(x, NPY_DOUBLE, 1, 0);
    if (ax == NULL) {
        goto fail;
    }
    lenxp = axp->dimensions[0];
    if (lenxp == 0) {
        PyErr_SetString(PyExc_ValueError,
                "array of sample points is empty");
        goto fail;
    }
    if (afp->dimensions[0] != lenxp) {
        PyErr_SetString(PyExc_ValueError,
                "fp and xp are not of the same length.");
        goto fail;
    }

    af = (NPY_AO*)PyArray_SimpleNew(ax->nd, ax->dimensions, NPY_DOUBLE);
    if (af == NULL) {
        goto fail;
    }
    lenx = PyArray_SIZE(ax);

    dy = (double *)PyArray_DATA(afp);
    dx = (double *)PyArray_DATA(axp);
    dz = (double *)PyArray_DATA(ax);
    dres = (double *)PyArray_DATA(af);

    /* Get left and right fill values. */
    if ((left == NULL) || (left == Py_None)) {
        lval = dy[0];
    }
    else {
        lval = PyFloat_AsDouble(left);
        if ((lval == -1) && PyErr_Occurred()) {
            goto fail;
        }
    }
    if ((right == NULL) || (right == Py_None)) {
        rval = dy[lenxp-1];
    }
    else {
        rval = PyFloat_AsDouble(right);
        if ((rval == -1) && PyErr_Occurred()) {
            goto fail;
        }
    }

    slopes = (double *) PyDataMem_NEW((lenxp - 1)*sizeof(double));
    for (i = 0; i < lenxp - 1; i++) {
        slopes[i] = (dy[i + 1] - dy[i])/(dx[i + 1] - dx[i]);
    }
    for (i = 0; i < lenx; i++) {
        indx = binary_search(dz[i], dx, lenxp);
        if (indx == -1) {
            dres[i] = lval;
        }
        else if (indx == lenxp - 1) {
            dres[i] = dy[indx];
        }
        else if (indx == lenxp) {
            dres[i] = rval;
        }
        else {
            dres[i] = slopes[indx]*(dz[i] - dx[indx]) + dy[indx];
        }
    }

    PyDataMem_FREE(slopes);
    Py_DECREF(afp);
    Py_DECREF(axp);
    Py_DECREF(ax);
    return (PyObject *)af;

fail:
    Py_XDECREF(afp);
    Py_XDECREF(axp);
    Py_XDECREF(ax);
    Py_XDECREF(af);
    return NULL;
}

/*
 * Converts a Python sequence into 'count' PyArrayObjects
 *
 * seq       - Input Python object, usually a tuple but any sequence works.
 * op        - Where the arrays are placed.
 * count     - How many arrays there should be (errors if it doesn't match).
 * paramname - The name of the parameter that produced 'seq'.
 */
static int sequence_to_arrays(PyObject *seq,
                                PyArrayObject **op, int count,
                                char *paramname)
{
    int i;

    if (!PySequence_Check(seq) || PySequence_Size(seq) != count) {
        PyErr_Format(PyExc_ValueError,
                "parameter %s must be a sequence of length %d",
                paramname, count);
        return -1;
    }

    for (i = 0; i < count; ++i) {
        PyObject *item = PySequence_GetItem(seq, i);
        if (item == NULL) {
            while (--i >= 0) {
                Py_DECREF(op[i]);
                op[i] = NULL;
            }
            return -1;
        }

        op[i] = (PyArrayObject *)PyArray_FromAny(item, NULL, 0, 0, 0, NULL);
        if (op[i] == NULL) {
            while (--i >= 0) {
                Py_DECREF(op[i]);
                op[i] = NULL;
            }
            Py_DECREF(item);
            return -1;
        }

        Py_DECREF(item);
    }

    return 0;
}

/* Inner loop for unravel_index */
static int
ravel_multi_index_loop(int ravel_ndim, npy_intp *ravel_dims,
                        npy_intp *ravel_strides,
                        npy_intp count,
                        NPY_CLIPMODE *modes,
                        char **coords, npy_intp *coords_strides)
{
    int i;
    npy_intp j, m;

    while (count--) {
        npy_intp raveled = 0;
        for (i = 0; i < ravel_ndim; ++i) {
            m = ravel_dims[i];
            j = *(npy_intp *)coords[i];
            switch (modes[i]) {
                case NPY_RAISE:
                    if(j < 0 || j>=m) {
                        PyErr_SetString(PyExc_ValueError,
                              "invalid entry in coordinates array");
                        return NPY_FAIL;
                    }
                    break;
                case NPY_WRAP:
                    if(j < 0) {
                        j += m;
                        if(j < 0) {
                            j = j%m;
                            if(j != 0) {
                                j += m;
                            }
                        }
                    }
                    else if(j >= m) {
                        j -= m;
                        if(j >= m) {
                            j = j%m;
                        }
                    }
                    break;
                case NPY_CLIP:
                    if(j < 0) {
                        j = 0;
                    }
                    else if(j >= m) {
                        j = m-1;
                    }
                    break;

            }
            raveled += j * ravel_strides[i];

            coords[i] += coords_strides[i];
        }
        *(npy_intp *)coords[ravel_ndim] = raveled;
        coords[ravel_ndim] += coords_strides[ravel_ndim];
    }

    return NPY_SUCCEED;
}

/* ravel_multi_index implementation - see add_newdocs.py */
static PyObject *
arr_ravel_multi_index(PyObject *self, PyObject *args, PyObject *kwds)
{
    int i, s;
    PyObject *mode0=NULL, *coords0=NULL;
    PyArrayObject *ret = NULL;
    PyArray_Dims dimensions={0,0};
    npy_intp ravel_strides[NPY_MAXDIMS];
    PyArray_ORDER order = NPY_CORDER;
    NPY_CLIPMODE modes[NPY_MAXDIMS];

    PyArrayObject *op[NPY_MAXARGS];
    PyArray_Descr *dtype[NPY_MAXARGS];
    npy_uint32 op_flags[NPY_MAXARGS];

    NpyIter *iter = NULL;

    char *kwlist[] = {"multi_index", "dims", "mode", "order", NULL};

    memset(op, 0, sizeof(op));
    dtype[0] = NULL;

    if(!PyArg_ParseTupleAndKeywords(args, kwds,
                        "OO&|OO&:ravel_multi_index", kwlist,
                     &coords0,
                     PyArray_IntpConverter, &dimensions,
                     &mode0,
                     PyArray_OrderConverter, &order)) {
        goto fail;
    }

    if (dimensions.len+1 > NPY_MAXARGS) {
        PyErr_SetString(PyExc_ValueError,
                    "too many dimensions passed to ravel_multi_index");
        goto fail;
    }

    if(!PyArray_ConvertClipmodeSequence(mode0, modes, dimensions.len)) {
       goto fail;
    }

    switch (order) {
        case NPY_CORDER:
            s = 1;
            for (i = dimensions.len-1; i >= 0; --i) {
                ravel_strides[i] = s;
                s *= dimensions.ptr[i];
            }
            break;
        case NPY_FORTRANORDER:
            s = 1;
            for (i = 0; i < dimensions.len; ++i) {
                ravel_strides[i] = s;
                s *= dimensions.ptr[i];
            }
            break;
        default:
            PyErr_SetString(PyExc_ValueError,
                            "only 'C' or 'F' order is permitted");
            goto fail;
    }

    /* Get the multi_index into op */
    if (sequence_to_arrays(coords0, op, dimensions.len, "multi_index") < 0) {
        goto fail;
    }


    for (i = 0; i < dimensions.len; ++i) {
        op_flags[i] = NPY_ITER_READONLY|
                      NPY_ITER_ALIGNED;
    }
    op_flags[dimensions.len] = NPY_ITER_WRITEONLY|
                               NPY_ITER_ALIGNED|
                               NPY_ITER_ALLOCATE;
    dtype[0] = PyArray_DescrFromType(NPY_INTP);
    for (i = 1; i <= dimensions.len; ++i) {
        dtype[i] = dtype[0];
    }

    iter = NpyIter_MultiNew(dimensions.len+1, op, NPY_ITER_BUFFERED|
                                                  NPY_ITER_EXTERNAL_LOOP|
                                                  NPY_ITER_ZEROSIZE_OK,
                                                  NPY_KEEPORDER,
                                                  NPY_SAME_KIND_CASTING,
                                                  op_flags, dtype);
    if (iter == NULL) {
        goto fail;
    }

    if (NpyIter_GetIterSize(iter) != 0) {
        NpyIter_IterNextFunc *iternext;
        char **dataptr;
        npy_intp *strides;
        npy_intp *countptr;

        iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            goto fail;
        }
        dataptr = NpyIter_GetDataPtrArray(iter);
        strides = NpyIter_GetInnerStrideArray(iter);
        countptr = NpyIter_GetInnerLoopSizePtr(iter);

        do {
            if (ravel_multi_index_loop(dimensions.len, dimensions.ptr,
                        ravel_strides, *countptr, modes,
                        dataptr, strides) != NPY_SUCCEED) {
                goto fail;
            }
        } while(iternext(iter));
    }

    ret = NpyIter_GetOperandArray(iter)[dimensions.len];
    Py_INCREF(ret);

    Py_DECREF(dtype[0]);
    for (i = 0; i < dimensions.len; ++i) {
        Py_XDECREF(op[i]);
    }
    PyDimMem_FREE(dimensions.ptr);
    NpyIter_Deallocate(iter);
    return PyArray_Return(ret);

fail:
    Py_XDECREF(dtype[0]);
    for (i = 0; i < dimensions.len; ++i) {
        Py_XDECREF(op[i]);
    }
    if (dimensions.ptr) {
        PyDimMem_FREE(dimensions.ptr);
    }
    if (iter != NULL) {
        NpyIter_Deallocate(iter);
    }
    return NULL;
}

/* C-order inner loop for unravel_index */
static int
unravel_index_loop_corder(int unravel_ndim, npy_intp *unravel_dims,
                        npy_intp unravel_size, npy_intp count,
                        char *indices, npy_intp indices_stride,
                        npy_intp *coords)
{
    int i;
    npy_intp val;

    while (count--) {
        val = *(npy_intp *)indices;
        if (val < 0 || val >= unravel_size) {
            PyErr_SetString(PyExc_ValueError,
                  "invalid entry in index array");
            return NPY_FAIL;
        }
        for (i = unravel_ndim-1; i >= 0; --i) {
            coords[i] = val % unravel_dims[i];
            val /= unravel_dims[i];
        }
        coords += unravel_ndim;
        indices += indices_stride;
    }

    return NPY_SUCCEED;
}

/* Fortran-order inner loop for unravel_index */
static int
unravel_index_loop_forder(int unravel_ndim, npy_intp *unravel_dims,
                        npy_intp unravel_size, npy_intp count,
                        char *indices, npy_intp indices_stride,
                        npy_intp *coords)
{
    int i;
    npy_intp val;

    while (count--) {
        val = *(npy_intp *)indices;
        if (val < 0 || val >= unravel_size) {
            PyErr_SetString(PyExc_ValueError,
                  "invalid entry in index array");
            return NPY_FAIL;
        }
        for (i = 0; i < unravel_ndim; ++i) {
            *coords++ = val % unravel_dims[i];
            val /= unravel_dims[i];
        }
        indices += indices_stride;
    }

    return NPY_SUCCEED;
}

/* unravel_index implementation - see add_newdocs.py */
static PyObject *
arr_unravel_index(PyObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *indices0 = NULL, *ret_tuple = NULL;
    PyArrayObject *ret_arr = NULL;
    PyArrayObject *indices = NULL;
    PyArray_Descr *dtype = NULL;
    PyArray_Dims dimensions={0,0};
    PyArray_ORDER order = PyArray_CORDER;
    npy_intp unravel_size;

    NpyIter *iter = NULL;
    int i, ret_ndim;
    npy_intp ret_dims[NPY_MAXDIMS], ret_strides[NPY_MAXDIMS];

    char *kwlist[] = {"indices", "dims", "order", NULL};

    if(!PyArg_ParseTupleAndKeywords(args, kwds, "OO&|O&:unravel_index",
                    kwlist,
                    &indices0,
                    PyArray_IntpConverter, &dimensions,
                    PyArray_OrderConverter, &order)) {
        goto fail;
    }

    if (dimensions.len == 0) {
        PyErr_SetString(PyExc_ValueError,
                "dims must have at least one value");
        goto fail;
    }

    unravel_size = PyArray_MultiplyList(dimensions.ptr, dimensions.len);

    if(!PyArray_Check(indices0)) {
        indices = (PyArrayObject*)PyArray_FromAny(indices0,
                                                    NULL, 0, 0, 0, NULL);
        if(indices == NULL) {
            goto fail;
        }
    }
    else {
        indices = (PyArrayObject *)indices0;
        Py_INCREF(indices);
    }

    dtype = PyArray_DescrFromType(NPY_INTP);
    if (dtype == NULL) {
        goto fail;
    }

    iter = NpyIter_New(indices, NPY_ITER_READONLY|
                                NPY_ITER_ALIGNED|
                                NPY_ITER_BUFFERED|
                                NPY_ITER_ZEROSIZE_OK|
                                NPY_ITER_DONT_NEGATE_STRIDES|
                                NPY_ITER_MULTI_INDEX,
                                NPY_KEEPORDER, NPY_SAME_KIND_CASTING,
                                dtype);
    if (iter == NULL) {
        goto fail;
    }

    /*
     * Create the return array with a layout compatible with the indices
     * and with a dimension added to the end for the multi-index
     */
    ret_ndim = PyArray_NDIM(indices) + 1;
    if (NpyIter_GetShape(iter, ret_dims) != NPY_SUCCEED) {
        goto fail;
    }
    ret_dims[ret_ndim-1] = dimensions.len;
    if (NpyIter_CreateCompatibleStrides(iter,
                dimensions.len*sizeof(npy_intp), ret_strides) != NPY_SUCCEED) {
        goto fail;
    }
    ret_strides[ret_ndim-1] = sizeof(npy_intp);

    /* Remove the multi-index and inner loop */
    if (NpyIter_RemoveMultiIndex(iter) != NPY_SUCCEED) {
        goto fail;
    }
    if (NpyIter_EnableExternalLoop(iter) != NPY_SUCCEED) {
        goto fail;
    }

    ret_arr = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type, dtype,
                            ret_ndim, ret_dims, ret_strides, NULL, 0, NULL);
    dtype = NULL;
    if (ret_arr == NULL) {
        goto fail;
    }

    if (order == NPY_CORDER) {
        if (NpyIter_GetIterSize(iter) != 0) {
            NpyIter_IterNextFunc *iternext;
            char **dataptr;
            npy_intp *strides;
            npy_intp *countptr, count;
            npy_intp *coordsptr = (npy_intp *)PyArray_DATA(ret_arr);

            iternext = NpyIter_GetIterNext(iter, NULL);
            if (iternext == NULL) {
                goto fail;
            }
            dataptr = NpyIter_GetDataPtrArray(iter);
            strides = NpyIter_GetInnerStrideArray(iter);
            countptr = NpyIter_GetInnerLoopSizePtr(iter);

            do {
                count = *countptr;
                if (unravel_index_loop_corder(dimensions.len, dimensions.ptr,
                            unravel_size, count, *dataptr, *strides,
                            coordsptr) != NPY_SUCCEED) {
                    goto fail;
                }
                coordsptr += count*dimensions.len;
            } while(iternext(iter));
        }
    }
    else if (order == NPY_FORTRANORDER) {
        if (NpyIter_GetIterSize(iter) != 0) {
            NpyIter_IterNextFunc *iternext;
            char **dataptr;
            npy_intp *strides;
            npy_intp *countptr, count;
            npy_intp *coordsptr = (npy_intp *)PyArray_DATA(ret_arr);

            iternext = NpyIter_GetIterNext(iter, NULL);
            if (iternext == NULL) {
                goto fail;
            }
            dataptr = NpyIter_GetDataPtrArray(iter);
            strides = NpyIter_GetInnerStrideArray(iter);
            countptr = NpyIter_GetInnerLoopSizePtr(iter);

            do {
                count = *countptr;
                if (unravel_index_loop_forder(dimensions.len, dimensions.ptr,
                            unravel_size, count, *dataptr, *strides,
                            coordsptr) != NPY_SUCCEED) {
                    goto fail;
                }
                coordsptr += count*dimensions.len;
            } while(iternext(iter));
        }
    }
    else {
        PyErr_SetString(PyExc_ValueError,
                        "only 'C' or 'F' order is permitted");
        goto fail;
    }

    /* Now make a tuple of views, one per index */
    ret_tuple = PyTuple_New(dimensions.len);
    if (ret_tuple == NULL) {
        goto fail;
    }
    for (i = 0; i < dimensions.len; ++i) {
        PyArrayObject *view;

        view = (PyArrayObject *)PyArray_New(&PyArray_Type, ret_ndim-1,
                                ret_dims, NPY_INTP,
                                ret_strides,
                                PyArray_BYTES(ret_arr) + i*sizeof(npy_intp),
                                0, 0, NULL);
        if (view == NULL) {
            goto fail;
        }
        Py_INCREF(ret_arr);
        view->base = (PyObject *)ret_arr;
        PyTuple_SET_ITEM(ret_tuple, i, PyArray_Return(view));
    }

    Py_DECREF(ret_arr);
    Py_XDECREF(indices);
    PyDimMem_FREE(dimensions.ptr);
    NpyIter_Deallocate(iter);

    return ret_tuple;

fail:
    Py_XDECREF(ret_tuple);
    Py_XDECREF(ret_arr);
    Py_XDECREF(dtype);
    Py_XDECREF(indices);
    if (dimensions.ptr) {
        PyDimMem_FREE(dimensions.ptr);
    }
    if (iter != NULL) {
        NpyIter_Deallocate(iter);
    }
    return NULL;
}


static PyTypeObject *PyMemberDescr_TypePtr = NULL;
static PyTypeObject *PyGetSetDescr_TypePtr = NULL;
static PyTypeObject *PyMethodDescr_TypePtr = NULL;

/* Can only be called if doc is currently NULL */
static PyObject *
arr_add_docstring(PyObject *NPY_UNUSED(dummy), PyObject *args)
{
    PyObject *obj;
    PyObject *str;
    char *docstr;
    static char *msg = "already has a docstring";

    /* Don't add docstrings */
    if (Py_OptimizeFlag > 1) {
        Py_INCREF(Py_None);
        return Py_None;
    }
#if defined(NPY_PY3K)
    if (!PyArg_ParseTuple(args, "OO!", &obj, &PyUnicode_Type, &str)) {
        return NULL;
    }

    docstr = PyBytes_AS_STRING(PyUnicode_AsUTF8String(str));
#else
    if (!PyArg_ParseTuple(args, "OO!", &obj, &PyString_Type, &str)) {
        return NULL;
    }

    docstr = PyString_AS_STRING(str);
#endif

#define _TESTDOC1(typebase) (Py_TYPE(obj) == &Py##typebase##_Type)
#define _TESTDOC2(typebase) (Py_TYPE(obj) == Py##typebase##_TypePtr)
#define _ADDDOC(typebase, doc, name) do {                               \
        Py##typebase##Object *new = (Py##typebase##Object *)obj;        \
        if (!(doc)) {                                                   \
            doc = docstr;                                               \
        }                                                               \
        else {                                                          \
            PyErr_Format(PyExc_RuntimeError, "%s method %s", name, msg); \
            return NULL;                                                \
        }                                                               \
    } while (0)

    if (_TESTDOC1(CFunction)) {
        _ADDDOC(CFunction, new->m_ml->ml_doc, new->m_ml->ml_name);
    }
    else if (_TESTDOC1(Type)) {
        _ADDDOC(Type, new->tp_doc, new->tp_name);
    }
    else if (_TESTDOC2(MemberDescr)) {
        _ADDDOC(MemberDescr, new->d_member->doc, new->d_member->name);
    }
    else if (_TESTDOC2(GetSetDescr)) {
        _ADDDOC(GetSetDescr, new->d_getset->doc, new->d_getset->name);
    }
    else if (_TESTDOC2(MethodDescr)) {
        _ADDDOC(MethodDescr, new->d_method->ml_doc, new->d_method->ml_name);
    }
    else {
        PyObject *doc_attr;

        doc_attr = PyObject_GetAttrString(obj, "__doc__");
        if (doc_attr != NULL && doc_attr != Py_None) {
            PyErr_Format(PyExc_RuntimeError, "object %s", msg);
            return NULL;
        }
        Py_XDECREF(doc_attr);

        if (PyObject_SetAttrString(obj, "__doc__", str) < 0) {
            PyErr_SetString(PyExc_TypeError,
                            "Cannot set a docstring for that object");
            return NULL;
        }
        Py_INCREF(Py_None);
        return Py_None;
    }

#undef _TESTDOC1
#undef _TESTDOC2
#undef _ADDDOC

    Py_INCREF(str);
    Py_INCREF(Py_None);
    return Py_None;
}


/*  PACKBITS
 *
 *  This function packs binary (0 or 1) 1-bit per pixel arrays
 *  into contiguous bytes.
 *
 */

static void
_packbits( void *In,
           int element_size,  /* in bytes */
           npy_intp in_N,
           npy_intp in_stride,
           void *Out,
           npy_intp out_N,
           npy_intp out_stride
)
{
    char build;
    int i, index;
    npy_intp out_Nm1;
    int maxi, remain, nonzero, j;
    char *outptr,*inptr;

    outptr = Out;    /* pointer to output buffer */
    inptr  = In;     /* pointer to input buffer */

    /*
     * Loop through the elements of In
     * Determine whether or not it is nonzero.
     *  Yes: set correspdoning bit (and adjust build value)
     *  No:  move on
     * Every 8th value, set the value of build and increment the outptr
     */

    remain = in_N % 8;                      /* uneven bits */
    if (remain == 0) {
        remain = 8;
    }
    out_Nm1 = out_N - 1;
    for (index = 0; index < out_N; index++) {
        build = 0;
        maxi = (index != out_Nm1 ? 8 : remain);
        for (i = 0; i < maxi; i++) {
            build <<= 1;
            nonzero = 0;
            for (j = 0; j < element_size; j++) {
                nonzero += (*(inptr++) != 0);
            }
            inptr += (in_stride - element_size);
            build += (nonzero != 0);
        }
        if (index == out_Nm1) build <<= (8-remain);
        /* printf("Here: %d %d %d %d\n",build,slice,index,maxi); */
        *outptr = build;
        outptr += out_stride;
    }
    return;
}


static void
_unpackbits(void *In,
        int NPY_UNUSED(el_size),  /* unused */
        npy_intp in_N,
        npy_intp in_stride,
        void *Out,
        npy_intp NPY_UNUSED(out_N),
        npy_intp out_stride
        )
{
    unsigned char mask;
    int i, index;
    char *inptr, *outptr;

    outptr = Out;
    inptr  = In;
    for (index = 0; index < in_N; index++) {
        mask = 128;
        for (i = 0; i < 8; i++) {
            *outptr = ((mask & (unsigned char)(*inptr)) != 0);
            outptr += out_stride;
            mask >>= 1;
        }
        inptr += in_stride;
    }
    return;
}

/* Fixme -- pack and unpack should be separate routines */
static PyObject *
pack_or_unpack_bits(PyObject *input, int axis, int unpack)
{
    PyArrayObject *inp;
    PyObject *new = NULL;
    PyObject *out = NULL;
    npy_intp outdims[MAX_DIMS];
    int i;
    void (*thefunc)(void *, int, npy_intp, npy_intp, void *, npy_intp, npy_intp);
    PyArrayIterObject *it, *ot;

    inp = (PyArrayObject *)PyArray_FROM_O(input);

    if (inp == NULL) {
        return NULL;
    }
    if (unpack) {
        if (PyArray_TYPE(inp) != NPY_UBYTE) {
            PyErr_SetString(PyExc_TypeError,
                    "Expected an input array of unsigned byte data type");
            goto fail;
        }
    }
    else if (!PyArray_ISINTEGER(inp)) {
        PyErr_SetString(PyExc_TypeError,
                "Expected an input array of integer data type");
        goto fail;
    }

    new = PyArray_CheckAxis(inp, &axis, 0);
    Py_DECREF(inp);
    if (new == NULL) {
        return NULL;
    }
    /* Handle zero-dim array separately */
    if (PyArray_SIZE(new) == 0) {
        return PyArray_Copy((PyArrayObject *)new);
    }

    if (PyArray_NDIM(new) == 0) {
        if (unpack) {
            /* Handle 0-d array by converting it to a 1-d array */
            PyObject *temp;
            PyArray_Dims newdim = {NULL, 1};
            npy_intp shape = 1;

            newdim.ptr = &shape;
            temp = PyArray_Newshape((PyArrayObject *)new, &newdim, NPY_CORDER);
            if (temp == NULL) {
                goto fail;
            }
            Py_DECREF(new);
            new = temp;
        }
        else {
            char *optr, *iptr;
            out = (PyArrayObject *)PyArray_New(Py_TYPE(new), 0, NULL, NPY_UBYTE,
                    NULL, NULL, 0, 0, NULL);
            if (out == NULL) {
                goto fail;
            }
            optr = PyArray_DATA(out);
            iptr = PyArray_DATA(new);
            *optr = 0;
            for (i = 0; i<PyArray_ITEMSIZE(new); i++) {
                if (*iptr != 0) {
                    *optr = 1;
                    break;
                }
                iptr++;
            }
            goto finish;
        }
    }


    /* Setup output shape */
    for (i=0; i<PyArray_NDIM(new); i++) {
        outdims[i] = PyArray_DIM(new, i);
    }

    if (unpack) {
        /* Multiply axis dimension by 8 */
        outdims[axis] <<= 3;
        thefunc = _unpackbits;
    }
    else {
        /*
         * Divide axis dimension by 8
         * 8 -> 1, 9 -> 2, 16 -> 2, 17 -> 3 etc..
         */
        outdims[axis] = ((outdims[axis] - 1) >> 3) + 1;
        thefunc = _packbits;
    }

    /* Create output array */
    out = (PyArrayObject *)PyArray_New(Py_TYPE(new),
                        PyArray_NDIM(new), outdims, NPY_UBYTE,
                        NULL, NULL, 0, PyArray_ISFORTRAN(new), NULL);
    if (out == NULL) {
        goto fail;
    }
    /* Setup iterators to iterate over all but given axis */
    it = (PyArrayIterObject *)PyArray_IterAllButAxis((PyObject *)new, &axis);
    ot = (PyArrayIterObject *)PyArray_IterAllButAxis((PyObject *)out, &axis);
    if (it == NULL || ot == NULL) {
        Py_XDECREF(it);
        Py_XDECREF(ot);
        goto fail;
    }

    while(PyArray_ITER_NOTDONE(it)) {
        thefunc(PyArray_ITER_DATA(it), PyArray_ITEMSIZE(new),
                PyArray_DIM(new, axis), PyArray_STRIDE(new, axis),
                PyArray_ITER_DATA(ot), PyArray_DIM(out, axis),
                PyArray_STRIDE(out, axis));
        PyArray_ITER_NEXT(it);
        PyArray_ITER_NEXT(ot);
    }
    Py_DECREF(it);
    Py_DECREF(ot);

finish:
    Py_DECREF(new);
    return out;

fail:
    Py_XDECREF(new);
    Py_XDECREF(out);
    return NULL;
}


static PyObject *
io_pack(PyObject *NPY_UNUSED(self), PyObject *args, PyObject *kwds)
{
    PyObject *obj;
    int axis = NPY_MAXDIMS;
    static char *kwlist[] = {"in", "axis", NULL};

    if (!PyArg_ParseTupleAndKeywords( args, kwds, "O|O&" , kwlist,
                &obj, PyArray_AxisConverter, &axis)) {
        return NULL;
    }
    return pack_or_unpack_bits(obj, axis, 0);
}

static PyObject *
io_unpack(PyObject *NPY_UNUSED(self), PyObject *args, PyObject *kwds)
{
    PyObject *obj;
    int axis = NPY_MAXDIMS;
    static char *kwlist[] = {"in", "axis", NULL};

    if (!PyArg_ParseTupleAndKeywords( args, kwds, "O|O&" , kwlist,
                &obj, PyArray_AxisConverter, &axis)) {
        return NULL;
    }
    return pack_or_unpack_bits(obj, axis, 1);
}

/* The docstrings for many of these methods are in add_newdocs.py. */
static struct PyMethodDef methods[] = {
    {"_insert", (PyCFunction)arr_insert,
        METH_VARARGS | METH_KEYWORDS, arr_insert__doc__},
    {"bincount", (PyCFunction)arr_bincount,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"digitize", (PyCFunction)arr_digitize,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"interp", (PyCFunction)arr_interp,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"ravel_multi_index", (PyCFunction)arr_ravel_multi_index,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"unravel_index", (PyCFunction)arr_unravel_index,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"add_docstring", (PyCFunction)arr_add_docstring,
        METH_VARARGS, NULL},
    {"packbits", (PyCFunction)io_pack,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"unpackbits", (PyCFunction)io_unpack,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {NULL, NULL, 0, NULL}    /* sentinel */
};

static void
define_types(void)
{
    PyObject *tp_dict;
    PyObject *myobj;

    tp_dict = PyArrayDescr_Type.tp_dict;
    /* Get "subdescr" */
    myobj = PyDict_GetItemString(tp_dict, "fields");
    if (myobj == NULL) {
        return;
    }
    PyGetSetDescr_TypePtr = Py_TYPE(myobj);
    myobj = PyDict_GetItemString(tp_dict, "alignment");
    if (myobj == NULL) {
        return;
    }
    PyMemberDescr_TypePtr = Py_TYPE(myobj);
    myobj = PyDict_GetItemString(tp_dict, "newbyteorder");
    if (myobj == NULL) {
        return;
    }
    PyMethodDescr_TypePtr = Py_TYPE(myobj);
    return;
}

#if defined(NPY_PY3K)
static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_compiled_base",
        NULL,
        -1,
        methods,
        NULL,
        NULL,
        NULL,
        NULL
};
#endif

#if defined(NPY_PY3K)
#define RETVAL m
PyObject *PyInit__compiled_base(void)
#else
#define RETVAL
PyMODINIT_FUNC
init_compiled_base(void)
#endif
{
    PyObject *m, *d;

#if defined(NPY_PY3K)
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule("_compiled_base", methods);
#endif
    if (!m) {
        return RETVAL;
    }

    /* Import the array objects */
    import_array();

    /* Add some symbolic constants to the module */
    d = PyModule_GetDict(m);

    /*
     * PyExc_Exception should catch all the standard errors that are
     * now raised instead of the string exception "numpy.lib.error".
     * This is for backward compatibility with existing code.
     */
    PyDict_SetItemString(d, "error", PyExc_Exception);


    /* define PyGetSetDescr_Type and PyMemberDescr_Type */
    define_types();

    return RETVAL;
}
