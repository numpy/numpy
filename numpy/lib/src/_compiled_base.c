#include "Python.h"
#include "structmember.h"
#include "numpy/noprefix.h"
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
 * bincount accepts one or two arguments. The first is an array of
 * non-negative integers and the second, if present, is an array of weights,
 * which must be promotable to double.  Call these arguments list and
 * weight. Both must be one-dimensional with len(weight) == len(list). If
 * weight is not present then bincount(list)[i] is the number of occurrences
 * of i in list.  If weight is present then bincount(self,list, weight)[i]
 * is the sum of all weight[j] where list [j] == i.  Self is not used.
 */
static PyObject *
arr_bincount(PyObject *NPY_UNUSED(self), PyObject *args, PyObject *kwds)
{
    PyArray_Descr *type;
    PyObject *list = NULL, *weight=Py_None;
    PyObject *lst=NULL, *ans=NULL, *wts=NULL;
    intp *numbers, *ians, len , mxi, mni, ans_size;
    int i;
    double *weights , *dans;
    static char *kwlist[] = {"list", "weights", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O",
                kwlist, &list, &weight)) {
            goto fail;
    }
    if (!(lst = PyArray_ContiguousFromAny(list, PyArray_INTP, 1, 1))) {
            goto fail;
    }
    len = PyArray_SIZE(lst);
    numbers = (intp *) PyArray_DATA(lst);
    mxi = mxx(numbers, len);
    mni = mnx(numbers, len);
    if (numbers[mni] < 0) {
        PyErr_SetString(PyExc_ValueError,
                "The first argument of bincount must be non-negative");
        goto fail;
    }
    ans_size = numbers [mxi] + 1;
    type = PyArray_DescrFromType(PyArray_INTP);
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

/*
 * binary_search accepts three arguments: a numeric value and
 * a numeric array and its length. It assumes that the array is sorted in
 * increasing order. It returns the index of the array's
 * largest element which is <= the value. It will return -1 if
 * the value is less than the least element of the array.
 * self is not used
 */
static npy_intp
binary_search(double dval, double dlist [], npy_intp len)
{
    npy_intp bottom , top , middle, result;

    if (dval < dlist [0]) {
        result = -1;
    }
    else {
        bottom = 0;
        top = len - 1;
        while (bottom < top) {
            middle = (top + bottom) / 2;
            if (dlist [middle] < dval) {
                bottom = middle + 1;
            }
            else if (dlist [middle] > dval) {
                top = middle - 1;
            }
            else {
                return middle;
            }
        }
        if (dlist [bottom] > dval) {
            result = bottom - 1;
        }
        else {
            result = bottom;
        }
    }
    return result;
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

    slopes = (double *) PyDataMem_NEW((lenxp-1)*sizeof(double));
    for (i = 0; i < lenxp - 1; i++) {
        slopes[i] = (dy[i + 1] - dy[i])/(dx[i + 1] - dx[i]);
    }
    for (i = 0; i < lenx; i++) {
        indx = binary_search(dz[i], dx, lenxp);
        if (indx < 0) {
            dres[i] = lval;
        }
        else if (indx >= lenxp - 1) {
            dres[i] = rval;
        }
        else {
            dres[i] = slopes[indx]*(dz[i]-dx[indx]) + dy[indx];
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

    docstr = PyUnicode_AS_DATA(str);
#warning XXX -- is this correct at all!?
#else
    if (!PyArg_ParseTuple(args, "OO!", &obj, &PyString_Type, &str)) {
        return NULL;
    }

    docstr = PyString_AS_STRING(str);
#endif

#define _TESTDOC1(typebase) (obj->ob_type == &Py##typebase##_Type)
#define _TESTDOC2(typebase) (obj->ob_type == Py##typebase##_TypePtr)
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
            ubyte *optr, *iptr;
            out = PyArray_New(new->ob_type, 0, NULL, NPY_UBYTE,
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
    out = PyArray_New(new->ob_type, PyArray_NDIM(new), outdims, PyArray_UBYTE,
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

static struct PyMethodDef methods[] = {
    {"_insert", (PyCFunction)arr_insert,
        METH_VARARGS | METH_KEYWORDS, arr_insert__doc__},
    {"bincount", (PyCFunction)arr_bincount,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"digitize", (PyCFunction)arr_digitize,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"interp", (PyCFunction)arr_interp,
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
    PyGetSetDescr_TypePtr = myobj->ob_type;
    myobj = PyDict_GetItemString(tp_dict, "alignment");
    if (myobj == NULL) {
        return;
    }
    PyMemberDescr_TypePtr = myobj->ob_type;
    myobj = PyDict_GetItemString(tp_dict, "newbyteorder");
    if (myobj == NULL) {
        return;
    }
    PyMethodDescr_TypePtr = myobj->ob_type;
    return;
}

/* Initialization function for the module (*must* be called init<name>) */
PyMODINIT_FUNC init_compiled_base(void) {
    PyObject *m, *d, *s;

    /* Create the module and add the functions */
    m = Py_InitModule("_compiled_base", methods);

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

    return;
}
