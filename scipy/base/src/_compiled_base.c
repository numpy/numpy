#include "Python.h"
#include "scipy/arrayobject.h"

static PyObject *ErrorObject;
#define Py_Try(BOOLEAN) {if (!(BOOLEAN)) goto fail;}
#define Py_Assert(BOOLEAN,MESS) {if (!(BOOLEAN)) {			\
	    PyErr_SetString(ErrorObject, (MESS));			\
	    goto fail;}							\
    }

static intp 
incr_slot_ (double x, double *bins, intp lbins)
{
    intp i ;
    for ( i = 0 ; i < lbins ; i ++ )
	if ( x < bins [i] )
	    return i ;
    return lbins ;
}

static intp 
decr_slot_ (double x, double * bins, intp lbins)
{
    intp i ;
    for ( i = lbins - 1 ; i >= 0; i -- )
	if (x < bins [i])
	    return i + 1 ;
    return 0 ;
}

static int 
monotonic_ (double * a, int lena)
{
    int i;
    if (a [0] <= a [1]) /* possibly monotonic increasing */
	{
	    for (i = 1 ; i < lena - 1; i ++)
		if (a [i] > a [i + 1]) return 0 ;
	    return 1 ;
	}
    else              /* possibly monotonic decreasing */
	{
	    for (i = 1 ; i < lena - 1; i ++)
		if (a [i] < a [i + 1]) return 0 ;
	    return -1 ;
	}    
}


static char arr_bincount__doc__[] = "";

static intp
mxx (intp *i , intp len)
{
    /* find the index of the maximum element of an integer array */
    intp mx = 0, max = i[0] ;
    intp j ;
    for ( j = 1 ; j < len; j ++ )
	if ( i [j] > max )
	    {max = i [j] ;
	    mx = j ;}
    return mx;
}

static intp 
mnx (intp *i , intp len)
{
    /* find the index of the minimum element of an integer array */
    intp mn = 0, min = i [0] ;
    intp j ;
    for ( j = 1 ; j < len; j ++ )
	if ( i [j] < min )
	    {min = i [j] ;
	    mn = j ;}
    return mn;
}


static PyObject *
arr_bincount(PyObject *self, PyObject *args, PyObject *kwds)
{
     /* histogram accepts one or two arguments. The first is an array
      * of non-negative integers and the second, if present, is an
      * array of weights, which must be promotable to double.
      * Call these arguments list and weight. Both must be one-
      * dimensional. len (weight) == len(list)
      * If weight is not present:
      *   histogram (list) [i] is the number of occurrences of i in list.
      * If weight is present:
      *   histogram (list, weight) [i] is the sum of all weight [j]
      * where list [j] == i.                                              */
     /* self is not used */
    PyArray_Typecode type = {PyArray_INTP, 0, 0};
    PyObject *list = NULL, *weight=Py_None ;
    PyObject *lst=NULL, *ans=NULL, *wts=NULL;
    intp *numbers, *ians, len , mxi, mni, ans_size;
    int i;
    double *weights , *dans;
    static char *kwlist[] = {"list", "weights", NULL};

    Py_Try(PyArg_ParseTupleAndKeywords(args, kwds, "O|O", kwlist,
				       &list, &weight));
    Py_Try(lst = PyArray_ContiguousFromObject(list, PyArray_INTP, 1, 1));
    len = PyArray_SIZE(lst);
    numbers = (intp *) PyArray_DATA(lst);
    mxi = mxx (numbers, len) ;
    mni = mnx (numbers, len) ;
    Py_Assert(numbers[mni] >= 0, 
	      "First argument of bincount must be nonnegative.");
    ans_size = numbers [mxi] + 1 ;
    if (weight == Py_None) {
	Py_Try(ans = PyArray_Zeros(1, &ans_size, &type));
	ians = (intp *)(PyArray_DATA(ans));
	for (i = 0 ; i < len ; i++)
	    ians [numbers [i]] += 1 ;
	Py_DECREF(lst);
    }
    else {
	Py_Try(wts = PyArray_ContiguousFromObject(weight, 
						  PyArray_DOUBLE, 1, 1));
	weights = (double *)PyArray_DATA (wts);
	Py_Assert(PyArray_SIZE(wts) == len, "bincount: length of weights " \
		  "does not match that of list");
	type.type_num = PyArray_DOUBLE;
	Py_Try(ans = PyArray_Zeros(1, &ans_size, &type));
	dans = (double *)PyArray_DATA (ans);
	for (i = 0 ; i < len ; i++) {
	    dans[numbers[i]] += weights[i];
	}
	Py_DECREF(lst);
	Py_DECREF(wts);
    }
    return ans;

 fail:
    Py_XDECREF(lst);
    Py_XDECREF(wts);
    return NULL;
}

static char arr_digitize__doc__[] = "";

static PyObject *
arr_digitize(PyObject *self, PyObject *args, PyObject *kwds)
{
    /* digitize (x, bins) returns an array of python integers the same
       length of x. The values i returned are such that
       bins [i - 1] <= x < bins [i] if bins is monotonically increasing,
       or bins [i - 1] > x >= bins [i] if bins is monotonically decreasing.
       Beyond the bounds of bins, returns either i = 0 or i = len (bins)
       as appropriate.                                                      */
    /* self is not used */
    PyObject *ox, *obins ;
    PyObject *ax=NULL, *abins=NULL, *aret;
    double *dx, *dbins ;       
    intp lbins, lx ;             /* lengths */
    intp *iret;
    int m, i ;
    static char *kwlist[] = {"x", "bins", NULL};
    PyArray_Typecode type = {PyArray_DOUBLE, sizeof(double), 0};

    Py_Try(PyArg_ParseTupleAndKeywords(args, kwds, "OO", kwlist, 
				       &ox, &obins));

    Py_Try(ax=PyArray_FromAny(ox, &type, 1, 1, CARRAY_FLAGS));
    Py_Try(abins = PyArray_FromAny(obins, &type, 1, 1, CARRAY_FLAGS));

    lx = PyArray_SIZE(ax);
    dx = (double *)PyArray_DATA(ax);
    lbins = PyArray_SIZE(abins);
    dbins = (double *)PyArray_DATA(abins);
    Py_Try(aret = PyArray_FromDims(1, &lx, PyArray_INTP));
    iret = (intp *)PyArray_DATA(aret);

    Py_Assert(lx > 0 && lbins > 0, 
	      "x and bins both must have nonzero length.");
    
    if (lbins == 1)  {
	for (i=0 ; i<lx ; i++)
            if (dx [i] >= dbins[0])
                iret[i] = 1;
            else 
                iret[i] = 0;
    }
    else {
        m = monotonic_ (dbins, lbins) ;
	if ( m == -1 ) {
            for ( i = 0 ; i < lx ; i ++ )
		iret [i] = decr_slot_ (dx [i], dbins, lbins) ;
	}
        else if ( m == 1 ) {
            for ( i = 0 ; i < lx ; i ++ )
                iret [i] = incr_slot_ ((float)dx [i], dbins, lbins) ;
        }
        else Py_Assert(0, "bins must be montonic increasing or decreasing..");
    }

    Py_DECREF(ax);
    Py_DECREF(abins);
    return aret;

 fail:
    Py_XDECREF(ax);
    Py_XDECREF(abins);
    return NULL;
}



static char arr_insert__doc__[] = "Insert vals sequentially into equivalent 1-d positions indicated by mask.";

static PyObject *
arr_insert(PyObject *self, PyObject *args, PyObject *kwdict)
{
    /* Returns input array with values inserted sequentially into places 
       indicated by the mask
    */
    PyObject *mask=NULL, *vals=NULL;
    PyArrayObject *ainput=NULL, *amask=NULL, *avals=NULL, 
	*avalscast=NULL, *tmp=NULL;
    int numvals, totmask, sameshape;
    char *input_data, *mptr, *vptr, *zero=NULL;
    int melsize, delsize, copied, nd;
    int *instrides, *inshape;
    int mindx, rem_indx, indx, i, k, objarray;
  
    static char *kwlist[] = {"input","mask","vals",NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwdict, "O&OO", kwlist, 
				     PyArray_Converter, &ainput, 
				     &mask, &vals))
	return NULL;

    
    amask = (PyArrayObject *) PyArray_FROM_OF(mask, CARRAY_FLAGS);
    if (amask == NULL) return NULL;
    /* Cast an object array */
    if (amask->descr->type_num == PyArray_OBJECT) {
	tmp = (PyArrayObject *)PyArray_Cast(amask, PyArray_INTP);
	if (tmp == NULL) goto fail;
	Py_DECREF(amask);
	amask = tmp;
    }

    sameshape = 1;
    if (amask->nd == ainput->nd) {
	for (k=0; k < amask->nd; k++) 
	    if (amask->dimensions[k] != ainput->dimensions[k])
		sameshape = 0;
    }
    else { /* Test to see if amask is 1d */
	if (amask->nd != 1) sameshape = 0;
	else if ((PyArray_SIZE(ainput)) != PyArray_SIZE(amask)) sameshape = 0;
    }
    if (!sameshape) {
	PyErr_SetString(PyExc_ValueError, 
			"Mask array must be 1D or same shape as input array.");
	goto fail;
    }

    avals = (PyArrayObject *)PyArray_FromObject(vals, PyArray_NOTYPE, 0, 1);
    if (avals == NULL) goto fail;
    avalscast = (PyArrayObject *)PyArray_Cast(avals, ainput->descr->type_num);
    if (avalscast == NULL) goto fail;
    Py_DECREF(avals);

    numvals = PyArray_SIZE(avalscast);
    nd = ainput->nd;
    input_data = ainput->data;
    mptr = amask->data;
    melsize = amask->descr->elsize;
    vptr = avalscast->data;
    delsize = avalscast->descr->elsize;
    zero = PyArray_Zero(amask);
    objarray = (ainput->descr->type_num == PyArray_OBJECT);
  
    /* Handle zero-dimensional case separately */
    if (nd == 0) {
	if (memcmp(mptr,zero,melsize) != 0) {
	    /* Copy value element over to input array */
	    memcpy(input_data,vptr,delsize);
	    if (objarray) Py_INCREF(*((PyObject **)vptr));
	}
	Py_DECREF(amask);
	Py_DECREF(avalscast);
	Py_INCREF(Py_None);
	PyDataMem_FREE(zero);
	return Py_None;
    }

    /* Walk through mask array, when non-zero is encountered
       copy next value in the vals array to the input array.
       If we get through the value array, repeat it as necessary. 
    */
    totmask = PyArray_SIZE(amask);
    copied = 0;
    instrides = ainput->strides;
    inshape = ainput->dimensions;
    for (mindx = 0; mindx < totmask; mindx++) { 
	if (memcmp(mptr,zero,melsize) != 0) {      
	    /* compute indx into input array 
	     */
	    rem_indx = mindx;
	    indx = 0;
	    for(i=nd-1; i > 0; --i) {
		indx += (rem_indx % inshape[i]) * instrides[i];
		rem_indx /= inshape[i];
	    }
	    indx += rem_indx * instrides[0];
	    /* fprintf(stderr, "mindx = %d, indx=%d\n", mindx, indx); */
	    /* Copy value element over to input array */
	    memcpy(input_data+indx,vptr,delsize);
	    if (objarray) Py_INCREF(*((PyObject **)vptr));
	    vptr += delsize;
	    copied += 1;
	    /* If we move past value data.  Reset */
	    if (copied >= numvals) vptr = avalscast->data;
	}
	mptr += melsize;
    }

    Py_DECREF(amask);
    Py_DECREF(avalscast);
    Py_INCREF(Py_None);
    PyDataMem_FREE(zero);
    return Py_None;
  
 fail:
    PyDataMem_FREE(zero);
    Py_XDECREF(amask);
    Py_XDECREF(avals);
    Py_XDECREF(avalscast);
    return NULL;
}


static struct PyMethodDef methods[] = {
    {"_insert",	 (PyCFunction)arr_insert, METH_VARARGS | METH_KEYWORDS, 
     arr_insert__doc__},
    {"bincount", (PyCFunction)arr_bincount,  
     METH_VARARGS | METH_KEYWORDS, arr_bincount__doc__},
     {"digitize", (PyCFunction)arr_digitize, METH_VARARGS | METH_KEYWORDS,
     arr_digitize__doc__},
    {NULL, NULL}    /* sentinel */
};

/* Initialization function for the module (*must* be called initArray) */

DL_EXPORT(void) init_compiled_base(void) {
    PyObject *m, *d, *s;
  
    /* Create the module and add the functions */
    m = Py_InitModule("scipy.base._compiled_base", methods); 

    /* Import the array and ufunc objects */
    import_array();

    /* Add some symbolic constants to the module */
    d = PyModule_GetDict(m);

    s = PyString_FromString("0.4");
    PyDict_SetItemString(d, "__version__", s);
    Py_DECREF(s);

    ErrorObject = PyString_FromString("scipy.base._compiled_base.error");
    PyDict_SetItemString(d, "error", ErrorObject);

    /* Check for errors */
    if (PyErr_Occurred())
	    Py_FatalError("can't initialize module _compiled_base");
}
