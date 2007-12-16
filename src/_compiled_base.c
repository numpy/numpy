#include "Python.h"
#include "structmember.h"
#include "numpy/noprefix.h"

static PyObject *ErrorObject;
#define Py_Try(BOOLEAN) {if (!(BOOLEAN)) goto fail;}
#define Py_Assert(BOOLEAN,MESS) {if (!(BOOLEAN)) {      \
            PyErr_SetString(ErrorObject, (MESS));       \
            goto fail;}                                 \
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
    PyArray_Descr *type;
    PyObject *list = NULL, *weight=Py_None ;
    PyObject *lst=NULL, *ans=NULL, *wts=NULL;
    intp *numbers, *ians, len , mxi, mni, ans_size;
    int i;
    double *weights , *dans;
    static char *kwlist[] = {"list", "weights", NULL};


    Py_Try(PyArg_ParseTupleAndKeywords(args, kwds, "O|O", kwlist,
                                       &list, &weight));
    Py_Try(lst = PyArray_ContiguousFromAny(list, PyArray_INTP, 1, 1));
    len = PyArray_SIZE(lst);
    numbers = (intp *) PyArray_DATA(lst);
    mxi = mxx (numbers, len) ;
    mni = mnx (numbers, len) ;
    Py_Assert(numbers[mni] >= 0,
              "irst argument of bincount must be non-negative");
    ans_size = numbers [mxi] + 1 ;
    type = PyArray_DescrFromType(PyArray_INTP);
    if (weight == Py_None) {
        Py_Try(ans = PyArray_Zeros(1, &ans_size, type, 0));
        ians = (intp *)(PyArray_DATA(ans));
        for (i = 0 ; i < len ; i++)
            ians [numbers [i]] += 1 ;
        Py_DECREF(lst);
    }
    else {
        Py_Try(wts = PyArray_ContiguousFromAny(weight,
                                               PyArray_DOUBLE, 1, 1));
        weights = (double *)PyArray_DATA (wts);
        Py_Assert(PyArray_SIZE(wts) == len, "bincount: length of weights " \
                  "does not match that of list");
        type = PyArray_DescrFromType(PyArray_DOUBLE);
        Py_Try(ans = PyArray_Zeros(1, &ans_size, type, 0));
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
    Py_XDECREF(ans);
    return NULL;
}


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
    PyObject *ax=NULL, *abins=NULL, *aret=NULL;
    double *dx, *dbins ;
    intp lbins, lx ;             /* lengths */
    intp *iret;
    int m, i ;
    static char *kwlist[] = {"x", "bins", NULL};
    PyArray_Descr *type;

    Py_Try(PyArg_ParseTupleAndKeywords(args, kwds, "OO", kwlist,
                                       &ox, &obins));

    type = PyArray_DescrFromType(PyArray_DOUBLE);
    Py_Try(ax=PyArray_FromAny(ox, type, 1, 1, CARRAY, NULL));
    Py_INCREF(type);
    Py_Try(abins = PyArray_FromAny(obins, type, 1, 1, CARRAY, NULL));

    lx = PyArray_SIZE(ax);
    dx = (double *)PyArray_DATA(ax);
    lbins = PyArray_SIZE(abins);
    dbins = (double *)PyArray_DATA(abins);
    Py_Try(aret = PyArray_SimpleNew(1, &lx, PyArray_INTP));
    iret = (intp *)PyArray_DATA(aret);

    Py_Assert(lx > 0 && lbins > 0,
              "x and bins both must have non-zero length");

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
                iret [i] = decr_slot_ ((double)dx [i], dbins, lbins) ;
        }
        else if ( m == 1 ) {
            for ( i = 0 ; i < lx ; i ++ )
                iret [i] = incr_slot_ ((double)dx [i], dbins, lbins) ;
        }
        else Py_Assert(0, "bins must be montonically increasing or decreasing");
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

static PyObject *
arr_insert(PyObject *self, PyObject *args, PyObject *kwdict)
{
    /* Returns input array with values inserted sequentially into places
       indicated by the mask
    */
    PyObject *mask=NULL, *vals=NULL;
    PyArrayObject *ainput=NULL, *amask=NULL, *avals=NULL,
        *tmp=NULL;
    int numvals, totmask, sameshape;
    char *input_data, *mptr, *vptr, *zero=NULL;
    int melsize, delsize, copied, nd;
    intp *instrides, *inshape;
    int mindx, rem_indx, indx, i, k, objarray;

    static char *kwlist[] = {"input","mask","vals",NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwdict, "O&OO", kwlist,
                                     PyArray_Converter, &ainput,
                                     &mask, &vals))
        goto fail;

    amask = (PyArrayObject *) PyArray_FROM_OF(mask, CARRAY);
    if (amask == NULL) goto fail;
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
        PyErr_SetString(PyExc_TypeError,
                        "mask array must be 1-d or same shape as input array");
        goto fail;
    }

    avals = (PyArrayObject *)PyArray_FromObject(vals, ainput->descr->type_num, 0, 1);
    if (avals == NULL) goto fail;

    numvals = PyArray_SIZE(avals);
    nd = ainput->nd;
    input_data = ainput->data;
    mptr = amask->data;
    melsize = amask->descr->elsize;
    vptr = avals->data;
    delsize = avals->descr->elsize;
    zero = PyArray_Zero(amask);
    if (zero == NULL)
        goto fail;
    objarray = (ainput->descr->type_num == PyArray_OBJECT);

    /* Handle zero-dimensional case separately */
    if (nd == 0) {
        if (memcmp(mptr,zero,melsize) != 0) {
            /* Copy value element over to input array */
            memcpy(input_data,vptr,delsize);
            if (objarray) Py_INCREF(*((PyObject **)vptr));
        }
        Py_DECREF(amask);
        Py_DECREF(avals);
        PyDataMem_FREE(zero);
        Py_DECREF(ainput);
        Py_INCREF(Py_None);
        return Py_None;
    }

    /* Walk through mask array, when non-zero is encountered
       copy next value in the vals array to the input array.
       If we get through the value array, repeat it as necessary.
    */
    totmask = (int) PyArray_SIZE(amask);
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
            if (copied >= numvals) vptr = avals->data;
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

static npy_intp
binary_search(double dval, double dlist [], npy_intp len)
{
    /* binary_search accepts three arguments: a numeric value and
     * a numeric array and its length. It assumes that the array is sorted in
     * increasing order. It returns the index of the array's
     * largest element which is <= the value. It will return -1 if
     * the value is less than the least element of the array. */
    /* self is not used */
    npy_intp bottom , top , middle, result;

    if (dval < dlist [0])
        result = -1 ;
    else {
        bottom = 0;
        top = len - 1;
        while (bottom < top) {
            middle = (top + bottom) / 2 ;
            if (dlist [middle] < dval)
                bottom = middle + 1 ;
            else if (dlist [middle] > dval)
                top = middle - 1 ;
            else
                return middle ;
        }
        if (dlist [bottom] > dval)
            result = bottom - 1 ;
        else
            result = bottom ;
    }

    return result ;
}

static PyObject *
arr_interp(PyObject *self, PyObject *args, PyObject *kwdict)
{

    PyObject *fp, *xp, *x;
    PyObject *left=NULL, *right=NULL;
    PyArrayObject *afp=NULL, *axp=NULL, *ax=NULL, *af=NULL;
    npy_intp i, lenx, lenxp, indx;
    double lval, rval;
    double *dy, *dx, *dz, *dres, *slopes;

    static char *kwlist[] = {"x", "xp", "fp", "left", "right", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwdict, "OOO|OO", kwlist,
                                     &x, &xp, &fp, &left, &right))
        return NULL;


    afp = (NPY_AO*)PyArray_ContiguousFromAny(fp, NPY_DOUBLE, 1, 1);
    if (afp == NULL) return NULL;
    axp = (NPY_AO*)PyArray_ContiguousFromAny(xp, NPY_DOUBLE, 1, 1);
    if (axp == NULL) goto fail;
    ax = (NPY_AO*)PyArray_ContiguousFromAny(x, NPY_DOUBLE, 1, 0);
    if (ax == NULL) goto fail;

    lenxp = axp->dimensions[0];
    if (afp->dimensions[0] != lenxp) {
        PyErr_SetString(PyExc_ValueError, "interp: fp and xp are not the same length.");
        goto fail;
    }

    af = (NPY_AO*)PyArray_SimpleNew(ax->nd, ax->dimensions, NPY_DOUBLE);
    if (af == NULL) goto fail;

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
        if ((lval==-1) && PyErr_Occurred())
            goto fail;
    }
    if ((right == NULL) || (right == Py_None)) {
        rval = dy[lenxp-1];
    }
    else {
        rval = PyFloat_AsDouble(right);
        if ((rval==-1) && PyErr_Occurred())
            goto fail;
    }

    slopes = (double *) PyDataMem_NEW((lenxp-1)*sizeof(double));
    for (i=0; i < lenxp-1; i++) {
        slopes[i] = (dy[i+1] - dy[i])/(dx[i+1]-dx[i]);
    }
    for (i=0; i<lenx; i++) {
        indx = binary_search(dz[i], dx, lenxp);
        if (indx < 0)
            dres[i] = lval;
        else if (indx >= lenxp - 1)
            dres[i] = rval;
        else
            dres[i] = slopes[indx]*(dz[i]-dx[indx]) + dy[indx];
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



static PyTypeObject *PyMemberDescr_TypePtr=NULL;
static PyTypeObject *PyGetSetDescr_TypePtr=NULL;
static PyTypeObject *PyMethodDescr_TypePtr=NULL;

/* Can only be called if doc is currently NULL
 */
static PyObject *
arr_add_docstring(PyObject *dummy, PyObject *args)
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

    if (!PyArg_ParseTuple(args, "OO!", &obj, &PyString_Type, &str))
        return NULL;

    docstr = PyString_AS_STRING(str);

#define _TESTDOC1(typebase) (obj->ob_type == &Py##typebase##_Type)
#define _TESTDOC2(typebase) (obj->ob_type == Py##typebase##_TypePtr)
#define _ADDDOC(typebase, doc, name) {                                  \
        Py##typebase##Object *new = (Py##typebase##Object *)obj;        \
        if (!(doc)) {                                                   \
            doc = docstr;                                               \
        }                                                               \
        else {                                                          \
            PyErr_Format(PyExc_RuntimeError,                            \
                         "%s method %s",name, msg);                     \
            return NULL;                                                \
        }                                                               \
    }

    if _TESTDOC1(CFunction)
                    _ADDDOC(CFunction, new->m_ml->ml_doc, new->m_ml->ml_name)
        else if _TESTDOC1(Type)
                             _ADDDOC(Type, new->tp_doc, new->tp_name)
            else if _TESTDOC2(MemberDescr)
                                 _ADDDOC(MemberDescr, new->d_member->doc, new->d_member->name)
                else if _TESTDOC2(GetSetDescr)
                                     _ADDDOC(GetSetDescr, new->d_getset->doc, new->d_getset->name)
                    else if _TESTDOC2(MethodDescr)
                                         _ADDDOC(MethodDescr, new->d_method->ml_doc,
                                                 new->d_method->ml_name)
                        else {
                            PyErr_SetString(PyExc_TypeError,
                                            "Cannot set a docstring for that object");
                            return NULL;
                        }

#undef _TESTDOC1
#undef _TESTDOC2
#undef _ADDDOC

    Py_INCREF(str);
    Py_INCREF(Py_None);
    return Py_None;
}


static char packbits_doc[] = 
"out = numpy.packbits(myarray)\n\n"
"  myarray = an array whose (assumed binary) elements you want to\n"
"             pack into bits (must be of integer type)\n\n"
"   This routine packs the elements of a binary-valued dataset into a\n"
"   1-D NumPy array of type uint8 ('B') whose bits correspond to\n"
"   the logical (0 or nonzero) value of the input elements. \n\n"
"   If myarray has more dimensions than 2 it packs each slice (the last\n"
"   2 dimensions --- rows*columns) separately.  The number of elements\n"
"   per slice (rows*columns) is then important to know to be able to unpack\n"
"   the data later.\n\n"
"     Example:\n"
"     >>> a = array([[[1,0,1],\n"
"     ...             [0,1,0]],\n"
"     ...            [[1,1,0],\n"
"     ...             [0,0,1]]])\n"
"     >>> b = numpy.packbits(a)\n"
"     >>> b\n"
"     array([168, 196], 'b')\n\n"
"     Note that 168 = 128 + 32 + 8\n"
"               196 = 128 + 64 + 4";

static char unpackbits_doc[] = 
"out = numpy.unpackbits(myarray, elements_per_slice {, out_type} )\n\n"
"     myarray =        Array of integer type whose least\n"
"                      significant byte is a bit-field for the\n"
"                      resulting output array.\n\n"
"     elements_per_slice = Necessary for interpretation of myarray.\n"
"                          This is how many elements in the\n "
"                         rows*columns of original packed structure.\n\nOPTIONAL\n"
"     out_type =       The type of output array to populate with 1's\n"
"                      and 0's.  Must be an integer type.\n\n\nThe output array\n"
"                      will be a 1-D array of 1's and zero's";


/*  PACKBITS


    This function packs binary (0 or 1) 1-bit per pixel images
        into bytes for writing to disk. 

*/

void packbits(
	      char	In[],
              int       element_size,  /* in bytes */
	      char	Out[],
              int       total_elements,
              int       els_per_slice
	     )
{
  char          build;
  int           i,index,slice,slices,out_bytes;
  int           maxi, remain, nonzero, j;
  char          *outptr,*inptr;

  outptr = Out;                          /* pointer to output buffer */
  inptr  = In;                           /* pointer to input buffer */
  slices = total_elements/els_per_slice;
  out_bytes = ceil( (float) els_per_slice / 8);     /* number of bytes in each slice */
  remain = els_per_slice % 8;                      /* uneven bits */
  if (remain == 0) remain = 8;           /* */
  /*  printf("Start: %d %d %d %d %d\n",inM,MN,slices,out_bytes,remain);
   */
  for (slice = 0; slice < slices; slice++) {
    for (index = 0; index < out_bytes; index++) {
      build = 0;
      maxi = (index != out_bytes - 1 ? 8 : remain);
      for (i = 0; i < maxi ; i++) {
        build <<= 1;                 /* shift bits left one bit */
        nonzero = 0;
        for (j = 0; j < element_size; j++)  /* determine if this number is non-zero */
          nonzero += (*(inptr++) != 0);
        build += (nonzero > 0);                   /* add to this bit if the input value is non-zero */
      }
      if (index == out_bytes - 1) build <<= (8-remain);
      /*      printf("Here: %d %d %d %d\n",build,slice,index,maxi); 
       */
      *(outptr++) = build;
    }
  }
  return;
}


void unpackbits(
		char    In[],
		int     in_element_size,
	        char    Out[],
                int     element_size,
	        int     total_elements,
                int     els_per_slice
               )
{
  unsigned char mask;
  int           i,index,slice,slices,out_bytes;
  int           maxi, remain;
  char          *outptr,*inptr;

  outptr = Out;
  inptr  = In;
  if (PyArray_ISNBO(PyArray_BIG)) {
    outptr += (element_size - 1);
    inptr  += (in_element_size - 1);
  }
  slices = total_elements / els_per_slice;
  out_bytes = ceil( (float) els_per_slice / 8);
  remain = els_per_slice % 8;
  if (remain == 0) remain = 8;
  /*  printf("Start: %d %d %d %d %d\n",inM,MN,slices,out_bytes,remain);
   */
  for (slice = 0; slice < slices; slice++) {
    for (index = 0; index < out_bytes; index++) {
      maxi = (index != out_bytes - 1 ? 8 : remain);
      mask = 128;
      for (i = 0; i < maxi ; i++) {
        *outptr = ((mask & (unsigned char)(*inptr)) > 0);
        outptr += element_size;
        mask >>= 1;
      }
      /*      printf("Here: %d %d %d %d\n",build,slice,index,maxi); 
       */
      inptr += in_element_size;
    }
  }
  return;
}

static PyObject *
numpyio_pack(PyObject *self, PyObject *args)  /* args: in */
{
  PyArrayObject *arr = NULL, *out = NULL;
  PyObject *obj;
  int      els_per_slice;
  int      out_size;
  int      type;

  if (!PyArg_ParseTuple( args, "O" , &obj))
    return NULL;
  
  type = PyArray_ObjectType(obj,0);
  if ((arr = (PyArrayObject *)PyArray_ContiguousFromObject(obj,type,0,0)) == NULL)
    return NULL;

  if (!PyArray_ISINTEGER(arr))
    PYSETERROR("Expecting an input array of integer data type");

  /* Get size information from input array and make a 1-D output array of bytes */

  els_per_slice = PyArray_DIM(arr, PyArray_NDIM(arr)-1);
  if (PyArray_NDIM(arr) > 1) 
    els_per_slice =  els_per_slice * PyArray_DIM(arr, PyArray_NDIM(arr)-2);
  
  out_size = (PyArray_SIZE(arr)/els_per_slice)*ceil ( (float) els_per_slice / 8);

  if ((out = (PyArrayObject *)PyArray_FromDims(1,&out_size,PyArray_UBYTE))==NULL) {
      goto fail;
  }
  
  packbits(PyArray_DATA(arr),PyArray_ITEMSIZE(arr),PyArray_DATA(out),
	   PyArray_SIZE(arr), els_per_slice);

  Py_DECREF(arr);
  return out;

 fail:
  Py_XDECREF(arr);
  return NULL;

}

static PyObject *
numpyio_unpack(PyObject *self, PyObject *args)  /* args: in, out_type */
{
  PyArrayObject *arr = NULL, *out=NULL;
  PyObject *obj;
  int      els_per_slice, arrsize;
  int      out_size;

  if (!PyArg_ParseTuple( args, "Oi|c" , &obj, &els_per_slice, &out_type))
    return NULL;
  
  if (els_per_slice < 1)
    PYSETERROR("Second argument is elements_per_slice and it must be >= 1");

  if ((arr = (PyArrayObject *)PyArray_ContiguousFromObject(obj,NPY_UBYTE,0,0)) == NULL)
    return NULL;

  arrsize = PyArray_SIZE(arr);

  if ((arrsize % (int) (ceil( (float) els_per_slice / 8))) != 0)
    PYSETERROR("That cannot be the number of elements per slice for this array size");

  if (!PyArray_ISINTEGER(arr))
    PYSETERROR("Can only unpack arrays that are of integer type");

  /* Make an 1-D output array of type out_type */

  out_size = els_per_slice * arrsize / ceil( (float) els_per_slice / 8);

  if ((out = (PyArrayObject *)PyArray_FromDims(1,&out_size,out_type))==NULL)
      goto fail;

  if (out->descr->type_num > PyArray_LONG) {
    PYSETERROR("Can only unpack bits into integer type.");
  }
  
  unpackbits(arr->data,arr->descr->elsize,out->data,out->descr->elsize,out_size,els_per_slice);

  Py_DECREF(arr);
  return PyArray_Return(out);

 fail:
  Py_XDECREF(out);
  Py_XDECREF(arr);
  return NULL;
}

static struct PyMethodDef methods[] = {
    {"_insert",  (PyCFunction)arr_insert, METH_VARARGS | METH_KEYWORDS,
     arr_insert__doc__},
    {"bincount", (PyCFunction)arr_bincount,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"digitize", (PyCFunction)arr_digitize, METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"interp", (PyCFunction)arr_interp, METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"add_docstring", (PyCFunction)arr_add_docstring, METH_VARARGS,
     NULL},
    {"packbits",  numpyio_pack,       1, packbits_doc},
    {"unpackbits", numpyio_unpack,     1, unpackbits_doc},
    {NULL, NULL}    /* sentinel */
};

static void
define_types(void)
{
    PyObject *tp_dict;
    PyObject *myobj;

    tp_dict = PyArrayDescr_Type.tp_dict;
    /* Get "subdescr" */
    myobj = PyDict_GetItemString(tp_dict, "fields");
    if (myobj == NULL) return;
    PyGetSetDescr_TypePtr = myobj->ob_type;
    myobj = PyDict_GetItemString(tp_dict, "alignment");
    if (myobj == NULL) return;
    PyMemberDescr_TypePtr = myobj->ob_type;
    myobj = PyDict_GetItemString(tp_dict, "newbyteorder");
    if (myobj == NULL) return;
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

    s = PyString_FromString("0.5");
    PyDict_SetItemString(d, "__version__", s);
    Py_DECREF(s);

    ErrorObject = PyString_FromString("numpy.lib._compiled_base.error");
    PyDict_SetItemString(d, "error", ErrorObject);
    Py_DECREF(ErrorObject);


    /* define PyGetSetDescr_Type and PyMemberDescr_Type */
    define_types();

    return;
}
