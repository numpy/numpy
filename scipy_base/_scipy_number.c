/* Numeric's source code for array-object ufuncs

   Only thing changed is the coercion model (select_types and setup_matrices)

   When no savespace bit is present then...
   Scalars (Python Objects) only change INT to FLOAT or FLOAT to COMPLEX but
   otherwise do not cause upcasting. 

*/

#define SIZE(mp) (_PyArray_multiply_list((mp)->dimensions, (mp)->nd))
#define NBYTES(mp) ((mp)->descr->elsize * SIZE(mp))
/* Obviously this needs some work. */
#define ISCONTIGUOUS(m) ((m)->flags & CONTIGUOUS)
#define PyArray_CONTIGUOUS(m) (ISCONTIGUOUS(m) ? Py_INCREF(m), m : \
(PyArrayObject *)(PyArray_ContiguousFromObject((PyObject *)(m), \
(m)->descr->type_num, 0,0)))

#ifndef max
#define max(x,y) (x)>(y)?(x):(y)
#endif
#ifndef min
#define min(x,y) (x)>(y)?(y):(x)
#endif

static int scipy_compare_lists(int *l1, int *l2, int n) {
    int i;
    for(i=0;i<n;i++) {
        if (l1[i] != l2[i]) return 0;
    }
    return 1;
}

static int scipy_get_stride(PyArrayObject *mp, int d) {
    return mp->strides[d];
}


static PyObject *scipy_array_copy(PyArrayObject *m1) {
    PyArrayObject *ret = 
	(PyArrayObject *)PyArray_FromDims(m1->nd, m1->dimensions, m1->descr->type_num);
	
    if (PyArray_CopyArray(ret, m1) == -1) return NULL;
	
    return (PyObject *)ret;
}

#define SCALARBIT 2048
#define MAKEFROMSCALAR(obj) (((PyArrayObject *)(obj))->descr->type_num |= SCALARBIT)
#define PyArray_ISFROMSCALAR(obj) ((((PyArrayObject*)(obj))->descr->type_num & SCALARBIT))
#define OFFSCALAR(obj) (((PyArrayObject *)(obj))->descr->type_num &= ~((int) SCALARBIT))
#define PYNOSCALAR 0
#define PYINTPOS 1
#define PYINTNEG 2
#define PYFLOAT  3
#define PYCOMPLEX 4


/* This function is called while searching for an appropriate ufunc

   It should return a 0 if coercion of thistype to neededtype is not safe.

   It uses PyArray_CanCastSafely but adds special logic to allow Python 
   scalars to be downcast within the same kind if they are in the presence
   of arrays.

 */
static int scipy_cancoerce(char thistype, char neededtype, char scalar) {

    if (scalar==PYNOSCALAR) return PyArray_CanCastSafely(thistype, neededtype);
    if (scalar==PYINTPOS) 
        return (neededtype >= PyArray_UBYTE);
    if (scalar==PYINTNEG)
	return ((neededtype >= PyArray_SBYTE) && (neededtype != PyArray_USHORT) && (neededtype != PyArray_UINT));
    if (scalar==PYFLOAT)
        return (neededtype >= PyArray_FLOAT);
    if (scalar==PYCOMPLEX)
        return (neededtype >= PyArray_CFLOAT);
    return 1; /* should never get here... */   
}

static int scipy_select_types(PyUFuncObject *self, char *arg_types, void **data, 
			      PyUFuncGenericFunction *function, char *scalars) {
    int i=0, j;
    int k=0;
    char largest_savespace = 0, real_type;

    for (j=0; j<self->nin; j++) {
	real_type = arg_types[j] & ~((char )SAVESPACEBIT);
	if ((arg_types[j] & SAVESPACEBIT) && (real_type > largest_savespace)) 
	    largest_savespace = real_type;
    }

    if (largest_savespace == 0) {

        /* start search for signature at first reasonable choice (first array-based
           type --- won't use scalar for this check)*/
        while(k<self->nin && scalars[k] != PYNOSCALAR) k++;
        if (k == self->nin) {  /* no arrays */
	    /* so use usual coercion rules -- ignore scalar handling */
	    for (j=0; j<self->nin; j++) {
		scalars[j] = PYNOSCALAR;
	    }
	    k = 0;
	}
        while (i<self->ntypes && arg_types[k] > self->types[i*self->nargs+k]) i++;
        
        /* Signature search */
        for(;i<self->ntypes; i++) {
            for(j=0; j<self->nin; j++) {
                if (!scipy_cancoerce(arg_types[j], self->types[i*self->nargs+j],
                                     scalars[j])) break;
            }
            if (j == self->nin) break; /* Found signature that will work */
            /* Otherwise, increment i and check next signature */
        }
        if(i>=self->ntypes) {
            PyErr_SetString(PyExc_TypeError, 
                            "function not supported for these types, and can't coerce to supported types");
            return -1;
        }

        /* reset arg_types to those needed for this signature */
        for(j=0; j<self->nargs; j++) 
            arg_types[j] = (self->types[i*self->nargs+j] & ~((char )SAVESPACEBIT));
    }
    else {
	while(i<self->ntypes && largest_savespace > self->types[i*self->nargs]) i++;
	if (i>=self->ntypes || largest_savespace < self->types[i*self->nargs]) {
	    PyErr_SetString(PyExc_TypeError,
			    "function not supported for the spacesaver array with the largest typecode.");
	    return -1;
	}
		
	for(j=0; j<self->nargs; j++)  /* Input arguments */
	    arg_types[j] = (self->types[i*self->nargs+j] | SAVESPACEBIT);                
    }
    
    
    *data = self->data[i];
    *function = self->functions[i];
	
    return 0;
}

static int scipy_setup_matrices(PyUFuncObject *self, PyObject *args,  
				PyUFuncGenericFunction *function, void **data,
				PyArrayObject **mps, char *arg_types) {
    int nargs, i;
    char *scalars=NULL;
    PyObject *obj;
    int temp;
	 
    nargs = PyTuple_Size(args);
    if ((nargs != self->nin) && (nargs != self->nin+self->nout)) {
	PyErr_SetString(PyExc_ValueError, "invalid number of arguments");
	return -1;
    }

    scalars = calloc(self->nin, sizeof(char));
    if (scalars == NULL) {
      PyErr_NoMemory();
      return -1;
    }

    /* Determine the types of the input arguments. */
    for(i=0; i<self->nin; i++) {
        obj = PyTuple_GET_ITEM(args,i);
	arg_types[i] = (char)PyArray_ObjectType(obj, 0);
	if (PyArray_Check(obj)) {
	    if (PyArray_ISSPACESAVER(obj)) 
		arg_types[i] |= SAVESPACEBIT;
	    else if (PyArray_ISFROMSCALAR(obj)) {
		temp = OFFSCALAR(obj);
		if (temp == PyArray_LONG) {
		    if (((long *)(((PyArrayObject *)obj)->data))[0] < 0) 
			scalars[i] = PYINTNEG;
		    else scalars[i] = PYINTPOS;
		}
		else if (temp == PyArray_DOUBLE) scalars[i] = PYFLOAT;
		else if (temp == PyArray_CDOUBLE) scalars[i] = PYCOMPLEX;
	    }
	}
	else {
	    if (PyInt_Check(obj)) {
		if (PyInt_AS_LONG(obj) < 0) scalars[i] = PYINTNEG;
		else scalars[i] = PYINTPOS;
	    }
	    else if (PyFloat_Check(obj)) scalars[i] = PYFLOAT;
	    else if (PyComplex_Check(obj)) scalars[i] = PYCOMPLEX;
	}
    }
	

    /* Select an appropriate function for these argument types. */
    temp = scipy_select_types(self, arg_types, data, function, scalars);
    free(scalars);
    if (temp == -1) return -1;

    /* Coerce input arguments to the right types. */
    for(i=0; i<self->nin; i++) {
	if ((mps[i] = (PyArrayObject *)PyArray_FromObject(PyTuple_GET_ITEM(args,
									   i),
							  arg_types[i], 0, 0)) == NULL) {
	    return -1;
	}
    }
	
    /* Check the return arguments, and INCREF. */
    for(i = self->nin;i<nargs; i++) {
	mps[i] = (PyArrayObject *)PyTuple_GET_ITEM(args, i);
	Py_INCREF(mps[i]);
	if (!PyArray_Check((PyObject *)mps[i])) {
	    PyErr_SetString(PyExc_TypeError, "return arrays must be of arraytype");
	    return -1;
	}
	if (mps[i]->descr->type_num != (arg_types[i] & ~((char )SAVESPACEBIT))) {
	    PyErr_SetString(PyExc_TypeError, "return array has incorrect type");
	    return -1;
	}
    }
	
    return nargs;
}

static int scipy_setup_return(PyUFuncObject *self, int nd, int *dimensions, int steps[MAX_DIMS][MAX_ARGS], 
                        PyArrayObject **mps, char *arg_types) {
    int i, j;
	
	
    /* Initialize the return matrices, or check if provided. */
    for(i=self->nin; i<self->nargs; i++) {
	if (mps[i] == NULL) {
	    if ((mps[i] = (PyArrayObject *)PyArray_FromDims(nd, dimensions,
							    arg_types[i])) == NULL)
		return -1;
	} else {
	    if (!scipy_compare_lists(mps[i]->dimensions, dimensions, nd)) {
		PyErr_SetString(PyExc_ValueError, "invalid return array shape");
		return -1;
	    }
	}
	for(j=0; j<mps[i]->nd; j++) {
	    steps[j][i] = scipy_get_stride(mps[i], j+mps[i]->nd-nd);
	}
	/* Small hack to keep purify happy (no UMR's for 0d array's) */
	if (mps[i]->nd == 0) steps[0][i] = 0;
    }
    return 0;
}

static int scipy_optimize_loop(int steps[MAX_DIMS][MAX_ARGS], int *loop_n, int n_loops) {
  int j, tmp;
	
#define swap(x, y) tmp = (x), (x) = (y), (y) = tmp
	
    /* Here should go some code to "compress" the loops. */
	
    if (n_loops > 1 && (loop_n[n_loops-1] < loop_n[n_loops-2]) ) {
	swap(loop_n[n_loops-1], loop_n[n_loops-2]);
	for (j=0; j<MAX_ARGS; j++) {
	    swap(steps[n_loops-1][j], steps[n_loops-2][j]);
	}
    }
    return n_loops;
	
#undef swap
}


static int scipy_setup_loop(PyUFuncObject *self, PyObject *args, PyUFuncGenericFunction *function, void **data,
	       int steps[MAX_DIMS][MAX_ARGS], int *loop_n, PyArrayObject **mps) {
    int i, j, nargs, nd, n_loops, tmp;
    int dimensions[MAX_DIMS];
    char arg_types[MAX_ARGS];
	
    nargs = scipy_setup_matrices(self, args, function, data, mps, arg_types);
    if (nargs < 0) return -1;
	
    /* The return matrices will have the same number of dimensions as the largest input array. */
    for(i=0, nd=0; i<self->nin; i++) nd = max(nd, mps[i]->nd);
	
    /* Setup the loop. This can be optimized later. */
    n_loops = 0;
	
    for(i=0; i<nd; i++) {
	dimensions[i] = 1;
	for(j=0; j<self->nin; j++) {
	    if (i + mps[j]->nd-nd  >= 0) tmp = mps[j]->dimensions[i + mps[j]->nd-nd];
	    else tmp = 1; 
			
	    if (tmp == 1) {  
		steps[n_loops][j] = 0;
	    } else {
		if (dimensions[i] == 1) dimensions[i] = tmp;
		else if (dimensions[i] != tmp) {
		    PyErr_SetString(PyExc_ValueError, "frames are not aligned");
		    return -1;
		}
		steps[n_loops][j] = scipy_get_stride(mps[j], i + mps[j]->nd-nd);
	    }
	}
	loop_n[n_loops] = dimensions[i];
	n_loops++;
    }
	
    /* Small hack to keep purify happy (no UMR's for 0d array's) */
    if (nd == 0) {
	for(j=0; j<self->nin; j++) steps[0][j] = 0;
    }
	
    if (scipy_setup_return(self, nd, dimensions, steps, mps, arg_types) == -1) return -1;
	
    n_loops = scipy_optimize_loop(steps, loop_n, n_loops);
	
    return n_loops;
}

static int scipy_PyUFunc_GenericFunction(PyUFuncObject *self, PyObject *args, PyArrayObject **mps) {
    int steps[MAX_DIMS][MAX_ARGS];
    int i, loop, n_loops, loop_i[MAX_DIMS], loop_n[MAX_DIMS];
    char *pointers[MAX_ARGS], *resets[MAX_DIMS][MAX_ARGS];
    void *data;
    PyUFuncGenericFunction function;
	
    if (self == NULL) {
	PyErr_SetString(PyExc_ValueError, "function not supported");
	return -1;
    }
	
    n_loops = scipy_setup_loop(self, args, &function, &data, steps, loop_n, mps);
    if (n_loops == -1) return -1;
	
    for(i=0; i<self->nargs; i++) pointers[i] = mps[i]->data;
	
    errno = 0;
    if (n_loops == 0) {
	n_loops = 1;
	function(pointers, &n_loops, steps[0], data);
    } else {
	/* This is the inner loop to actually do the computation. */
	loop=-1;
	while(1) {
	    while (loop < n_loops-2) {
		loop++;
		loop_i[loop] = 0;
		for(i=0; i<self->nin+self->nout; i++) { resets[loop][i] = pointers[i]; }
	    }
			
	    function(pointers, loop_n+(n_loops-1), steps[n_loops-1], data);
			
	    while (loop >= 0 && !(++loop_i[loop] < loop_n[loop]) && loop >= 0) loop--;
	    if (loop < 0) break;
	    for(i=0; i<self->nin+self->nout; i++) { pointers[i] = resets[loop][i] + steps[loop][i]*loop_i[loop]; }
	}
    }
    if (PyErr_Occurred()) return -1;
	
    /* Cleanup the returned matrices so that scalars will be returned as python scalars */
    /*  We don't use this in SciPy --- will disable checking for all ufuncs */
    /*
    if (self->check_return) {
        for(i=self->nin; i<self->nout+self->nin; i++) check_array(mps[i]);
        if (errno != 0) {
            math_error();
            return -1;
        }
    }
    */

    return 0;
}

/* -------------------------------------------------------------- */

typedef struct {
    PyUFuncObject *add, 
	*subtract, 
	*multiply, 
	*divide, 
	*remainder, 
	*power, 
	*negative, 
	*absolute;
    PyUFuncObject *invert, 
	*left_shift, 
	*right_shift, 
	*bitwise_and, 
	*bitwise_xor,
	*bitwise_or;
    PyUFuncObject *less,     /* Added by Scott N. Gunyan */
        *less_equal,         /* for rich comparisons */
        *equal,
        *not_equal,
        *greater,
        *greater_equal;
    PyUFuncObject *floor_divide, /* Added by Bruce Sherwood */
        *true_divide;            /* for floor and true divide */
} NumericOps;


static NumericOps sn_ops;

#define GET(op) sn_ops.op = (PyUFuncObject *)PyDict_GetItemString(dict, #op)

static int scipy_SetNumericOps(PyObject *dict) {
    GET(add);
    GET(subtract);
    GET(multiply);
    GET(divide);
    GET(remainder);
    GET(power);
    GET(negative);
    GET(absolute);
    GET(invert);
    GET(left_shift);
    GET(right_shift);
    GET(bitwise_and);
    GET(bitwise_or);
    GET(bitwise_xor);
    GET(less);         /* Added by Scott N. Gunyan */
    GET(less_equal);   /* for rich comparisons */
    GET(equal);
    GET(not_equal);
    GET(greater);
    GET(greater_equal);
    GET(floor_divide);  /* Added by Bruce Sherwood */
    GET(true_divide);   /* for floor and true divide */
    return 0;
}

/* This is getting called */ 
static int scipy_array_coerce(PyArrayObject **pm, PyObject **pw) {
    PyObject *new_op;
    char isscalar = 0;

    if (PyInt_Check(*pw) || PyFloat_Check(*pw) || PyComplex_Check(*pw)) {
	isscalar = 1;
    }
    if ((new_op = PyArray_FromObject(*pw, PyArray_NOTYPE, 0, 0)) 
	== NULL) 
	return -1;
    Py_INCREF(*pm);
    *pw = new_op;
    if (isscalar) MAKEFROMSCALAR(*pw);
    return 0;
 }

static PyObject *PyUFunc_BinaryFunction(PyUFuncObject *s, PyArrayObject *mp1, PyObject *mp2) {
    PyObject *arglist;
    PyArrayObject *mps[3];

    arglist = Py_BuildValue("(OO)", mp1, mp2);
    mps[0] = mps[1] = mps[2] = NULL;
    if (scipy_PyUFunc_GenericFunction(s, arglist, mps) == -1) {
	Py_DECREF(arglist);
	Py_XDECREF(mps[0]); Py_XDECREF(mps[1]); Py_XDECREF(mps[2]);
	return NULL;
    }
	
    Py_DECREF(mps[0]); Py_DECREF(mps[1]);
    Py_DECREF(arglist);
    return PyArray_Return(mps[2]);
}

/*This method adds the augmented assignment*/
/*functionality that was made available in Python 2.0*/
static PyObject *PyUFunc_InplaceBinaryFunction(PyUFuncObject *s, PyArrayObject *mp1, PyObject *mp2) {
    PyObject *arglist;
    PyArrayObject *mps[3];
	
    arglist = Py_BuildValue("(OOO)", mp1, mp2, mp1);
	
    mps[0] = mps[1] = mps[2] = NULL;
    if (scipy_PyUFunc_GenericFunction(s, arglist, mps) == -1) {
	Py_DECREF(arglist);
	Py_XDECREF(mps[0]); Py_XDECREF(mps[1]); Py_XDECREF(mps[2]);
	return NULL;
    }
	
    Py_DECREF(mps[0]); Py_DECREF(mps[1]);
    Py_DECREF(arglist);
    return PyArray_Return(mps[2]);
}

static PyObject *PyUFunc_UnaryFunction(PyUFuncObject *s, PyArrayObject *mp1) {
    PyObject *arglist;
    PyArrayObject *mps[3];
	
    arglist = Py_BuildValue("(O)", mp1);
	
    mps[0] = mps[1] = NULL;
    if (scipy_PyUFunc_GenericFunction(s, arglist, mps) == -1) {
	Py_DECREF(arglist);
	Py_XDECREF(mps[0]); Py_XDECREF(mps[1]);
	return NULL;
    }
	
    Py_DECREF(mps[0]);
    Py_DECREF(arglist);
    return PyArray_Return(mps[1]);
}

/* Could add potential optimizations here for special casing certain conditions...*/

static PyObject *scipy_array_add(PyArrayObject *m1, PyObject *m2) {
    return PyUFunc_BinaryFunction(sn_ops.add, m1, m2);
}
static PyObject *scipy_array_subtract(PyArrayObject *m1, PyObject *m2) {
    return PyUFunc_BinaryFunction(sn_ops.subtract, m1, m2);
}
static PyObject *scipy_array_multiply(PyArrayObject *m1, PyObject *m2) {
    return PyUFunc_BinaryFunction(sn_ops.multiply, m1, m2);
}
static PyObject *scipy_array_divide(PyArrayObject *m1, PyObject *m2) {
    return PyUFunc_BinaryFunction(sn_ops.divide, m1, m2);
}
static PyObject *scipy_array_remainder(PyArrayObject *m1, PyObject *m2) {
    return PyUFunc_BinaryFunction(sn_ops.remainder, m1, m2);
}
static PyObject *scipy_array_power(PyArrayObject *m1, PyObject *m2) {
    return PyUFunc_BinaryFunction(sn_ops.power, m1, m2);
}
static PyObject *scipy_array_negative(PyArrayObject *m1) { 
    return PyUFunc_UnaryFunction(sn_ops.negative, m1);
}
static PyObject *scipy_array_absolute(PyArrayObject *m1) { 
    return PyUFunc_UnaryFunction(sn_ops.absolute, m1);
}
static PyObject *scipy_array_invert(PyArrayObject *m1) { 
    return PyUFunc_UnaryFunction(sn_ops.invert, m1);
}
static PyObject *scipy_array_left_shift(PyArrayObject *m1, PyObject *m2) {
    return PyUFunc_BinaryFunction(sn_ops.left_shift, m1, m2);
}
static PyObject *scipy_array_right_shift(PyArrayObject *m1, PyObject *m2) {
    return PyUFunc_BinaryFunction(sn_ops.right_shift, m1, m2);
}
static PyObject *scipy_array_bitwise_and(PyArrayObject *m1, PyObject *m2) {
    return PyUFunc_BinaryFunction(sn_ops.bitwise_and, m1, m2);
}
static PyObject *scipy_array_bitwise_or(PyArrayObject *m1, PyObject *m2) {
    return PyUFunc_BinaryFunction(sn_ops.bitwise_or, m1, m2);
}
static PyObject *scipy_array_bitwise_xor(PyArrayObject *m1, PyObject *m2) {
    return PyUFunc_BinaryFunction(sn_ops.bitwise_xor, m1, m2);
}


/*These methods add the augmented assignment*/
/*functionality that was made available in Python 2.0*/
static PyObject *scipy_array_inplace_add(PyArrayObject *m1, PyObject *m2) {
    return PyUFunc_InplaceBinaryFunction(sn_ops.add, m1, m2);
}
static PyObject *scipy_array_inplace_subtract(PyArrayObject *m1, PyObject *m2) {
    return PyUFunc_InplaceBinaryFunction(sn_ops.subtract, m1, m2);
}
static PyObject *scipy_array_inplace_multiply(PyArrayObject *m1, PyObject *m2) {
    return PyUFunc_InplaceBinaryFunction(sn_ops.multiply, m1, m2);
}
static PyObject *scipy_array_inplace_divide(PyArrayObject *m1, PyObject *m2) {
    return PyUFunc_InplaceBinaryFunction(sn_ops.divide, m1, m2);
}
static PyObject *scipy_array_inplace_remainder(PyArrayObject *m1, PyObject *m2) {
    return PyUFunc_InplaceBinaryFunction(sn_ops.remainder, m1, m2);
}
static PyObject *scipy_array_inplace_power(PyArrayObject *m1, PyObject *m2) {
    return PyUFunc_InplaceBinaryFunction(sn_ops.power, m1, m2);
}
static PyObject *scipy_array_inplace_left_shift(PyArrayObject *m1, PyObject *m2) {
    return PyUFunc_InplaceBinaryFunction(sn_ops.left_shift, m1, m2);
}
static PyObject *scipy_array_inplace_right_shift(PyArrayObject *m1, PyObject *m2) {
    return PyUFunc_InplaceBinaryFunction(sn_ops.right_shift, m1, m2);
}
static PyObject *scipy_array_inplace_bitwise_and(PyArrayObject *m1, PyObject *m2) {
    return PyUFunc_InplaceBinaryFunction(sn_ops.bitwise_and, m1, m2);
}
static PyObject *scipy_array_inplace_bitwise_or(PyArrayObject *m1, PyObject *m2) {
    return PyUFunc_InplaceBinaryFunction(sn_ops.bitwise_or, m1, m2);
}
static PyObject *scipy_array_inplace_bitwise_xor(PyArrayObject *m1, PyObject *m2) {
    return PyUFunc_InplaceBinaryFunction(sn_ops.bitwise_xor, m1, m2);
}

/*Added by Bruce Sherwood Dec 2001*/
/*These methods add the floor and true division*/
/*functionality that was made available in Python 2.2*/
static PyObject *scipy_array_floor_divide(PyArrayObject *m1, PyObject *m2) {
    return PyUFunc_BinaryFunction(sn_ops.floor_divide, m1, m2);
}
static PyObject *scipy_array_true_divide(PyArrayObject *m1, PyObject *m2) {
    return PyUFunc_BinaryFunction(sn_ops.true_divide, m1, m2);
}
static PyObject *scipy_array_inplace_floor_divide(PyArrayObject *m1, PyObject *m2) {
    return PyUFunc_InplaceBinaryFunction(sn_ops.floor_divide, m1, m2);
}
static PyObject *scipy_array_inplace_true_divide(PyArrayObject *m1, PyObject *m2) {
    return PyUFunc_InplaceBinaryFunction(sn_ops.true_divide, m1, m2);
}
/*End of methods added by Bruce Sherwood*/

/* Array evaluates as "true" if any of the elements are non-zero */
static int scipy_array_nonzero(PyArrayObject *mp) {
    char *zero;
    PyArrayObject *self;
    char *data;
    int i, s, elsize;
	
    self = PyArray_CONTIGUOUS(mp);
    zero = self->descr->zero;

    s = SIZE(self);
    elsize = self->descr->elsize;
    data = self->data;
    for(i=0; i<s; i++, data+=elsize) {
	if (memcmp(zero, data, elsize) != 0) break;
    }
	
    Py_DECREF(self);

    return i!=s;
}

static PyObject *scipy_array_divmod(PyArrayObject *op1, PyObject *op2) {
    PyObject *divp, *modp, *result;

    divp = scipy_array_divide(op1, op2);
    if (divp == NULL) return NULL;
    modp = scipy_array_remainder(op1, op2);
    if (modp == NULL) {
	Py_DECREF(divp);
	return NULL;
    }
    result = Py_BuildValue("OO", divp, modp);
    Py_DECREF(divp);
    Py_DECREF(modp);
    return result;
}


static PyObject *scipy_array_int(PyArrayObject *v) {        
    PyObject *pv, *pv2;
    if (PyArray_SIZE(v) != 1) {
	PyErr_SetString(PyExc_TypeError, "only length-1 arrays can be converted to Python scalars.");
	return NULL;
    }
    pv = v->descr->getitem(v->data);
    if (pv == NULL) return NULL;
    if (pv->ob_type->tp_as_number == 0) {
	PyErr_SetString(PyExc_TypeError, "cannot convert to an int, scalar object is not a number.");
	Py_DECREF(pv);
	return NULL;
    }
    if (pv->ob_type->tp_as_number->nb_int == 0) {
	PyErr_SetString(PyExc_TypeError, "don't know how to convert scalar number to int");
	Py_DECREF(pv);
	return NULL;
    }

    pv2 = pv->ob_type->tp_as_number->nb_int(pv);
    Py_DECREF(pv);
    return pv2;        
}

static PyObject *scipy_array_float(PyArrayObject *v) {
    PyObject *pv, *pv2;
    if (PyArray_SIZE(v) != 1) {
	PyErr_SetString(PyExc_TypeError, "only length-1 arrays can be converted to Python scalars.");
	return NULL;
    }
    pv = v->descr->getitem(v->data);
    if (pv == NULL) return NULL;
    if (pv->ob_type->tp_as_number == 0) {
	PyErr_SetString(PyExc_TypeError, "cannot convert to an int, scalar object is not a number.");
	Py_DECREF(pv);
	return NULL;
    }
    if (pv->ob_type->tp_as_number->nb_float == 0) {
	PyErr_SetString(PyExc_TypeError, "don't know how to convert scalar number to float");
	Py_DECREF(pv);
	return NULL;
    }
    pv2 = pv->ob_type->tp_as_number->nb_float(pv);
    Py_DECREF(pv);
    return pv2;        
}

static PyObject *scipy_array_long(PyArrayObject *v) {        
    PyObject *pv, *pv2;
    if (PyArray_SIZE(v) != 1) {
	PyErr_SetString(PyExc_TypeError, "only length-1 arrays can be converted to Python scalars.");
	return NULL;
    }
    pv = v->descr->getitem(v->data);
    if (pv->ob_type->tp_as_number == 0) {
	PyErr_SetString(PyExc_TypeError, "cannot convert to an int, scalar object is not a number.");
	return NULL;
    }
    if (pv->ob_type->tp_as_number->nb_long == 0) {
	PyErr_SetString(PyExc_TypeError, "don't know how to convert scalar number to long");
	return NULL;
    }
    pv2 = pv->ob_type->tp_as_number->nb_long(pv);
    Py_DECREF(pv);
    return pv2;        
}

static PyObject *scipy_array_oct(PyArrayObject *v) {        
    PyObject *pv, *pv2;
    if (PyArray_SIZE(v) != 1) {
	PyErr_SetString(PyExc_TypeError, "only length-1 arrays can be converted to Python scalars.");
	return NULL;
    }
    pv = v->descr->getitem(v->data);
    if (pv->ob_type->tp_as_number == 0) {
	PyErr_SetString(PyExc_TypeError, "cannot convert to an int, scalar object is not a number.");
	return NULL;
    }
    if (pv->ob_type->tp_as_number->nb_oct == 0) {
	PyErr_SetString(PyExc_TypeError, "don't know how to convert scalar number to oct");
	return NULL;
    }
    pv2 = pv->ob_type->tp_as_number->nb_oct(pv);
    Py_DECREF(pv);
    return pv2;        
}

static PyObject *scipy_array_hex(PyArrayObject *v) {        
    PyObject *pv, *pv2;
    if (PyArray_SIZE(v) != 1) {
	PyErr_SetString(PyExc_TypeError, "only length-1 arrays can be converted to Python scalars.");
	return NULL;
    }
    pv = v->descr->getitem(v->data);
    if (pv->ob_type->tp_as_number == 0) {
	PyErr_SetString(PyExc_TypeError, "cannot convert to an int, scalar object is not a number.");
	return NULL;
    }
    if (pv->ob_type->tp_as_number->nb_hex == 0) {
	PyErr_SetString(PyExc_TypeError, "don't know how to convert scalar number to hex");
	return NULL;
    }
    pv2 = pv->ob_type->tp_as_number->nb_hex(pv);
    Py_DECREF(pv);
    return pv2;        
}



/* ---------- */

static PyObject *scipy_ufunc_call(PyUFuncObject *self, PyObject *args) {
    int i;
    PyTupleObject *ret;
    PyArrayObject *mps[MAX_ARGS];
	
    /* Initialize all array objects to NULL to make cleanup easier if something goes wrong. */
    for(i=0; i<self->nargs; i++) mps[i] = NULL;
	
    if (scipy_PyUFunc_GenericFunction(self, args, mps) == -1) {
	for(i=0; i<self->nargs; i++) if (mps[i] != NULL) Py_DECREF(mps[i]);
	return NULL;
    }
	
    for(i=0; i<self->nin; i++) Py_DECREF(mps[i]);
	
    if (self->nout == 1) { 
	return PyArray_Return(mps[self->nin]); 
    } else {  
	ret = (PyTupleObject *)PyTuple_New(self->nout);
	for(i=0; i<self->nout; i++) {
	    PyTuple_SET_ITEM(ret, i, PyArray_Return(mps[i+self->nin]));
	}
	return (PyObject *)ret;
    }
}
