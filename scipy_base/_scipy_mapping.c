char *scipy_index2ptr(PyArrayObject *mp, int i) {
    if (i==0 && (mp->nd == 0 || mp->dimensions[0] > 0)) 
	return mp->data;
	
    if (mp->nd>0 &&  i>0 && i < mp->dimensions[0]) {
	return mp->data+i*mp->strides[0];
    }
    PyErr_SetString(PyExc_IndexError,"index out of bounds");      
    return NULL;
}

static PyObject *scipy_array_item(PyArrayObject *self, int i) {
    char *item;

    if ((item = scipy_index2ptr(self, i)) == NULL) return NULL;
	
    if(self->nd > 1) {
	PyArrayObject *r;
        r = (PyArrayObject *)PyArray_FromDimsAndDataAndDescr(self->nd-1,
                                                             self->dimensions+1,
                                                             self->descr,
                                                             item);
        if (r == NULL) return NULL;
        memmove(r->strides, self->strides+1, sizeof(int)*(r->nd));
	r->base = (PyObject *)self;
	r->flags = (self->flags & (CONTIGUOUS | SAVESPACE));
        r->flags |= OWN_DIMENSIONS | OWN_STRIDES;
	Py_INCREF(self);
	return (PyObject*)r;
    } else {

	/* I would like to do this, but it requires a fix to several places of code.
	   fprintf(stderr,"Getting a Python scalar by indexing a rank-0 array is obsolete: use a.toscalar().\n");
	*/
	return self->descr->getitem(item);
    } 
}

static int scipy_PyArray_CopyObject(PyArrayObject *dest, PyObject *src_object) {
    PyArrayObject *src;
    PyObject *tmp;
    int ret, n_new, n_old;
    char *new_string;
	
    /* Special function added here to try and make arrays of strings
       work out. */
    if ((dest->descr->type_num == PyArray_CHAR) && dest->nd > 0 
	&& PyString_Check(src_object)) {
	n_new = dest->dimensions[dest->nd-1];
	n_old = PyString_Size(src_object); 
	if (n_new > n_old) {
	    new_string = (char *)malloc(n_new*sizeof(char));
	    memmove(new_string, 
		   PyString_AS_STRING((PyStringObject *)src_object),
		   n_old);
	    memset(new_string+n_old, ' ', n_new-n_old);
	    tmp = PyString_FromStringAndSize(new_string, 
					     n_new);
	    free(new_string);
	    src_object = tmp;
	}
    }
    src = (PyArrayObject *)PyArray_FromObject(src_object,
					      dest->descr->type_num, 0,
					      dest->nd);
    if (src == NULL) return -1;
	
    ret = PyArray_CopyArray(dest, src);
    Py_DECREF(src);
    return ret;
}


static int scipy_array_ass_item(PyArrayObject *self, int i, PyObject *v) {
    PyObject *c=NULL;
    PyArrayObject *tmp;
    char *item;
    int ret;

     if (v == NULL) {
	PyErr_SetString(PyExc_ValueError, "Can't delete array elements.");
	return -1;
    }

    if (i < 0) i = i+self->dimensions[0];

    if (self->nd > 1) {
	if((tmp = (PyArrayObject *)scipy_array_item(self, i)) == NULL) return -1;
	ret = scipy_PyArray_CopyObject(tmp, v);
	Py_DECREF(tmp);
	return ret;   
    }
	
    if ((item = scipy_index2ptr(self, i)) == NULL) return -1;

    if(self->descr->type_num != PyArray_OBJECT && PyString_Check(v) && PyObject_Length(v) == 1) {
	char *s;
	if ((s=PyString_AsString(v)) == NULL) return -1;
	if(self->descr->type == 'c') {
	    ((char*)self->data)[i]=*s;
	    return 0;
	}
	if(s) c=PyInt_FromLong((long)*s);
	if(c) v=c;
    }

    self->descr->setitem(v, item);
    if(c) {Py_DECREF(c);}
    if(PyErr_Occurred()) return -1;
    return 0;
}



static int scipy_slice_GetIndices(PySliceObject *r, int length, 
				  int *start, int *stop, int *step)
{
    if (r->step == Py_None) {
	*step = 1;
    } else {
	if (!PyInt_Check(r->step)) return -1;
	*step = PyInt_AsLong(r->step);
    }
    if (r->start == Py_None) {
	*start = *step < 0 ? length-1 : 0;
    } else {
	if (!PyInt_Check(r->start)) return -1;
	*start = PyInt_AsLong(r->start);
	if (*start < 0) *start += length;
    }
    if (r->stop == Py_None) {
	*stop = *step < 0 ? -1 : length;
    } else {
	if (!PyInt_Check(r->stop)) return -1;
	*stop = PyInt_AsLong(r->stop);
	if (*stop < 0) *stop += length;
    }
    if (*step < 0) {
        if (*start > (length-1)) *start = length-1;
    } else {
        if (*start > length) *start = length;
    }
    if (*start < 0) *start = 0;
    if (*stop < -1) *stop = -1;
    else if (*stop > length) *stop = length;
    return 0;
}



static int scipy_get_slice(PyObject *op, int max, int *np, int *sp) {
    int start, stop, step;
	
    if (PySlice_Check(op)) {
	if (scipy_slice_GetIndices((PySliceObject *)op, max, 
				   &start, &stop, &step) == -1) return -1;
		
	if (step != 0) {
	    if (step < 0) *np = (stop-start+1+step)/step;
	    else *np = (stop-start-1+step)/step;
	} else {
	    if (stop == start) {
		*np = 0; step = 1;
	    }
	    else return -1;
	}
	if (*np < 0) *np = 0;
	*sp = step;
	return start;
    }  
    return -1;
}

#define PseudoIndex -1
#define RubberIndex -2
#define SingleIndex -3

static int scipy_parse_subindex(PyObject *op, int *step_size, int *n_steps, int max) {
    int i, tmp;
	
    if (op == Py_None) {
	*n_steps = PseudoIndex;
	return 0;
    }
	
    if (op == Py_Ellipsis) {
	*n_steps = RubberIndex;
	return 0;
    }
	
    if (PySlice_Check(op)) {
	if ((i = scipy_get_slice(op, max, n_steps, step_size)) >= 0) {
	    return i;
	} else {
	    PyErr_SetString(PyExc_IndexError, "invalid slice");
	    return -1;
	}
    }
	
    if (PyInt_Check(op)) {
	*n_steps=SingleIndex;
	*step_size=0;
	tmp = PyInt_AsLong(op);
	if (tmp < 0) tmp += max;
	if (tmp >= max || tmp < 0) {
	    PyErr_SetString(PyExc_IndexError, "invalid index");
	    return -1;
	}
	return tmp;
    } 

    PyErr_SetString(PyExc_IndexError, 
		    "each subindex must be either a slice, an integer, Ellipsis, or NewAxis");
    return -1;
}


static int scipy_parse_index(PyArrayObject *self, PyObject *op, 
			     int *dimensions, int *strides, int *offset_ptr) {
    int i, j, n;
    int nd_old, nd_new, start, offset, n_add, n_pseudo;
    int step_size, n_steps;
    PyObject *op1=NULL;
    int is_slice;


    if (PySlice_Check(op) || op == Py_Ellipsis) {
	n = 1;
	op1 = op;
	Py_INCREF(op);  
	/* this relies on the fact that n==1 for loop below */
	is_slice = 1;
    }
    else {
	if (!PySequence_Check(op)) {
	    PyErr_SetString(PyExc_IndexError, 
			    "index must be either an int or a sequence");
	    return -1;
	}
	n = PySequence_Length(op);
	is_slice = 0;
    }
	
    nd_old = nd_new = 0;
	
    offset = 0;
    for(i=0; i<n; i++) {
	if (!is_slice) {
	    if (!(op1=PySequence_GetItem(op, i))) {
		PyErr_SetString(PyExc_IndexError, "invalid index");
		return -1;
	    }
	}
	
	start = scipy_parse_subindex(op1, &step_size, &n_steps, 
				     nd_old < self->nd ? self->dimensions[nd_old] : 0);
	Py_DECREF(op1);
	if (start == -1) break;
		
	if (n_steps == PseudoIndex) {
	    dimensions[nd_new] = 1; strides[nd_new] = 0; nd_new++;
	} else {
	    if (n_steps == RubberIndex) {
		for(j=i+1, n_pseudo=0; j<n; j++) {
		    op1 = PySequence_GetItem(op, j);
		    if (op1 == Py_None) n_pseudo++;
		    Py_DECREF(op1);
		}
		n_add = self->nd-(n-i-n_pseudo-1+nd_old);
		if (n_add < 0) {
		    PyErr_SetString(PyExc_IndexError, "too many indices");
		    return -1;
		}
		for(j=0; j<n_add; j++) {
		    dimensions[nd_new] = self->dimensions[nd_old];
		    strides[nd_new] = self->strides[nd_old];
		    nd_new++; nd_old++;
		}
	    } else {
		if (nd_old >= self->nd) {
		    PyErr_SetString(PyExc_IndexError, "too many indices");
		    return -1;
		}
		offset += self->strides[nd_old]*start;
		nd_old++;
		if (n_steps != SingleIndex) {
		    dimensions[nd_new] = n_steps;
		    strides[nd_new] = step_size*self->strides[nd_old-1];
		    nd_new++;
		}
	    }
	}
    }
    if (i < n) return -1;
    n_add = self->nd-nd_old;
    for(j=0; j<n_add; j++) {
	dimensions[nd_new] = self->dimensions[nd_old];
	strides[nd_new] = self->strides[nd_old];
	nd_new++; nd_old++;
    }     
    *offset_ptr = offset;
    return nd_new;
}





/* Code to handle accessing Array objects as sequence objects */
static int scipy_array_length(PyArrayObject *self) {
    if (self->nd != 0) {
	return self->dimensions[0];
    } else {
	return 1; /* Because a[0] works on 0d arrays. */
    }
}

#define SWAP(a, b, type) { type t = (a); (a) = (b); (b) = t; }

static int scipy_makecontiguous(PyArrayObject *self) {
    PyArrayObject *tmp;
    int ret;
    
    tmp = (PyArrayObject *)PyArray_ContiguousFromObject((PyObject *)self, 
							self->descr->type_num, 
							0, 0);
    if (tmp==NULL) return -1;

    /* Now, swap the fields of tmp and self */
    SWAP(tmp->data, self->data, char *);
    SWAP(tmp->strides, self->strides, int *);
    SWAP(tmp->base, self->base, PyObject *);
    SWAP(tmp->flags, self->flags, int); 
   
    /* Decrement tmp (will actually be self fields) */
    Py_DECREF(tmp);
    return 0;
}

static PyArrayObject *scipy_onearray_index(PyArrayObject *self, PyObject *op, int typenum) {
    PyObject *tup, *tmp, *optmp;
    PyArrayObject *opa, *other;
    int i, Nel, Nout, elsize, dims[1];
    unsigned char *ptr;
    char *inptr, *optr;

    if (!PyArray_ISCONTIGUOUS(self)) {		
	if (scipy_makecontiguous(self)==-1) return NULL;
    }
    opa = (PyArrayObject *)PyArray_ContiguousFromObject 
	(op,((PyArrayObject *)op)->descr->type_num, 0, 0);
    if (opa==NULL) return NULL;
    tup = Py_BuildValue("(i)",-1);
    if (tup==NULL) { Py_DECREF(opa); return NULL;}
    tmp = PyArray_Reshape(self, tup);
    optmp = PyArray_Reshape(opa, tup);
    Py_DECREF(tup);
    if ((tmp==NULL) || (optmp == NULL)) { 
	Py_DECREF(opa);
	Py_XDECREF(tmp);
	Py_XDECREF(optmp); 
	return NULL;
    }
    if ( typenum == PyArray_UBYTE) {
	Nel = PyArray_SIZE((PyArrayObject *)tmp);
	if (Nel != PyArray_SIZE((PyArrayObject *)optmp)) {
	    PyErr_SetString(PyExc_IndexError,"shape mismatch between array and mask"); 
	    Py_DECREF(opa); Py_DECREF(tmp); Py_DECREF(optmp);
	    return NULL;
	}
	
	/* count size of 1-d output array */
	Nout = 0;
	ptr = (unsigned char *)((PyArrayObject *)optmp)->data;
	for (i = 0; i < Nel; i++) {
	    if (*ptr != 0) Nout++;
	    ptr++;
	}
	
	/* construct output array */
	dims[0] = Nout;
	other = (PyArrayObject *)PyArray_FromDims 
	    (1, dims, ((PyArrayObject *)tmp)->descr->type_num);
	
	/* populate output array (other) */
	ptr = (unsigned char *)((PyArrayObject *)optmp)->data;
	inptr = ((PyArrayObject *)tmp)->data;
	optr = other->data;
	elsize = other->descr->elsize;
	for (i=0; i < Nel; i++) {
	    if (*ptr != 0) {
		memcpy(optr, inptr, elsize);
		optr += elsize;
	    }
	    ptr++;
	    inptr += elsize;
	}
    }
    else {
	other = (PyArrayObject *) PyArray_Take(tmp, optmp, 0);
    }
    Py_DECREF(tmp);
    Py_DECREF(optmp);
    Py_DECREF(opa);
    return other; 

}


/* Called when treating array object like a mapping -- called first from 
   Python when using a[object] */
static PyObject *scipy_array_subscript(PyArrayObject *self, PyObject *op) {
    int dimensions[MAX_DIMS], strides[MAX_DIMS];
    int nd, offset, i, elsize, typenum;
    unsigned char flag;
    PyArrayObject *other;
    
	
    if (PyInt_Check(op)) {
	i = PyInt_AsLong(op);
	if (i < 0 && self->nd > 0) i = i+self->dimensions[0]; 
	return scipy_array_item(self, i);
    }
    
    if (PyArray_Check(op)) {
	typenum = ((PyArrayObject*)op)->descr->type_num;
	flag = ((typenum == PyArray_INT) || (typenum == PyArray_LONG));
	flag |= ((typenum == PyArray_UBYTE));
	flag |= ((typenum == PyArray_SHORT) || (typenum == PyArray_SBYTE));
#ifdef PyArray_UNSIGNED_TYPES
	flag |= ((typenum == PyArray_UINT) || (typenum == PyArray_USHORT));
#endif
	if (flag) {
	    other = scipy_onearray_index(self, op, typenum);
	    if (other == NULL) return NULL;
	    return (PyObject *)other;
	}
    }
	
    if ((nd = scipy_parse_index(self, op, dimensions, strides, &offset)) 
	== -1) {
	return NULL;
    }
	
    if ((other = (PyArrayObject *)PyArray_FromDimsAndDataAndDescr(nd, 
								  dimensions,
								  self->descr,
								  self->data+offset)) == NULL) {
	return NULL;
    }
    memmove(other->strides, strides, sizeof(int)*other->nd);
    other->base = (PyObject *)self;
    Py_INCREF(self);
	
    elsize=other->descr->elsize;
    /* Check to see if other is CONTIGUOUS:  see if strides match 
       dimensions */
    for (i=other->nd-1; i>=0; i--) {
	if (other->strides[i] == elsize) {
	    elsize *= other->dimensions[i];
	} else {
	    break;
	}
    }
    if (i >= 0) other->flags &= ~CONTIGUOUS; 

    /* Maintain SAVESPACE flag on selection */
    if (self->flags & SAVESPACE) other->flags |= SAVESPACE;
	
    return (PyObject *)other;
}

static PyObject *scipy_array_subscript_nice(PyArrayObject *self, PyObject *op) {
    PyObject *ret;
       
    if ((ret = scipy_array_subscript(self, op)) == NULL) return NULL;
    if (PyArray_Check(ret)) return PyArray_Return((PyArrayObject *)ret);
    else return ret;
}


/* Similar to PyArray_PutMask but it doesn't check for contiguous ArrayObject self
   and uses UBYTE instead of LONG in the mask
 */
static PyObject *scipy_PyArray_PutMask(PyArrayObject *self, PyObject *mask0, 
				       PyObject* values0) {
    PyArrayObject  *mask, *values;
    int i, chunk, ni, max_item, nv;
    char *src, *dest;
    unsigned char *ptr;

    mask = NULL;
    values = NULL;
    max_item = PyArray_SIZE(self);
    dest = self->data;
    chunk = self->descr->elsize;

    if (!PyArray_ISCONTIGUOUS(((PyArrayObject *)mask0))) {
	mask = (PyArrayObject *)PyArray_ContiguousFromObject(mask0, PyArray_UBYTE, 0, 0);
	if (mask == NULL) goto fail;
    }
    else {
        mask = (PyArrayObject *)mask0;
	Py_INCREF(mask);
    }
    ni = PyArray_SIZE(mask);
    if (ni != max_item) {
	PyErr_SetString(PyExc_IndexError, "mask and data must be the same size.");
	goto fail;
    }

    values = (PyArrayObject *)PyArray_ContiguousFromObject(values0, 
							   self->descr->type, 0, 0);
    if (values == NULL) goto fail;
    nv = PyArray_SIZE(values);   /* zero if null array */
    ptr = (unsigned char *)mask->data;
    if (nv > 0) {
        for(i=0; i<ni; i++) {
            src = values->data + chunk * (i % nv);
            if (*ptr) {
                memmove(dest + i * chunk, src, chunk);
            }
	    ptr++;
        }
    }

    Py_XDECREF(values);
    Py_XDECREF(mask);
    Py_INCREF(Py_None);
    return Py_None;
	
 fail:
    Py_XDECREF(mask);
    Py_XDECREF(values);
    return NULL;
}


/* Another assignment hacked by using CopyObject.  */

static int scipy_array_ass_sub(PyArrayObject *self, PyObject *index, PyObject *op) {
    int ret;
    PyArrayObject *tmp;
    int typenum;
    unsigned char flag;
	
    if (op == NULL) {
	PyErr_SetString(PyExc_ValueError, 
			"Can't delete array elements.");
	return -1;
    }
	
    if (PyInt_Check(index)) 
	return scipy_array_ass_item(self, PyInt_AsLong(index), op);

    if (PyArray_Check(index)) {
	typenum = ((PyArrayObject*)index)->descr->type_num;
	/* if index is typecode 'b' -- unsigned byte then use putmask */
	if ( typenum == PyArray_UBYTE) {
	    if (!PyArray_ISCONTIGUOUS(self)) {
		ret = scipy_makecontiguous(self);
		if (ret == -1) return -1;
	    }
	    /* XXXX This will upcast index to long unnecessarily --- may need to write own XXXX */
	    tmp = (PyArrayObject *)scipy_PyArray_PutMask(self, index, op);
	    if (tmp == NULL) return -1;
	    Py_DECREF(tmp);
	    return 0;
	}
	flag = ((typenum == PyArray_INT) || (typenum == PyArray_LONG));
	flag |= ((typenum == PyArray_SHORT) || (typenum == PyArray_SBYTE));
#ifdef PyArray_UNSIGNED_TYPES
	flag |= ((typenum == PyArray_UINT) || (typenum == PyArray_USHORT));
#endif
	if (flag) {
	    /* put index */
	    if (!PyArray_ISCONTIGUOUS(self)) {
		ret = scipy_makecontiguous(self);
		if (ret == -1) return -1;
	    }
	    tmp = (PyArrayObject *)PyArray_Put((PyObject *)self, index, op);
	    if (tmp == NULL) return -1;
	    Py_DECREF(tmp);
	    return 0;
      }
    }

    if ((tmp = (PyArrayObject *)scipy_array_subscript(self, index)) == NULL)
	return -1; 
    ret = scipy_PyArray_CopyObject(tmp, op);
    Py_DECREF(tmp);
	
    return ret;
}
