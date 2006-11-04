
/* Should only be used if x is known to be an nd-array */
#define _ARET(x) PyArray_Return((PyArrayObject *)(x))

static PyObject *
array_take(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
	int dimension=MAX_DIMS;
	PyObject *indices;
        PyArrayObject *out=NULL;
        NPY_CLIPMODE mode=NPY_RAISE;
	static char *kwlist[] = {"indices", "axis", "out", "mode", NULL};

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O&O&O&", kwlist,
					 &indices, PyArray_AxisConverter,
					 &dimension,
                                         PyArray_OutputConverter,
                                         &out,
                                         PyArray_ClipmodeConverter,
                                         &mode))
		return NULL;

	return _ARET(PyArray_TakeFrom(self, indices, dimension, out, mode));
}

static PyObject *
array_fill(PyArrayObject *self, PyObject *args)
{
	PyObject *obj;
	if (!PyArg_ParseTuple(args, "O", &obj))
		return NULL;
	if (PyArray_FillWithScalar(self, obj) < 0) return NULL;
	Py_INCREF(Py_None);
	return Py_None;
}

static PyObject *
array_put(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
	PyObject *indices, *values;
        NPY_CLIPMODE mode=NPY_RAISE;
	static char *kwlist[] = {"indices", "values", "mode", NULL};

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|O&", kwlist,
					 &indices, &values,
                                         PyArray_ClipmodeConverter,
                                         &mode))
		return NULL;
	return PyArray_PutTo(self, values, indices, mode);
}

static PyObject *
array_reshape(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
        PyArray_Dims newshape;
        PyObject *ret;
	PyArray_ORDER order=PyArray_CORDER;
	int n;

	if (kwds != NULL) {
		PyObject *ref;
		ref = PyDict_GetItemString(kwds, "order");
                if (ref == NULL) {
                        PyErr_SetString(PyExc_TypeError, 
                                        "invalid keyword argument");
                        return NULL;
                }
		if ((PyArray_OrderConverter(ref, &order) == PY_FAIL))
			return NULL;
	}

	n = PyTuple_Size(args);
	if (n <= 1) {
                if (PyTuple_GET_ITEM(args, 0) == Py_None)
                        return PyArray_View(self, NULL, NULL);
		if (!PyArg_ParseTuple(args, "O&", PyArray_IntpConverter,
				      &newshape)) return NULL;
	}
        else {
		if (!PyArray_IntpConverter(args, &newshape)) {
			if (!PyErr_Occurred()) {
				PyErr_SetString(PyExc_TypeError,
						"invalid shape");
			}
			goto fail;
		}
	}
	ret = PyArray_Newshape(self, &newshape, order);
	PyDimMem_FREE(newshape.ptr);
        return ret;

 fail:
	PyDimMem_FREE(newshape.ptr);
	return NULL;
}

static PyObject *
array_squeeze(PyArrayObject *self, PyObject *args)
{
        if (!PyArg_ParseTuple(args, "")) return NULL;
        return PyArray_Squeeze(self);
}

static PyObject *
array_view(PyArrayObject *self, PyObject *args)
{
	PyObject *otype=NULL;
        PyArray_Descr *type=NULL;

	if (!PyArg_ParseTuple(args, "|O", &otype)) return NULL;

	if (otype) {
		if (PyType_Check(otype) &&			\
		    PyType_IsSubtype((PyTypeObject *)otype,
				     &PyArray_Type)) {
			return PyArray_View(self, NULL,
                                            (PyTypeObject *)otype);
                }
		else {
			if (PyArray_DescrConverter(otype, &type) == PY_FAIL)
				return NULL;
		}
	}
	return PyArray_View(self, type, NULL);
}

static PyObject *
array_argmax(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
	int axis=MAX_DIMS;
        PyArrayObject *out=NULL;
	static char *kwlist[] = {"axis", "out", NULL};

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O&O&", kwlist,
					 PyArray_AxisConverter,
					 &axis,
                                         PyArray_OutputConverter,
                                         &out))
		return NULL;

	return _ARET(PyArray_ArgMax(self, axis, out));
}

static PyObject *
array_argmin(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
	int axis=MAX_DIMS;
        PyArrayObject *out=NULL;
	static char *kwlist[] = {"axis", "out", NULL};

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O&O&", kwlist,
					 PyArray_AxisConverter,
					 &axis,
                                         PyArray_OutputConverter,
                                         &out))
		return NULL;

	return _ARET(PyArray_ArgMin(self, axis, out));
}

static PyObject *
array_max(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
	int axis=MAX_DIMS;
        PyArrayObject *out=NULL;
	static char *kwlist[] = {"axis", "out", NULL};

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O&O&", kwlist,
					 PyArray_AxisConverter,
					 &axis,
                                         PyArray_OutputConverter,
                                         &out))
		return NULL;

	return PyArray_Max(self, axis, out);
}

static PyObject *
array_ptp(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
	int axis=MAX_DIMS;
        PyArrayObject *out=NULL;
	static char *kwlist[] = {"axis", "out", NULL};

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O&O&", kwlist,
					 PyArray_AxisConverter,
					 &axis,
                                         PyArray_OutputConverter,
                                         &out))
		return NULL;

	return PyArray_Ptp(self, axis, out);
}


static PyObject *
array_min(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
	int axis=MAX_DIMS;
        PyArrayObject *out=NULL;
	static char *kwlist[] = {"axis", "out", NULL};

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O&O&", kwlist,
					 PyArray_AxisConverter,
					 &axis,
                                         PyArray_OutputConverter,
                                         &out))
		return NULL;

	return PyArray_Min(self, axis, out);
}

static PyObject *
array_swapaxes(PyArrayObject *self, PyObject *args)
{
	int axis1, axis2;

	if (!PyArg_ParseTuple(args, "ii", &axis1, &axis2)) return NULL;

	return PyArray_SwapAxes(self, axis1, axis2);
}


/* steals typed reference */
/*OBJECT_API
 Get a subset of bytes from each element of the array
*/
static PyObject *
PyArray_GetField(PyArrayObject *self, PyArray_Descr *typed, int offset)
{
	PyObject *ret=NULL;

	if (offset < 0 || (offset + typed->elsize) > self->descr->elsize) {
		PyErr_Format(PyExc_ValueError,
			     "Need 0 <= offset <= %d for requested type "  \
			     "but received offset = %d",
			     self->descr->elsize-typed->elsize, offset);
		Py_DECREF(typed);
		return NULL;
	}
	ret = PyArray_NewFromDescr(self->ob_type,
				   typed,
				   self->nd, self->dimensions,
				   self->strides,
				   self->data + offset,
				   self->flags, (PyObject *)self);
	if (ret == NULL) return NULL;
	Py_INCREF(self);
	((PyArrayObject *)ret)->base = (PyObject *)self;

	PyArray_UpdateFlags((PyArrayObject *)ret, UPDATE_ALL);
	return ret;
}

static PyObject *
array_getfield(PyArrayObject *self, PyObject *args, PyObject *kwds)
{

        PyArray_Descr *dtype;
	int offset = 0;
	static char *kwlist[] = {"dtype", "offset", 0};

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|i", kwlist,
					 PyArray_DescrConverter,
					 &dtype, &offset)) return NULL;

	return _ARET(PyArray_GetField(self, dtype, offset));
}


/*OBJECT_API
 Set a subset of bytes from each element of the array
*/
static int
PyArray_SetField(PyArrayObject *self, PyArray_Descr *dtype,
		 int offset, PyObject *val)
{
	PyObject *ret=NULL;
	int retval = 0;

	if (offset < 0 || (offset + dtype->elsize) > self->descr->elsize) {
		PyErr_Format(PyExc_ValueError,
			     "Need 0 <= offset <= %d for requested type "  \
			     "but received offset = %d",
			     self->descr->elsize-dtype->elsize, offset);
		Py_DECREF(dtype);
		return -1;
	}
	ret = PyArray_NewFromDescr(self->ob_type,
				   dtype, self->nd, self->dimensions,
				   self->strides, self->data + offset,
				   self->flags, (PyObject *)self);
	if (ret == NULL) return -1;
	Py_INCREF(self);
	((PyArrayObject *)ret)->base = (PyObject *)self;

	PyArray_UpdateFlags((PyArrayObject *)ret, UPDATE_ALL);
	retval = PyArray_CopyObject((PyArrayObject *)ret, val);
	Py_DECREF(ret);
	return retval;
}

static PyObject *
array_setfield(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
        PyArray_Descr *dtype;
	int offset = 0;
	PyObject *value;
	static char *kwlist[] = {"value", "dtype", "offset", 0};

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO&|i", kwlist,
					 &value, PyArray_DescrConverter,
					 &dtype, &offset)) return NULL;

	if (PyArray_SetField(self, dtype, offset, value) < 0)
		return NULL;
	Py_INCREF(Py_None);
	return Py_None;
}

/* This doesn't change the descriptor just the actual data...
 */

/*OBJECT_API*/
static PyObject *
PyArray_Byteswap(PyArrayObject *self, Bool inplace)
{
        PyArrayObject *ret;
	intp size;
	PyArray_CopySwapNFunc *copyswapn;
	PyArrayIterObject *it;

        copyswapn = self->descr->f->copyswapn;
	if (inplace) {
                if (!PyArray_ISWRITEABLE(self)) {
                        PyErr_SetString(PyExc_RuntimeError,
                                        "Cannot byte-swap in-place on a " \
                                        "read-only array");
                        return NULL;
                }
		size = PyArray_SIZE(self);
		if (PyArray_ISONESEGMENT(self)) {
			copyswapn(self->data, self->descr->elsize, NULL, -1, size, 1, self);
		}
		else { /* Use iterator */
                        int axis = -1;
                        intp stride;
			it = (PyArrayIterObject *)                      \
				PyArray_IterAllButAxis((PyObject *)self, &axis);
                        stride = self->strides[axis];
                        size = self->dimensions[axis];
			while (it->index < it->size) {
                                copyswapn(it->dataptr, stride, NULL, -1, size, 1, self);
				PyArray_ITER_NEXT(it);
			}
			Py_DECREF(it);
		}

		Py_INCREF(self);
		return (PyObject *)self;
	}
	else {
                PyObject *new;
		if ((ret = (PyArrayObject *)PyArray_NewCopy(self,-1)) == NULL)
			return NULL;
                new = PyArray_Byteswap(ret, TRUE);
                Py_DECREF(new);
		return (PyObject *)ret;
	}
}


static PyObject *
array_byteswap(PyArrayObject *self, PyObject *args)
{
	Bool inplace=FALSE;

	if (!PyArg_ParseTuple(args, "|O&", PyArray_BoolConverter, &inplace))
		return NULL;

	return PyArray_Byteswap(self, inplace);
}

static PyObject *
array_tolist(PyArrayObject *self, PyObject *args)
{
        if (!PyArg_ParseTuple(args, "")) return NULL;
        return PyArray_ToList(self);
}


static PyObject *
array_tostring(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
	NPY_ORDER order=NPY_CORDER;
	static char *kwlist[] = {"order", NULL};

        if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O&", kwlist,
					 PyArray_OrderConverter,
					 &order)) return NULL;
        return PyArray_ToString(self, order);
}


/* This should grow an order= keyword to be consistent
 */

static PyObject *
array_tofile(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
	int ret;
        PyObject *file;
	FILE *fd;
        char *sep="";
	char *format="";
	static char *kwlist[] = {"file", "sep", "format", NULL};

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|ss", kwlist,
                                         &file, &sep, &format)) return NULL;

	if (PyString_Check(file) || PyUnicode_Check(file)) {
		file = PyObject_CallFunction((PyObject *)&PyFile_Type,
					     "Os", file, "wb");
		if (file==NULL) return NULL;
	}
	else {
		Py_INCREF(file);
	}
	fd = PyFile_AsFile(file);
	if (fd == NULL) {
		PyErr_SetString(PyExc_IOError, "first argument must be a " \
				"string or open file");
		Py_DECREF(file);
		return NULL;
	}
	ret = PyArray_ToFile(self, fd, sep, format);
	Py_DECREF(file);
	if (ret < 0) return NULL;
        Py_INCREF(Py_None);
        return Py_None;
}


static PyObject *
array_toscalar(PyArrayObject *self, PyObject *args) {
        int n, nd;
        n = PyTuple_GET_SIZE(args);
        
        if (n==1) {
                PyObject *obj;
                obj = PyTuple_GET_ITEM(args, 0);
                if (PyTuple_Check(obj)) {
                        args = obj;
                        n = PyTuple_GET_SIZE(args);
                }
        }
        
        if (n==0) {
                if (self->nd == 0 || PyArray_SIZE(self) == 1)
                        return self->descr->f->getitem(self->data, self);
                else {
                        PyErr_SetString(PyExc_ValueError, 
                                        "can only convert an array "    \
                                        " of size 1 to a Python scalar");
                        return NULL;
                }
        }
        else if (n != self->nd && (n > 1 || self->nd==0)) {
                PyErr_SetString(PyExc_ValueError, 
                                "incorrect number of indices for "      \
                                "array");
                return NULL;
        }
        else if (n==1) { /* allows for flat getting as well as 1-d case */
                intp value, loc, index, factor;
                intp factors[MAX_DIMS];
                value = PyArray_PyIntAsIntp(PyTuple_GET_ITEM(args, 0));
                if (error_converting(value)) {
                        PyErr_SetString(PyExc_ValueError, "invalid integer");
                        return NULL;
                }
                if (value >= PyArray_SIZE(self)) {
                        PyErr_SetString(PyExc_ValueError, 
                                        "index out of bounds");
                        return NULL;
                }
                if (self->nd == 1) {
                        value *= self->strides[0];
                        return self->descr->f->getitem(self->data + value,
                                                       self);
                }
                nd = self->nd;
                factor = 1;
                while (nd--) {
                        factors[nd] = factor;
                        factor *= self->dimensions[nd];
                }
                loc = 0;
                for (nd=0; nd < self->nd; nd++) {
                        index = value / factors[nd];
                        value = value % factors[nd];
                        loc += self->strides[nd]*index;
                }
                
                return self->descr->f->getitem(self->data + loc,
                                               self);
                
        }
        else {
                intp loc, index[MAX_DIMS];
                nd = PyArray_IntpFromSequence(args, index, MAX_DIMS);
                if (nd < n) return NULL;
                loc = 0;
                while (nd--) {
                        if (index[nd] < 0) 
                                index[nd] += self->dimensions[nd];
                        if (index[nd] < 0 || 
                            index[nd] >= self->dimensions[nd]) {
                                PyErr_SetString(PyExc_ValueError, 
                                                "index out of bounds");
                                return NULL;
                        }
                        loc += self->strides[nd]*index[nd];
                }
                return self->descr->f->getitem(self->data + loc, self);
        }
}

static PyObject *
array_setscalar(PyArrayObject *self, PyObject *args) {
        int n, nd;
        int ret = -1;
        PyObject *obj;
        n = PyTuple_GET_SIZE(args)-1;
        
        if (n < 0) {
                PyErr_SetString(PyExc_ValueError, 
                                "itemset must have at least one argument");
                return NULL;
        }
        obj = PyTuple_GET_ITEM(args, n);
        if (n==0) {
                if (self->nd == 0 || PyArray_SIZE(self) == 1) {
                        ret = self->descr->f->setitem(obj, self->data, self);
                }                
                else {
                        PyErr_SetString(PyExc_ValueError, 
                                        "can only place a scalar for an "
                                        " array of size 1");
                        return NULL;
                }
        }
        else if (n != self->nd && (n > 1 || self->nd==0)) {
                PyErr_SetString(PyExc_ValueError, 
                                "incorrect number of indices for "      \
                                "array");
                return NULL;
        }
        else if (n==1) { /* allows for flat setting as well as 1-d case */
                intp value, loc, index, factor;
                intp factors[MAX_DIMS];
                PyObject *indobj;

                indobj = PyTuple_GET_ITEM(args, 0);
                if (PyTuple_Check(indobj)) {
                        PyObject *res;
                        PyObject *newargs;
                        PyObject *tmp;
                        int i, nn;
                        nn = PyTuple_GET_SIZE(indobj);
                        newargs = PyTuple_New(nn+1);
                        Py_INCREF(obj);
                        for (i=0; i<nn; i++) {
                                tmp = PyTuple_GET_ITEM(indobj, i);
                                Py_INCREF(tmp);
                                PyTuple_SET_ITEM(newargs, i, tmp);
                        }
                        PyTuple_SET_ITEM(newargs, nn, obj);
                        /* Call with a converted set of arguments */
                        res = array_setscalar(self, newargs);
                        Py_DECREF(newargs);
                        return res;
                }
                value = PyArray_PyIntAsIntp(indobj);
                if (error_converting(value)) {
                        PyErr_SetString(PyExc_ValueError, "invalid integer");
                        return NULL;
                }
                if (value >= PyArray_SIZE(self)) {
                        PyErr_SetString(PyExc_ValueError, 
                                        "index out of bounds");
                        return NULL;
                }
                if (self->nd == 1) {
                        value *= self->strides[0];
                        ret = self->descr->f->setitem(obj, self->data + value, 
                                                      self);
                        goto finish;
                }
                nd = self->nd;
                factor = 1;
                while (nd--) {
                        factors[nd] = factor;
                        factor *= self->dimensions[nd];
                }
                loc = 0;
                for (nd=0; nd < self->nd; nd++) {
                        index = value / factors[nd];
                        value = value % factors[nd];
                        loc += self->strides[nd]*index;
                }
                
                ret = self->descr->f->setitem(obj, self->data + loc, self);
        }
        else {
                intp loc, index[MAX_DIMS];
                PyObject *tupargs;
                tupargs = PyTuple_GetSlice(args, 1, n+1);
                nd = PyArray_IntpFromSequence(tupargs, index, MAX_DIMS);
                Py_DECREF(tupargs);
                if (nd < n) return NULL;
                loc = 0;
                while (nd--) {
                        if (index[nd] < 0) 
                                index[nd] += self->dimensions[nd];
                        if (index[nd] < 0 || 
                            index[nd] >= self->dimensions[nd]) {
                                PyErr_SetString(PyExc_ValueError, 
                                                "index out of bounds");
                                return NULL;
                        }
                        loc += self->strides[nd]*index[nd];
                }
                ret = self->descr->f->setitem(obj, self->data + loc, self);
        }

 finish:
        if (ret < 0) return NULL;
        Py_INCREF(Py_None);
        return Py_None;
}


static PyObject *
array_cast(PyArrayObject *self, PyObject *args)
{
	PyArray_Descr *descr=NULL;
	PyObject *obj;

        if (!PyArg_ParseTuple(args, "O&", PyArray_DescrConverter,
			      &descr)) return NULL;

	if (descr == self->descr) {
		obj = _ARET(PyArray_NewCopy(self,0));
		Py_XDECREF(descr);
		return obj;
	}
	if (descr->names != NULL) {
		return PyArray_FromArray(self, descr, NPY_FORCECAST);
	}
	return PyArray_CastToType(self, descr, 0);
}

/* default sub-type implementation */


static PyObject *
array_wraparray(PyArrayObject *self, PyObject *args)
{
	PyObject *arr;
	PyObject *ret;

	if (PyTuple_Size(args) < 1) {
		PyErr_SetString(PyExc_TypeError,
				"only accepts 1 argument");
		return NULL;
	}
	arr = PyTuple_GET_ITEM(args, 0);
	if (!PyArray_Check(arr)) {
		PyErr_SetString(PyExc_TypeError,
				"can only be called with ndarray object");
		return NULL;
	}

	Py_INCREF(PyArray_DESCR(arr));
	ret = PyArray_NewFromDescr(self->ob_type,
				   PyArray_DESCR(arr),
				   PyArray_NDIM(arr),
				   PyArray_DIMS(arr),
				   PyArray_STRIDES(arr), PyArray_DATA(arr),
				   PyArray_FLAGS(arr), (PyObject *)self);
	if (ret == NULL) return NULL;
	Py_INCREF(arr);
	PyArray_BASE(ret) = arr;
	return ret;
}


static PyObject *
array_getarray(PyArrayObject *self, PyObject *args)
{
	PyArray_Descr *newtype=NULL;
	PyObject *ret;

	if (!PyArg_ParseTuple(args, "|O&", PyArray_DescrConverter,
			      &newtype)) return NULL;

	/* convert to PyArray_Type */
	if (!PyArray_CheckExact(self)) {
		PyObject *new;
		PyTypeObject *subtype = &PyArray_Type;

		if (!PyType_IsSubtype(self->ob_type, &PyArray_Type)) {
			subtype = &PyArray_Type;
		}

		Py_INCREF(PyArray_DESCR(self));
		new = PyArray_NewFromDescr(subtype,
					   PyArray_DESCR(self),
					   PyArray_NDIM(self),
					   PyArray_DIMS(self),
					   PyArray_STRIDES(self),
					   PyArray_DATA(self),
					   PyArray_FLAGS(self), NULL);
		if (new == NULL) return NULL;
		Py_INCREF(self);
		PyArray_BASE(new) = (PyObject *)self;
		self = (PyArrayObject *)new;
	}
	else {
		Py_INCREF(self);
	}

	if ((newtype == NULL) || \
	    PyArray_EquivTypes(self->descr, newtype)) {
		return (PyObject *)self;
	}
	else {
		ret = PyArray_CastToType(self, newtype, 0);
		Py_DECREF(self);
		return ret;
	}
}


static PyObject *
array_copy(PyArrayObject *self, PyObject *args)
{
	PyArray_ORDER fortran=PyArray_CORDER;
        if (!PyArg_ParseTuple(args, "|O&", PyArray_OrderConverter,
			      &fortran)) return NULL;

        return PyArray_NewCopy(self, fortran);
}


static PyObject *
array_resize(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
        PyArray_Dims newshape;
        PyObject *ret;
	int n;
	int refcheck = 1;
	PyArray_ORDER fortran=PyArray_ANYORDER;

	if (kwds != NULL) {
		PyObject *ref;
		ref = PyDict_GetItemString(kwds, "refcheck");
		if (ref) {
			refcheck = PyInt_AsLong(ref);
			if (refcheck==-1 && PyErr_Occurred()) {
				return NULL;
			}
		}
		ref = PyDict_GetItemString(kwds, "order");
		if (ref != NULL ||
		    (PyArray_OrderConverter(ref, &fortran) == PY_FAIL))
			return NULL;
	}
	n = PyTuple_Size(args);
	if (n <= 1) {
                if (PyTuple_GET_ITEM(args, 0) == Py_None) {
                        Py_INCREF(Py_None);
                        return Py_None;
                }
		if (!PyArg_ParseTuple(args, "O&", PyArray_IntpConverter,
				      &newshape)) return NULL;
	}
        else {
		if (!PyArray_IntpConverter(args, &newshape)) {
			if (!PyErr_Occurred()) {
				PyErr_SetString(PyExc_TypeError,
						"invalid shape");
			}
			return NULL;
		}
	}
	ret = PyArray_Resize(self, &newshape, refcheck, fortran);
        PyDimMem_FREE(newshape.ptr);
        if (ret == NULL) return NULL;
	Py_DECREF(ret);
	Py_INCREF(Py_None);
	return Py_None;
}

static PyObject *
array_repeat(PyArrayObject *self, PyObject *args, PyObject *kwds) {
	PyObject *repeats;
	int axis=MAX_DIMS;
	static char *kwlist[] = {"repeats", "axis", NULL};

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O&", kwlist,
					 &repeats, PyArray_AxisConverter,
					 &axis)) return NULL;

	return _ARET(PyArray_Repeat(self, repeats, axis));
}

static PyObject *
array_choose(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
	PyObject *choices;
	int n;
        PyArrayObject *out=NULL;
        NPY_CLIPMODE clipmode=NPY_RAISE;

	n = PyTuple_Size(args);
	if (n <= 1) {
		if (!PyArg_ParseTuple(args, "O", &choices))
			return NULL;
	}
        else {
		choices = args;
	}
        if (kwds && PyDict_Check(kwds)) {
                if (PyArray_OutputConverter(PyDict_GetItemString(kwds,
                                                                 "out"),
                                            &out) == PY_FAIL)
                        return NULL;
                if (PyArray_ClipmodeConverter(PyDict_GetItemString(kwds,
                                                                   "mode"),
                                              &clipmode) == PY_FAIL)
                        return NULL;
        }

	return _ARET(PyArray_Choose(self, choices, out, clipmode));
}

static PyObject *
array_sort(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
	int axis=-1;
	int val;
	PyArray_SORTKIND which=PyArray_QUICKSORT;
        PyObject *order=NULL;
        PyArray_Descr *saved=NULL;
        PyArray_Descr *newd;
        static char *kwlist[] = {"axis", "kind", "order", NULL};
        
        if (!PyArg_ParseTupleAndKeywords(args, kwds, "|iO&O", kwlist, &axis,
                                         PyArray_SortkindConverter, &which,
                                         &order))
                return NULL;

        if (order != NULL) {
                PyObject *new_name;
                saved = self->descr;
                if (saved->names == NULL) {
                        PyErr_SetString(PyExc_ValueError, "Cannot specify " \
                                        "order with no fields.");
                        return NULL;
                }
                new_name = PyObject_CallMethod(_numpy_internal, "_newnames",
                                               "OO", saved, order);
                if (new_name == NULL) return NULL;
                newd = PyArray_DescrNew(saved);
                newd->names = new_name;
                self->descr = newd;
        }

        val = PyArray_Sort(self, axis, which);
        if (order != NULL) {
                Py_XDECREF(self->descr);
                self->descr = saved;
        }
        if (val < 0) return NULL;
	Py_INCREF(Py_None);
	return Py_None;
}

static PyObject *
array_argsort(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
	int axis=-1;
	PyArray_SORTKIND which=PyArray_QUICKSORT;
	static char *kwlist[] = {"axis", "kind", NULL};

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "|iO&", kwlist, &axis,
					 PyArray_SortkindConverter, &which))
		return NULL;

	return _ARET(PyArray_ArgSort(self, axis, which));
}

static PyObject *
array_searchsorted(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
        static char *kwlist[] = {"keys", "side", NULL};
	PyObject *keys;
        NPY_SEARCHSIDE side = NPY_SEARCHLEFT;

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O&:searchsorted",
                                         kwlist, &keys,
                                         PyArray_SearchsideConverter, &side))
                return NULL;

	return _ARET(PyArray_SearchSorted(self, keys, side));
}

static void
_deepcopy_call(char *iptr, char *optr, PyArray_Descr *dtype,
	       PyObject *deepcopy, PyObject *visit)
{
	if (!PyDataType_REFCHK(dtype)) return;
	else if (PyDescr_HASFIELDS(dtype)) {
                PyObject *key, *value, *title=NULL;
                PyArray_Descr *new;
                int offset;
                Py_ssize_t pos=0;
                while (PyDict_Next(dtype->fields, &pos, &key, &value)) {
 			if (!PyArg_ParseTuple(value, "Oi|O", &new, &offset,
					      &title)) return;
                        _deepcopy_call(iptr + offset, optr + offset, new,
				       deepcopy, visit);
                }
        }
        else {
		PyObject **itemp, **otemp;
		PyObject *res;
		itemp = (PyObject **)iptr;
		otemp = (PyObject **)optr;
		Py_XINCREF(*itemp);
		/* call deepcopy on this argument */
		res = PyObject_CallFunctionObjArgs(deepcopy,
						   *itemp, visit, NULL);
		Py_XDECREF(*itemp);
		Py_XDECREF(*otemp);
		*otemp = res;
	}

}


static PyObject *
array_deepcopy(PyArrayObject *self, PyObject *args)
{
        PyObject* visit;
	char *optr;
        PyArrayIterObject *it;
        PyObject *copy, *ret, *deepcopy;

        if (!PyArg_ParseTuple(args, "O", &visit)) return NULL;
        ret = PyArray_Copy(self);
        if (PyDataType_REFCHK(self->descr)) {
                copy = PyImport_ImportModule("copy");
                if (copy == NULL) return NULL;
                deepcopy = PyObject_GetAttrString(copy, "deepcopy");
                Py_DECREF(copy);
                if (deepcopy == NULL) return NULL;
                it = (PyArrayIterObject *)PyArray_IterNew((PyObject *)self);
                if (it == NULL) {Py_DECREF(deepcopy); return NULL;}
                optr = PyArray_DATA(ret);
                while(it->index < it->size) {
			_deepcopy_call(it->dataptr, optr, self->descr,
				       deepcopy, visit);
			optr += self->descr->elsize;
                        PyArray_ITER_NEXT(it);
                }
                Py_DECREF(deepcopy);
                Py_DECREF(it);
        }
        return _ARET(ret);
}

/* Convert Array to flat list (using getitem) */
static PyObject *
_getlist_pkl(PyArrayObject *self)
{
	PyObject *theobject;
	PyArrayIterObject *iter=NULL;
	PyObject *list;
	PyArray_GetItemFunc *getitem;

	getitem = self->descr->f->getitem;
	iter = (PyArrayIterObject *)PyArray_IterNew((PyObject *)self);
	if (iter == NULL) return NULL;
	list = PyList_New(iter->size);
	if (list == NULL) {Py_DECREF(iter); return NULL;}
	while (iter->index < iter->size) {
		theobject = getitem(iter->dataptr, self);
		PyList_SET_ITEM(list, (int) iter->index, theobject);
		PyArray_ITER_NEXT(iter);
	}
	Py_DECREF(iter);
	return list;
}

static int
_setlist_pkl(PyArrayObject *self, PyObject *list)
{
	PyObject *theobject;
	PyArrayIterObject *iter=NULL;
	PyArray_SetItemFunc *setitem;

	setitem = self->descr->f->setitem;
	iter = (PyArrayIterObject *)PyArray_IterNew((PyObject *)self);
	if (iter == NULL) return -1;
	while(iter->index < iter->size) {
		theobject = PyList_GET_ITEM(list, (int) iter->index);
		setitem(theobject, iter->dataptr, self);
		PyArray_ITER_NEXT(iter);
	}
	Py_XDECREF(iter);
	return 0;
}


static PyObject *
array_reduce(PyArrayObject *self, PyObject *args)
{
        /* version number of this pickle type. Increment if we need to
           change the format. Be sure to handle the old versions in
           array_setstate. */
        const int version = 1;
	PyObject *ret=NULL, *state=NULL, *obj=NULL, *mod=NULL;
	PyObject *mybool, *thestr=NULL;
	PyArray_Descr *descr;

	/* Return a tuple of (callable object, arguments, object's state) */
	/*  We will put everything in the object's state, so that on UnPickle
	    it can use the string object as memory without a copy */

	ret = PyTuple_New(3);
	if (ret == NULL) return NULL;
	mod = PyImport_ImportModule("numpy.core.multiarray");
	if (mod == NULL) {Py_DECREF(ret); return NULL;}
	obj = PyObject_GetAttrString(mod, "_reconstruct");
	Py_DECREF(mod);
	PyTuple_SET_ITEM(ret, 0, obj);
	PyTuple_SET_ITEM(ret, 1,
			 Py_BuildValue("ONc",
				       (PyObject *)self->ob_type,
				       Py_BuildValue("(N)",
						     PyInt_FromLong(0)),
				       /* dummy data-type */
				       'b'));

	/* Now fill in object's state.  This is a tuple with
	   5 arguments

           1) an integer with the pickle version.
	   2) a Tuple giving the shape
	   3) a PyArray_Descr Object (with correct bytorder set)
	   4) a Bool stating if Fortran or not
	   5) a Python object representing the data (a string, or
	        a list or any user-defined object).

	   Notice because Python does not describe a mechanism to write
	   raw data to the pickle, this performs a copy to a string first
	*/

	state = PyTuple_New(5);
	if (state == NULL) {
		Py_DECREF(ret); return NULL;
	}
        PyTuple_SET_ITEM(state, 0, PyInt_FromLong(version));
	PyTuple_SET_ITEM(state, 1, PyObject_GetAttrString((PyObject *)self,
							  "shape"));
	descr = self->descr;
	Py_INCREF(descr);
	PyTuple_SET_ITEM(state, 2, (PyObject *)descr);
	mybool = (PyArray_ISFORTRAN(self) ? Py_True : Py_False);
	Py_INCREF(mybool);
	PyTuple_SET_ITEM(state, 3, mybool);
	if (PyDataType_FLAGCHK(self->descr, NPY_LIST_PICKLE)) {
		thestr = _getlist_pkl(self);
	}
	else {
                thestr = PyArray_ToString(self, NPY_ANYORDER);
	}
	if (thestr == NULL) {
		Py_DECREF(ret);
		Py_DECREF(state);
		return NULL;
	}
	PyTuple_SET_ITEM(state, 4, thestr);
	PyTuple_SET_ITEM(ret, 2, state);
	return ret;
}



static size_t _array_fill_strides(intp *, intp *, int, size_t, int, int *);

static int _IsAligned(PyArrayObject *);

static PyArray_Descr * _array_typedescr_fromstr(char *);

static PyObject *
array_setstate(PyArrayObject *self, PyObject *args)
{
	PyObject *shape;
	PyArray_Descr *typecode;
        int version = 1;
	int fortran;
	PyObject *rawdata;
	char *datastr;
	Py_ssize_t len;
	intp size, dimensions[MAX_DIMS];
	int nd;

	/* This will free any memory associated with a and
	   use the string in setstate as the (writeable) memory.
	*/
	if (!PyArg_ParseTuple(args, "(iO!O!iO)", &version, &PyTuple_Type,
			      &shape, &PyArrayDescr_Type, &typecode,
			      &fortran, &rawdata)) {
            PyErr_Clear();
            version = 0;
	    if (!PyArg_ParseTuple(args, "(O!O!iO)", &PyTuple_Type,
			      &shape, &PyArrayDescr_Type, &typecode,
			      &fortran, &rawdata)) {
		return NULL;
            }
        }

        /* If we ever need another pickle format, increment the version
           number. But we should still be able to handle the old versions.
           We've only got one right now. */
        if (version != 1 && version != 0) {
            PyErr_Format(PyExc_ValueError,
                         "can't handle version %d of numpy.ndarray pickle",
                         version);
            return NULL;
        }

	Py_XDECREF(self->descr);
	self->descr = typecode;
	Py_INCREF(typecode);
	nd = PyArray_IntpFromSequence(shape, dimensions, MAX_DIMS);
	if (nd < 0) return NULL;
	size = PyArray_MultiplyList(dimensions, nd);
	if (self->descr->elsize == 0) {
		PyErr_SetString(PyExc_ValueError, "Invalid data-type size.");
		return NULL;
	}
	if (size < 0 || size > MAX_INTP / self->descr->elsize) {
		PyErr_NoMemory();
		return NULL;
	}

	if (PyDataType_FLAGCHK(typecode, NPY_LIST_PICKLE)) {
		if (!PyList_Check(rawdata)) {
			PyErr_SetString(PyExc_TypeError,
					"object pickle not returning list");
			return NULL;
		}
	}
	else {
		if (!PyString_Check(rawdata)) {
			PyErr_SetString(PyExc_TypeError,
					"pickle not returning string");
			return NULL;
		}

		if (PyString_AsStringAndSize(rawdata, &datastr, &len))
			return NULL;

		if ((len != (self->descr->elsize * size))) {
			PyErr_SetString(PyExc_ValueError,
					"buffer size does not"	\
					" match array size");
			return NULL;
		}
	}

        if ((self->flags & OWNDATA)) {
		if (self->data != NULL)
			PyDataMem_FREE(self->data);
		self->flags &= ~OWNDATA;
        }
	Py_XDECREF(self->base);

	self->flags &= ~UPDATEIFCOPY;

        if (self->dimensions != NULL) {
                PyDimMem_FREE(self->dimensions);
		self->dimensions = NULL;
	}

	self->flags = DEFAULT;

	self->nd = nd;

	if (nd > 0) {
		self->dimensions = PyDimMem_NEW(nd * 2);
		self->strides = self->dimensions + nd;
		memcpy(self->dimensions, dimensions, sizeof(intp)*nd);
		(void) _array_fill_strides(self->strides, dimensions, nd,
					   (size_t) self->descr->elsize,
                                           (fortran ? FORTRAN : CONTIGUOUS),
					   &(self->flags));
	}

	if (!PyDataType_FLAGCHK(typecode, NPY_LIST_PICKLE)) {
                int swap=!PyArray_ISNOTSWAPPED(self);
		self->data = datastr;
		if (!_IsAligned(self) || swap) {
			intp num = PyArray_NBYTES(self);
			self->data = PyDataMem_NEW(num);
			if (self->data == NULL) {
				self->nd = 0;
				PyDimMem_FREE(self->dimensions);
				return PyErr_NoMemory();
			}
                        if (swap) { /* byte-swap on pickle-read */
				intp numels = num / self->descr->elsize;
                                self->descr->f->copyswapn(self->data, self->descr->elsize,
                                                          datastr, self->descr->elsize,
                                                          numels, 1, self);
				if (!PyArray_ISEXTENDED(self)) {
					self->descr = PyArray_DescrFromType(self->descr->type_num);
				}
				else {
					self->descr = PyArray_DescrNew(typecode);
					if (self->descr->byteorder == PyArray_BIG)
						self->descr->byteorder = PyArray_LITTLE;
					else if (self->descr->byteorder == PyArray_LITTLE)
						self->descr->byteorder = PyArray_BIG;
				}
				Py_DECREF(typecode);
                        }
                        else {
                                memcpy(self->data, datastr, num);
                        }
			self->flags |= OWNDATA;
			self->base = NULL;
		}
		else {
			self->base = rawdata;
			Py_INCREF(self->base);
		}
	}
	else {
		self->data = PyDataMem_NEW(PyArray_NBYTES(self));
		if (self->data == NULL) {
			self->nd = 0;
			self->data = PyDataMem_NEW(self->descr->elsize);
			if (self->dimensions) PyDimMem_FREE(self->dimensions);
			return PyErr_NoMemory();
		}
		if (PyDataType_FLAGCHK(self->descr, NPY_NEEDS_INIT))
                        memset(self->data, 0, PyArray_NBYTES(self));
		self->flags |= OWNDATA;
		self->base = NULL;
		if (_setlist_pkl(self, rawdata) < 0)
			return NULL;
	}

	PyArray_UpdateFlags(self, UPDATE_ALL);

	Py_INCREF(Py_None);
	return Py_None;
}

/*OBJECT_API*/
static int
PyArray_Dump(PyObject *self, PyObject *file, int protocol)
{
	PyObject *cpick=NULL;
	PyObject *ret;
	if (protocol < 0) protocol = 2;

	cpick = PyImport_ImportModule("cPickle");
	if (cpick==NULL) return -1;

	if PyString_Check(file) {
		file = PyFile_FromString(PyString_AS_STRING(file), "wb");
		if (file==NULL) return -1;
	}
	else Py_INCREF(file);
	ret = PyObject_CallMethod(cpick, "dump", "OOi", self,
				  file, protocol);
	Py_XDECREF(ret);
	Py_DECREF(file);
	Py_DECREF(cpick);
	if (PyErr_Occurred()) return -1;
	return 0;
}

/*OBJECT_API*/
static PyObject *
PyArray_Dumps(PyObject *self, int protocol)
{
	PyObject *cpick=NULL;
	PyObject *ret;
	if (protocol < 0) protocol = 2;

	cpick = PyImport_ImportModule("cPickle");
	if (cpick==NULL) return NULL;
	ret = PyObject_CallMethod(cpick, "dumps", "Oi", self, protocol);
	Py_DECREF(cpick);
	return ret;
}


static PyObject *
array_dump(PyArrayObject *self, PyObject *args)
{
	PyObject *file=NULL;
	int ret;

	if (!PyArg_ParseTuple(args, "O", &file))
		return NULL;
	ret = PyArray_Dump((PyObject *)self, file, 2);
	if (ret < 0) return NULL;
	Py_INCREF(Py_None);
	return Py_None;
}


static PyObject *
array_dumps(PyArrayObject *self, PyObject *args)
{
	if (!PyArg_ParseTuple(args, ""))
		return NULL;
	return PyArray_Dumps((PyObject *)self, 2);
}


static PyObject *
array_transpose(PyArrayObject *self, PyObject *args)
{
	PyObject *shape=Py_None;
	int n;
	PyArray_Dims permute;
	PyObject *ret;

	n = PyTuple_Size(args);
	if (n > 1) shape = args;
	else if (n == 1) shape = PyTuple_GET_ITEM(args, 0);

	if (shape == Py_None)
		ret = PyArray_Transpose(self, NULL);
	else {
		if (!PyArray_IntpConverter(shape, &permute)) return NULL;
		ret = PyArray_Transpose(self, &permute);
		PyDimMem_FREE(permute.ptr);
	}

	return ret;
}

/* Return typenumber from dtype2 unless it is NULL, then return 
   NPY_DOUBLE if dtype1->type_num is integer or bool
   and dtype1->type_num otherwise. 
*/
static int
_get_type_num_double(PyArray_Descr *dtype1, PyArray_Descr *dtype2)
{
        if (dtype2 != NULL)
                return dtype2->type_num;
        
        /* For integer or bool data-types */
        if (dtype1->type_num < NPY_FLOAT) {
                return NPY_DOUBLE;
        }
        else {
                return dtype1->type_num;
        }
}

#define _CHKTYPENUM(typ) ((typ) ? (typ)->type_num : PyArray_NOTYPE)

static PyObject *
array_mean(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
	int axis=MAX_DIMS;
	PyArray_Descr *dtype=NULL;
        PyArrayObject *out=NULL;
        int num;
	static char *kwlist[] = {"axis", "dtype", "out", NULL};

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O&O&O&", kwlist,
					 PyArray_AxisConverter,
					 &axis, PyArray_DescrConverter2,
					 &dtype,
                                         PyArray_OutputConverter,
                                         &out)) return NULL;        

        num = _get_type_num_double(self->descr, dtype);
	return PyArray_Mean(self, axis, num, out);
}

static PyObject *
array_sum(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
	int axis=MAX_DIMS;
	PyArray_Descr *dtype=NULL;
        PyArrayObject *out=NULL;
	static char *kwlist[] = {"axis", "dtype", "out", NULL};

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O&O&O&", kwlist,
					 PyArray_AxisConverter,
					 &axis, PyArray_DescrConverter2,
					 &dtype,
                                         PyArray_OutputConverter,
                                         &out)) return NULL;

	return PyArray_Sum(self, axis, _CHKTYPENUM(dtype), out);
}


static PyObject *
array_cumsum(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
	int axis=MAX_DIMS;
	PyArray_Descr *dtype=NULL;
        PyArrayObject *out=NULL;
	static char *kwlist[] = {"axis", "dtype", "out", NULL};

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O&O&O&", kwlist,
					 PyArray_AxisConverter,
					 &axis, PyArray_DescrConverter2,
					 &dtype,
                                         PyArray_OutputConverter,
                                         &out)) return NULL;

	return PyArray_CumSum(self, axis, _CHKTYPENUM(dtype), out);
}

static PyObject *
array_prod(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
	int axis=MAX_DIMS;
	PyArray_Descr *dtype=NULL;
        PyArrayObject *out=NULL;
	static char *kwlist[] = {"axis", "dtype", "out", NULL};

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O&O&O&", kwlist,
					 PyArray_AxisConverter,
					 &axis, PyArray_DescrConverter2,
					 &dtype,
                                         PyArray_OutputConverter,
                                         &out)) return NULL;

	return PyArray_Prod(self, axis, _CHKTYPENUM(dtype), out);
}

static PyObject *
array_cumprod(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
	int axis=MAX_DIMS;
	PyArray_Descr *dtype=NULL;
        PyArrayObject *out=NULL;
	static char *kwlist[] = {"axis", "dtype", "out", NULL};

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O&O&O&", kwlist,
					 PyArray_AxisConverter,
					 &axis, PyArray_DescrConverter2,
					 &dtype,
                                         PyArray_OutputConverter,
                                         &out)) return NULL;

	return PyArray_CumProd(self, axis, _CHKTYPENUM(dtype), out);
}


static PyObject *
array_any(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
	int axis=MAX_DIMS;
        PyArrayObject *out=NULL;
	static char *kwlist[] = {"axis", "out", NULL};

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O&O&", kwlist,
					 PyArray_AxisConverter,
					 &axis,
                                         PyArray_OutputConverter,
                                         &out))
		return NULL;

	return PyArray_Any(self, axis, out);
}


static PyObject *
array_all(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
	int axis=MAX_DIMS;
        PyArrayObject *out=NULL;
	static char *kwlist[] = {"axis", "out", NULL};

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O&O&", kwlist,
					 PyArray_AxisConverter,
					 &axis,
                                         PyArray_OutputConverter,
                                         &out))
		return NULL;

	return PyArray_All(self, axis, out);
}


static PyObject *
array_stddev(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
	int axis=MAX_DIMS;
	PyArray_Descr *dtype=NULL;
        PyArrayObject *out=NULL;
        int num;
	static char *kwlist[] = {"axis", "dtype", "out", NULL};

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O&O&O&", kwlist,
					 PyArray_AxisConverter,
					 &axis, PyArray_DescrConverter2,
					 &dtype,
                                         PyArray_OutputConverter,
                                         &out)) return NULL;

        num = _get_type_num_double(self->descr, dtype);
        return PyArray_Std(self, axis, num, out, 0);
}


static PyObject *
array_variance(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
	int axis=MAX_DIMS;
	PyArray_Descr *dtype=NULL;
        PyArrayObject *out=NULL;
        int num;
	static char *kwlist[] = {"axis", "dtype", "out", NULL};

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O&O&O&", kwlist,
					 PyArray_AxisConverter,
					 &axis, PyArray_DescrConverter2,
					 &dtype,
                                         PyArray_OutputConverter,
                                         &out)) return NULL;

        num = _get_type_num_double(self->descr, dtype);
	return PyArray_Std(self, axis, num, out, 1);
}


static PyObject *
array_compress(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
	int axis=MAX_DIMS;
	PyObject *condition;
        PyArrayObject *out=NULL;
	static char *kwlist[] = {"condition", "axis", "out", NULL};

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O&O&", kwlist,
					 &condition, PyArray_AxisConverter,
					 &axis,
                                         PyArray_OutputConverter,
                                         &out)) return NULL;

	return _ARET(PyArray_Compress(self, condition, axis, out));
}


static PyObject *
array_nonzero(PyArrayObject *self, PyObject *args)
{
	if (!PyArg_ParseTuple(args, "")) return NULL;

	return PyArray_Nonzero(self);
}


static PyObject *
array_trace(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
	int axis1=0, axis2=1, offset=0;
	PyArray_Descr *dtype=NULL;
        PyArrayObject *out=NULL;
	static char *kwlist[] = {"offset", "axis1", "axis2", "dtype", "out", NULL};

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "|iiiO&O&", kwlist,
					 &offset, &axis1, &axis2,
					 PyArray_DescrConverter2, &dtype,
                                         PyArray_OutputConverter, &out))
		return NULL;

	return _ARET(PyArray_Trace(self, offset, axis1, axis2,
				   _CHKTYPENUM(dtype), out));
}

#undef _CHKTYPENUM


static PyObject *
array_clip(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
	PyObject *min, *max;
        PyArrayObject *out=NULL;
	static char *kwlist[] = {"min", "max", "out", NULL};

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|O&", kwlist,
					 &min, &max,
                                         PyArray_OutputConverter,
                                         &out))
		return NULL;

	return _ARET(PyArray_Clip(self, min, max, out));
}


static PyObject *
array_conjugate(PyArrayObject *self, PyObject *args)
{

        PyArrayObject *out=NULL;
	if (!PyArg_ParseTuple(args, "|O&",
                              PyArray_OutputConverter,
                              &out)) return NULL;

	return PyArray_Conjugate(self, out);
}


static PyObject *
array_diagonal(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
	int axis1=0, axis2=1, offset=0;
	static char *kwlist[] = {"offset", "axis1", "axis2", NULL};

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "|iii", kwlist,
					 &offset, &axis1, &axis2))
		return NULL;

	return _ARET(PyArray_Diagonal(self, offset, axis1, axis2));
}


static PyObject *
array_flatten(PyArrayObject *self, PyObject *args)
{
	PyArray_ORDER fortran=PyArray_CORDER;

	if (!PyArg_ParseTuple(args, "|O&", PyArray_OrderConverter,
			      &fortran)) return NULL;

	return PyArray_Flatten(self, fortran);
}


static PyObject *
array_ravel(PyArrayObject *self, PyObject *args)
{
	PyArray_ORDER fortran=PyArray_CORDER;

	if (!PyArg_ParseTuple(args, "|O&", PyArray_OrderConverter,
			      &fortran)) return NULL;

	return PyArray_Ravel(self, fortran);
}


static PyObject *
array_round(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
	int decimals = 0;
        PyArrayObject *out=NULL;
	static char *kwlist[] = {"decimals", "out", NULL};

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "|iO&", kwlist,
					 &decimals, PyArray_OutputConverter,
                                         &out))
            return NULL;

	return _ARET(PyArray_Round(self, decimals, out));
}



static int _IsAligned(PyArrayObject *);
static Bool _IsWriteable(PyArrayObject *);

static PyObject *
array_setflags(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
	static char *kwlist[] = {"write", "align", "uic", NULL};
	PyObject *write=Py_None;
	PyObject *align=Py_None;
	PyObject *uic=Py_None;
	int flagback = self->flags;

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOO", kwlist,
					 &write, &align, &uic))
		return NULL;

	if (align != Py_None) {
		if (PyObject_Not(align)) self->flags &= ~ALIGNED;
		else if (_IsAligned(self)) self->flags |= ALIGNED;
		else {
			PyErr_SetString(PyExc_ValueError,
					"cannot set aligned flag of mis-"\
					"aligned array to True");
			return NULL;
		}
	}

	if (uic != Py_None) {
                if (PyObject_IsTrue(uic)) {
			self->flags = flagback;
                        PyErr_SetString(PyExc_ValueError,
                                        "cannot set UPDATEIFCOPY "       \
                                        "flag to True");
                        return NULL;
                }
                else {
                        self->flags &= ~UPDATEIFCOPY;
                        Py_XDECREF(self->base);
                        self->base = NULL;
                }
        }

        if (write != Py_None) {
                if (PyObject_IsTrue(write))
			if (_IsWriteable(self)) {
				self->flags |= WRITEABLE;
			}
			else {
				self->flags = flagback;
				PyErr_SetString(PyExc_ValueError,
						"cannot set WRITEABLE "	\
						"flag to True of this "	\
						"array");		\
				return NULL;
			}
                else
                        self->flags &= ~WRITEABLE;
        }

        Py_INCREF(Py_None);
        return Py_None;
}


static PyObject *
array_newbyteorder(PyArrayObject *self, PyObject *args)
{
	char endian = PyArray_SWAP;
	PyArray_Descr *new;

	if (!PyArg_ParseTuple(args, "|O&", PyArray_ByteorderConverter,
			      &endian)) return NULL;

	new = PyArray_DescrNewByteorder(self->descr, endian);
	if (!new) return NULL;
	return PyArray_View(self, new, NULL);

}

static PyMethodDef array_methods[] = {

	/* for subtypes */
	{"__array__", (PyCFunction)array_getarray,
            METH_VARARGS, NULL},
	{"__array_wrap__", (PyCFunction)array_wraparray,
            METH_VARARGS, NULL},

	/* for the copy module */
        {"__copy__", (PyCFunction)array_copy,
            METH_VARARGS, NULL},
        {"__deepcopy__", (PyCFunction)array_deepcopy,
            METH_VARARGS, NULL},

        /* for Pickling */
        {"__reduce__", (PyCFunction) array_reduce,
            METH_VARARGS, NULL},
	{"__setstate__", (PyCFunction) array_setstate,
            METH_VARARGS, NULL},
	{"dumps", (PyCFunction) array_dumps,
            METH_VARARGS, NULL},
	{"dump", (PyCFunction) array_dump,
            METH_VARARGS, NULL},

	/* Original and Extended methods added 2005 */
        {"all", (PyCFunction)array_all,
	    METH_VARARGS | METH_KEYWORDS, NULL},
	{"any", (PyCFunction)array_any,
	    METH_VARARGS | METH_KEYWORDS, NULL},
	{"argmax", (PyCFunction)array_argmax,
	    METH_VARARGS | METH_KEYWORDS, NULL},
	{"argmin", (PyCFunction)array_argmin,
	    METH_VARARGS | METH_KEYWORDS, NULL},
	{"argsort", (PyCFunction)array_argsort,
	    METH_VARARGS | METH_KEYWORDS, NULL},
        {"astype", (PyCFunction)array_cast,
            METH_VARARGS, NULL},
        {"byteswap", (PyCFunction)array_byteswap,
            METH_VARARGS, NULL},
	{"choose", (PyCFunction)array_choose,
	    METH_VARARGS | METH_KEYWORDS, NULL},
	{"clip", (PyCFunction)array_clip,
	    METH_VARARGS | METH_KEYWORDS, NULL},
	{"compress", (PyCFunction)array_compress,
	    METH_VARARGS | METH_KEYWORDS, NULL},
	{"conj", (PyCFunction)array_conjugate,
	    METH_VARARGS, NULL},
	{"conjugate", (PyCFunction)array_conjugate,
	    METH_VARARGS, NULL},
        {"copy", (PyCFunction)array_copy,
            METH_VARARGS, NULL},
	{"cumprod", (PyCFunction)array_cumprod,
	    METH_VARARGS | METH_KEYWORDS, NULL},
	{"cumsum", (PyCFunction)array_cumsum,
	    METH_VARARGS | METH_KEYWORDS, NULL},
	{"diagonal", (PyCFunction)array_diagonal,
	    METH_VARARGS | METH_KEYWORDS, NULL},
	{"fill", (PyCFunction)array_fill,
	    METH_VARARGS, NULL},
	{"flatten", (PyCFunction)array_flatten,
	    METH_VARARGS, NULL},
	{"getfield", (PyCFunction)array_getfield,
            METH_VARARGS | METH_KEYWORDS, NULL},
        {"item", (PyCFunction)array_toscalar,
            METH_VARARGS, NULL},
        {"itemset", (PyCFunction) array_setscalar,
            METH_VARARGS, NULL},
	{"max", (PyCFunction)array_max,
	    METH_VARARGS | METH_KEYWORDS, NULL},
	{"mean", (PyCFunction)array_mean,
	    METH_VARARGS | METH_KEYWORDS, NULL},
	{"min", (PyCFunction)array_min,
	    METH_VARARGS | METH_KEYWORDS, NULL},
	{"newbyteorder", (PyCFunction)array_newbyteorder,
	    METH_VARARGS, NULL},
	{"nonzero", (PyCFunction)array_nonzero,
	    METH_VARARGS, NULL},
	{"prod", (PyCFunction)array_prod,
	    METH_VARARGS | METH_KEYWORDS, NULL},
	{"ptp", (PyCFunction)array_ptp,
	    METH_VARARGS | METH_KEYWORDS, NULL},
	{"put",	(PyCFunction)array_put,
	    METH_VARARGS | METH_KEYWORDS, NULL},
	{"ravel", (PyCFunction)array_ravel,
	    METH_VARARGS, NULL},
	{"repeat", (PyCFunction)array_repeat,
	    METH_VARARGS | METH_KEYWORDS, NULL},
	{"reshape", (PyCFunction)array_reshape,
	    METH_VARARGS | METH_KEYWORDS, NULL},
        {"resize", (PyCFunction)array_resize,
            METH_VARARGS | METH_KEYWORDS, NULL},
	{"round", (PyCFunction)array_round,
	    METH_VARARGS | METH_KEYWORDS, NULL},
	{"searchsorted", (PyCFunction)array_searchsorted,
	    METH_VARARGS | METH_KEYWORDS, NULL},
	{"setfield", (PyCFunction)array_setfield,
            METH_VARARGS | METH_KEYWORDS, NULL},
	{"setflags", (PyCFunction)array_setflags,
	    METH_VARARGS | METH_KEYWORDS, NULL},
	{"sort", (PyCFunction)array_sort,
	    METH_VARARGS | METH_KEYWORDS, NULL},
	{"squeeze", (PyCFunction)array_squeeze,
	    METH_VARARGS, NULL},
	{"std", (PyCFunction)array_stddev,
	    METH_VARARGS | METH_KEYWORDS, NULL},
	{"sum", (PyCFunction)array_sum,
	    METH_VARARGS | METH_KEYWORDS, NULL},
	{"swapaxes", (PyCFunction)array_swapaxes,
	    METH_VARARGS, NULL},
	{"take", (PyCFunction)array_take,
	    METH_VARARGS | METH_KEYWORDS, NULL},
	{"tofile", (PyCFunction)array_tofile,
            METH_VARARGS | METH_KEYWORDS, NULL},
        {"tolist", (PyCFunction)array_tolist,
            METH_VARARGS, NULL},
        {"tostring", (PyCFunction)array_tostring,
            METH_VARARGS | METH_KEYWORDS, NULL},
	{"trace", (PyCFunction)array_trace,
	    METH_VARARGS | METH_KEYWORDS, NULL},
	{"transpose", (PyCFunction)array_transpose,
	    METH_VARARGS, NULL},
	{"var", (PyCFunction)array_variance,
	    METH_VARARGS | METH_KEYWORDS, NULL},
	{"view", (PyCFunction)array_view,
	    METH_VARARGS, NULL},
        {NULL,		NULL}		/* sentinel */
};

#undef _ARET
