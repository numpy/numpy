
#define _ARET(x) PyArray_Return((PyArrayObject *)(x))
static char doc_take[] = "a.take(indices, axis=None).  Selects the elements "\
	"in indices from array a along the given axis.";

static PyObject *
array_take(PyArrayObject *self, PyObject *args, PyObject *kwds) 
{
	int dimension=MAX_DIMS;
	PyObject *indices;
	static char *kwlist[] = {"indices", "axis", NULL};
	
	dimension=0;
	if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O&", kwlist, 
					 &indices, PyArray_AxisConverter,
					 &dimension))
		return NULL;
	
	return _ARET(PyArray_Take(self, indices, dimension));
}

static char doc_fill[] = "a.fill(value) places the scalar value at every"\
	"position in the array.";

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

static char doc_put[] = "a.put(indices, values) sets a.flat[n] = v[n] "\
	"for each n in indices. v can be scalar or shorter than indices, "\
	"will repeat.";

static PyObject *
array_put(PyArrayObject *self, PyObject *args, PyObject *kwds) 
{
	PyObject *indices, *values;
	static char *kwlist[] = {"indices", "values", NULL};
	
	if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO", kwlist,
					 &indices, &values))
		return NULL;
	return PyArray_Put(self, indices, values);
}

static char doc_putmask[] = "a.putmask(mask, values) sets a.flat[n] = v[n] "\
	"for each n where mask.flat[n] is TRUE. v can be scalar.";

static PyObject *
array_putmask(PyArrayObject *self, PyObject *args, PyObject *kwds) 
{
	PyObject *mask, *values;

	static char *kwlist[] = {"mask", "values", NULL};
	
	if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO", kwlist,
					 &mask, &values))
		return NULL;
	return PyArray_PutMask(self, mask, values);
}

/* Used to reshape a Fortran Array */
static void
_reverse_shape(PyArray_Dims *newshape)
{
	int i, n = newshape->len;
	intp *ptr = newshape->ptr;
	intp *eptr;
	intp tmp;
	int len = n >> 1;

	eptr = ptr+n-1;
	for(i=0; i<len; i++) {
		tmp = *eptr;
		*eptr-- = *ptr;
		*ptr++ = tmp;
	}
}

static char doc_reshape[] = \
	"self.reshape(d1, d2, ..., dn) Return a new array from this one. \n" \
	"\n  The new array must have the same number of elements as self. "\
	"Also\n   a copy of the data only occurs if necessary.";

static PyObject *
array_reshape(PyArrayObject *self, PyObject *args) 
{
        PyArray_Dims newshape;
        PyObject *ret, *tmp;
	int n;
	
	n = PyTuple_Size(args);
	if (n <= 1) {
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

	if (newshape.len == 1) {
		PyDimMem_FREE(newshape.ptr);
		return PyArray_Ravel(self, 0);
	}

	if ((newshape.len == 0) || PyArray_ISCONTIGUOUS(self)) {
		ret = PyArray_Newshape(self, &newshape);
	}
	else if PyArray_ISFORTRAN(self) {
		tmp = PyArray_Transpose(self, NULL);
		if (tmp == NULL) goto fail;
		_reverse_shape(&newshape);
		ret = PyArray_Newshape((PyArrayObject *)tmp, &newshape);
		Py_DECREF(tmp);
		if (ret == NULL) goto fail;
		tmp = PyArray_Transpose((PyArrayObject *)ret, NULL);
		Py_DECREF(ret);
		if (tmp == NULL) goto fail;
		ret = tmp;
	}
	else {
		tmp = PyArray_Copy(self);
		if (tmp==NULL) goto fail;
		ret = PyArray_Newshape((PyArrayObject *)tmp, &newshape);
		Py_DECREF(tmp);
	}
	PyDimMem_FREE(newshape.ptr);
        return _ARET(ret);

 fail:
	PyDimMem_FREE(newshape.ptr);
	return NULL;
}

static char doc_squeeze[] = "m.squeeze() eliminate all length-1 dimensions";

static PyObject *
array_squeeze(PyArrayObject *self, PyObject *args)
{
        if (!PyArg_ParseTuple(args, "")) return NULL;
        return _ARET(PyArray_Squeeze(self));
}



static char doc_view[] = "a.view(<dtype>) return a new view of array with same data.";

static PyObject *
array_view(PyArrayObject *self, PyObject *args)
{
        PyArray_Typecode type = {PyArray_NOTYPE, 0, 0};
	if (!PyArg_ParseTuple(args, "|O&",
                              PyArray_TypecodeConverter, &type)) 
                return NULL;

	return _ARET(PyArray_View(self, &type));
}

static char doc_argmax[] = "a.argmax(axis=None)";

static PyObject *
array_argmax(PyArrayObject *self, PyObject *args) 
{
	int axis=MAX_DIMS;
	
	if (!PyArg_ParseTuple(args, "|O&", PyArray_AxisConverter, 
			      &axis)) return NULL;
	
	return _ARET(PyArray_ArgMax(self, axis));
}

static char doc_argmin[] = "a.argmin(axis=None)";

static PyObject *
array_argmin(PyArrayObject *self, PyObject *args) 
{
	int axis=MAX_DIMS;
	
	if (!PyArg_ParseTuple(args, "|O&", PyArray_AxisConverter, 
			      &axis)) return NULL;
	
	return _ARET(PyArray_ArgMin(self, axis));
}

static char doc_max[] = "a.max(axis=None)";

static PyObject *
array_max(PyArrayObject *self, PyObject *args) 
{
	int axis=MAX_DIMS;
	
	if (!PyArg_ParseTuple(args, "|O&", PyArray_AxisConverter, 
			      &axis)) return NULL;
	
	return PyArray_Max(self, axis);
}

static char doc_ptp[] = "a.ptp(axis=None) a.max(axis)-a.min(axis)";

static PyObject *
array_ptp(PyArrayObject *self, PyObject *args) 
{
	int axis=MAX_DIMS;
	
	if (!PyArg_ParseTuple(args, "|O&", PyArray_AxisConverter, 
			      &axis)) return NULL;
	
	return PyArray_Ptp(self, axis);
}


static char doc_min[] = "a.min(axis=None)";

static PyObject *
array_min(PyArrayObject *self, PyObject *args) 
{
	int axis=MAX_DIMS;
	
	if (!PyArg_ParseTuple(args, "|O&", PyArray_AxisConverter, 
			      &axis)) return NULL;
	
	return PyArray_Min(self, axis);
}


static char doc_swapaxes[] = "a.swapaxes(axis1, axis2)  returns new view with axes swapped.";

static PyObject *
array_swapaxes(PyArrayObject *self, PyObject *args)
{
	int axis1, axis2;

	if (!PyArg_ParseTuple(args, "ii", &axis1, &axis2)) return NULL;

	return PyArray_SwapAxes(self, axis1, axis2);
}

static char doc_getfield[] = "m.field(type, offset) returns a field "\
	" of the given array as a certain type.  A field is a view of "\
	" the array's data with each itemsize determined by the given type"\
	" and the offset into the current array.";

static PyObject *
PyArray_GetField(PyArrayObject *self, PyArray_Typecode *type, 
				  int offset)
{
	PyObject *ret=NULL;

	if (offset < 0 || (offset + type->itemsize) > self->itemsize) {
		PyErr_Format(PyExc_ValueError,
			     "Need 0 <= offset <= %d for requested type "  \
			     "but received offset = %d",
			     self->itemsize-type->itemsize, offset);
		return NULL;
	}
	ret = PyArray_New(self->ob_type, self->nd, self->dimensions,
			  type->type_num, self->strides, 
			  self->data + offset,
			  type->itemsize, self->flags, (PyObject *)self);
	if (ret == NULL) return NULL;
	Py_INCREF(self);
	((PyArrayObject *)ret)->base = (PyObject *)self;
	PyArray_UpdateFlags((PyArrayObject *)ret, UPDATE_ALL_FLAGS);
	return ret;	
}

static PyObject *
array_getfield(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
	PyArray_Typecode typecode = {PyArray_NOTYPE, 0, 0};
	int offset = 0;
	static char *kwlist[] = {"dtype", "offset", 0};
	
	if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|i", kwlist,
					 PyArray_TypecodeConverter, 
					 &typecode, &offset)) return NULL;
	
	return _ARET(PyArray_GetField(self, &typecode, offset));
}

static PyObject *
PyArray_Byteswap(PyArrayObject *self, Bool inplace)
{
        PyArrayObject *ret;
	intp size;
	PyArray_CopySwapNFunc *copyswapn;
	PyArray_CopySwapFunc *copyswap;
	PyArrayIterObject *it;

	if (inplace) {
		copyswapn = self->descr->copyswapn;
		
		size = PyArray_SIZE(self);
		if (PyArray_ISONESEGMENT(self)) {
			copyswapn(self->data, NULL, size, 1, self->itemsize);
		}
		else { /* Use iterator */
			
			it = (PyArrayIterObject *)\
				PyArray_IterNew((PyObject *)self);
			copyswap = self->descr->copyswap;
			while (it->index < it->size) {
				copyswap(it->dataptr, NULL, 1, 
					 self->itemsize);
				PyArray_ITER_NEXT(it);
			}
			Py_DECREF(it);
		}

		if (self->flags & NOTSWAPPED) self->flags &= ~NOTSWAPPED;
		else self->flags |= NOTSWAPPED;
		
		Py_INCREF(self);
		return (PyObject *)self;
	}
	else {
		if ((ret = (PyArrayObject *)PyArray_Copy(self)) == NULL) 
			return NULL;
		
		size = PyArray_SIZE(self);
		
		/* set the NOTSWAPPED flag to opposite self */
		/* ret is always notswapped, 
		   PyArray_Copy has already swapped if self was swapped 
		   to begin with so just correct the flag in that case. */

		if (self->flags & NOTSWAPPED) {
			ret->descr->copyswapn(ret->data, NULL, size, 1, ret->itemsize);
			ret->flags &= ~NOTSWAPPED;
		}
		else { /* self was swapped, so now ret isn't */
			ret->flags |= NOTSWAPPED;
		}
		
		return (PyObject *)ret;
	}
}

static char doc_byteswap[] = "m.byteswap(False)  Swap the bytes in"\
	" the array.  Return the byteswapped array.  If the first argument"\
	" is TRUE, byteswap in-place and return a reference to self.";

static PyObject *
array_byteswap(PyArrayObject *self, PyObject *args) 
{
	Bool inplace=FALSE;
	
	if (!PyArg_ParseTuple(args, "|O&", PyArray_BoolConverter, &inplace))
		return NULL;
	
	return PyArray_Byteswap(self, inplace);
}

static char doc_tolist[] = "m.tolist().	 Copy the data portion of the array"\
	" to a hierarchical python list and return that list.";

static PyObject *
array_tolist(PyArrayObject *self, PyObject *args) 
{
        if (!PyArg_ParseTuple(args, "")) return NULL;
        if (self->nd <= 0) {
                PyErr_SetString(PyExc_ValueError, 
				"can't convert a 0-d array to a list");
                return NULL;
        }
	
        return PyArray_ToList(self);
}

static char doc_tostring[] = "m.tostring()  Construct a Python string "\
        "containing the raw bytes in the array";

static PyObject *
array_tostring(PyArrayObject *self, PyObject *args)
{
        if (!PyArg_ParseTuple(args, "")) return NULL;
        return PyArray_ToString(self);
}

static char doc_tofile[] = "m.tofile(fid, sep="") write the data to a file.";

static PyObject *
array_tofile(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
	int ret;
        PyObject *file;
	FILE *fd;
        char *sep="";
	char *format="";
	char *mode;
	static char *kwlist[] = {"file", "sep", "format", NULL};
        
	if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|ss", kwlist, 
                                         &file, &sep, &format)) return NULL;

	if (PyString_Check(file)) {
		if (sep == "") mode="wb";
		else mode="w";
		file = PyFile_FromString(PyString_AS_STRING(file), mode);
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

static char doc_toscalar[] = "m.toscalar().  Copy the first data point of "\
	"the array to a standard Python scalar and return it.";

static PyObject *
array_toscalar(PyArrayObject *self, PyObject *args) {
        if (!PyArg_ParseTuple(args, "")) return NULL;
	if (self->nd == 0 || PyArray_SIZE(self) == 1) 
		return self->descr->getitem(self->data, self);
	else {
		PyErr_SetString(PyExc_ValueError, "can only convert an"	\
				" array of size 1 to Python scalar.");
		return NULL;
	}
}

static char doc_cast[] = "m.astype(t).	Cast array m to type t.	 \n\n"\
	"t can be either a string representing a typecode, or a python type"\
	" object of type int, float, or complex.";

static PyObject *
array_cast(PyArrayObject *self, PyObject *args) 
{
	PyArray_Typecode typecode = {PyArray_NOTYPE, 0, 0};
	
        if (!PyArg_ParseTuple(args, "O&", PyArray_TypecodeConverter,
			      &typecode)) return NULL;

	if (typecode.type_num == PyArray_NOTYPE ||	\
	    typecode.type_num == PyArray_TYPE(self)) {
		return _ARET(PyArray_Copy(self));
	}
	return _ARET(PyArray_CastToType(self, &typecode));
}	  

/* default sub-type implementation */

static char doc_wraparray[] = "m.__array_wrap__(obj) returns an object of "\
	"type m from the ndarray object obj";

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

	ret = PyArray_New(self->ob_type, PyArray_NDIM(arr),
			  PyArray_DIMS(arr), PyArray_TYPE(arr),
			  PyArray_STRIDES(arr), PyArray_DATA(arr),
			  PyArray_ITEMSIZE(arr), 
			  PyArray_FLAGS(arr), (PyObject *)self);
	if (ret == NULL) return NULL;
	Py_INCREF(arr);
	PyArray_BASE(ret) = arr;
	return ret;
}

/* NO-OP --- just so all subclasses will have one by default. */
static PyObject *
array_finalize(PyArrayObject *self, PyObject *args)
{
	Py_INCREF(Py_None);
	return Py_None;
}


static char doc_array_getarray[] = "m.__array__(|dtype) just returns either a new reference to self if dtype is not given or a new array of provided data type if dtype is different from the current dtype of the array.";

static PyObject *
array_getarray(PyArrayObject *self, PyObject *args) 
{
	PyArray_Typecode newtype = {PyArray_NOTYPE, 0, 0};
	PyArray_Typecode oldtype = {PyArray_TYPE(self),
				    PyArray_ITEMSIZE(self),
				    0};
	PyObject *ret;

	if (!PyArg_ParseTuple(args, "|O&", &PyArray_TypecodeConverter,
			      &newtype)) return NULL;

	/* convert to PyArray_Type or PyBigArray_Type */
	if (!PyArray_CheckExact(self) || !PyBigArray_CheckExact(self)) {
		PyObject *new;
		PyTypeObject *subtype = &PyArray_Type;

		if (!PyType_IsSubtype(self->ob_type, &PyArray_Type)) {
			subtype = &PyBigArray_Type;
		}
		
		new = PyArray_New(subtype, PyArray_NDIM(self),
				  PyArray_DIMS(self), PyArray_TYPE(self),
				  PyArray_STRIDES(self), PyArray_DATA(self),
				  PyArray_ITEMSIZE(self), 
				  PyArray_FLAGS(self), NULL);
		if (new == NULL) return NULL;
		Py_INCREF(self);
		PyArray_BASE(new) = (PyObject *)self;
		self = (PyArrayObject *)new;
	}
	else {
		Py_INCREF(self);
	}
		
	if (newtype.type_num == PyArray_NOTYPE ||
	    PyArray_EquivalentTypes(&oldtype, &newtype)) {
		return (PyObject *)self;
	}
	else {
		ret = PyArray_CastToType(self, &newtype);
		Py_DECREF(self);
		return ret;
	}
}

static char doc_copy[] = "m.copy(|fortran). Return a copy of the array.\n"\
	"If fortran == 0 then the result is contiguous (default). \n"\
	"If fortran >  0 then the result has fortran data order. \n"\
	"If fortran <  0 then the result has fortran data order only if m\n"
	"   is already in fortran order.";

static PyObject *
array_copy(PyArrayObject *self, PyObject *args) 
{
	int fortran=0;
        if (!PyArg_ParseTuple(args, "|i", &fortran)) return NULL;
	
        return PyArray_NewCopy(self, fortran);
}

static char doc_resize[] = "self.resize(new_shape).  "\
	"Change size and shape of self inplace.\n"\
	"\n    Array must own its own memory and not be referenced by other " \
	"arrays\n    Returns None.";

static PyObject *
array_resize(PyArrayObject *self, PyObject *args) 
{
        PyArray_Dims newshape;
        PyObject *ret;
	int n;
	
	n = PyTuple_Size(args);
	if (n <= 1) {
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
	ret = PyArray_Resize(self, &newshape);
        PyDimMem_FREE(newshape.ptr);
	Py_DECREF(ret);
	Py_INCREF(Py_None);
	return Py_None;
}

static char doc_repeat[] = "a.repeat(repeats=, axis=None)\n"\
	"\n"\
	" Copy elements of a, repeats times.  The repeats argument must\n"\
	"  be a sequence of length a.shape[axis] or a scalar.";

static PyObject *
array_repeat(PyArrayObject *self, PyObject *args, PyObject *kwds) {
	PyObject *repeats;
	int axis=MAX_DIMS;
	static char *kwlist[] = {"repeats", "axis", NULL};
	
	if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O&", kwlist, 
					 &repeats, PyArray_AxisConverter,
					 &axis)) return NULL;
	
	return PyArray_Repeat(self, repeats, axis);
}

static char doc_choose[] = "a.choose(b0, b1, ..., bn)\n"\
	"\n"\
	"Return an array with elements chosen from 'a' at the positions\n"\
        "of the given arrays b_i.  The array 'a' should be an integer array\n"\
        "with entries from 0 to n+1, and the b_i arrays should have the same\n"\
        "shape as 'a'.";

static PyObject *
array_choose(PyArrayObject *self, PyObject *args) 
{
	PyObject *choices;
	int n;
	
	n = PyTuple_Size(args);
	if (n <= 1) {
		if (!PyArg_ParseTuple(args, "O", &choices))
			return NULL;
	}
        else {
		choices = args;
	}
	
	return _ARET(PyArray_Choose(self, choices));
}

static char doc_sort[] = "a.sort(<None>)";

static PyObject *
array_sort(PyArrayObject *self, PyObject *args) 
{
	int axis=MAX_DIMS;
	
	if (!PyArg_ParseTuple(args, "|O&", PyArray_AxisConverter, 
			      &axis)) return NULL;
	
	return _ARET(PyArray_Sort(self, axis));
}

static char doc_argsort[] = "a.argsort(<None>)\n"\
	"  Return the indexes into a that would sort it along the"\
	"  given axis (or <None> if the sorting should be done"\
	"  in terms of a.flat";

static PyObject *
array_argsort(PyArrayObject *self, PyObject *args) 
{
	int axis=MAX_DIMS;
	
	if (!PyArg_ParseTuple(args, "|O&", PyArray_AxisConverter, 
			      &axis)) return NULL;
	
	return _ARET(PyArray_ArgSort(self, axis));
}

static char doc_searchsorted[] = "a.searchsorted(v)\n"\
	" Assuming that a is a 1-D array, in ascending order and\n"\
	" represents bin boundaries, then a.searchsorted(values) gives an\n"\
	" array of bin numbers, giving the bin into which each value would\n"\
	" be placed.  This method is helpful for histograming.  \n"\
	" Note: No warning is given if the boundaries, in a, are not \n"\
	" in ascending order.";
;

static PyObject *
array_searchsorted(PyArrayObject *self, PyObject *args) 
{
	PyObject *values;
	
	if (!PyArg_ParseTuple(args, "O", &values)) return NULL;
	
	return _ARET(PyArray_SearchSorted(self, values));
}

static char doc_deepcopy[] = "Used if copy.deepcopy is called on an array.";

static PyObject *
array_deepcopy(PyArrayObject *self, PyObject *args) 
{
        PyObject* visit;
        PyObject **optr;
        PyArrayIterObject *it;
        PyObject *copy, *ret, *deepcopy, *temp, *res;

        if (!PyArg_ParseTuple(args, "O", &visit)) return NULL;
        ret = PyArray_Copy(self);
        if (PyArray_ISOBJECT(self)) {
                copy = PyImport_ImportModule("copy");
                if (copy == NULL) return NULL;
                deepcopy = PyObject_GetAttrString(copy, "deepcopy");
                if (deepcopy == NULL) return NULL;
                Py_DECREF(copy);
                it = (PyArrayIterObject *)PyArray_IterNew((PyObject *)self);
                if (it == NULL) {Py_DECREF(deepcopy); return NULL;}
                optr = (PyObject **)PyArray_DATA(ret);
                while(it->index < it->size) {
                        temp = *((PyObject **)it->dataptr);
                        Py_INCREF(temp);
                        /* call deepcopy on this argument */
                        res = PyObject_CallFunctionObjArgs(deepcopy, 
                                                           temp, visit, NULL);
                        Py_DECREF(temp);
                        Py_DECREF(*optr);
                        *optr++ = res;
                        PyArray_ITER_NEXT(it);
                }
                Py_DECREF(deepcopy);
                Py_DECREF(it);
        }
        return _ARET(ret);
}

/* Convert Object Array to flat list and pickle the flat list string */
static PyObject *
_getobject_pkl(PyArrayObject *self)
{
	PyObject *theobject;
	PyArrayIterObject *iter=NULL;
	PyObject *list;

	
	iter = (PyArrayIterObject *)PyArray_IterNew((PyObject *)self);
	if (iter == NULL) return NULL;
	list = PyList_New(iter->size);
	if (list == NULL) {Py_DECREF(iter); return NULL;}
	while (iter->index < iter->size) {
		theobject = *((PyObject **)iter->dataptr);
		Py_INCREF(theobject);
		PyList_SET_ITEM(list, (int) iter->index, theobject);
		PyArray_ITER_NEXT(iter);
	}
	Py_DECREF(iter);
	return list;
}

static int
_setobject_pkl(PyArrayObject *self, PyObject *list)
{
	PyObject *theobject;
	PyArrayIterObject *iter=NULL;
	int size;

	size = self->itemsize;
	
	iter = (PyArrayIterObject *)PyArray_IterNew((PyObject *)self);
	if (iter == NULL) return -1;
	while(iter->index < iter->size) {
		theobject = PyList_GET_ITEM(list, (int) iter->index);
		Py_INCREF(theobject);
		*((PyObject **)iter->dataptr) = theobject;
		PyArray_ITER_NEXT(iter);
	}
	Py_XDECREF(iter);
	return 0;
}


static char doc_reduce[] = "a.__reduce__()  for pickling.";

static PyObject *
array_reduce(PyArrayObject *self, PyObject *args)
{
	PyObject *ret=NULL, *state=NULL, *obj=NULL, *mod=NULL;
	PyObject *mybool, *thestr=NULL;

	/* Return a tuple of (callable object, arguments, object's state) */
	/*  We will put everything in the object's state, so that on UnPickle
	    it can use the string object as memory without a copy */

	ret = PyTuple_New(3);
	if (ret == NULL) return NULL;
	mod = PyImport_ImportModule("scipy.base.multiarray");
	if (mod == NULL) return NULL;
	obj = PyObject_GetAttrString(mod, "empty");
	Py_DECREF(mod);
	if (obj == NULL) return NULL;
	PyTuple_SET_ITEM(ret, 0, obj);
	PyTuple_SET_ITEM(ret, 1, 
			 Py_BuildValue("NNN",
				       Py_BuildValue("(N)",
						     PyInt_FromLong(0)),
				       PyObject_GetAttrString((PyObject *)self,
							      "dtypechar"),
				       PyInt_FromLong((long) 0)));
	
	/* Now fill in object's state.  This is a tuple with 
	   4 arguments

	   1) a Tuple giving the shape
	   2) a string giving the typestr (> or < indicates byte order
	      if swap is possibly needed)
	   3) a Bool stating if Fortran or not
	   4) a binary string with the data (or a list for Object arrays)

	   Notice because Python does not describe a mechanism to write 
	   raw data to the pickle, this performs a copy to a string first
	*/

	state = PyTuple_New(4);
	if (state == NULL) {
		Py_DECREF(ret); return NULL;
	}
	PyTuple_SET_ITEM(state, 0, PyObject_GetAttrString((PyObject *)self, 
							  "shape"));	
	PyTuple_SET_ITEM(state, 1, PyObject_GetAttrString((PyObject *)self, 
							  "dtypestr"));
	mybool = (PyArray_ISFORTRAN(self) ? Py_True : Py_False);
	Py_INCREF(mybool);
	PyTuple_SET_ITEM(state, 2, mybool);
	if (PyArray_ISOBJECT(self)) {
		thestr = _getobject_pkl(self);
	}
	else {
                thestr = PyArray_ToString(self);
	}
	if (thestr == NULL) {
		Py_DECREF(ret);
		Py_DECREF(state);
		return NULL;
	}
	PyTuple_SET_ITEM(state, 3, thestr);
	PyTuple_SET_ITEM(ret, 2, state);
	return ret;
}

static char doc_setstate[] = "a.__setstate__(tuple) for unpickling.";

/*
	   1) a Tuple giving the shape
	   2) a string giving the typestr (> or < indicates byte order
	      if swap is possibly needed)
	   3) a Bool stating if Fortran or not
	   4) a binary string with the data (or a list if Object array) 
*/

static intp _array_fill_strides(intp *, intp *, int, intp, int, int *);

static PyObject *
array_setstate(PyArrayObject *self, PyObject *args)
{
	PyObject *shape;
	PyArray_Typecode typecode;
	long fortran;
	PyObject *rawdata;
	char *datastr;
	const char *typestr;
	int len, typestrlen;
	int swap;
	intp dimensions[MAX_DIMS];
	int nd;
	
	/* This will free any memory associated with a and
	   use the string in setstate as the (writeable) memory.
	*/
	if (!PyArg_ParseTuple(args, "(O!z#iO)", &PyTuple_Type,
			      &shape, &typestr, &typestrlen,
			      &fortran, &rawdata))
		return NULL;

	if (_array_typecode_fromstr((char *)typestr, &swap, &typecode) < 0)
		return NULL;

	self->descr = PyArray_DescrFromType(typecode.type_num);
	self->itemsize = typecode.itemsize;
	nd = PyArray_IntpFromSequence(shape, dimensions, MAX_DIMS);
	if (typecode.type_num == PyArray_OBJECT) {
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

		if ((len != (self->itemsize *				\
			     (int) PyArray_MultiplyList(dimensions, nd)))) {
			PyErr_SetString(PyExc_ValueError, 
					"buffer size does not"	\
					" match array size");
			return NULL;
		}
	}

        if ((self->flags & OWN_DATA)) {
		if (self->data != NULL)
			PyDataMem_FREE(self->data);
		self->flags &= ~OWN_DATA;
        }
	Py_XDECREF(self->base);

	self->flags &= ~UPDATEIFCOPY;

        if (self->dimensions != NULL) {
                PyDimMem_FREE(self->dimensions); 
		self->dimensions = NULL;
	}

	self->flags = DEFAULT_FLAGS;

	self->nd = nd;

	if (nd > 0) {
		self->dimensions = PyDimMem_NEW(nd * 2);
		self->strides = self->dimensions + nd;
		memcpy(self->dimensions, dimensions, sizeof(intp)*nd);
		(void) _array_fill_strides(self->strides, dimensions, nd,
					   self->itemsize, fortran, 
					   &(self->flags));
	}

	if (typecode.type_num != PyArray_OBJECT) {
		self->data = datastr;
		self->base = rawdata;
		Py_INCREF(self->base);
	}
	else {
		self->data = PyDataMem_NEW(PyArray_NBYTES(self));
		if (self->data == NULL) { 
			self->nd = 0;
			self->data = PyDataMem_NEW(self->itemsize);
			if (self->dimensions) PyDimMem_FREE(self->dimensions);
			return PyErr_NoMemory();
		}
		self->flags |= OWN_DATA;
		self->base = NULL;
		if (_setobject_pkl(self, rawdata) < 0) 
			return NULL;
	}

	PyArray_UpdateFlags(self, UPDATE_ALL_FLAGS);

	if (swap) self->flags &= ~NOTSWAPPED;
	
	Py_INCREF(Py_None);
	return Py_None;	
}

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


static char doc_dump[] = "m.dump(file)";

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

static char doc_dumps[] = "m.dumps()";

static PyObject *
array_dumps(PyArrayObject *self, PyObject *args)
{
	if (!PyArg_ParseTuple(args, ""))
		return NULL;
	return PyArray_Dumps((PyObject *)self, 2);
}


static char doc_transpose[] = "m.transpose(<None>)";

static PyObject *
array_transpose(PyArrayObject *self, PyObject *args) 
{
	PyObject *shape=Py_None;
	int n;

	n = PyTuple_Size(args);
	if (n > 1) shape = args;
	else if (n == 1) shape = PyTuple_GET_ITEM(args, 0);
	
	return _ARET(PyArray_Transpose(self, shape));
}

static char doc_mean[] = "a.mean(axis=None, rtype=None)\n\n"\
  "Average the array over the given axis.  If the axis is None, average\n"\
  "over all dimensions of the array.\n"\
  "\n"\
  "If an integer axis is given, this equals:\n"\
  "    a.sum(axis, rtype) * 1.0 / len(a)\n"\
  "\n"\
  "If axis is None, this equals:\n"\
  "     a.sum(axis, rtype) * 1.0 / product(a.shape)\n"\
  "\n"\
  "The optional rtype argument is the data type for intermediate\n"\
  "calculations in the sum.";


static PyObject *
array_mean(PyArrayObject *self, PyObject *args, PyObject *kwds) 
{
	int axis=MAX_DIMS;
	PyArray_Typecode rtype = {PyArray_NOTYPE, 0, 0};
	static char *kwlist[] = {"axis", "rtype", NULL};
	
	if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O&O&", kwlist,
					 PyArray_AxisConverter, 
					 &axis, PyArray_TypecodeConverter,
					 &rtype)) return NULL;

	return _ARET(PyArray_Mean(self, axis, rtype.type_num));
}

static char doc_sum[] = "a.sum(axis=None, rtype=None)\n\n"\
  "Sum the array over the given axis.  If the axis is None, sum over all\n"\
  "dimensions of the array.\n"\
  "\n"\
  "The optional rtype argument is the data type for the returned value\n"\
  "and intermediate calculations.  The default is to upcast (promote)\n"\
  "smaller integer types to the platform-dependent int.  For example, on\n"\
  "32-bit platforms:\n"\
  "\n"\
  "    a.dtype                         default sum() rtype\n"\
  "    ---------------------------------------------------\n"\
  "    bool, int8, int16, int32        int32\n"\
  "\n"\
  "Examples:\n"\
  "\n"\
  ">>> array([0.5, 1.5]).sum()\n"\
  "2.0\n"\
  ">>> array([0.5, 1.5]).sum(rtype=int32)\n"\
  "1\n"\
  ">>> array([[0, 1], [0, 5]]).sum()\n"\
  "array([0, 6])\n"\
  ">>> array([[0, 1], [0, 5]]).sum(axis=1)\n"\
  "array([1, 5])";

static PyObject *
array_sum(PyArrayObject *self, PyObject *args, PyObject *kwds) 
{
	int axis=MAX_DIMS;
	PyArray_Typecode rtype = {PyArray_NOTYPE, 0, 0};	
	static char *kwlist[] = {"axis", "rtype", NULL};

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O&O&", kwlist, 
					 PyArray_AxisConverter, 
					 &axis, PyArray_TypecodeConverter,
					 &rtype)) return NULL;
	
	return PyArray_Sum(self, axis, rtype.type_num);
}


static char doc_cumsum[] = "a.cumsum(axis=None, rtype=None)";

static PyObject *
array_cumsum(PyArrayObject *self, PyObject *args, PyObject *kwds) 
{
	int axis=MAX_DIMS;
	PyArray_Typecode rtype = {PyArray_NOTYPE, 0, 0};
	static char *kwlist[] = {"axis", "rtype", NULL};
	
	if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O&O&", kwlist, 
					 PyArray_AxisConverter, 
					 &axis, PyArray_TypecodeConverter,
					 &rtype)) return NULL;
	
	return PyArray_CumSum(self, axis, rtype.type_num);
}

static char doc_prod[] = "a.prod(axis=None, rtype=None)";

static PyObject *
array_prod(PyArrayObject *self, PyObject *args, PyObject *kwds) 
{
	int axis=MAX_DIMS;
	PyArray_Typecode rtype = {PyArray_NOTYPE, 0, 0};
	static char *kwlist[] = {"axis", "rtype", NULL};
	
	if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O&O&", kwlist, 
					 PyArray_AxisConverter, 
					 &axis, PyArray_TypecodeConverter,
					 &rtype)) return NULL;
	
	return PyArray_Prod(self, axis, rtype.type_num);
}


static char doc_cumprod[] = "a.cumprod(axis=None, rtype=None)";

static PyObject *
array_cumprod(PyArrayObject *self, PyObject *args, PyObject *kwds) 
{
	int axis=MAX_DIMS;
	PyArray_Typecode rtype = {PyArray_NOTYPE, 0, 0};
	static char *kwlist[] = {"axis", "rtype", NULL};
	
	if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O&O&", kwlist, 
					 PyArray_AxisConverter, 
					 &axis, PyArray_TypecodeConverter,
					 &rtype)) return NULL;
	
	return PyArray_CumProd(self, axis, rtype.type_num);
}


static char doc_any[] = "a.any(axis=None)";

static PyObject *
array_any(PyArrayObject *self, PyObject *args) 
{
	int axis=MAX_DIMS;
	
	if (!PyArg_ParseTuple(args, "|O&", PyArray_AxisConverter, 
			      &axis)) return NULL;
	
	return PyArray_Any(self, axis);
}

static char doc_all[] = "a.all(axis=None)";

static PyObject *
array_all(PyArrayObject *self, PyObject *args) 
{
	int axis=MAX_DIMS;
	
	if (!PyArg_ParseTuple(args, "|O&", PyArray_AxisConverter, 
			      &axis)) return NULL;
	
	return PyArray_All(self, axis);
}


static char doc_stddev[] = "a.std(axis=None, rtype=None)";

static PyObject *
array_stddev(PyArrayObject *self, PyObject *args, PyObject *kwds) 
{
	int axis=MAX_DIMS;
	PyArray_Typecode rtype = {PyArray_NOTYPE, 0, 0};
	static char *kwlist[] = {"axis", "rtype", NULL};
	
	if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O&O&", kwlist, 
					 PyArray_AxisConverter, 
					 &axis, PyArray_TypecodeConverter,
					 &rtype)) return NULL;
	
	return PyArray_Std(self, axis, rtype.type_num);
}

static char doc_compress[] = "a.compress(condition=, axis=None)";

static PyObject *
array_compress(PyArrayObject *self, PyObject *args, PyObject *kwds) 
{
	int axis=MAX_DIMS;
	PyObject *condition;	
	static char *kwlist[] = {"condition", "axis", NULL};
	
	if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O&", kwlist, 
					 &condition, PyArray_AxisConverter,
					 &axis)) return NULL;

	return _ARET(PyArray_Compress(self, condition, axis));
}

static char doc_nonzero[] = "a.nonzero() return a tuple of indices referencing"\
	"the elements of a that are nonzero.";

static PyObject *
array_nonzero(PyArrayObject *self, PyObject *args)
{
	if (!PyArg_ParseTuple(args, "")) return NULL;

	return _ARET(PyArray_Nonzero(self));
}


static char doc_trace[] = "a.trace(offset=0, axis1=0, axis2=1, rtype=None) \n"\
	"return the sum along the offset diagonal of the arrays indicated\n" \
	"axis1 and axis2.";

static PyObject *
array_trace(PyArrayObject *self, PyObject *args, PyObject *kwds) 
{
	int axis1=0, axis2=1, offset=0;
	PyArray_Typecode rtype = {PyArray_NOTYPE, 0, 0};
	static char *kwlist[] = {"offset", "axis1", "axis2", "rtype", NULL};
	
	if (!PyArg_ParseTupleAndKeywords(args, kwds, "|iiiO&", kwlist, 
					 &offset, &axis1, &axis2,
					 PyArray_TypecodeConverter, &rtype))
		return NULL;
	
	return _ARET(PyArray_Trace(self, offset, axis1, axis2, rtype.type_num));
}


static char doc_clip[] = "a.clip(min=, max=)";

static PyObject *
array_clip(PyArrayObject *self, PyObject *args, PyObject *kwds) 
{
	PyObject *min, *max;
	static char *kwlist[] = {"min", "max", NULL};

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO", kwlist,
					 &min, &max)) 
		return NULL;
	
	return _ARET(PyArray_Clip(self, min, max));
}

static char doc_conj[] = "a.conj()";

static char doc_conjugate[] = "a.conjugate()";

static PyObject *
array_conjugate(PyArrayObject *self, PyObject *args) 
{

	if (!PyArg_ParseTuple(args, "")) return NULL;
	
	return PyArray_Conjugate(self);
}


static char doc_diagonal[] = "a.diagonal(offset=0, axis1=0, axis2=1)";

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

static char doc_flatten[] = "a.flatten([fortran]) return a 1-d array (always copy)";

static PyObject *
array_flatten(PyArrayObject *self, PyObject *args)
{
	Bool fortran=FALSE;

	if (!PyArg_ParseTuple(args, "|O&",
			      PyArray_BoolConverter, &fortran)) 
		return NULL;

	return PyArray_Flatten(self, (int) fortran);
}

static char doc_ravel[] = "a.ravel([fortran]) return a 1-d array (copy only if needed)";

static PyObject *
array_ravel(PyArrayObject *self, PyObject *args)
{
	Bool fortran=FALSE;

	if (!PyArg_ParseTuple(args, "|O&", PyArray_BoolConverter,
			      &fortran)) return NULL;

	return PyArray_Ravel(self, (int) fortran);
}



static char doc_setflags[] = "a.setflags(write=None, swap=None, align=None, uic=None)";

static int _IsAligned(PyArrayObject *);
static Bool _IsWriteable(PyArrayObject *);

static PyObject *
array_setflags(PyArrayObject *self, PyObject *args, PyObject *kwds)
{
	static char *kwlist[] = {"write", "swap", "align", "uic", NULL};
	PyObject *write=Py_None;
	PyObject *swap=Py_None;
	PyObject *align=Py_None;
	PyObject *uic=Py_None;
	int flagback = self->flags;
		
	if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOOO", kwlist,
					 &write, &swap, &align, &uic))
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
                                        "cannot set UPDATEIFCOPY"       \
                                        "flag to True");
                        return NULL;
                }
                else {
                        self->flags &= ~UPDATEIFCOPY;
                        Py_DECREF(self->base);
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
        
        if (swap != Py_None) {
                if (PyObject_IsTrue(swap))
                        self->flags &= ~NOTSWAPPED;
                else 
                        self->flags |= NOTSWAPPED;
        }
        Py_INCREF(Py_None);
        return Py_None;
}

static PyMethodDef array_methods[] = {
        {"tolist",	 (PyCFunction)array_tolist,	1, doc_tolist},
        {"toscalar", (PyCFunction)array_toscalar, METH_VARARGS, doc_toscalar},
	{"tofile", (PyCFunction)array_tofile, 
         METH_VARARGS | METH_KEYWORDS, doc_tofile},
        {"tostring", (PyCFunction)array_tostring, METH_VARARGS, doc_tostring},
        {"byteswap",   (PyCFunction)array_byteswap,	1, doc_byteswap},
        {"astype", (PyCFunction)array_cast, 1, doc_cast},
	{"getfield", (PyCFunction)array_getfield, 
	 METH_VARARGS | METH_KEYWORDS, doc_getfield},
        {"copy", (PyCFunction)array_copy, 1, doc_copy},  
        {"resize", (PyCFunction)array_resize, 1, doc_resize}, 

	/* for subtypes */
	{"__array__", (PyCFunction)array_getarray, 1, doc_array_getarray},
	{"__array_wrap__", (PyCFunction)array_wraparray, 1, doc_wraparray},
	/* default version so it is found... -- only used for subclasses */
	{"__array_finalize__", (PyCFunction)array_finalize, 1, NULL},
	
	
	/* for the copy module */
        {"__copy__", (PyCFunction)array_copy, 1, doc_copy},	 
        {"__deepcopy__", (PyCFunction)array_deepcopy, 1, doc_deepcopy},  
	
        /* for Pickling */
        {"__reduce__", (PyCFunction) array_reduce, 1, doc_reduce},	
	{"__setstate__", (PyCFunction) array_setstate, 1, doc_setstate},
	{"dumps", (PyCFunction) array_dumps, 1, doc_dumps},
	{"dump", (PyCFunction) array_dump, 1, doc_dump},

	/* Extended methods added 2005 */
	{"fill", (PyCFunction)array_fill,
	 METH_VARARGS, doc_fill},
	{"transpose",	(PyCFunction)array_transpose, 
	 METH_VARARGS, doc_transpose},
	{"take",	(PyCFunction)array_take, 
	 METH_VARARGS|METH_KEYWORDS, doc_take},
	{"put",	(PyCFunction)array_put, 
	 METH_VARARGS|METH_KEYWORDS, doc_put},
	{"putmask",	(PyCFunction)array_putmask, 
	 METH_VARARGS|METH_KEYWORDS, doc_putmask},
	{"repeat",	(PyCFunction)array_repeat, 
	 METH_VARARGS|METH_KEYWORDS, doc_repeat},
	{"choose",	(PyCFunction)array_choose, 
	 METH_VARARGS, doc_choose},	
	{"sort",	(PyCFunction)array_sort, 
	 METH_VARARGS, doc_sort},
	{"argsort",	(PyCFunction)array_argsort, 
	 METH_VARARGS, doc_argsort},
	{"searchsorted",  (PyCFunction)array_searchsorted, 
	 METH_VARARGS, doc_searchsorted},	
	{"argmax",	(PyCFunction)array_argmax, 
	 METH_VARARGS, doc_argmax},
	{"argmin",  (PyCFunction)array_argmin,
	 METH_VARARGS, doc_argmin},
	{"reshape",	(PyCFunction)array_reshape, 
	 METH_VARARGS, doc_reshape},
	{"squeeze",	(PyCFunction)array_squeeze,
	 METH_VARARGS, doc_squeeze},
	{"view",  (PyCFunction)array_view, 
	 METH_VARARGS, doc_view},
	{"swapaxes", (PyCFunction)array_swapaxes,
	 METH_VARARGS, doc_swapaxes},
	{"max", (PyCFunction)array_max,
	 METH_VARARGS, doc_max},
	{"min", (PyCFunction)array_min,
	 METH_VARARGS, doc_min},
	{"ptp", (PyCFunction)array_ptp,
	 METH_VARARGS, doc_ptp},
	{"mean", (PyCFunction)array_mean,
	 METH_VARARGS|METH_KEYWORDS, doc_mean},
	{"trace", (PyCFunction)array_trace,
	 METH_VARARGS|METH_KEYWORDS, doc_trace},
	{"diagonal", (PyCFunction)array_diagonal,
	 METH_VARARGS|METH_KEYWORDS, doc_diagonal},
	{"clip", (PyCFunction)array_clip,
	 METH_VARARGS|METH_KEYWORDS, doc_clip},
	{"conj", (PyCFunction)array_conjugate,
	 METH_VARARGS, doc_conj},
	{"conjugate", (PyCFunction)array_conjugate,
	 METH_VARARGS, doc_conjugate},
	{"nonzero", (PyCFunction)array_nonzero,
	 METH_VARARGS, doc_nonzero},
	{"std", (PyCFunction)array_stddev,
	 METH_VARARGS|METH_KEYWORDS, doc_stddev},
	{"sum", (PyCFunction)array_sum,
	 METH_VARARGS|METH_KEYWORDS, doc_sum},
	{"cumsum", (PyCFunction)array_cumsum,
	 METH_VARARGS|METH_KEYWORDS, doc_cumsum},
	{"prod", (PyCFunction)array_prod,
	 METH_VARARGS|METH_KEYWORDS, doc_prod},
	{"cumprod", (PyCFunction)array_cumprod,
	 METH_VARARGS|METH_KEYWORDS, doc_cumprod},
	{"all", (PyCFunction)array_all,
	 METH_VARARGS, doc_all},
	{"any", (PyCFunction)array_any,
	 METH_VARARGS, doc_any},
	{"compress", (PyCFunction)array_compress,
	 METH_VARARGS|METH_KEYWORDS, doc_compress},
	{"flatten", (PyCFunction)array_flatten,
	 METH_VARARGS, doc_flatten},
	{"ravel", (PyCFunction)array_ravel,
	 METH_VARARGS, doc_ravel},
	{"setflags", (PyCFunction)array_setflags,
	 METH_VARARGS|METH_KEYWORDS, doc_setflags},
        {NULL,		NULL}		/* sentinel */
};

#undef _ARET


