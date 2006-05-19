/* Basearray*/


static PyGetSetDef basearray_getsetlist[] = {
	{"__array_data__",
	 (getter)array_dataptr_get,
	 NULL,
	 "Array protocol: data"},
	{"__array_typestr__",
	 (getter)array_typestr_get,
	 NULL,
	 "Array protocol: typestr"},
	{"__array_descr__",
	 (getter)array_protocol_descr_get,
	 NULL,
	 "Array protocol: descr"},
	{"__array_shape__",
	 (getter)array_shape_get,
	 NULL,
	 "Array protocol: shape"},
	{"__array_strides__",
	 (getter)array_protocol_strides_get,
	 NULL,
	 "Array protocol: strides"},
        {"__array_struct__",
         (getter)array_struct_get,
         NULL,
         "Array protocol: struct"},
	{NULL, NULL, NULL, NULL},  /* Sentinel */
};



static char BaseArraytype__doc__[] =
        "Base Array";


static PyTypeObject PyBaseArray_Type = {
        PyObject_HEAD_INIT(NULL)
        0,					  /*ob_size*/
        "numpy.basearray",		          /*tp_name*/
        sizeof(PyArrayObject),		          /*tp_basicsize*/
        0,					  /*tp_itemsize*/
        /* methods */
        (destructor)array_dealloc,		  /*tp_dealloc  */
        (printfunc)NULL,			  /*tp_print*/
        0,					  /*tp_getattr*/
        0,					  /*tp_setattr*/
        (cmpfunc)0,		                  /*tp_compare*/
        0,		                          /*tp_repr*/ /* XXX fill in */
        0,			                  /*tp_as_number*/
        0,	                                  /*tp_as_sequence*/
        0,			                  /*tp_as_mapping*/
        (hashfunc)0,			          /*tp_hash*/
        (ternaryfunc)0,			          /*tp_call*/
        (reprfunc)0,	                          /*tp_str*/ /* XXX fill in */

        (getattrofunc)0,			  /*tp_getattro*/
        (setattrofunc)0,			  /*tp_setattro*/
        &array_as_buffer,                      	  /*tp_as_buffer*/
        (Py_TPFLAGS_DEFAULT
         | Py_TPFLAGS_BASETYPE
         | Py_TPFLAGS_CHECKTYPES),                /*tp_flags*/
        /*Documentation string */
        BaseArraytype__doc__,			  /*tp_doc*/

        (traverseproc)0,			  /*tp_traverse */
        (inquiry)0,			          /*tp_clear */
        (richcmpfunc)0,	                          /*tp_richcompare */
        offsetof(PyArrayObject, weakreflist),     /*tp_weaklistoffset */

        /* Iterator support (use standard) */

        (getiterfunc)0,	                          /* tp_iter */
        (iternextfunc)0,			  /* tp_iternext */

        /* Sub-classing (new-style object) support */

        0,              			  /* tp_methods */
        0,					  /* tp_members */
        basearray_getsetlist,		          /* tp_getset */
        0,					  /* tp_base */
        0,					  /* tp_dict */
        0,					  /* tp_descr_get */
        0,					  /* tp_descr_set */
        0,					  /* tp_dictoffset */
        (initproc)0,		                  /* tp_init */
        array_alloc,	                          /* tp_alloc */
        (newfunc)array_new,		          /* tp_new */
        _pya_free,	                          /* tp_free */
        0,					  /* tp_is_gc */
        0,					  /* tp_bases */
        0,					  /* tp_mro */
        0,					  /* tp_cache */
        0,					  /* tp_subclasses */
        0					  /* tp_weaklist */
};

#define _ARET(x) PyArray_Return((PyArrayObject *)(x))


static PyObject *
basearray_getitem(PyObject *dummy, PyObject *args) 
{
	PyObject *barray, *key;
    
	if (!PyArg_ParseTuple(args, "OO", &barray, &key)) return NULL;
	
	if (!PyBaseArray_Check(barray)) {
		PyErr_SetString(PyExc_ValueError, 
                "can only get item from basearray instances");
		return NULL;
	}
        return array_subscript_nice(barray, key);
}

static PyObject *
array_setitem(PyObject *barray, PyObject *args)
{
	PyObject *key, *val;
    
	if (!PyArg_ParseTuple(args, "OO", &key, &val)) return NULL;
	
        if (array_ass_sub(barray, key, val) == -1) return NULL;
        Py_INCREF(Py_None);
        return Py_None;
}



static PyObject *
basearray_getshape(PyObject *dummy, PyObject *args)
{
	PyObject *barray;
    
	if (!PyArg_ParseTuple(args, "O", &barray)) return NULL;
	
	if (!PyBaseArray_Check(barray)) {
		PyErr_SetString(PyExc_ValueError, 
                "can only get shape of basearray instances");
		return NULL;
	}
        return array_shape_get(barray);
}

static PyObject *
basearray_setshape(PyObject *dummy, PyObject *args)
{
	PyObject *barray, *val;
    
	if (!PyArg_ParseTuple(args, "OO", &barray, &val)) return NULL;
	
	if (!PyBaseArray_Check(barray)) {
		PyErr_SetString(PyExc_ValueError, 
                "can only set shape of basearray instances");
		return NULL;
	}
        if (array_shape_set(barray, val) == -1) return NULL;
        Py_INCREF(Py_None);
        return Py_None;
}


static PyObject *
basearray_getstrides(PyObject *dummy, PyObject *args)
{
	PyObject *barray;
    
	if (!PyArg_ParseTuple(args, "O", &barray)) return NULL;
	
	if (!PyBaseArray_Check(barray)) {
		PyErr_SetString(PyExc_ValueError, 
                "can only get strides of basearray instances");
		return NULL;
	}
        return array_strides_get(barray);
}

static PyObject *
basearray_setstrides(PyObject *dummy, PyObject *args)
{
	PyObject *barray, *val;
    
	if (!PyArg_ParseTuple(args, "OO", &barray, &val)) return NULL;
	
	if (!PyBaseArray_Check(barray)) {
		PyErr_SetString(PyExc_ValueError, 
                "can only set strides of basearray instances");
		return NULL;
	}
        if (array_strides_set(barray, val) == -1) return NULL;
        Py_INCREF(Py_None);
        return Py_None;
}

static PyObject *
basearray_getdescr(PyObject *dummy, PyObject *args)
{
	PyObject *barray;
    
	if (!PyArg_ParseTuple(args, "O", &barray)) return NULL;
	
	if (!PyBaseArray_Check(barray)) {
		PyErr_SetString(PyExc_ValueError, 
                "can only get dtype of basearray instances");
		return NULL;
	}
        return array_descr_get(barray);
}

static PyObject *
basearray_setdescr(PyObject *dummy, PyObject *args)
{
	PyObject *barray, *val;
    
	if (!PyArg_ParseTuple(args, "OO", &barray, &val)) return NULL;
	
	if (!PyBaseArray_Check(barray)) {
		PyErr_SetString(PyExc_ValueError, 
                "can only set dtype of basearray instances");
		return NULL;
	}
        if (array_descr_set(barray, val) == -1) return NULL;
        Py_INCREF(Py_None);
        return Py_None;
}


static PyObject *
basearray_getreal(PyObject *dummy, PyObject *args)
{
	PyObject *barray;
    
	if (!PyArg_ParseTuple(args, "O", &barray)) return NULL;
	
	if (!PyBaseArray_Check(barray)) {
		PyErr_SetString(PyExc_ValueError, 
                "can only get real part of basearray instances");
		return NULL;
	}
        return array_real_get(barray);
}

static PyObject *
basearray_setreal(PyObject *dummy, PyObject *args)
{
	PyObject *barray, *val;
    
	if (!PyArg_ParseTuple(args, "OO", &barray, &val)) return NULL;
	
	if (!PyBaseArray_Check(barray)) {
		PyErr_SetString(PyExc_ValueError, 
                "can only set real part of basearray instances");
		return NULL;
	}
        if (array_real_set(barray, val) == -1) return NULL;
        Py_INCREF(Py_None);
        return Py_None;
}


static PyObject *
basearray_getimag(PyObject *dummy, PyObject *args)
{
	PyObject *barray;
    
	if (!PyArg_ParseTuple(args, "O", &barray)) return NULL;
	
	if (!PyBaseArray_Check(barray)) {
		PyErr_SetString(PyExc_ValueError, 
                "can only get imaginary part of basearray instances");
		return NULL;
	}
        return array_imag_get(barray);
}

static PyObject *
basearray_setimag(PyObject *dummy, PyObject *args)
{
	PyObject *barray, *val;
    
	if (!PyArg_ParseTuple(args, "OO", &barray, &val)) return NULL;
	
	if (!PyBaseArray_Check(barray)) {
		PyErr_SetString(PyExc_ValueError, 
                "can only set imaginary part of basearray instances");
		return NULL;
	}
        if (array_imag_set(barray, val) == -1) return NULL;
        Py_INCREF(Py_None);
        return Py_None;
}

static PyObject *
basearray_getflags(PyObject *dummy, PyObject *args)
{
        PyObject *barray;
    
	if (!PyArg_ParseTuple(args, "O", &barray)) return NULL;
	
	if (!PyBaseArray_Check(barray)) {
		PyErr_SetString(PyExc_ValueError, 
                "can only get flags of basearray instances");
		return NULL;
	}
        return PyArray_NewFlagsObject(barray);
}


static PyObject *
basearray_copy(PyObject *dummy, PyObject *args) {
        PyObject *barray;
    	PyArray_ORDER fortran=PyArray_CORDER;
        if (!PyArg_ParseTuple(args, "O|O&", &barray, PyArray_OrderConverter,
			      &fortran)) return NULL;
        
        if (!PyBaseArray_Check(barray)) {
		PyErr_SetString(PyExc_ValueError, 
                "can only copy basearray instances");
		return NULL;
	}
        
        return PyArray_NewCopy(barray, fortran);
}


static PyObject *
basearray_view(PyObject *dummy, PyObject *args) {
	PyObject *barray, *otype=NULL;
        PyArray_Descr *type=NULL;

	if (!PyArg_ParseTuple(args, "O|O", &barray, &otype)) return NULL;

        if (!PyBaseArray_Check(barray)) {
		PyErr_SetString(PyExc_ValueError, 
                "can only create views of basearray instances");
		return NULL;
	}
        
	if (otype) {
		if (PyType_Check(otype) &&			\
		    PyType_IsSubtype((PyTypeObject *)otype, 
				     &PyArray_Type)) {
			return PyArray_View(barray, NULL, 
                                            (PyTypeObject *)otype);
                }
		else {
			if (PyArray_DescrConverter(otype, &type) == PY_FAIL) 
				return NULL;
		}
	}
        
        return PyArray_View(barray, type, NULL);
}

static PyObject *
basearray_astype(PyArrayObject *dummy, PyObject *args) 
{
	PyArray_Descr *descr=NULL;
	PyObject *barray, *obj;
	
        if (!PyArg_ParseTuple(args, "OO&", &barray, PyArray_DescrConverter,
			      &descr)) return NULL;
	
        if (!PyBaseArray_Check(barray)) {
		PyErr_SetString(PyExc_ValueError, 
                "can only apply astype to basearray instances");
		return NULL;
	}
        
	if (descr == ((PyArrayObject*)barray)->descr) {
		obj = _ARET(PyArray_NewCopy(barray,0));
		Py_XDECREF(descr);
		return obj;
	}
	return _ARET(PyArray_CastToType(barray, descr, 0));
}	


static PyObject *
basearray_wrap(PyObject *dummy, PyObject *args)
{
	PyObject *barray, *arr, *ret;
    
	if (!PyArg_ParseTuple(args, "OO", &barray, &arr)) return NULL;
	
	if (!PyBaseArray_Check(barray)) {
		PyErr_SetString(PyExc_ValueError, 
                "can only wrap basearray instances");
		return NULL;
	}
        
	Py_INCREF(PyArray_DESCR(arr));
	ret = PyArray_NewFromDescr(barray->ob_type, 
				   PyArray_DESCR(arr),
				   PyArray_NDIM(arr),
				   PyArray_DIMS(arr), 
				   PyArray_STRIDES(arr), PyArray_DATA(arr),
				   PyArray_FLAGS(arr), (PyObject *)barray);
	if (ret == NULL) return NULL;
	Py_INCREF(arr);
	PyArray_BASE(ret) = arr;
	return ret;
}



static PyObject *
basearray_sort(PyArrayObject *dummy, PyObject *args, PyObject *kwds) 
{
        PyObject *barray;
	int axis=-1;
	int val;
	PyArray_SORTKIND which=PyArray_QUICKSORT;
	static char *kwlist[] = {"array", "axis", "kind", NULL};
	
	if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|iO&", 
                                         kwlist,  &barray, &axis,
					 PyArray_SortkindConverter, &which))
		return NULL;
	
	if (!PyBaseArray_Check(barray)) {
		PyErr_SetString(PyExc_ValueError, 
                "can only sort basearray instances");
		return NULL;
	}
        
	val = PyArray_Sort(barray, axis, which);
	if (val < 0) return NULL;
	Py_INCREF(Py_None);
	return Py_None;
}


static PyObject *
basearray_newfromobject(PyTypeObject *dummy, PyObject *args, PyObject *kwds)
{
	PyObject *subtype, *op, *ret=NULL;
	static char *kwd[] = {"subtype", "object", "dtype", "order", NULL};
	int nd;
	PyArray_Descr *type = NULL;
	PyArray_ORDER order=PyArray_ANYORDER;
	int flags = ENSURECOPY;

	if(!PyArg_ParseTupleAndKeywords(args, kwds, "OO|O&O&", kwd, &subtype, &op, 
					PyArray_DescrConverter2,&type, 
					PyArray_OrderConverter, &order)) 
		return NULL;

        if (!PyObject_IsSubclass(subtype, &PyBaseArray_Type)) {
            PyErr_SetString(PyExc_TypeError, "subtype must derive from basearray");
            return NULL;
        }

        if (order == PyArray_CORDER) 
                flags |= CONTIGUOUS;
	else if (order == PyArray_FORTRANORDER)
                flags |= CONTIGUOUS;


	if ((ret = PyArray_CheckFromAny(op, type, 0, 0, flags, NULL)) == NULL) 
		return NULL;

        if (subtype != ret->ob_type) {
                PyObject *arr = ret;
            	Py_INCREF(PyArray_DESCR(arr));
                ret = PyArray_NewFromDescr(subtype, 
                                           PyArray_DESCR(arr),
                                           PyArray_NDIM(arr),
                                           PyArray_DIMS(arr), 
                                           PyArray_STRIDES(arr), PyArray_DATA(arr),
                                           PyArray_FLAGS(arr), NULL);
                if (ret == NULL) {
                    Py_DECREF(arr);
                    return NULL;
                }
                PyArray_BASE(ret) = arr;
        }
        return ret;
}        
 
    
#undef _ARET