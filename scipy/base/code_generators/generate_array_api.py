#  doc is comment_documentation

# use list so order is preserved.
objectapi_list = [
    (r"""Set internal structure with number functions that all
    arrays will use
    """,
     'SetNumericOps','PyObject *dict','int'),

    (r"""Get dictionary showing number functions that all
    arrays will use
    """,
     'GetNumericOps','void','PyObject *'),


    (r"""For object arrays, increment all internal references.
    """,
     'INCREF','PyArrayObject *','int'),

    (r"""Decrement all internal references for object arrays.
    """,
     'XDECREF','PyArrayObject *','int'),

    (r"""Set the array print function to be a Python function.
    """,
     'SetStringFunction','PyObject *op, int repr','void'),

    (r"""Get the PyArray_Descr structure for a type.
    """,
     'DescrFromType','int','PyArray_Descr *'),

    (r"""Get pointer to zero of correct type for array.
    """,
     'Zero', 'PyArrayObject *', 'char *'),

    (r"""Get pointer to one of correct type for array
    """,
     'One', 'PyArrayObject *', 'char *'),


    (r"""Cast an array to a different type.
    """,
     'Cast','PyArrayObject *, int','PyObject *'),

    (r"""Cast an array using typecode structure.
    """,
     'CastToType','PyArrayObject *, PyArray_Typecode *','PyObject *'),

    (r"""Cast to an already created array.
    """,
     'CastTo', 'PyArrayObject *, PyArrayObject *', 'int'),

    (r"""Check the type coercion rules.
    """,
     'CanCastSafely','int fromtype, int totype','int'),

    (r"""
    """,
     'CanCastTo','PyArray_Typecode *, PyArray_Typecode *', 'Bool'),

    (r"""Return the typecode of the array a Python object would be
    converted to
    """,
     'ObjectType','PyObject *, int','int'),

    (r"""
    """,
     'ArrayType','PyObject *, PyArray_Typecode *, PyArray_Typecode *','void'),

    (r"""
    """,
     'ConvertToCommonType', 'PyObject *, int *', 'PyArrayObject **'),

    (r"""Return type typecode from array scalar.
    """,
     'TypecodeFromScalar','PyObject *, PyArray_Typecode *','void'),

    (r"""Compute the size of an array (in number of items)
    """,
     'Size','PyObject *','intp'),

    (r"""Get scalar-equivalent to 0-d array
    """,
     'Scalar', 'void *, int, int, int', 'PyObject *'),

    (r"""Get scalar-equivalent to 0-d array
    """,
     'ToScalar', 'void *, PyArrayObject *', 'PyObject *'),

    (r"""Get 0-dim array from scalar
    """,
     'FromScalar', 'PyObject *, PyArray_Typecode *', 'PyObject *'),

    (r"""Convert to c-type
    """,
     'ScalarAsCtype', 'PyObject *, void *', 'void'),

    (r"""Register Data type
    """,
     'RegisterDataType', 'PyTypeObject *', 'int'),

    (r"""Insert Descr Table
    """,
     'RegisterDescrForType', 'int, PyArray_Descr *', 'int'),

    (r"""Construct an empty array from dimensions and typenum
    """,
     'FromDims','int nd, int *, int typenum','PyObject *'),

    (r"""Construct an array from dimensions, typenum, and a pointer
    to the data.  Python will never free this (unless you later set
    the OWN_DATA flag).
    """,
     'FromDimsAndData','int, int *, int, char *','PyObject *'),

    (r"""Construct an array from an arbitrary Python Object.
    Last two integers are min_dimensions, and max_dimensions.
    If max_dimensions = 0, then any number of dimensions are allowed.
    Fix the dimension by setting min_dimension == max_dimension.
    If the array is already contiguous (and aligned and not swapped)
    no copy is done, just a new reference created.
    Base-class ndarray is returned.
    """,
     'ContiguousFromObject',
     'PyObject *, int typenum, int, int',
     'PyObject *'),

    (r"""Construct an array from an arbitrary Python Object.
    Last two integers are min_dimensions, and max_dimensions.
    If max_dimensions = 0, then any number of dimensions are allowed.
    Fix the dimension by setting min_dimension == max_dimension.
    If the array is already contiguous (and aligned and not swapped)
    no copy is done, just a new reference created.  Subclasses
    passed through.
    """,
     'ContiguousFromAny',
     'PyObject *, int typenum, int, int',
     'PyObject *'),


    (r"""Same as ContiguousFromObject except ensure a copy.
    """,
     'CopyFromObject','PyObject *, int, int, int','PyObject *'),

    (r"""Can return a discontiguous array (but aligned and byteswapped)
    """,
     'FromObject','PyObject *, int, int, int','PyObject *'),

    (r"""
    """,
     'FromAny', 'PyObject *, PyArray_Typecode *, int, int, int', 'PyObject *'),

    (r"""
    """,
     'EnsureArray', 'PyObject *', 'PyObject *'),

    (r"""
    """,
     'FromFile', 'FILE *, PyArray_Typecode *, intp, char *','PyObject *'),

    (r"""
    """,
     'FromBuffer', 'PyObject *, PyArray_Typecode *, intp, intp, int',
     'PyObject *'),

    (r"""Return either an array or the appropriate Python object if the
    array is 0d and matches a Python type.
    """,
     'Return','PyArrayObject *','PyObject *'),

    (r"""Get a subset of bytes from each element of the array
    """,
     'GetField', 'PyArrayObject *, PyArray_Typecode *, int', 'PyObject *'),

    (r"""
    """,
     'Byteswap', 'PyArrayObject *, Bool', 'PyObject *'),

    (r"""Resize (reallocate data).  Only works if nothing else is
    referencing this array and it is contiguous.
    """,
     'Resize','PyArrayObject *ap, PyArray_Dims *newshape','PyObject *'),

    (r"""Copy an array.
    """,
     'NewCopy','PyArrayObject *, int','PyObject *'),

    (r"""Like FromDimsAndData but uses the Descr structure instead of
    typecode as input.
    """,
     'FromDimsAndDataAndDescr','int, int *, PyArray_Descr *, char *',
     'PyObject *'),

    (r"""Copy an Array into another array.
    """,
     'CopyInto', 'PyArrayObject *dest, PyArrayObject *src', 'int'),

    (r"""To List
    """,
     'ToList','PyArrayObject *', 'PyObject *'),

    (r"""To File
    """,
     'ToFile','PyArrayObject *, FILE *, char *, char *', 'int'),

    (r"""
    """,
     'Dump', 'PyObject *, PyObject *, int', 'int'),

    (r"""
    """,
     'Dumps', 'PyObject *, int', 'PyObject *'),


    (r"""Is the typenum valid?
    """,
     'ValidType','int','int'),

    (r"""Update Several Flags at once.
    """,
     'UpdateFlags','PyArrayObject *, int','void'),

    (r"""Generic new array creation routine.
    """,
     'New','PyTypeObject *, int nd, intp *dims, int type, intp *strides, void *data, int itemsize, int fortran, PyObject *obj', 'PyObject *'),

    (r"""Get Priority from object
    """,
     'GetPriority', 'PyObject *, double', 'double'),

    (r"""Get Iterator.
    """,
     'IterNew','PyObject *', 'PyObject *'),

    (r"""Get MultiIterator,
    """,
     'MultiIterNew','int, ...', 'PyObject *'),

    (r"""
    """,
     'PyIntAsInt', 'PyObject *', 'int'),

    (r"""
    """,
     'PyIntAsIntp','PyObject *', 'intp'),

    (r"""
    """,
     'Broadcast', 'PyArrayMultiIterObject *', 'int'),

    (r"""
    """,
     'FillObjectArray', 'PyArrayObject *, PyObject *','void'),

    (r"""
    """,
     'FillWithScalar','PyArrayObject *, PyObject *', 'int'),

    (r"""
    """,
     'CheckStrides', 'int, int, intp, intp *, intp *', 'Bool')

    ]

multiapi_list = [
    (r"""Return Transpose.
    """,
     'Transpose','PyArrayObject *, PyArray_Dims *','PyObject *'),

    (r"""Take
    """,
     'Take','PyArrayObject *, PyObject *, int axis','PyObject *'),

    (r"""Put values into an array
    """,
     'Put','PyArrayObject *arr, PyObject *items, PyObject *values','PyObject *'),

    (r"""Put values into an array according to a mask.
    """,
     'PutMask','PyArrayObject *arr, PyObject *mask, PyObject *values','PyObject *'),

    (r"""Repeat the array.
    """,
     'Repeat','PyArrayObject *, PyObject *, int','PyObject *'),

    (r"""Numeric.choose()
    """,
     'Choose','PyArrayObject *, PyObject *','PyObject *'),

    (r"""Sort an array
    """,
     'Sort','PyArrayObject *, int', 'PyObject *'),

    (r"""ArgSort an array
    """,
     'ArgSort','PyArrayObject *, int','PyObject *'),

    (r"""Numeric.searchsorted(a,v)
    """,
     'SearchSorted','PyArrayObject *, PyObject *','PyObject *'),

    (r"""ArgMax
    """,
     'ArgMax','PyArrayObject *, int','PyObject *'),

    (r"""ArgMin
    """,
     'ArgMin','PyArrayObject *, int','PyObject *'),

    (r"""Reshape an array
    """,
     'Reshape','PyArrayObject *, PyObject *','PyObject *'),

    (r"""New shape for an array
    """,
     'Newshape','PyArrayObject *, PyArray_Dims *','PyObject *'),

    (r"""
    """,
     'Squeeze','PyArrayObject *','PyObject *'),

    (r"""View
    """,
     'View','PyArrayObject *, PyArray_Typecode *','PyObject *'),

    (r"""SwapAxes
    """,
     'SwapAxes','PyArrayObject *, int, int','PyObject *'),

    (r"""Max
    """,
     'Max','PyArrayObject *, int','PyObject *'),

    (r"""Min
    """,
     'Min','PyArrayObject *, int','PyObject *'),

    (r"""Ptp
    """,
     'Ptp','PyArrayObject *, int','PyObject *'),

    (r"""Mean
    """,
     'Mean','PyArrayObject *, int, int','PyObject *'),

    (r"""Trace
    """,
     'Trace','PyArrayObject *, int, int, int, int','PyObject *'),

    (r"""Diagonal
    """,
     'Diagonal','PyArrayObject *, int, int, int','PyObject *'),

    (r"""Clip
    """,
     'Clip','PyArrayObject *, PyObject *, PyObject *','PyObject *'),

    (r"""Conjugate
    """,
     'Conjugate','PyArrayObject *','PyObject *'),

    (r"""Nonzero
    """,
     'Nonzero','PyArrayObject *','PyObject *'),

    (r"""Std
    """,
     'Std','PyArrayObject *, int, int','PyObject *'),

    (r"""Sum
    """,
     'Sum','PyArrayObject *, int, int','PyObject *'),

    (r"""CumSum
    """,
     'CumSum','PyArrayObject *, int, int','PyObject *'),

    (r"""Prod
    """,
     'Prod','PyArrayObject *, int, int','PyObject *'),

    (r"""CumProd
    """,
     'CumProd','PyArrayObject *, int, int','PyObject *'),

    (r"""All
    """,
     'All','PyArrayObject *, int','PyObject *'),

    (r"""Any
    """,
     'Any','PyArrayObject *, int','PyObject *'),

    (r"""Compress
    """,
     'Compress','PyArrayObject *, PyObject *, int','PyObject *'),

    (r"""Flatten
    """,
     'Flatten','PyArrayObject *, int','PyObject *'),

    (r"""Ravel
    """,
     'Ravel','PyArrayObject *, int','PyObject *'),

    (r"""Multiply a List
    """,
     'MultiplyList','intp *lp, int n','intp'),

    (r"""Multiply a List of ints
    """,
     'MultiplyIntList','int *lp, int n','int'),

    (r"""Produce a pointer into array
    """,
     'GetPtr', 'PyArrayObject *, intp*', 'char *'),

    (r"""Compare Lists
    """,
     'CompareLists','intp *, intp *, int n','int'),

    (r"""Simulat a C-array
    """,
     "AsCArray",'PyObject **, void *ptr, intp *, int, int, int','int'),

    (r"""Convert to a 1D C-array
    """,
     'As1D','PyObject **, char **ptr, int *d1, int typecode','int'),

    (r"""Convert to a 2D C-array
    """,
     'As2D','PyObject **, char ***ptr, int *d1, int *d2, int typecode','int'),

    (r"""Free pointers created if As2D is called
    """,
     'Free','PyObject *, void *','int'),

    (r"""Useful to pass as converter function for O& processing in
    PyArgs_ParseTuple.
    """,
     'Converter','PyObject *, PyObject **','int'),

    (r"""PyArray_IntpFromSequence
    """,
     'IntpFromSequence', 'PyObject *, intp *, int', 'int'),

    (r"""Concatenate an arbitrary Python sequence into
     an array.
    """,
     'Concatenate','PyObject *, int','PyObject *'),

    (r"""Numeric.innerproduct(a,v)
    """,
     'InnerProduct','PyObject *, PyObject *','PyObject *'),

    (r"""Numeric.matrixproduct(a,v)
    """,
     'MatrixProduct','PyObject *, PyObject *','PyObject *'),

    (r"""Fast Copy and Transpose
    """,
     'CopyAndTranspose','PyObject *','PyObject *'),

    (r"""Numeric.correlate(a1,a2,mode)
    """,
     'Correlate','PyObject *, PyObject *, int mode','PyObject *'),

    (r"""Typestr converter
    """,
     'TypestrConvert', 'int, int', 'int'),

    (r"""Get typenum from an object -- a converter function
    """,
     'TypecodeConverter','PyObject *, PyArray_Typecode *', 'int'),

    (r"""Get intp chunk from sequence
    """,
     'IntpConverter', 'PyObject *, PyArray_Dims *', 'int'),

    (r"""Get buffer chunk from object
    """,
     'BufferConverter', 'PyObject *, PyArray_Chunk *', 'int'),

    (r"""Get axis from an object (possibly None) -- a converter function,
    """,
     'AxisConverter','PyObject *, int *', 'int'),

    (r"""Convert an object to true / false
    """,
     'BoolConverter','PyObject *, Bool *', 'int'),

    (r"""
    """,
     'EquivalentTypes', 'PyArray_Typecode *, PyArray_Typecode *', 'Bool'),

    (r"""
    """,
     'EquivArrTypes', 'PyArrayObject *, PyArrayObject *', 'Bool'),

    (r"""
    """,
     'EquivalentTypenums', 'int, int', 'Bool'),

    (r"""Zeros
    """,
     'Zeros', 'int, intp *, PyArray_Typecode *', 'PyObject *'),

    (r"""Empty
    """,
     'Empty', 'int, intp *, PyArray_Typecode *', 'PyObject *'),


    (r"""Where
    """,
      'Where', 'PyObject *, PyObject *, PyObject *', 'PyObject *'),

    (r"""Arange
    """,
     'Arange', 'double, double, double, int', 'PyObject *')

    ]


types = ['Generic','Numeric','Integer','SignedInteger','UnsignedInteger', 'Inexact',
         'Floating', 'ComplexFloating', 'Flexible', 'Character',
         'Bool','Byte','Short','Int', 'Long', 'LongLong', 'UByte', 'UShort',
         'UInt', 'ULong', 'ULongLong', 'Float', 'Double', 'LongDouble',
         'CFloat', 'CDouble', 'CLongDouble', 'Object', 'String', 'Unicode',
         'Void']

# API fixes for __arrayobject_api.h

fixed = 5
numtypes = len(types) + fixed
numobject = len(objectapi_list) + numtypes
nummulti = len(multiapi_list)
numtotal = numobject + nummulti


module_list = []
extension_list = []
init_list = []

# setup types
for k, atype in enumerate(types):
    num = fixed + k
    astr = "        (void *) &Py%sArrType_Type," % types[k]
    init_list.append(astr)
    astr = "static PyTypeObject Py%sArrType_Type;" % types[k]
    module_list.append(astr)
    astr = "#define Py%sArrType_Type (*(PyTypeObject *)PyArray_API[%d])" % \
           (types[k], num)
    extension_list.append(astr)


#setup object API
for k, item in enumerate(objectapi_list):
    num = numtypes + k
    astr = "static %s PyArray_%s \\\n       (%s);" % \
           (item[3],item[1],item[2])
    module_list.append(astr)
    astr = "#define PyArray_%s \\\n        (*(%s (*)(%s)) \\\n"\
           "         PyArray_API[%d])" % (item[1],item[3],item[2],num)
    extension_list.append(astr)
    astr = "        (void *) PyArray_%s," % item[1]
    init_list.append(astr)


##outstr = r"""
###ifdef _ARRAYOBJECT

##static PyTypeObject PyArray_Type;
##static PyTypeObject PyArrayIter_Type;

##%s


###else

###define PyArray_Type (*(PyTypeObject *)PyArray_API[0])
###define PyArrayIter_Type (*(PyTypeObject *)PyArray_API[1])

##%s

###endif
##""" % ('\n'.join(module_list), '\n'.join(extension_list))

### Write out to header
##fid = open('__arrayobject_api.h','w')
##fid.write(outstr)
##fid.close()


##outstr = r"""
##/* Export only these pointers */

##void *arrayobject_API[] = {
##        (void *) &PyArray_Type,
##        (void *) &PyArrayIter_Type,
##%s
##};
##""" % '\n'.join(init_list)

###Write out to c-code
##fid = open('__arrayobject_api.c','w')
##fid.write(outstr)
##fid.close()


#module_list = []
#extension_list = []
#init_list = []

# setup multiarray module API
for k, item in enumerate(multiapi_list):
    num = numobject + k
    astr = "static %s PyArray_%s \\\n       (%s);" % \
           (item[3],item[1],item[2])
    module_list.append(astr)
    astr = "#define PyArray_%s \\\n        (*(%s (*)(%s)) \\\n"\
           "         PyArray_API[%d])" % (item[1],item[3],item[2],num)
    extension_list.append(astr)
    astr = "        (void *) PyArray_%s," % item[1]
    init_list.append(astr)


outstr = r"""
#ifdef _MULTIARRAYMODULE

static PyTypeObject PyBigArray_Type;
static PyTypeObject PyArray_Type;
static PyTypeObject PyArrayIter_Type;
static PyTypeObject PyArrayMapIter_Type;
static PyTypeObject PyArrayMultiIter_Type;
static int PyArray_NUMUSERTYPES=0;

%s

#else

#if defined(PY_ARRAY_UNIQUE_SYMBOL)
#define PyArray_API PY_ARRAY_UNIQUE_SYMBOL
#endif

#if defined(NO_IMPORT) || defined(NO_IMPORT_ARRAY)
extern void **PyArray_API;
#else
#if defined(PY_ARRAY_UNIQUE_SYMBOL)
void **PyArray_API;
#else
static void **PyArray_API=NULL;
#endif
#endif

#define PyBigArray_Type (*(PyTypeObject *)PyArray_API[0])
#define PyArray_Type (*(PyTypeObject *)PyArray_API[1])
#define PyArrayIter_Type (*(PyTypeObject *)PyArray_API[2])
#define PyArrayMultiIter_Type (*(PyTypeObject *)PyArray_API[3])
#define PyArray_NUMUSERTYPES (*(int *)PyArray_API[4])

%s

#if !defined(NO_IMPORT_ARRAY) && !defined(NO_IMPORT)
static int
import_array(void)
{
  PyObject *numpy = PyImport_ImportModule("scipy.base.multiarray");
  PyObject *c_api = NULL;
  if (numpy == NULL) return -1;
  c_api = PyObject_GetAttrString(numpy, "_ARRAY_API");
  if (c_api == NULL) {Py_DECREF(numpy); return -1;}
  if (PyCObject_Check(c_api)) {
      PyArray_API = (void **)PyCObject_AsVoidPtr(c_api);
  }
  Py_DECREF(c_api);
  Py_DECREF(numpy);
  if (PyArray_API == NULL) return -1;
  return 0;
}
#endif

#endif

""" % ('\n'.join(module_list),
       '\n'.join(extension_list))

# Write to header
fid = open('__multiarray_api.h','w')
fid.write(outstr)
fid.close()


outstr = r"""
/* These pointers will be stored in the C-object for use in other
    extension modules
*/

void *PyArray_API[] = {
        (void *) &PyBigArray_Type,
        (void *) &PyArray_Type,
        (void *) &PyArrayIter_Type,
        (void *) &PyArrayMultiIter_Type,
        (int *) &PyArray_NUMUSERTYPES,
%s
};
""" % '\n'.join(init_list)

# Write to c-code
fid = open('__multiarray_api.c','w')
fid.write(outstr)
fid.close()
