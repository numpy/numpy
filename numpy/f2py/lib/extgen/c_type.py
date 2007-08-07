"""
C types.


"""

__all__ = ['CType', 'CTypeAlias', 'CTypeFuncAlias', 'CTypePtr', 'CTypeStruct', 'CDecl',
           'CTypePython']

from base import Component

class CTypeBase(Component):

    template = '%(name)s'
    template_typedef = ''
    default_container_label = '<IGNORE>'
    default_component_class_name = 'CType'

    @property
    def provides(self):
        return '%s_%s' % (self.__class__.__name__, self.name)

    def initialize(self, name, *components):
        self.name = name
        map(self.add, components)
        return self

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, ', '.join([repr(self.name)]+[repr(c) for (c,l) in self.components]))
    
    def update_containers(self):
        self.container_TypeDef += self.evaluate(self.template_typedef)

    def __str__(self):
        return self.name

    def get_pyarg_fmt(self, arg):
        if arg.input_intent=='hide': return None
        return 'O'

    def get_pyarg_obj(self, arg):
        if arg.input_intent=='hide': return None
        return '&' + arg.pycvar
    
    def get_pyret_fmt(self, arg):
        if arg.output_intent=='return':
            # set_converters ensures tha all return values a new references
            return 'N'
        return
    
    def get_pyret_obj(self, arg):
        if arg.output_intent=='return':
            return arg.retpycvar
        return

    def set_Decl(self, arg):
        if arg.input_intent!='hide':
            arg += CDecl(self, '%s = NULL' % (arg.pycvar))

        if arg.output_intent!='hide':
            arg += CDecl(self, '%s = NULL' % (arg.retpycvar))

    def set_converters(self, arg):
        """
        Notes for user:
          if arg is intent(optional, in, out) and not specified
          as function argument then function may created but
          it must then have *new reference* (ie use Py_INCREF
          unless it is a new reference already).
        """
        # this method is called from PyCFunction.update_containers(),
        # note that self.parent is None put arg.parent is PyCFunction
        # instance.
        eval_a = arg.evaluate
        FromPyObj = arg.container_FromPyObj
        PyObjFrom = arg.container_PyObjFrom
        if arg.output_intent=='return':
            if arg.input_intent in ['optional', 'extra']:
                FromPyObj += eval_a('''\
if (!(%(pycvar)s==NULL)) {
  /* make %(pycvar)r a new reference */
  %(retpycvar)s = %(pycvar)s;
  Py_INCREF((PyObject*)%(retpycvar)s);
}
''')
                PyObjFrom += eval_a('''\
if (%(retpycvar)s==NULL) {
  /* %(pycvar)r was not specified */
  if (%(pycvar)s==NULL) {
    %(retpycvar)s = Py_None;
    Py_INCREF((PyObject*)%(retpycvar)s);
  } else {
    %(retpycvar)s = %(pycvar)s;
    /* %(pycvar)r must be a new reference or expect a core dump. */
  }
} elif (!(%(retpycvar)s == %(pycvar)s)) {
  /* a new %(retpycvar)r was created, undoing %(pycvar)s new reference */
  Py_DECREF((PyObject*)%(pycvar)s);
}
''')
            elif arg.input_intent=='hide':
                PyObjFrom += eval_a('''\
if (%(retpycvar)s==NULL) {
  %(retpycvar)s = Py_None;
  Py_INCREF((PyObject*)%(retpycvar)s);
} /* else %(retpycvar)r must be a new reference or expect a core dump. */
''')
            elif arg.input_intent=='required':
                 FromPyObj += eval_a('''\
/* make %(pycvar)r a new reference */
%(retpycvar)s = %(pycvar)s;
Py_INCREF((PyObject*)%(retpycvar)s);
''')
                 PyObjFrom += eval_a('''\
if (!(%(retpycvar)s==%(pycvar)s)) {
  /* a new %(retpycvar)r was created, undoing %(pycvar)r new reference */
  /* %(retpycvar)r must be a new reference or expect a core dump. */
  Py_DECREF((PyObject*)%(pycvar)s);
}
''')
        return

class _CatchTypeDef(Component): # for doctest
    template = '%(TypeDef)s'
    default_container_label = '<IGNORE>'
    container_options = dict(TypeDef=dict(default=''))
    def initialize(self, ctype):
        self.add(ctype)
        return self
    
class CType(CTypeBase):

    """ CType(<name>)

    Represents any predefined type in C.

    >>> cint = CType('int')
    >>> print cint
    int
    >>> _CatchTypeDef(cint).generate()
    ''
    """

    def initialize(self, name):
        if isinstance(name, CTypeBase):
            return name
        if isinstance(name, type) or name in ['cell', 'generator', 'cobject', 'instance']:
            return CTypePython(name)
        try:
            return Component.get(name)
        except KeyError:
            pass
        self.name = name
        return self
    def update_containers(self):
        pass

    def set_pyarg_decl(self, arg):
        pass
    def set_titles(self, arg):
        pass

class CTypeAlias(CTypeBase):

    """ CTypeAlias(<name>, <ctype>)

    >>> aint = CTypeAlias('aint', 'int')
    >>> print aint
    aint
    >>> print _CatchTypeDef(aint).generate()
    typedef int aint;
    """

    template_typedef = 'typedef %(ctype_name)s %(name)s;'

    def initialize(self, name, ctype):
        self.name = name
        if isinstance(ctype, str): ctype = CType(ctype)
        self.ctype_name = ctype.name
        self.add(ctype)
        return self
    
class CTypeFuncAlias(CTypeBase):

    """
    CTypeFuncAlias(<name>, <return ctype>, *(<argument ctypes>))

    >>> ifunc = CTypeFuncAlias('ifunc', 'int')
    >>> print ifunc
    ifunc
    >>> print _CatchTypeDef(ifunc).generate()
    typedef int (*ifunc)(void);
    >>> ifunc += 'double'
    >>> print _CatchTypeDef(ifunc).generate()
    typedef int (*ifunc)(double);
    """

    template_typedef = 'typedef %(RCType)s (*%(name)s)(%(ACType)s);'
    container_options = dict(RCType = dict(default='void'),
                             ACType = dict(default='void', separator=', '))
    component_container_map = dict(CType = 'ACType')
    default_component_class_name = 'CType'

    def initialize(self, name, *components):
        self.name = name
        if components:
            self.add(components[0], 'RCType')
        map(self.add, components[1:])
        return self

class CTypePtr(CTypeBase):

    """
    CTypePtr(<ctype>)

    >>> int_ptr = CTypePtr('int')
    >>> print int_ptr
    int_ptr
    >>> print _CatchTypeDef(int_ptr).generate()
    typedef int* int_ptr;
    >>> int_ptr_ptr = CTypePtr(int_ptr)
    >>> print int_ptr_ptr
    int_ptr_ptr
    >>> print _CatchTypeDef(int_ptr_ptr).generate()
    typedef int* int_ptr;
    typedef int_ptr* int_ptr_ptr;
    """

    template_typedef = 'typedef %(ctype_name)s* %(name)s;'

    def initialize(self, ctype):
        if isinstance(ctype, str): ctype = CType(ctype)
        self.name = '%s_ptr' % (ctype)
        self.ctype_name = ctype.name
        self.add(ctype)
        return self

class CTypeStruct(CTypeBase):

    """
    CTypeStruct(<name>, *(<declarations>))

    >>> s = CTypeStruct('s', CDecl('int','a'))
    >>> print s
    s
    >>> print _CatchTypeDef(s).generate()
    typedef struct {
      int a;
    } s;
    >>> s += CDecl(CTypeFuncAlias('ft'), 'f')
    >>> print _CatchTypeDef(s).generate()
    typedef void (*ft)(void);
    typedef struct {
      int a;
      ft f;
    } s;

    """

    container_options = dict(Decl = dict(default='<KILLLINE>', use_indent=True))
    default_component_class_name = None #'CDecl'
    component_container_map = dict(CDecl='Decl')

    template_typedef = '''\
typedef struct {
  %(Decl)s
} %(name)s;'''

    def initialize(self, name, *components):
        self.name = name
        map(self.add, components)
        return self

class CDecl(Component):

    """
    CDecl(<ctype>, *(<names with or without initialization>))

    >>> ad = CDecl('int')
    >>> ad.generate()
    ''
    >>> ad += 'a'
    >>> print ad.generate()
    int a;
    >>> ad += 'b'
    >>> print ad.generate()
    int a, b;
    >>> ad += 'c = 1'
    >>> print ad.generate()
    int a, b, c = 1;
    """

    template = '%(CTypeName)s %(Names)s;'
    container_options = dict(Names=dict(default='<KILLLINE>', separator=', '),
                             CTypeName=dict())
    default_component_class_name = 'str'
    component_container_map = dict(str = 'Names')

    def initialize(self, ctype, *names):
        if isinstance(ctype, str): ctype = CType(ctype)
        self.add(ctype, 'CTypeName')
        map(self.add, names)
        return self

class CTypePython(CTypeBase):

    """ CTypePython(<python type object or 'cobject' or 'cell' or 'generator'>)

    >>> from __init__ import * #doctest: +ELLIPSIS
    Ignoring...
    >>> m = ExtensionModule('test_CTypePython')
    >>> f = PyCFunction('func')
    >>> f += PyCArgument('i', int, output_intent='return')
    >>> f += PyCArgument('l', long, output_intent='return')
    >>> f += PyCArgument('f', float, output_intent='return')
    >>> f += PyCArgument('c', complex, output_intent='return')
    >>> f += PyCArgument('s', str, output_intent='return')
    >>> f += PyCArgument('u', unicode, output_intent='return')
    >>> f += PyCArgument('t', tuple, output_intent='return')
    >>> f += PyCArgument('lst', list, output_intent='return')
    >>> f += PyCArgument('d', dict, output_intent='return')
    >>> f += PyCArgument('set', set, output_intent='return')
    >>> f += PyCArgument('o1', object, output_intent='return')
    >>> f += PyCArgument('o2', object, output_intent='return')
    >>> m += f
    >>> b = m.build() #doctest: +ELLIPSIS
    exec_command...
    >>> b.func(23, 23l, 1.2, 1+2j, 'hello', u'hei', (2,'a'), [-2], {3:4}, set([1,2]), 2, '15')
    (23, 23L, 1.2, (1+2j), 'hello', u'hei', (2, 'a'), [-2], {3: 4}, set([1, 2]), 2, '15')

    >>> print b.func.__doc__
      func(i, l, f, c, s, u, t, lst, d, set, o1, o2) -> (i, l, f, c, s, u, t, lst, d, set, o1, o2)
    <BLANKLINE>
    Required arguments:
      i - a python int object
      l - a python long object
      f - a python float object
      c - a python complex object
      s - a python str object
      u - a python unicode object
      t - a python tuple object
      lst - a python list object
      d - a python dict object
      set - a python set object
      o1 - a python object
      o2 - a python object
    <BLANKLINE>
    Return values:
      i - a python int object
      l - a python long object
      f - a python float object
      c - a python complex object
      s - a python str object
      u - a python unicode object
      t - a python tuple object
      lst - a python list object
      d - a python dict object
      set - a python set object
      o1 - a python object
      o2 - a python object

    >>> import numpy
    >>> m = ExtensionModule('test_CTypePython_numpy')
    >>> f = PyCFunction('func_int')
    >>> f += PyCArgument('i1', numpy.int8, output_intent='return')
    >>> f += PyCArgument('i2', numpy.int16, output_intent='return')
    >>> f += PyCArgument('i3', numpy.int32, output_intent='return')
    >>> f += PyCArgument('i4', numpy.int64, output_intent='return')
    >>> m += f
    >>> f = PyCFunction('func_uint')
    >>> f += PyCArgument('i1', numpy.uint8, output_intent='return')
    >>> f += PyCArgument('i2', numpy.uint16, output_intent='return')
    >>> f += PyCArgument('i3', numpy.uint32, output_intent='return')
    >>> f += PyCArgument('i4', numpy.uint64, output_intent='return')
    >>> m += f
    >>> f = PyCFunction('func_float')
    >>> f += PyCArgument('f1', numpy.float32, output_intent='return')
    >>> f += PyCArgument('f2', numpy.float64, output_intent='return')
    >>> f += PyCArgument('f3', numpy.float128, output_intent='return')
    >>> m += f
    >>> f = PyCFunction('func_complex')
    >>> f += PyCArgument('c1', numpy.complex64, output_intent='return')
    >>> f += PyCArgument('c2', numpy.complex128, output_intent='return')
    >>> f += PyCArgument('c3', numpy.complex256, output_intent='return')
    >>> m += f
    >>> f = PyCFunction('func_array')
    >>> f += PyCArgument('a1', numpy.ndarray, output_intent='return')
    >>> m += f
    >>> #f = PyCFunction('func_c_int')
    >>> #f += PyCArgument('i1', 'c_int', output_intent='return')
    >>> #m += f
    >>> b = m.build() #doctest: +ELLIPSIS
    exec_command...
    >>> b.func_int(numpy.int8(-2), numpy.int16(-3), numpy.int32(-4), numpy.int64(-5))
    (-2, -3, -4, -5)
    >>> b.func_uint(numpy.uint8(-1), numpy.uint16(-1), numpy.uint32(-1), numpy.uint64(-1))
    (255, 65535, 4294967295, 18446744073709551615)
    >>> b.func_float(numpy.float32(1.2),numpy.float64(1.2),numpy.float128(1.2))
    (1.20000004768, 1.2, 1.19999999999999995559)
    >>> b.func_complex(numpy.complex64(1+2j),numpy.complex128(1+2j),numpy.complex256(1+2j))
    ((1+2j), (1+2j), (1.0+2.0j))
    >>> b.func_array(numpy.array([1,2]))
    array([1, 2])
    >>> b.func_array(numpy.array(2))
    array(2)
    >>> b.func_array(2)
    Traceback (most recent call last):
    ...
    TypeError: argument 1 must be numpy.ndarray, not int
    >>> b.func_array(numpy.int8(2))
    Traceback (most recent call last):
    ...
    TypeError: argument 1 must be numpy.ndarray, not numpy.int8
    >>> #b.func_c_int(2)
    """

    typeinfo_map = dict(
        # <key>: (<type object in C>, <C type>, <PyArgFmt>)
        int = ('PyInt_Type', 'PyIntObject*', 'O!'),
        long = ('PyLong_Type', 'PyLongObject*', 'O!'),
        float = ('PyFloat_Type', 'PyFloatObject*', 'O!'),
        complex = ('PyComplex_Type', 'PyComplexObject*', 'O!'),
        str = ('PyString_Type', 'PyStringObject*', 'S'),
        unicode = ('PyUnicode_Type', 'PyUnicodeObject*', 'U'),
        buffer = ('PyBuffer_Type', 'PyBufferObject*', 'O!'),
        tuple = ('PyTuple_Type', 'PyTupleObject*', 'O!'),
        list = ('PyList_Type', 'PyListObject*', 'O!'),
        dict = ('PyDict_Type', 'PyDictObject*', 'O!'),
        file = ('PyFile_Type', 'PyFileObject*', 'O!'),
        instance = ('PyInstance_Type', 'PyObject*', 'O!'),
        function = ('PyFunction_Type', 'PyFunctionObject*', 'O!'),
        method = ('PyMethod_Type', 'PyObject*', 'O!'),
        module = ('PyModule_Type', 'PyObject*', 'O!'),
        iter = ('PySeqIter_Type', 'PyObject*', 'O!'),
        property = ('PyProperty_Type', 'PyObject*', 'O!'),
        slice = ('PySlice_Type', 'PyObject*', 'O!'),
        cell = ('PyCell_Type', 'PyCellObject*', 'O!'),
        generator = ('PyGen_Type', 'PyGenObject*', 'O!'),
        set = ('PySet_Type', 'PySetObject*', 'O!'),
        frozenset = ('PyFrozenSet_Type', 'PySetObject*', 'O!'),
        cobject = (None, 'PyCObject*', 'O'),
        type = ('PyType_Type', 'PyTypeObject*', 'O!'),
        object = (None, 'PyObject*', 'O'),
        numpy_ndarray = ('PyArray_Type', 'PyArrayObject*', 'O!'),
        numpy_descr = ('PyArrayDescr_Type','PyArray_Descr', 'O!'),
        numpy_ufunc = ('PyUFunc_Type', 'PyUFuncObject*', 'O!'),
        numpy_iter = ('PyArrayIter_Type', 'PyArrayIterObject*', 'O!'),
        numpy_multiiter = ('PyArrayMultiIter_Type', 'PyArrayMultiIterObject*', 'O!'),
        numpy_int8 = ('PyInt8ArrType_Type', 'PyInt8ScalarObject*', 'O!'),
        numpy_int16 = ('PyInt16ArrType_Type', 'PyInt16ScalarObject*', 'O!'),
        numpy_int32 = ('PyInt32ArrType_Type', 'PyInt32ScalarObject*', 'O!'),
        numpy_int64 = ('PyInt64ArrType_Type', 'PyInt64ScalarObject*', 'O!'),
        numpy_int128 = ('PyInt128ArrType_Type', 'PyInt128ScalarObject*', 'O!'),
        numpy_uint8 = ('PyUInt8ArrType_Type', 'PyUInt8ScalarObject*', 'O!'),
        numpy_uint16 = ('PyUInt16ArrType_Type', 'PyUInt16ScalarObject*', 'O!'),
        numpy_uint32 = ('PyUInt32ArrType_Type', 'PyUInt32ScalarObject*', 'O!'),
        numpy_uint64 = ('PyUInt64ArrType_Type', 'PyUInt64ScalarObject*', 'O!'),
        numpy_uint128 = ('PyUInt128ArrType_Type', 'PyUInt128ScalarObject*', 'O!'),
        numpy_float16 = ('PyFloat16ArrType_Type', 'PyFloat16ScalarObject*', 'O!'),
        numpy_float32 = ('PyFloat32ArrType_Type', 'PyFloat32ScalarObject*', 'O!'),
        numpy_float64 = ('PyFloat64ArrType_Type', 'PyFloat64ScalarObject*', 'O!'),
        numpy_float80 = ('PyFloat80ArrType_Type', 'PyFloat80ScalarObject*', 'O!'),
        numpy_float96 = ('PyFloat96ArrType_Type', 'PyFloat96ScalarObject*', 'O!'),
        numpy_float128 = ('PyFloat128ArrType_Type', 'PyFloat128ScalarObject*', 'O!'),
        numpy_complex32 = ('PyComplex32ArrType_Type', 'PyComplex32ScalarObject*', 'O!'),
        numpy_complex64 = ('PyComplex64ArrType_Type', 'PyComplex64ScalarObject*', 'O!'),
        numpy_complex128 = ('PyComplex128ArrType_Type', 'PyComplex128ScalarObject*', 'O!'),
        numpy_complex160 = ('PyComplex160ArrType_Type', 'PyComplex160ScalarObject*', 'O!'),
        numpy_complex192 = ('PyComplex192ArrType_Type', 'PyComplex192ScalarObject*', 'O!'),
        numpy_complex256 = ('PyComplex256ArrType_Type', 'PyComplex256ScalarObject*', 'O!'),
        numeric_array = ('PyArray_Type', 'PyArrayObject*', 'O!'),
        )
    
    def initialize(self, typeobj):
        m = self.typeinfo_map

        key = None
        if isinstance(typeobj, type):
            if typeobj.__module__=='__builtin__':
                key = typeobj.__name__
                if key=='array':
                    key = 'numeric_array'
            elif typeobj.__module__=='numpy':
                key = 'numpy_' + typeobj.__name__
        elif isinstance(typeobj, str):
            key = typeobj
            if key.startswith('numpy_'):
                k = key[6:]
                named_scalars = ['byte','short','int','long','longlong',
                                 'ubyte','ushort','uint','ulong','ulonglong',
                                 'intp','uintp',
                                 'float_','double',
                                 'longfloat','longdouble',
                                 'complex_',
                                 ]
                if k in named_scalars:
                    import numpy
                    key = 'numpy_' + getattr(numpy, k).__name__

        try: item = m[key]
        except KeyError:
            raise NotImplementedError('%s: need %s support' % (self.__class__.__name__, typeobj))

        self.typeobj_name = key
        self.ctypeobj = item[0]
        self.name = item[1]
        self.pyarg_fmt = item[2]            

        if key.startswith('numpy_'):
            self.add(Component.get('arrayobject.h'), 'Header')
            self.add(Component.get('import_array'), 'ModuleInit')

        if key.startswith('numeric_'):
            raise NotImplementedError(self.__class__.__name__ + ': Numeric support')
        return self

    def set_titles(self, arg):
        if self.typeobj_name == 'object':
            tn = 'a python ' + self.typeobj_name
        else:
            if self.typeobj_name.startswith('numpy_'):
                tn = 'a numpy.' + self.typeobj_name[6:] + ' object'
            else:
                tn = 'a python ' + self.typeobj_name + ' object'
        if arg.input_intent!='hide':
            r = ''
            if arg.input_title: r = ', ' + arg.input_title
            arg.input_title = tn + r
        if arg.output_intent!='hide':
            r = ''
            if arg.output_title: r = ', ' + arg.output_title
            arg.output_title = tn + r

    def get_pyarg_fmt(self, arg):
        if arg.input_intent=='hide': return None
        return self.pyarg_fmt

    def get_pyarg_obj(self, arg):
        if arg.input_intent=='hide': return None
        if self.pyarg_fmt=='O!':
            return '&%s, &%s' % (self.ctypeobj, arg.pycvar)
        return '&' + arg.pycvar




class CInt(CType):
    name = provides = 'int'
    def initialize(self): return self
    def get_pyarg_fmt(self, arg): return 'i'
    def get_pyarg_obj(self, arg): return '&' + arg.cvar
    def get_pyret_fmt(self, arg): return 'i'
    def get_pyret_obj(self, arg): return arg.cvar

def register():
    Component.register(
        CInt(),
        )

def _test():
    import doctest
    doctest.testmod()
    
if __name__ == "__main__":
    _test()
