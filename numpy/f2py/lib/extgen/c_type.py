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
        raise NotImplementedError('%s.get_pyarg_fmt()' % (self.__class__.__name__))

    def get_pyarg_obj(self, arg):
        raise NotImplementedError('%s.get_pyarg_obj()' % (self.__class__.__name__))
    
    def get_pyret_fmt(self, arg):
        raise NotImplementedError('%s.get_pyret_fmt()' % (self.__class__.__name__))
    
    def get_pyret_obj(self, arg):
        raise NotImplementedError('%s.get_pyret_obj()' % (self.__class__.__name__))

    def get_init_value(self, arg):
        return

    def set_Decl(self, arg):
        init_value = self.get_init_value(arg)
        if init_value:
            init =  ' = %s' % (init_value)
        else:
            init = ''
        if arg.pycvar and arg.pycvar==arg.retpycvar:
            arg += CDecl(self, '%s%s' % (arg.pycvar, init))
        else:
            if arg.input_intent!='hide':
                arg += CDecl(self, '%s%s' % (arg.pycvar, init))
            if arg.output_intent!='hide':
                arg += CDecl(self, '%s%s' % (arg.retpycvar, init))

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

        argfmt = self.get_pyarg_fmt(arg)
        retfmt = self.get_pyret_fmt(arg)

        if arg.output_intent=='return':
            if arg.input_intent in ['optional', 'extra']:
                if retfmt in 'SON':
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
                if retfmt in 'SON':
                    PyObjFrom += eval_a('''\
if (%(retpycvar)s==NULL) {
  %(retpycvar)s = Py_None;
  Py_INCREF((PyObject*)%(retpycvar)s);
} /* else %(retpycvar)r must be a new reference or expect a core dump. */
''')
            elif arg.input_intent=='required':
                if retfmt in 'SON':
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
        if isinstance(name, type):
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

    >>> m = ExtensionModule('test_CTypePython_c')
    >>> f = PyCFunction('func_c_int')
    >>> f += PyCArgument('i1', 'c_char', output_intent='return')
    >>> f += PyCArgument('i2', 'c_short', output_intent='return')
    >>> f += PyCArgument('i3', 'c_int', output_intent='return')
    >>> f += PyCArgument('i4', 'c_long', output_intent='return')
    >>> f += PyCArgument('i5', 'c_long_long', output_intent='return')
    >>> m += f
    >>> f = PyCFunction('func_c_unsigned_int')
    >>> f += PyCArgument('i1', 'c_unsigned_char', output_intent='return')
    >>> f += PyCArgument('i2', 'c_unsigned_short', output_intent='return')
    >>> f += PyCArgument('i3', 'c_unsigned_int', output_intent='return')
    >>> f += PyCArgument('i4', 'c_unsigned_long', output_intent='return')
    >>> f += PyCArgument('i5', 'c_unsigned_long_long', output_intent='return')
    >>> m += f
    >>> f = PyCFunction('func_c_float')
    >>> f += PyCArgument('f1', 'c_float', output_intent='return')
    >>> f += PyCArgument('f2', 'c_double', output_intent='return')
    >>> m += f
    >>> f = PyCFunction('func_c_complex')
    >>> f += PyCArgument('c1', 'c_Py_complex', output_intent='return')
    >>> m += f
    >>> f = PyCFunction('func_c_string')
    >>> f += PyCArgument('s1', 'c_const_char_ptr', output_intent='return')
    >>> f += PyCArgument('s2', 'c_const_char_ptr', output_intent='return')
    >>> f += PyCArgument('s3', 'c_Py_UNICODE', output_intent='return')
    >>> f += PyCArgument('s4', 'c_char1', output_intent='return')
    >>> m += f
    >>> b = m.build() #doctest: +ELLIPSIS
    exec_command...
    >>> b.func_c_int(2,3,4,5,6)
    (2, 3, 4, 5, 6L)
    >>> b.func_c_unsigned_int(-1,-1,-1,-1,-1)
    (255, 65535, 4294967295, 18446744073709551615L, 18446744073709551615L)
    >>> b.func_c_float(1.2,1.2)
    (1.2000000476837158, 1.2)
    >>> b.func_c_complex(1+2j)
    (1+2j)
    >>> b.func_c_string('hei', None, u'tere', 'b')
    ('hei', None, u'tere', 'b')

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
    """

    typeinfo_map = dict(
        # <key>: (<type object in C>, <C type>, <ArgFmt>, <RetFmt>, <init value in C decl stmt>)
        int = ('PyInt_Type', 'PyIntObject*', 'O!', 'N', 'NULL'),
        long = ('PyLong_Type', 'PyLongObject*', 'O!', 'N', 'NULL'),
        float = ('PyFloat_Type', 'PyFloatObject*', 'O!', 'N', 'NULL'),
        complex = ('PyComplex_Type', 'PyComplexObject*', 'O!', 'N', 'NULL'),
        str = ('PyString_Type', 'PyStringObject*', 'S', 'N', 'NULL'),
        unicode = ('PyUnicode_Type', 'PyUnicodeObject*', 'U', 'N', 'NULL'),
        buffer = ('PyBuffer_Type', 'PyBufferObject*', 'O!', 'N', 'NULL'),
        tuple = ('PyTuple_Type', 'PyTupleObject*', 'O!', 'N', 'NULL'),
        list = ('PyList_Type', 'PyListObject*', 'O!', 'N', 'NULL'),
        dict = ('PyDict_Type', 'PyDictObject*', 'O!', 'N', 'NULL'),
        file = ('PyFile_Type', 'PyFileObject*', 'O!', 'N', 'NULL'),
        instance = ('PyInstance_Type', 'PyObject*', 'O!', 'N', 'NULL'),
        function = ('PyFunction_Type', 'PyFunctionObject*', 'O!', 'N', 'NULL'),
        method = ('PyMethod_Type', 'PyObject*', 'O!', 'N', 'NULL'),
        module = ('PyModule_Type', 'PyObject*', 'O!', 'N', 'NULL'),
        iter = ('PySeqIter_Type', 'PyObject*', 'O!', 'N', 'NULL'),
        property = ('PyProperty_Type', 'PyObject*', 'O!', 'N', 'NULL'),
        slice = ('PySlice_Type', 'PyObject*', 'O!', 'N', 'NULL'),
        cell = ('PyCell_Type', 'PyCellObject*', 'O!', 'N', 'NULL'),
        generator = ('PyGen_Type', 'PyGenObject*', 'O!', 'N', 'NULL'),
        set = ('PySet_Type', 'PySetObject*', 'O!', 'N', 'NULL'),
        frozenset = ('PyFrozenSet_Type', 'PySetObject*', 'O!', 'N', 'NULL'),
        cobject = (None, 'PyCObject*', 'O', 'N', 'NULL'),
        type = ('PyType_Type', 'PyTypeObject*', 'O!', 'N', 'NULL'),
        object = (None, 'PyObject*', 'O', 'N', 'NULL'),
        numpy_ndarray = ('PyArray_Type', 'PyArrayObject*', 'O!', 'N', 'NULL'),
        numpy_descr = ('PyArrayDescr_Type','PyArray_Descr', 'O!', 'N', 'NULL'),
        numpy_ufunc = ('PyUFunc_Type', 'PyUFuncObject*', 'O!', 'N', 'NULL'),
        numpy_iter = ('PyArrayIter_Type', 'PyArrayIterObject*', 'O!', 'N', 'NULL'),
        numpy_multiiter = ('PyArrayMultiIter_Type', 'PyArrayMultiIterObject*', 'O!', 'N', 'NULL'),
        numpy_int8 = ('PyInt8ArrType_Type', 'PyInt8ScalarObject*', 'O!', 'N', 'NULL'),
        numpy_int16 = ('PyInt16ArrType_Type', 'PyInt16ScalarObject*', 'O!', 'N', 'NULL'),
        numpy_int32 = ('PyInt32ArrType_Type', 'PyInt32ScalarObject*', 'O!', 'N', 'NULL'),
        numpy_int64 = ('PyInt64ArrType_Type', 'PyInt64ScalarObject*', 'O!', 'N', 'NULL'),
        numpy_int128 = ('PyInt128ArrType_Type', 'PyInt128ScalarObject*', 'O!', 'N', 'NULL'),
        numpy_uint8 = ('PyUInt8ArrType_Type', 'PyUInt8ScalarObject*', 'O!', 'N', 'NULL'),
        numpy_uint16 = ('PyUInt16ArrType_Type', 'PyUInt16ScalarObject*', 'O!', 'N', 'NULL'),
        numpy_uint32 = ('PyUInt32ArrType_Type', 'PyUInt32ScalarObject*', 'O!', 'N', 'NULL'),
        numpy_uint64 = ('PyUInt64ArrType_Type', 'PyUInt64ScalarObject*', 'O!', 'N', 'NULL'),
        numpy_uint128 = ('PyUInt128ArrType_Type', 'PyUInt128ScalarObject*', 'O!', 'N', 'NULL'),
        numpy_float16 = ('PyFloat16ArrType_Type', 'PyFloat16ScalarObject*', 'O!', 'N', 'NULL'),
        numpy_float32 = ('PyFloat32ArrType_Type', 'PyFloat32ScalarObject*', 'O!', 'N', 'NULL'),
        numpy_float64 = ('PyFloat64ArrType_Type', 'PyFloat64ScalarObject*', 'O!', 'N', 'NULL'),
        numpy_float80 = ('PyFloat80ArrType_Type', 'PyFloat80ScalarObject*', 'O!', 'N', 'NULL'),
        numpy_float96 = ('PyFloat96ArrType_Type', 'PyFloat96ScalarObject*', 'O!', 'N', 'NULL'),
        numpy_float128 = ('PyFloat128ArrType_Type', 'PyFloat128ScalarObject*', 'O!', 'N', 'NULL'),
        numpy_complex32 = ('PyComplex32ArrType_Type', 'PyComplex32ScalarObject*', 'O!', 'N', 'NULL'),
        numpy_complex64 = ('PyComplex64ArrType_Type', 'PyComplex64ScalarObject*', 'O!', 'N', 'NULL'),
        numpy_complex128 = ('PyComplex128ArrType_Type', 'PyComplex128ScalarObject*', 'O!', 'N', 'NULL'),
        numpy_complex160 = ('PyComplex160ArrType_Type', 'PyComplex160ScalarObject*', 'O!', 'N', 'NULL'),
        numpy_complex192 = ('PyComplex192ArrType_Type', 'PyComplex192ScalarObject*', 'O!', 'N', 'NULL'),
        numpy_complex256 = ('PyComplex256ArrType_Type', 'PyComplex256ScalarObject*', 'O!', 'N', 'NULL'),
        numeric_array = ('PyArray_Type', 'PyArrayObject*', 'O!', 'N', 'NULL'),
        c_char = (None, 'char', 'b', 'b', '0'),
        c_unsigned_char = (None, 'unsigned char', 'B', 'B', '0'),
        c_short = (None, 'short int', 'h', 'h', '0'),
        c_unsigned_short = (None, 'unsigned short int', 'H', 'H', '0'),
        c_int = (None,'int', 'i', 'i', '0'),
        c_unsigned_int = (None,'unsigned int', 'I', 'I', '0'),
        c_long = (None,'long', 'l', 'l', '0'),
        c_unsigned_long = (None,'unsigned long', 'k', 'k', '0'),
        c_long_long = (None,'PY_LONG_LONG', 'L', 'L', '0'),
        c_unsigned_long_long = (None,'unsigned PY_LONG_LONG', 'K', 'K', '0'),        
        c_Py_ssize_t = (None,'Py_ssize_t', 'n', 'n', '0'),
        c_char1 = (None,'char', 'c', 'c', '"\\0"'),
        c_float = (None,'float', 'f', 'f', '0.0'),
        c_double = (None,'double', 'd', 'd', '0.0'),
        c_Py_complex = (None,'Py_complex', 'D', 'D', '{0.0, 0.0}'),
        c_const_char_ptr = (None,'const char *', 'z', 'z', 'NULL'),
        c_Py_UNICODE = (None,'Py_UNICODE*','u','u', 'NULL'),
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
        self.pyret_fmt = item[3]
        self.cinit_value = item[4]
        
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
            elif self.typeobj_name.startswith('c_'):
                n = self.typeobj_name[2:]
                if not n.startswith('Py_'):
                    n = ' '.join(n.split('_'))
                tn = 'a to C ' + n + ' convertable object'
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

    def get_pyret_fmt(self, arg):
        if arg.output_intent=='hide': return None
        return self.pyret_fmt

    def get_pyret_obj(self, arg):
        if arg.output_intent=='return':
            if self.get_pyret_fmt(arg)=='D':
                return '&' + arg.retpycvar
            return arg.retpycvar
        return

    def get_init_value(self, arg):
        return self.cinit_value

def register():
    Component.register(
        )

def _test():
    import doctest
    doctest.testmod()
    
if __name__ == "__main__":
    _test()
