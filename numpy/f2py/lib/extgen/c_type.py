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
    
    def update_containers(self):
        self.container_TypeDef += self.evaluate(self.template_typedef)

    def __str__(self):
        return self.name

    def get_pyret_fmt(self, input_intent_hide = True):
        if input_intent_hide: return 'O'
        return 'N'

    def get_pyret_arg(self, cname):
        return cname

    def get_pyarg_fmt(self):
        return 'O'

    def get_pyarg_arg(self, cname):
        return '&%s' % (cname)

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
        if isinstance(name, type) or name in ['cell', 'generator', 'cobject']:
            return CTypePython(name)
        try:
            return Component.get(name)
        except KeyError:
            pass
        self.name = name
        return self
    def update_containers(self):
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

    #>>> print b.func.__doc__
    
    """
    @property
    def provides(self):
        return self.typeobj_name
    def initialize(self, typeobj):
        if isinstance(typeobj, type):
            self.typeobj_name = typeobj.__name__
        elif isinstance(typeobj, str):
            self.typeobj_name = typeobj
        else:
            raise ValueError('%s: unexpected input type %r' \
                             % (self.__class__.__name__, type(typeobj)))
        self.ctypeobj = dict(int='PyInt_Type',
                             long='PyLong_Type',
                             float='PyFloat_Type',
                             complex='PyComplex_Type',
                             str='PyString_Type',
                             unicode='PyUnicode_Type',
                             buffer='PyBuffer_Type',
                             tuple='PyTuple_Type',
                             list='PyList_Type',
                             dict='PyDict_Type',
                             file='PyFile_Type',
                             instance='PyInstance_Type',
                             function='PyFunction_Type',
                             method='PyMethod_Type',
                             module='PyModule_Type',
                             iter='PySeqIter_Type',
                             property='PyProperty_Type',
                             slice='PySlice_Type',
                             cell='PyCell_Type',
                             generator='PyGen_Type',
                             set='PySet_Type',
                             frozenset='PyFrozenSet_Type',
                             type='PyType_Type',
                             ).get(self.typeobj_name)
        pyctypes = dict(int='PyIntObject',
                        long='PyLongObject',
                        float='PyFloatObject',
                        complex='PyComplexObject',
                        str='PyStringObject',
                        unicode='PyUnicodeObject',
                        buffer='PyBufferObject',
                        tuple='PyTupleObject',
                        list='PyListObject',
                        dict='PyDictObject',
                        file='PyFileObject',
                        instance='PyObject',
                        function='PyFunctionObject',
                        method='PyObject',
                        module='PyObject',
                        iter='PyObject',
                        property='PyObject',
                        slice='PyObject',
                        cobject='PyCObject', # XXX: check key
                        cell='PyCellObject',
                        type='PyTypeObject',
                        generator='PyGenObject',
                        set='PySetObject',
                        frozenset='PySetObject',
                        object='PyObject',
                        )
        try:
            self.name = pyctypes[self.typeobj_name] + '*'
        except KeyError:
            raise NotImplementedError('%s: need %s support' % (self.__class__.__name__, typeobj))
        return self

    def set_titles(self, arg):
        if self.typeobj_name == 'object':
            tn = 'a python ' + self.typeobj_name
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

    def set_pyarg_decl(self, arg):
        if arg.input_intent=='hide':
            arg += CDecl(self, '%s = Py_None' % (arg.pycvar))
        else:
            arg += CDecl(self, '%s = NULL' % (arg.pycvar))

    def get_pyarg_fmt(self, arg):
        return dict(object='O', str='S', unicode='U', cobject='O'
                    ).get(self.typeobj_name, 'O!') # XXX: check cobject

    def get_pyarg_obj(self, arg):
        if self.typeobj_name in ['object', 'str', 'unicode', 'cobject']: # XXX: check cobject
            return '&' + arg.pycvar
        return '&%s, &%s' % (self.ctypeobj, arg.pycvar)
    
    def get_pyret_fmt(self, arg):
        if arg.input_intent=='hide':
            return 'O'
        # return 'N' if arg is constructed inside PyCFunction
        return 'O'
    def get_pyret_obj(self, arg): return arg.pycvar


class CInt(CType):
    name = provides = 'int'
    def initialize(self): return self
    def set_pyarg_decl(self, arg):
        #arg += CDecl(Component.get('PyObject*'), '%s = NULL' % (arg.pycvar))
        arg += CDecl(self, '%s = 0' % (arg.cvar))
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
