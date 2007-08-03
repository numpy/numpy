"""
Defines C type declaration templates:

  CTypeAlias(name, ctype)  --- typedef ctype name;
  CTypeFunction(name, rtype, atype1, atype2,..) --- typedef rtype (*name)(atype1, atype2,...);
  CTypeStruct(name, (name1,type1), (name2,type2), ...) --- typedef struct { type1 name1; type2 name2; .. } name;
  CTypePtr(ctype) --- ctype *
  CInt(), CLong(), ... --- int, long, ...
  CPyObject()

The instances of CTypeBase have the following public methods and properties:

  - .asPtr()
  - .declare(name)
"""


from base import Base

class CTypeBase(Base):

    def declare(self, name):
        return '%s %s;' % (self.typename, name)

    def __str__(self):
        return self.typename

    def asPtr(self):
        return CTypePtr(self)

class CTypeAlias(CTypeBase):

    def __new__(cls, typename, ctype):
        obj = Base.__new__(cls)
        assert isinstance(ctype, CTypeBase),`type(ctype)`
        obj.add(typename, ctype)
        return obj

    @property
    def typename(self): return self.components[0][0]
    @property
    def ctype(self): return self.components[0][1]

    def local_generate(self, params=None):
        container = self.get_container('TypeDef')
        container.add(self.typename, 'typedef %s %s;' % (self.ctype, self.typename))
        return self.declare(params)


class CTypeFunction(CTypeBase):

    def __new__(cls, typename, rctype, *arguments):
        obj = Base.__new__(cls)
        assert isinstance(rctype, CTypeBase),`type(rctype)`
        obj.add(typename, rctype)
        for i in range(len(arguments)):
            a = arguments[i]
            assert isinstance(a, CTypeBase),`type(a)`
            obj.add('_a%i' % (i), a)
        return obj

    @property
    def typename(self): return self.components[0][0]
    @property
    def rctype(self): return self.components[0][1]
    @property
    def arguments(self): return [v for n,v in self.components[1:]]

    def local_generate(self, params=None):
        container = self.get_container('TypeDef')
        container.add(self.typename, 'typedef %s (*%s)(%s);' \
                      % (self.rctype, self.typename,
                         ', '.join([str(ctype) for ctype in self.arguments])))
        return self.declare(params)

class CTypeStruct(CTypeBase):

    def __new__(cls, typename, *components):
        obj = Base.__new__(cls, typename)
        for n,v in components:
            assert isinstance(v,CTypeBase),`type(v)`
            obj.add(n,v)
        return obj

    @property
    def typename(self): return self._args[0]

    def local_generate(self, params=None):
        container = self.get_container('TypeDef')
        decls = [ctype.declare(name) for name, ctype in self.components]
        if decls:
            d = 'typedef struct {\n  %s\n} %s;' % ('\n  '.join(decls),self.typename)
        else:
            d = 'typedef struct {} %s;' % (self.typename)
        container.add(self.typename, d)
        return self.declare(params)

class CTypePtr(CTypeBase):

    def __new__(cls, ctype):
        obj = Base.__new__(cls)
        assert isinstance(ctype, CTypeBase),`type(ctype)`
        obj.add('*', ctype)
        return obj

    @property
    def ctype(self): return self.components[0][1]
    
    @property
    def typename(self):
        return self.ctype.typename + '*'

    def local_generate(self, params=None):
        return self.declare(params)

class CTypeDefined(CTypeBase):

    @property
    def typename(self): return self._args[0]

class CTypeIntrinsic(CTypeDefined):

    def __new__(cls, typename):
        return Base.__new__(cls, typename)

class CPyObject(CTypeDefined):
    def __new__(cls):
        return Base.__new__(cls, 'PyObject')

class CInt(CTypeIntrinsic):

    def __new__(cls):
        return Base.__new__(cls, 'int')

    def local_generate(self, params=None):
        container = self.get_container('CAPICode')
        code = '''\
static int pyobj_to_int(PyObject *obj, int* value) {
  int status = 1;
  if (PyInt_Check(obj)) {
    *value = PyInt_AS_LONG(obj);
    status = 0;
  }
  return status;
}
'''
        container.add('pyobj_to_int', code)
        code = '''\
static PyObject* pyobj_from_int(int* value) {
  return PyInt_FromLong(*value);
}
'''
        container.add('pyobj_from_int', code)

        return self.declare(params)

