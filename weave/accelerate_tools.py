#**************************************************************************#
#* FILE   **************    accelerate_tools.py    ************************#
#**************************************************************************#
#* Author: Patrick Miller February  9 2002                                *#
#**************************************************************************#
"""
accelerate_tools contains the interface for on-the-fly building of
C++ equivalents to Python functions.
"""
#**************************************************************************#

from types import InstanceType,FunctionType,IntType,FloatType,StringType,TypeType,XRangeType
import inspect
import md5
import weave
import imp
from bytecodecompiler import CXXCoder,Type_Descriptor,Function_Descriptor

def CStr(s):
    "Hacky way to get legal C string from Python string"
    if s is None: return '""'
    assert type(s) == StringType,"Only None and string allowed"
    r = repr('"'+s) # Better for embedded quotes
    return '"'+r[2:-1]+'"'


##################################################################
#                         CLASS INSTANCE                         #
##################################################################
class Instance(Type_Descriptor):
    cxxtype = 'PyObject*'
    
    def __init__(self,prototype):
	self.prototype	= prototype
	return

    def check(self,s):
        return "PyInstance_Check(%s)"%s

    def inbound(self,s):
        return s

    def outbound(self,s):
        return s,0

    def get_attribute(self,name):
        proto = getattr(self.prototype,name)
        T = lookup_type(proto)
        code = 'tempPY = PyObject_GetAttrString(%%(rhs)s,"%s");\n'%name
        convert = T.inbound('tempPY')
        code += '%%(lhsType)s %%(lhs)s = %s;\n'%convert
        return T,code

    def set_attribute(self,name):
        proto = getattr(self.prototype,name)
        T = lookup_type(proto)
        convert,owned = T.outbound('%(rhs)s')
        code = 'tempPY = %s;'%convert
        if not owned:
            code += ' Py_INCREF(tempPY);'
        code += ' PyObject_SetAttrString(%%(lhs)s,"%s",tempPY);'%name
        code += ' Py_DECREF(tempPY);\n'
        return T,code

##################################################################
#                          CLASS BASIC                           #
##################################################################
class Basic(Type_Descriptor):
    owned = 1
    def check(self,s):
        return "%s(%s)"%(self.checker,s)
    def inbound(self,s):
        return "%s(%s)"%(self.inbounder,s)
    def outbound(self,s):
        return "%s(%s)"%(self.outbounder,s),self.owned

class Basic_Number(Basic):
    def literalizer(self,s):
        return str(s)
    def binop(self,symbol,a,b):
        assert symbol in ['+','-','*','/'],symbol
        return '%s %s %s'%(a,symbol,b),self

class Integer(Basic_Number):
    cxxtype = "long"
    checker = "PyInt_Check"
    inbounder = "PyInt_AsLong"
    outbounder = "PyInt_FromLong"

class Double(Basic_Number):
    cxxtype = "double"
    checker = "PyFloat_Check"
    inbounder = "PyFloat_AsDouble"
    outbounder = "PyFloat_FromDouble"

class String(Basic):
    cxxtype = "char*"
    checker = "PyString_Check"
    inbounder = "PyString_AsString"
    outbounder = "PyString_FromString"

    def literalizer(self,s):
        return CStr(s)

# -----------------------------------------------
# Singletonize the type names
# -----------------------------------------------
Integer = Integer()
Double = Double()
String = String()

import Numeric

class Vector(Type_Descriptor):
    cxxtype = 'PyArrayObject*'
    refcount = 1
    dims = 1
    module_init_code = 'import_array();\n'
    inbounder = "(PyArrayObject*)"
    outbounder = "(PyObject*)"
    owned = 0 # Convertion is by casting!

    prerequisites = Type_Descriptor.prerequisites+\
                   ['#include "Numeric/arrayobject.h"']
    dims = 1
    def check(self,s):
        return "PyArray_Check(%s) && ((PyArrayObject*)%s)->nd == %d &&  ((PyArrayObject*)%s)->descr->type_num == %s"%(
            s,s,self.dims,s,self.typecode)

    def inbound(self,s):
        return "%s(%s)"%(self.inbounder,s)
    def outbound(self,s):
        return "%s(%s)"%(self.outbounder,s),self.owned

    def getitem(self,A,v,t):
        assert self.dims == len(v),'Expect dimension %d'%self.dims
        code = '*((%s*)(%s->data'%(self.cxxbase,A)
        for i in range(self.dims):
            # assert that ''t[i]'' is an integer
            code += '+%s*%s->strides[%d]'%(v[i],A,i)
        code += '))'
        return code,self.pybase
    def setitem(self,A,v,t):
        return self.getitem(A,v,t)

class matrix(Vector):
    dims = 2

class IntegerVector(Vector):
    typecode = 'PyArray_INT'
    cxxbase = 'int'
    pybase = Integer

class Integermatrix(matrix):
    typecode = 'PyArray_INT'
    cxxbase = 'int'
    pybase = Integer

class LongVector(Vector):
    typecode = 'PyArray_LONG'
    cxxbase = 'long'
    pybase = Integer

class Longmatrix(matrix):
    typecode = 'PyArray_LONG'
    cxxbase = 'long'
    pybase = Integer

class DoubleVector(Vector):
    typecode = 'PyArray_DOUBLE'
    cxxbase = 'double'
    pybase = Double

class Doublematrix(matrix):
    typecode = 'PyArray_DOUBLE'
    cxxbase = 'double'
    pybase = Double


##################################################################
#                          CLASS XRANGE                          #
##################################################################
class XRange(Type_Descriptor):
    cxxtype = 'XRange'
    prerequisites = ['''
    class XRange {
    public:
    XRange(long aLow, long aHigh, long aStep=1)
    : low(aLow),high(aHigh),step(aStep)
    {
    }
    XRange(long aHigh)
    : low(0),high(aHigh),step(1)
    {
    }
    long low;
    long high;
    long step;
    };''']

# -----------------------------------------------
# Singletonize the type names
# -----------------------------------------------
IntegerVector = IntegerVector()
Integermatrix = Integermatrix()
LongVector = LongVector()
Longmatrix = Longmatrix()
DoubleVector = DoubleVector()
Doublematrix = Doublematrix()
XRange = XRange()


typedefs = {
    IntType: Integer,
    FloatType: Double,
    StringType: String,
    (Numeric.ArrayType,1,'i'): IntegerVector,
    (Numeric.ArrayType,2,'i'): Integermatrix,
    (Numeric.ArrayType,1,'l'): LongVector,
    (Numeric.ArrayType,2,'l'): Longmatrix,
    (Numeric.ArrayType,1,'d'): DoubleVector,
    (Numeric.ArrayType,2,'d'): Doublematrix,
    XRangeType : XRange,
    }

import math
functiondefs = {
    (len,(String,)):
    Function_Descriptor(code='strlen(%s)',return_type=Integer),
    
    (len,(LongVector,)):
    Function_Descriptor(code='PyArray_Size((PyObject*)%s)',return_type=Integer),

    (float,(Integer,)):
    Function_Descriptor(code='(double)(%s)',return_type=Double),
    
    (range,(Integer,Integer)):
    Function_Descriptor(code='XRange(%s)',return_type=XRange),

    (range,(Integer)):
    Function_Descriptor(code='XRange(%s)',return_type=XRange),

    (math.sin,(Double,)):
    Function_Descriptor(code='sin(%s)',return_type=Double),

    (math.cos,(Double,)):
    Function_Descriptor(code='cos(%s)',return_type=Double),

    (math.sqrt,(Double,)):
    Function_Descriptor(code='sqrt(%s)',return_type=Double),
    }
    


##################################################################
#                      FUNCTION LOOKUP_TYPE                      #
##################################################################
def lookup_type(x):
    T = type(x)
    try:
        return typedefs[T]
    except:
        import Numeric
        if isinstance(T,Numeric.ArrayType):
            return typedefs[(T,len(x.shape),x.typecode())]
        elif T == InstanceType:
            return Instance(x)
        else:
            raise NotImplementedError,T

##################################################################
#                        class ACCELERATE                        #
##################################################################
class accelerate:
    
    def __init__(self, function, *args, **kw):
        assert type(function) == FunctionType
        self.function = function
        self.module = inspect.getmodule(function)
        if self.module is None:
            import __main__
            self.module = __main__
        self.__call_map = {}

    def __cache(self,*args):
        raise TypeError

    def __call__(self,*args):
        try:
            return self.__cache(*args)
        except TypeError:
            # Figure out type info -- Do as tuple so its hashable
            signature = tuple( map(lookup_type,args) )
            
            # If we know the function, call it
            try:
                fast = self.__call_map[signature]
            except:
                fast = self.singleton(signature)
                self.__cache = fast
                self.__call_map[signature] = fast
            return fast(*args)

    def signature(self,*args):
        # Figure out type info -- Do as tuple so its hashable
        signature = tuple( map(lookup_type,args) )
        return self.singleton(signature)


    def singleton(self,signature):
        identifier = self.identifier(signature)
        
        # Generate a new function, then call it
        f = self.function

        # See if we have an accelerated version of module
        try:
            print 'lookup',self.module.__name__+'_weave'
            accelerated_module = __import__(self.module.__name__+'_weave')
            print 'have accelerated',self.module.__name__+'_weave'
            fast = getattr(accelerated_module,identifier)
            return fast
        except ImportError:
            accelerated_module = None
        except AttributeError:
            pass

        P = self.accelerate(signature,identifier)

        E = weave.ext_tools.ext_module(self.module.__name__+'_weave')
        E.add_function(P)
        E.generate_file()
        weave.build_tools.build_extension(self.module.__name__+'_weave.cpp',verbose=2)

        if accelerated_module:
            raise NotImplementedError,'Reload'
        else:
            accelerated_module = __import__(self.module.__name__+'_weave')

        fast = getattr(accelerated_module,identifier)
        return fast

    def identifier(self,signature):
        # Build an MD5 checksum
        f = self.function
        co = f.func_code
        identifier = str(signature)+\
                     str(co.co_argcount)+\
                     str(co.co_consts)+\
                     str(co.co_varnames)+\
                     co.co_code
        return 'F'+md5.md5(identifier).hexdigest()
        
    def accelerate(self,signature,identifier):
        P = Python2CXX(self.function,signature,name=identifier)
        return P

    def code(self,*args):
        if len(args) != self.function.func_code.co_argcount:
            raise TypeError,'%s() takes exactly %d arguments (%d given)'%(
                self.function.__name__,
                self.function.func_code.co_argcount,
                len(args))
        signature = tuple( map(lookup_type,args) )
        ident = self.function.__name__
        return self.accelerate(signature,ident).function_code()
        

##################################################################
#                        CLASS PYTHON2CXX                        #
##################################################################
class Python2CXX(CXXCoder):
    def typedef_by_value(self,v):
        T = lookup_type(v)
        if T not in self.used:
            self.used.append(T)
        return T

    def function_by_signature(self,signature):
        descriptor = functiondefs[signature]
        if descriptor.return_type not in self.used:
            self.used.append(descriptor.return_type)
        return descriptor

    def __init__(self,f,signature,name=None):
        # Make sure function is a function
        import types
        assert type(f) == FunctionType
        # and check the input type signature
        assert reduce(lambda x,y: x and y,
                      map(lambda x: isinstance(x,Type_Descriptor),
                          signature),
                      1),'%s not all type objects'%signature
        self.arg_specs = []
        self.customize = weave.base_info.custom_info()

        CXXCoder.__init__(self,f,signature,name)

        return

    def function_code(self):
        code = self.wrapped_code()
        for T in self.used:
            if T != None and T.module_init_code:
                self.customize.add_module_init_code(T.module_init_code)
        return code

    def python_function_definition_code(self):
        return '{ "%s", wrapper_%s, METH_VARARGS, %s },\n'%(
            self.name,
            self.name,
            CStr(self.function.__doc__))
