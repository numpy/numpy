#
# Author: Travis Oliphant
#

from Numeric import *
import types
from arraymap import arraymap
from shape_base import squeeze,atleast_1d
from type_check import isscalar

__all__ = ['general_function']

class general_function:
    """
 general_function(somefunction)  Generalized Function class.

  Description:
 
    Define a generalized function which takes nested sequence
    objects or Numeric arrays as inputs and returns a
    Numeric array as output, evaluating the function over successive
    tuples of the input arrays like the python map function except it uses
    the broadcasting rules of Numeric Python.

  Input:

    somefunction -- a Python function or method

  Example:

    def myfunc(a,b):
        if a > b:
            return a-b
        else
            return a+b

    gfunc = general_function(myfunc)

    >>> gfunc([1,2,3,4],2)
    array([3,4,1,2])

    """
    def __init__(self,pyfunc,otypes=None,doc=None):
        if not callable(pyfunc) or type(pyfunc) is types.ClassType:
            raise TypeError, "Object is not a callable Python object."
        self.thefunc = pyfunc
        if doc is None:
            self.__doc__ = pyfunc.__doc__
        else:
            self.__doc__ = doc
        if otypes is None:
            self.otypes=''
        else:
            if isinstance(otypes,types.StringType):
                self.otypes=otypes
            else:
                raise ValueError, "Output types must be a string."

    def __call__(self,*args):
        for arg in args:
            try:
                n = len(arg)
                if (n==0):
                    return self.zerocall(args)
            except (AttributeError, TypeError):
                pass
        return squeeze(arraymap(self.thefunc,args,self.otypes))

    def zerocall(self,args):
        # one of the args was a zeros array
        #  return zeros for each output
        #  first --- find number of outputs
        newargs = []
        args = atleast_1d(*args)
        for arg in args:
            if arg.typecode() != 'O':
                newargs.append(1.1)
            else:
                newargs.append(arg[0])
        newargs = tuple(newargs)
        res = self.thefunc(*newargs)
        if isscalar(res):
            return zeros((0,),'d')
        else:
            return (zeros((0,),'d'),)*len(res)

#-----------------------------------------------------------------------------
# Test Routines
#-----------------------------------------------------------------------------

def test(level=10):
    from scipy_test.testing import module_test
    module_test(__name__,__file__,level=level)

def test_suite(level=1):
    from scipy_test.testing import module_test_suite
    return module_test_suite(__name__,__file__,level=level)
