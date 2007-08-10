"""
Python Extensions Generator
"""

__all__ = ['Component']

from base import Component

for _m in ['utils', 'c_support', 'py_support', 'setup_py']:
    exec 'from %s import *' % (_m)
    exec 'import %s as _m' % (_m)
    __all__.extend(_m.__all__)

#from pyc_function import PyCFunction
#from pyc_argument import PyCArgument
#from c_code import CCode

#import c_type
#from c_type import *
#__all__ += c_type.__all__
#import c_struct
#from c_struct import *
#__all__ += c_struct.__all__#

#import predefined_components
#import converters
#c_type.register()
