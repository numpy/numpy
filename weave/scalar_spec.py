from base_spec import base_converter
import scalar_info
#from Numeric import *
from types import *

# the following typemaps are for 32 bit platforms.  A way to do this
# general case? maybe ask numeric types how long they are and base
# the decisions on that.

numeric_to_c_type_mapping = {}

numeric_to_c_type_mapping['T'] = 'T' # for templates
numeric_to_c_type_mapping['F'] = 'std::complex<float> '
numeric_to_c_type_mapping['D'] = 'std::complex<double> '
numeric_to_c_type_mapping['f'] = 'float'
numeric_to_c_type_mapping['d'] = 'double'
numeric_to_c_type_mapping['1'] = 'char'
numeric_to_c_type_mapping['b'] = 'unsigned char'
numeric_to_c_type_mapping['s'] = 'short'
numeric_to_c_type_mapping['i'] = 'int'
# not strictly correct, but shoulld be fine fo numeric work.
# add test somewhere to make sure long can be cast to int before using.
numeric_to_c_type_mapping['l'] = 'int'

# standard Python numeric type mappings.
numeric_to_c_type_mapping[type(1)]  = 'int'
numeric_to_c_type_mapping[type(1.)] = 'double'
numeric_to_c_type_mapping[type(1.+1.j)] = 'std::complex<double> '
#hmmm. The following is likely unsafe...
numeric_to_c_type_mapping[type(1L)]  = 'int'

class scalar_converter(base_converter):
    _build_information = [scalar_info.scalar_info()]        

    def type_spec(self,name,value):
        # factory
        new_spec = self.__class__()
        new_spec.name = name
        new_spec.numeric_type = type(value)
        return new_spec
    
    def declaration_code(self,inline=0):
        type = numeric_to_c_type_mapping[self.numeric_type]
        func_type = self.type_name
        name = self.name
        var_name = self.retrieve_py_variable(inline)
        template = '%(type)s %(name)s = '\
                        'convert_to_%(func_type)s (%(var_name)s,"%(name)s");\n'
        code = template % locals()
        return code

    def __repr__(self):
        msg = "(%s:: name: %s, type: %s)" % \
               (self.type_name,self.name, self.numeric_type)
        return msg
    def __cmp__(self,other):
        #only works for equal
        return cmp(self.name,other.name) or \
               cmp(self.numeric_type,other.numeric_type) or \
               cmp(self.__class__, other.__class__)

class int_converter(scalar_converter):
    type_name = 'int'
    def type_match(self,value):
        return type(value) in [IntType, LongType]
        
    def local_dict_code(self):
        code = 'local_dict["%s"] = Py::Int(%s);\n' % (self.name,self.name)        
        return code
    
class float_converter(scalar_converter):
    type_name = 'float'
    def type_match(self,value):
        return type(value) in [FloatType]
    def local_dict_code(self):
        code = 'local_dict["%s"] = Py::Float(%s);\n' % (self.name,self.name)        
        return code

class complex_converter(scalar_converter):
    type_name = 'complex'
    def type_match(self,value):
        return type(value) in [ComplexType]
    def local_dict_code(self):
        code = 'local_dict["%s"] = Py::Complex(%s.real(),%s.imag());\n' % \
                (self.name,self.name,self.name)        
        return code

def test(level=10):
    from scipy_base.testing import module_test
    module_test(__name__,__file__,level=level)

def test_suite(level=1):
    from scipy_base.testing import module_test_suite
    return module_test_suite(__name__,__file__,level=level)
        