from c_spec import common_base_converter
from c_spec import num_to_c_types
from Numeric import *
from types import *
import os


num_typecode = {}
num_typecode['c'] = 'PyArray_CHAR'
num_typecode['1'] = 'PyArray_SBYTE'
num_typecode['b'] = 'PyArray_UBYTE'
num_typecode['s'] = 'PyArray_SHORT'
num_typecode['w'] = 'PyArray_USHORT'
num_typecode['i'] = 'PyArray_INT' # PyArray_INT has troubles ?? What does this note mean ??
num_typecode['u'] = 'PyArray_UINT'
num_typecode['l'] = 'PyArray_LONG'
num_typecode['f'] = 'PyArray_FLOAT'
num_typecode['d'] = 'PyArray_DOUBLE'
num_typecode['F'] = 'PyArray_CFLOAT'
num_typecode['D'] = 'PyArray_CDOUBLE'

type_check_code = \
"""
class numpy_type_handler
{
public:
    void conversion_numpy_check_type(PyArrayObject* arr_obj, int numeric_type,
                                     const char* name)
    {
        // Make sure input has correct numeric type.
        // allow character and byte to match
        // also allow int and long to match
        int arr_type = arr_obj->descr->type_num;
        if ( arr_type != numeric_type &&
            !(numeric_type == PyArray_CHAR  && arr_type == PyArray_SBYTE) &&
            !(numeric_type == PyArray_SBYTE && arr_type == PyArray_CHAR)  &&
            !(numeric_type == PyArray_INT   && arr_type == PyArray_LONG)  &&
            !(numeric_type == PyArray_LONG  && arr_type == PyArray_INT)) 
        {

            char* type_names[20] = {"char","unsigned byte","byte", "short", "unsigned short",
                                    "int", "unsigned int", "long", "float", "double", 
                                    "complex float","complex double", "object","ntype",
                                    "unkown"};
            char msg[500];
            sprintf(msg,"Conversion Error: received '%s' typed array instead of '%s' typed array for variable '%s'",
                    type_names[arr_type],type_names[numeric_type],name);
            throw_error(PyExc_TypeError,msg);    
        }
    }
    
    void numpy_check_type(PyArrayObject* arr_obj, int numeric_type, const char* name)
    {
        // Make sure input has correct numeric type.
        int arr_type = arr_obj->descr->type_num;
        if ( arr_type != numeric_type &&
            !(numeric_type == PyArray_CHAR  && arr_type == PyArray_SBYTE) &&
            !(numeric_type == PyArray_SBYTE && arr_type == PyArray_CHAR)  &&
            !(numeric_type == PyArray_INT   && arr_type == PyArray_LONG)  &&
            !(numeric_type == PyArray_LONG  && arr_type == PyArray_INT)) 
        {
            char* type_names[20] = {"char","unsigned byte","byte", "short", 
                                    "unsigned short", "int", "unsigned int",
                                    "long", "float", "double", 
                                    "complex float", "complex double", 
                                    "object","ntype","unkown"};
            char msg[500];
            sprintf(msg,"received '%s' typed array instead of '%s' typed array for variable '%s'",
                    type_names[arr_type],type_names[numeric_type],name);
            throw_error(PyExc_TypeError,msg);    
        }
    }
};

numpy_type_handler x__numpy_type_handler = numpy_type_handler();
#define conversion_numpy_check_type x__numpy_type_handler.conversion_numpy_check_type
#define numpy_check_type x__numpy_type_handler.numpy_check_type

"""

size_check_code = \
"""
class numpy_size_handler
{
public:
    void conversion_numpy_check_size(PyArrayObject* arr_obj, int Ndims, 
                                     const char* name)
    {
        if (arr_obj->nd != Ndims)
        {
            char msg[500];
            sprintf(msg,"Conversion Error: received '%d' dimensional array instead of '%d' dimensional array for variable '%s'",
                    arr_obj->nd,Ndims,name);
            throw_error(PyExc_TypeError,msg);
        }    
    }
    
    void numpy_check_size(PyArrayObject* arr_obj, int Ndims, const char* name)
    {
        if (arr_obj->nd != Ndims)
        {
            char msg[500];
            sprintf(msg,"received '%d' dimensional array instead of '%d' dimensional array for variable '%s'",
                    arr_obj->nd,Ndims,name);
            throw_error(PyExc_TypeError,msg);
        }    
    }
};

numpy_size_handler x__numpy_size_handler = numpy_size_handler();
#define conversion_numpy_check_size x__numpy_size_handler.conversion_numpy_check_size
#define numpy_check_size x__numpy_size_handler.numpy_check_size

"""

numeric_init_code = \
"""
Py_Initialize();
import_array();
PyImport_ImportModule("Numeric");
"""
    
class array_converter(common_base_converter):

    def init_info(self):
        common_base_converter.init_info(self)
        self.type_name = 'numpy'
        self.check_func = 'PyArray_Check'    
        self.c_type = 'PyArrayObject*'
        self.return_type = 'PyArrayObject*'
        self.to_c_return = '(PyArrayObject*) py_obj'
        self.matching_types = [ArrayType]
        self.headers = ['"Numeric/arrayobject.h"','<complex>','<math.h>']
        self.support_code = [size_check_code, type_check_code]
        self.module_init_code = [numeric_init_code]    
               
    def get_var_type(self,value):
        return value.typecode()
    
    def template_vars(self,inline=0):
        res = common_base_converter.template_vars(self,inline)    
        if hasattr(self,'var_type'):
            res['num_type'] = num_to_c_types[self.var_type]
            res['num_typecode'] = num_typecode[self.var_type]
        res['array_name'] = self.name + "_array"
        return res
         
    def declaration_code(self,templatize = 0,inline=0):
        code = '%(py_var)s = %(var_lookup)s;\n'   \
               '%(c_type)s %(array_name)s = %(var_convert)s;\n'  \
               'conversion_numpy_check_type(%(array_name)s,%(num_typecode)s,"%(name)s");\n' \
               'int* N%(name)s = %(array_name)s->dimensions;\n' \
               'int* S%(name)s = %(array_name)s->strides;\n' \
               'int D%(name)s = %(array_name)s->nd;\n' \
               '%(num_type)s* %(name)s = (%(num_type)s*) %(array_name)s->data;\n' 
        code = code % self.template_vars(inline=inline)
        return code
