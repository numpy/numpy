"""
    build_info holds classes that define the information
    needed for building C++ extension modules for Python that
    handle different data types.  The information includes
    such as include files, libraries, and even code snippets.
       
    array_info -- for building functions that use Python
                  Numeric arrays.
"""

import base_info
import standard_array_spec
from Numeric import *
from types import *
import os

blitz_support_code =  \
"""

// This should be declared only if they are used by some function
// to keep from generating needless warnings. for now, we'll always
// declare them.

int _beg = blitz::fromStart;
int _end = blitz::toEnd;
blitz::Range _all = blitz::Range::all();

template<class T, int N>
static blitz::Array<T,N> convert_to_blitz(PyArrayObject* arr_obj,const char* name)
{    
    blitz::TinyVector<int,N> shape(0);
    blitz::TinyVector<int,N> strides(0);
    int stride_acc = 1;
    //for (int i = N-1; i >=0; i--)
    for (int i = 0; i < N; i++)
    {
        shape[i] = arr_obj->dimensions[i];
        strides[i] = arr_obj->strides[i]/sizeof(T);
    }
    //return blitz::Array<T,N>((T*) arr_obj->data,shape,        
    return blitz::Array<T,N>((T*) arr_obj->data,shape,strides,
                             blitz::neverDeleteData);
}

template<class T, int N>
static blitz::Array<T,N> py_to_blitz(PyArrayObject* arr_obj,const char* name)
{
    
    blitz::TinyVector<int,N> shape(0);
    blitz::TinyVector<int,N> strides(0);
    int stride_acc = 1;
    //for (int i = N-1; i >=0; i--)
    for (int i = 0; i < N; i++)
    {
        shape[i] = arr_obj->dimensions[i];
        strides[i] = arr_obj->strides[i]/sizeof(T);
    }
    //return blitz::Array<T,N>((T*) arr_obj->data,shape,        
    return blitz::Array<T,N>((T*) arr_obj->data,shape,strides,
                             blitz::neverDeleteData);
}
"""

import os, blitz_spec
local_dir,junk = os.path.split(os.path.abspath(blitz_spec.__file__))   
blitz_dir = os.path.join(local_dir,'blitz-20001213')

# The need to warn about compilers made the info_object method in
# converters necessary and also this little class necessary.  
# The spec/info unification needs to continue so that this can 
# incorporated into the spec somehow.

class array_info(base_info.custom_info):    
    # throw error if trying to use msvc compiler
    
    def check_compiler(self,compiler):        
        msvc_msg = 'Unfortunately, the blitz arrays used to support numeric' \
                   ' arrays will not compile with MSVC.' \
                   '  Please try using mingw32 (www.mingw.org).'
        if compiler == 'msvc':
            return ValueError, self.msvc_msg        


class array_converter(standard_array_spec.array_converter):
    def init_info(self):
        standard_array_spec.array_converter.init_info(self)
        blitz_headers = ['"blitz/array.h"','"Numeric/arrayobject.h"',
                          '<complex>','<math.h>']
        self.headers.extend(blitz_headers)
        self.include_dirs = [blitz_dir]
        self.support_code.append(blitz_support_code)
        
        # type_name is used to setup the initial type conversion.  Even
        # for blitz conversion, the first step is to convert it to a
        # standard numpy array.
        #self.type_name = 'blitz'
        self.type_name = 'numpy'
        
    def info_object(self):
        return array_info()

    def type_spec(self,name,value):
        new_spec = standard_array_spec.array_converter.type_spec(self,name,value)
        new_spec.dims = len(value.shape)
        if new_spec.dims > 11:
            msg = "Error converting variable '" + name + "'.  " \
                  "blitz only supports arrays up to 11 dimensions."
            raise ValueError, msg
        return new_spec

    def template_vars(self,inline=0):
        res = standard_array_spec.array_converter.template_vars(self,inline)    
        if hasattr(self,'dims'):
            res['dims'] = self.dims
        return res    

    def declaration_code(self,templatize = 0,inline=0):
        code = '%(py_var)s = %(var_lookup)s;\n'   \
               '%(c_type)s %(array_name)s = %(var_convert)s;\n'  \
               'conversion_numpy_check_type(%(array_name)s,%(num_typecode)s,"%(name)s");\n' \
               'conversion_numpy_check_size(%(array_name)s,%(dims)s,"%(name)s");\n' \
               'blitz::Array<%(num_type)s,%(dims)d> %(name)s =' \
               ' convert_to_blitz<%(num_type)s,%(dims)d>(%(array_name)s,"%(name)s");\n' \
               'blitz::TinyVector<int,%(dims)d> N%(name)s = %(name)s.shape();\n'
        code = code % self.template_vars(inline=inline)
        return code

    def __cmp__(self,other):
        #only works for equal
        return ( cmp(self.name,other.name) or  
                 cmp(self.var_type,other.var_type) or 
                 cmp(self.dims, other.dims) or 
                 cmp(self.__class__, other.__class__) )

