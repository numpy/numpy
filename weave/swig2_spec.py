"""
This module allows one to use SWIG2 (SWIG version >= 1.3) wrapped
objects from Weave.  SWIG-1.3 wraps objects differently from SWIG-1.1.

The code here is based on wx_spec.py.  However, this module is more
like a template for any SWIG2 wrapped converter.  To wrap any special
code that uses SWIG the user simply needs to override the defaults in
the swig2_converter class.  These special circumstances arise when one
has wrapped code that uses C++ namespaces.  However, for most
straightforward SWIG wrappers this converter should fine out of the
box.

This code also has support to automatically handle SWIG wrapped
objects that use SWIG_COBJECT_TYPES.  These use a PyCObject instead of
a string to store the opaque pointer.

By default this code assumes that the user will not link with the SWIG
runtime library (libswigpy under *nix).  In this case no type checking
will be performed by SWIG.

To turn on type checking and link with the SWIG runtime library, there
are two approaches.

 1. If you are writing a customized converter based on this code then
    in the overloaded init_info, just call swig2_converter.init_info
    with runtime=1 and add the swig runtime library to the libraries
    loaded.

 2. If you are using the default swig2_converter you need to add two
    keyword arguments to your weave.inline call:

     a. Add a define_macros=[('SWIG_NOINCLUDE', None)]

     b. Add the swigpy library to the libraries like so:
        libraries=['swigpy']

Prabhu Ramachandran <prabhu@aero.iitm.ernet.in>
"""

import common_info
from c_spec import common_base_converter
import converters
import swigptr2

#----------------------------------------------------------------------
# This code obtains the C++ pointer given a a SWIG2 wrapped C++ object
# in Python.
#----------------------------------------------------------------------

swig2_py_to_c_template = \
"""
class %(type_name)s_handler
{
public:    
    %(c_type)s convert_to_%(type_name)s(PyObject* py_obj, const char* name)
    {
        %(c_type)s c_ptr;
        swig_type_info *ty = SWIG_TypeQuery("%(c_type)s");
        // work on this error reporting...
        if (SWIG_ConvertPtr(py_obj, (void **) &c_ptr, ty,
            SWIG_POINTER_EXCEPTION | 0) == -1) {
            handle_conversion_error(py_obj,"%(type_name)s", name);
        }
        %(inc_ref_count)s
        return c_ptr;
    }
    
    %(c_type)s py_to_%(type_name)s(PyObject* py_obj,const char* name)
    {
        %(c_type)s c_ptr;
        swig_type_info *ty = SWIG_TypeQuery("%(c_type)s");
        // work on this error reporting...
        if (SWIG_ConvertPtr(py_obj, (void **) &c_ptr, ty,
            SWIG_POINTER_EXCEPTION | 0) == -1) {
            handle_bad_type(py_obj,"%(type_name)s", name);
        }
        %(inc_ref_count)s
        return c_ptr;
    }
};

%(type_name)s_handler x__%(type_name)s_handler = %(type_name)s_handler();
#define convert_to_%(type_name)s(py_obj,name) \\
        x__%(type_name)s_handler.convert_to_%(type_name)s(py_obj,name)
#define py_to_%(type_name)s(py_obj,name) \\
        x__%(type_name)s_handler.py_to_%(type_name)s(py_obj,name)

"""

#----------------------------------------------------------------------
# This code generates a new SWIG pointer object given a C++ pointer.
#
# Important note: The thisown flag of the returned object is set to 0
# by default.
#----------------------------------------------------------------------

swig2_c_to_py_template = """
PyObject* %(type_name)s_to_py(void *obj)
{
    swig_type_info *ty = SWIG_TypeQuery("%(c_type)s");
    return SWIG_NewPointerObj(obj, ty, 0);
}
"""

class swig2_converter(common_base_converter):
    """ A converter for SWIG >= 1.3 wrapped objects."""
    def __init__(self, class_name="undefined", pycobj=0):
        """If `pycobj` is True, then code is generated to deal with a
        PyCObject.

        """
        self.class_name = class_name
        self.pycobj = pycobj # This is on if a PyCObject has been used.
        common_base_converter.__init__(self)

    def init_info(self, runtime=0):
        """Keyword arguments:
        
          runtime -- If false (default), the user does not need to
          link to the swig runtime (libswipy).  In this case no SWIG
          type checking is performed.  If true, the user must link to
          the swipy runtime library and in this case type checking
          will be performed.  This option is useful when you derive a
          subclass of this one for your object converters.          

        """
        common_base_converter.init_info(self)
        # These are generated on the fly instead of defined at 
        # the class level.
        self.type_name = self.class_name
        self.c_type = self.class_name + "*"
        self.return_type = self.class_name + "*"
        self.to_c_return = None # not used
        self.check_func = None # not used

        if self.pycobj:
            self.define_macros.append(("SWIG_COBJECT_TYPES", None))

        if runtime:
            self.define_macros.append(("SWIG_NOINCLUDE", None))
        self.support_code.append(swigptr2.swigptr2_code)
    
    def type_match(self,value):
        """ This is a generic type matcher for SWIG-1.3 objects.  For
        specific instances, override this method.  The method also
        handles cases where SWIG uses a PyCObject for the `this`
        attribute and not a string.

        """
        is_match = 0
        if hasattr(value, 'this'):
            if type(value.this) == type('str'):
                try:
                    data = value.this.split('_')
                    if data[2] == 'p':
                        is_match = 1
                except AttributeError:
                    pass
            elif str(type(value.this)) == "<type 'PyCObject'>":
                is_match = 1
        return is_match

    def generate_build_info(self):
        if self.class_name != "undefined":
            res = common_base_converter.generate_build_info(self)
        else:
            # if there isn't a class_name, we don't want the
            # support_code to be included
            import base_info
            res = base_info.base_info()
        return res
        
    def py_to_c_code(self):
        return swig2_py_to_c_template % self.template_vars()

    def c_to_py_code(self):
        return swig2_c_to_py_template % self.template_vars()
                    
    def type_spec(self,name,value):
        """ This returns a generic type converter for SWIG-1.3
        objects.  For specific instances, override this function if
        necessary."""
        # factory
        pycobj = 0
        if type(value.this) == type('str'):
            class_name = value.this.split('_')[-1]
        else: # PyCObject case
            class_name = value.__class__.__name__
            if class_name[-3:] == 'Ptr':
                class_name = class_name[:-3]
            pycobj = 1
        new_spec = self.__class__(class_name, pycobj)
        new_spec.name = name
        return new_spec

    def __cmp__(self,other):
        #only works for equal
        res = -1
        try:
            res = cmp(self.name,other.name) or \
                  cmp(self.__class__, other.__class__) or \
                  cmp(self.class_name, other.class_name) or \
                  cmp(self.type_name,other.type_name)
        except:
            pass
        return res

#----------------------------------------------------------------------
# Uncomment the next line if you want this to be a default converter
# that is magically invoked by inline.
#----------------------------------------------------------------------
#converters.default.insert(0, swig2_converter())
