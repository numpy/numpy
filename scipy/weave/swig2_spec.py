"""
This module allows one to use SWIG2 (SWIG version >= 1.3) wrapped
objects from Weave.  SWIG-1.3 wraps objects differently from SWIG-1.1.

The code here is based on wx_spec.py.  However, this module is more
like a template for any SWIG2 wrapped converter.  To wrap any special
code that uses SWIG the user simply needs to override the defaults in
the swig2_converter class.  These special circumstances arise when one
has wrapped code that uses C++ namespaces.  However, for most
straightforward SWIG wrappers this converter should work fine out of
the box.

Newer versions of SWIG (>=1.3.22) represent the wrapped object using a
PyCObject and also a PySwigObject (>=1.3.24).  This code supports all
of these options transparently.

Since SWIG-1.3.x is under intense development there are several issues
to consider when using the swig2_converter.

 1. For SWIG versions <= 1.3.19, the runtime code was built either
    into the module or into a separate library called libswigpy (or
    something like that).  In the latter case, the users Python
    modules were linked to this library and shared type information
    (this was common for large projects with several modules that
    needed to share type information).  If you are using multiple
    inheritance and want to be certain that type coercions from a
    derived class to a base class are done correctly, you will need to
    link to the libswigpy library.  You will then need to add these to
    the keyword arguments passed along to `weave.inline`:

      a. Add a define_macros=[('SWIG_NOINCLUDE', None)]

      b. Add the swigpy library to the libraries like so:
         libraries=['swigpy']

      c. If the libswigpy is in a non-standard location add the path
         to the library_dirs argument as
         `library_dirs=['/usr/local/lib']` or whatever.

    OTOH if you do not need to link to libswigpy (this is likely if
    you are not using multiple inheritance), then you do not need the
    above.  However you are likely to get an annoying message of the
    form::

      WARNING: swig_type_info is NULL.

    for each SWIG object you are inlining (during each call).  To
    avoid this add a define_macros=[('NO_SWIG_WARN', None)].

 2. Since keeping track of a separate runtime is a pain, for SWIG
    versions >= 1.3.23 the type information was stored inside a
    special module.  Thus in these versions there is no need to link
    to this special SWIG runtime library.  This module handles these
    cases automatically and nothing special need be done.

    Using modules wrapped with different SWIG versions simultaneously.
    Lets say you have library 'A' that is wrapped using SWIG version
    1.3.20.  Then lets say you have a library 'B' wrapped using
    version 1.3.24.  Now if you want to use both in weave.inline, we
    have a serious problem.  The trouble is that both 'A' and 'B' may
    use different and incompatible runtime layouts.  It is impossible
    to get the type conversions right in these cases.  Thus it is
    strongly advised that you use one version of SWIG to wrap all of
    the code that you intend to inline using weave.  Note that you can
    certainly use SWIG-1.3.23 for everything and do not have to use
    the latest and greatest SWIG to use weave.inline.  Just make sure
    that when inlining SWIG wrapped objects that all such objects use
    the same runtime layout.  By default, if you are using different
    versions and do need to inline these objects, the latest layout
    will be assumed.  This might leave you with holes in your feet,
    but you have been warned.  You can force the converter to use a
    specific runtime version if you want (see the
    `swig2_converter.__init__` method and its documentation).


Prabhu Ramachandran <prabhu_r@users.sf.net>
"""

import sys
import common_info
from c_spec import common_base_converter
import converters
import swigptr2


#----------------------------------------------------------------------
# Commonly used functions for the type query.  This is done mainly to
# avoid code duplication.
#----------------------------------------------------------------------
swig2_common_code = \
'''
swig_type_info *
Weave_SWIG_TypeQuery(const char *name) {
    swig_type_info *ty = SWIG_TypeQuery(name);
#ifndef NO_SWIG_WARN
    if (ty == NULL) {
        printf("WARNING: swig_type_info is NULL.\\n");
    }
#endif
    return ty;
}
'''
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
        swig_type_info *ty = Weave_SWIG_TypeQuery("%(c_type)s");
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
        swig_type_info *ty = Weave_SWIG_TypeQuery("%(c_type)s");
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
    swig_type_info *ty = Weave_SWIG_TypeQuery("%(c_type)s");
    return SWIG_NewPointerObj(obj, ty, 0);
}
"""

class swig2_converter(common_base_converter):
    """ A converter for SWIG >= 1.3 wrapped objects."""
    def __init__(self, class_name="undefined", pycobj=0, runtime_version=None):
        """Initializes the instance.

        Parameters
        ----------

        - class_name : `string`

          Name of class, this is set dynamically at build time by the
          `type_spec` method.

        - pycobj : `int`

          If `pycobj` is 0 then code is generated to deal with string
          representations of the SWIG wrapped pointer.  If it is 1,
          then code is generated to deal with a PyCObject.  If it is 2
          then code is generated to deal with with PySwigObject.

        - runtime_version : `int`

          Specifies the SWIG_RUNTIME_VERSION to use.  Defaults to
          `None`.  In this case the runtime is automatically
          determined.  This option is useful if you want to force the
          runtime_version to be a specific one and override the
          auto-detected one.

        """
        self.class_name = class_name
        self.pycobj = pycobj # This is on if a PyCObject has been used.
        self.runtime_version = runtime_version
        common_base_converter.__init__(self)

    def _get_swig_runtime_version(self):
        """This method tries to deduce the SWIG runtime version.  If
        the SWIG runtime layout changes, the `SWIG_TypeQuery` function
        will not work properly.
        """
        versions = []
        for key in sys.modules.keys():
            idx = key.find('swig_runtime_data')
            if idx > -1:
                ver = int(key[idx+17:])
                if ver not in versions:
                    versions.append(ver)
        nver = len(versions)
        if nver == 0:
            return 0
        elif nver == 1:
            return versions[0]
        else:
            print "WARNING: Multiple SWIG versions detected.  No version was"
            print "explicitly specified.  Using the highest possible version."
            return max(versions)

    def init_info(self, runtime=0):
        """Keyword arguments:
        
          runtime -- If false (default), the user does not need to
          link to the swig runtime (libswipy).  Newer versions of SWIG
          (>=1.3.23) do not need to build a SWIG runtime library at
          all.  In these versions of SWIG the swig_type_info is stored
          in a common module.  swig_type_info stores the type
          information and the type converters to cast pointers
          correctly.

          With earlier versions of SWIG (<1.3.22) one has to either
          link the weave module with a SWIG runtime library
          (libswigpy) in order to get the swig_type_info.  Thus, if
          `runtime` is True, the user must link to the swipy runtime
          library and in this case type checking will be performed.
          With these versions of SWIG, if runtime is `False`, no type
          checking is done.

        """
        common_base_converter.init_info(self)
        # These are generated on the fly instead of defined at 
        # the class level.
        self.type_name = self.class_name
        self.c_type = self.class_name + "*"
        self.return_type = self.class_name + "*"
        self.to_c_return = None # not used
        self.check_func = None # not used

        if self.pycobj == 1:
            self.define_macros.append(("SWIG_COBJECT_TYPES", None))
            self.define_macros.append(("SWIG_COBJECT_PYTHON", None))
        elif self.pycobj == 2:
            self.define_macros.append(("SWIG_COBJECT_TYPES", None))
            
            
        if self.runtime_version is None:
            self.runtime_version = self._get_swig_runtime_version()

        rv = self.runtime_version
        if rv == 0:
            # The runtime option is only useful for older versions of
            # SWIG.
            if runtime:
                self.define_macros.append(("SWIG_NOINCLUDE", None))
            self.support_code.append(swigptr2.swigptr2_code_v0)
        elif rv == 1:
            self.support_code.append(swigptr2.swigptr2_code_v1)
        elif rv == 2:
            self.support_code.append(swigptr2.swigptr2_code_v2)
        else:
            raise AssertionError, "Unsupported version of the SWIG runtime:", rv

        self.support_code.append(swig2_common_code)

    def _get_swig_type(self, value):
        """Given the object in the form of `value`, this method
        returns information on the SWIG internal object repesentation
        type.  Different versions of SWIG use different object
        representations.  This method provides information on the type
        of internal representation.

        Currently returns one of ['', 'str', 'pycobj', 'pyswig'].
        """
        swig_typ = ''
        if hasattr(value, 'this'):
            type_this = type(value.this)
            type_str = str(type_this)
            if type_this == type('str'):
                try:
                    data = value.this.split('_')
                    if data[2] == 'p':
                        swig_typ = 'str'
                except AttributeError:
                    pass
            elif type_str == "<type 'PyCObject'>":
                swig_typ = 'pycobj'
            elif type_str.find('PySwig') > -1:
                swig_typ = 'pyswig'

        return swig_typ        
    
    def type_match(self,value):
        """ This is a generic type matcher for SWIG-1.3 objects.  For
        specific instances, override this method.  The method also
        handles cases where SWIG uses a PyCObject for the `this`
        attribute and not a string.

        """
        if self._get_swig_type(value):
            return 1
        else:
            return 0

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
        swig_ob_type = self._get_swig_type(value)
        pycobj = 0
        if swig_ob_type == 'str':
            class_name = value.this.split('_')[-1]
        elif swig_ob_type == 'pycobj':
            pycobj = 1
        elif swig_ob_type == 'pyswig':
            pycobj = 2
        else:
            raise AssertionError, "Does not look like a SWIG object: %s"%value

        if pycobj:
            class_name = value.__class__.__name__
            if class_name[-3:] == 'Ptr':
                class_name = class_name[:-3]
            
        new_spec = self.__class__(class_name, pycobj, self.runtime_version)
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
