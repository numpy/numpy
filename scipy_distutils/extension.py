"""distutils.extension

Provides the Extension class, used to describe C/C++ extension
modules in setup scripts.

Overridden to support f2py.
"""

# created 2000/05/30, Greg Ward

__revision__ = "$Id$"

from distutils.extension import Extension as old_Extension

import re
cxx_ext_re = re.compile(r'.*[.](cpp|cxx|cc)\Z',re.I).match
fortran_pyf_ext_re = re.compile(r'.*[.](f90|f95|f77|for|ftn|f|pyf)\Z',re.I).match

class Extension(old_Extension):
    def __init__ (self, name, sources,
                  include_dirs=None,
                  define_macros=None,
                  undef_macros=None,
                  library_dirs=None,
                  libraries=None,
                  runtime_library_dirs=None,
                  extra_objects=None,
                  extra_compile_args=None,
                  extra_link_args=None,
                  export_symbols=None,
                  f2py_options=None
                 ):
        old_Extension.__init__(self,name, sources,
                               include_dirs,
                               define_macros,
                               undef_macros,
                               library_dirs,
                               libraries,
                               runtime_library_dirs,
                               extra_objects,
                               extra_compile_args,
                               extra_link_args,
                               export_symbols)
        self.f2py_options = f2py_options or []

    def has_cxx_sources(self):
        for source in self.sources:
            if cxx_ext_re(source):
                return 1
        return 0

    def has_f2py_sources(self):
        for source in self.sources:
            if fortran_pyf_ext_re(source):
                return 1
        return 0

# class Extension
