
import re
import os
import sys

from distutils.ccompiler import *
from distutils import ccompiler

import log

if sys.platform == 'win32':
    compiler_class['mingw32'] = ('mingw32ccompiler', 'Mingw32CCompiler',
                                 "Mingw32 port of GNU C Compiler for Win32"\
                                 "(for MSC built Python)")
    if os.environ.get('OSTYPE','')=='msys':
        # On windows platforms, we want to default to mingw32 (gcc)
        # because msvc can't build blitz stuff.
        log.info('Setting mingw32 as default compiler for nt.')
        ccompiler._default_compilers = (('nt', 'mingw32'),) \
                                       + ccompiler._default_compilers

_distutils_new_compiler = new_compiler
def new_compiler (plat=None,
                  compiler=None,
                  verbose=0,
                  dry_run=0,
                  force=0):
    # Try first C compilers from scipy_distutils.
    if plat is None:
        plat = os.name
    try:
        if compiler is None:
            compiler = get_default_compiler(plat)
        (module_name, class_name, long_description) = compiler_class[compiler]
    except KeyError:
        msg = "don't know how to compile C/C++ code on platform '%s'" % plat
        if compiler is not None:
            msg = msg + " with '%s' compiler" % compiler
        raise DistutilsPlatformError, msg

    module_name = "scipy_distutils." + module_name
    try:
        __import__ (module_name)
    except ImportError, msg:
        print msg
        module_name = module_name[6:]
        try:
            __import__(module_name)
        except ImportError:
            raise DistutilsModuleError, \
                  "can't compile C/C++ code: unable to load module '%s'" % \
                  module_name
    try:
        module = sys.modules[module_name]
        klass = vars(module)[class_name]
    except KeyError:
        raise DistutilsModuleError, \
              ("can't compile C/C++ code: unable to find class '%s' " +
               "in module '%s'") % (class_name, module_name)
    print '*'*80
    print module_name,klass
    print '*'*80
    return klass(None, dry_run, force)
