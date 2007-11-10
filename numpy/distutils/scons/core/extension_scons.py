#! /usr/bin/env python
# Last Change: Fri Oct 19 11:00 AM 2007 J

# Module for support to build python extension. scons specific code goes here.
import sys
from copy import deepcopy

from distutils.unixccompiler import UnixCCompiler
from numpy.distutils.misc_util import msvc_runtime_library

from extension import get_pythonlib_dir, get_python_inc

def built_with_mstools(env):
    """Return True if built with MS tools (compiler + linker)."""
    return env['cc_opt'] == 'msvc'

def built_with_mingw(env):
    """Return true if built with mingw compiler."""
    return env['cc_opt'] == 'mingw'

def get_pythonlib_name(debug = 0):
    """Return the name of python library (necessary to link on NT with
    mingw."""
    # Yeah, distutils burried the link option on NT deep down in
    # Extension module, we cannot reuse it !
    if debug == 1:
	template = 'python%d%d_d'
    else:
	template = 'python%d%d'

    return template % (sys.hexversion >> 24, 
		       (sys.hexversion >> 16) & 0xff)


def PythonExtension(env, target, source, *args, **kw):
    # XXX Check args and kw
    # XXX: Some things should not be set here... Actually, this whole
    # thing is a mess.
    if env.has_key('LINKFLAGS'):
        LINKFLAGS = deepcopy(env['LINKFLAGS'])
    else:
        LINKFLAGS = []

    if env.has_key('CPPPATH'):
        CPPPATH = deepcopy(env['CPPPATH'])
    else:
        CPPPATH = []

    if env.has_key('LIBPATH'):
        LIBPATH = deepcopy(env['LIBPATH'])
    else:
        LIBPATH = []

    if env.has_key('LIBS'):
        LIBS = deepcopy(env['LIBS'])
    else:
        LIBS = []

    CPPPATH.append(get_python_inc())
    if sys.platform == 'win32': 
        if built_with_mstools(env):
            # XXX: We add the path where to find python lib (or any other
            # version, of course). This seems to be necessary for MS compilers.
            #env.AppendUnique(LIBPATH = get_pythonlib_dir())
	    LIBPATH.append(get_pythonlib_dir())
    	elif built_with_mingw(env):
	    # XXX: this part should be moved elsewhere (mingw abstraction
	    # for python)

	    # This is copied from mingw32ccompiler.py in numpy.distutils
	    # (not supported by distutils.)

	    # Include the appropiate MSVC runtime library if Python was
	    # built with MSVC >= 7.0 (MinGW standard is msvcrt)
            py_runtime_library = msvc_runtime_library()
	    LIBPATH.append(get_pythonlib_dir())
	    LIBS.extend([get_pythonlib_name(), py_runtime_library])

    elif sys.platform == "darwin":
        # XXX: When those should be used ? (which version of Mac OS X ?)
        LINKFLAGS += ' -undefined dynamic_lookup '
    else:
	pass

    # Use LoadableModule because of Mac OS X
    # ... but scons has a bug (#issue 1669) with mingw and Loadable
    # Module, so use SharedLibrary with mingw.
    if built_with_mingw(env):
        wrap = env.SharedLibrary(target, source, SHLIBPREFIX = '', 
                                 #LDMODULESUFFIX = '$PYEXTSUFFIX', 
                                 SHLIBSUFFIX = '$PYEXTSUFFIX', 
                                 LINKFLAGS = LINKFLAGS, 
                                 LIBS = LIBS, 
                                 LIBPATH = LIBPATH, 
                                 CPPPATH = CPPPATH, 
				 *args, **kw)
    else:
        wrap = env.LoadableModule(target, source, SHLIBPREFIX = '', 
                                  LDMODULESUFFIX = '$PYEXTSUFFIX', 
                                  SHLIBSUFFIX = '$PYEXTSUFFIX', 
                                  LINKFLAGS = LINKFLAGS, 
                                  LIBS = LIBS, 
                                  LIBPATH = LIBPATH, 
                                  CPPPATH = CPPPATH, *args, **kw)
    return wrap
