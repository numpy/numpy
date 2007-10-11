#! /usr/bin/env python
# Last Change: Thu Oct 11 03:00 PM 2007 J

# Module for support to build python extension. scons specific code goes here.
import sys
from copy import deepcopy

from extension import get_pythonlib_dir, get_python_inc

def built_with_mstools(env):
    """Return True if built with MS tools (compiler + linker)."""
    return env.has_key('MSVS')

def PythonExtension(env, target, source, *args, **kw):
    # XXX Check args and kw
    # XXX: Some things should not be set here...
    if env.has_key('LINKFLAGS'):
        LINKFLAGS = deepcopy(env['LINKFLAGS'])
    else:
        LINKFLAGS = []

    if env.has_key('CPPPPATH'):
        CPPPATH = deepcopy(env['CPPPATH'])
    else:
        CPPPATH = []

    CPPPATH.append(get_python_inc())
    if sys.platform == 'win32': 
        if built_with_mstools(env):
            # XXX: We add the path where to find python2.5.lib (or any other
            # version, of course). This seems to be necessary for MS compilers.
            env.AppendUnique(LIBPATH = get_pythonlib_dir())
    elif sys.platform == "darwin":
        # XXX: When those should be used ? (which version of Mac OS X ?)
        LINKFLAGS += ' -undefined dynamic_lookup '

    # Use LoadableModule because of Mac OS X
    wrap = env.LoadableModule(target, source, SHLIBPREFIX = '', 
                             LDMODULESUFFIX = '$PYEXTSUFFIX', LINKFLAGS = LINKFLAGS, 
                             CPPPATH = CPPPATH, *args, **kw)
    return wrap
