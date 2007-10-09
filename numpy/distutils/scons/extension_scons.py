#! /usr/bin/env python
# Last Change: Tue Oct 09 04:00 PM 2007 J

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
        # XXX: pyext should definitely not be set here
        pyext = '.pyd'
        if built_with_mstools(env):
            # # XXX is the export necessary ? (this seems to work wo)
            # LINKFLAGS += " /EXPORT:init%s " % target[0]

            # XXX: We add the path where to find python2.5.lib (or any other
            # version, of course). This seems to be necessary for MS compilers.
            env.AppendUnique(LIBPATH = get_pythonlib_dir())
    else:
        pyext = env['SHLIBSUFFIX']
    wrap = env.SharedLibrary(target, source, SHLIBPREFIX = '', 
                             SHLIBSUFFIX = pyext, LINKFLAGS = LINKFLAGS, 
                             CPPPATH = CPPPATH, *args, **kw)
    return wrap
