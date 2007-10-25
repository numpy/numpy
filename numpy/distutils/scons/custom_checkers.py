#! /usr/bin/env python
# Last Change: Thu Oct 25 02:00 PM 2007 J

# Module for custom, common checkers for numpy (and scipy)
import os.path
from copy import deepcopy

from libinfo_scons import NumpyCheckLib
from testcode_snippets import cblas_sgemm as cblas_src

def _check_include_and_run(context, name, cpppath, headers, run_src, libs, libpath):
    """This is a basic implementation for generic "test include and run"
    testers.
    
    For example, for library foo, which implements function do_foo, and with
    include header foo.h, this will:
        - test that foo.h is found and compilable by the compiler
        - test that the given source code can be compiled. The source code
          should contain a simple program with the function.
          
    Arguments:
        - name: name of the library
        - cpppath: list of directories
        - headers: list of headers
        - run_src: the code for the run test
        - libs: list of libraries to link
        - libpath: list of library path."""
    context.Message('Checking for %s ... ' % name)
    env = context.env

    # Check headers are available
    oldCPPPATH = (env.has_key('CPPPATH') and deepcopy(env['CPPPATH'])) or []
    env.Append(CPPPATH = cpppath)
    # XXX: handle context
    src = '\n'.join(headers)

    ret = context.TryCompile(src, '.c')
    if not ret:
        env.Replace(CPPPATH = oldCPPPATH)
        context.Result('Failed: %s include not found' % name)
        return 0

    # Check a simple cblas example works
    oldLIBPATH = (env.has_key('LIBPATH') and deepcopy(env['LIBPATH'])) or []
    oldLIBS = (env.has_key('LIBS') and deepcopy(env['LIBS'])) or []
    oldRPATH = (env.has_key('RPATH') and deepcopy(env['RPATH'])) or []
    env.Append(LIBPATH = libpath)
    env.Append(LIBS = libs)
    env.Append(RPATH = libpath)

    ret = context.TryLink(run_src, '.c')
    if not ret:
        env.Replace(LIBS = oldLIBS)
        env.Replace(LIBPATH = oldLIBPATH)
        env.Replace(RPATH = oldRPATH)
        context.Result('Failed: %s test could not be linked and run' % name)
        return 0

    context.Result(ret)
    return ret
     
