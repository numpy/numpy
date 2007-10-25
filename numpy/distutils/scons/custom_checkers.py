#! /usr/bin/env python
# Last Change: Thu Oct 25 04:00 PM 2007 J

# Module for custom, common checkers for numpy (and scipy)
import sys
import os.path
from copy import deepcopy
from distutils.util import get_platform

from libinfo_scons import NumpyCheckLib
from testcode_snippets import cblas_sgemm as cblas_src

def _check_include_and_run(context, name, cpppath, headers, run_src, libs,
                           libpath, linkflags, cflags):
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
        - libpath: list of library path.
        - linkflags: list of link flags to add."""
    context.Message('Checking for %s ... ' % name)
    env = context.env

    #----------------------------
    # Check headers are available
    #----------------------------
    oldCPPPATH = (env.has_key('CPPPATH') and deepcopy(env['CPPPATH'])) or []
    oldCFLAGS = (env.has_key('CFLAGS') and deepcopy(env['CFLAGS'])) or []
    env.Append(CPPPATH = cpppath)
    env.Append(CFLAGS = cflags)
    # XXX: handle context
    hcode = ['#include <%s>' % h for h in headers]
    # HACK: we add cpppath in the command of the source, to add dependency of
    # the check on the cpppath.
    hcode.extend(['#if 0', '%s' % cpppath, '#endif\n'])
    src = '\n'.join(hcode)

    ret = context.TryCompile(src, '.c')
    if not ret:
        env.Replace(CPPPATH = oldCPPPATH)
        env.Replace(CFLAGS = oldCFLAGS)
        context.Result('Failed: %s include not found' % name)
        return 0

    #------------------------------
    # Check a simple example works
    #------------------------------
    oldLIBPATH = (env.has_key('LIBPATH') and deepcopy(env['LIBPATH'])) or []
    oldLIBS = (env.has_key('LIBS') and deepcopy(env['LIBS'])) or []
    # XXX: RPATH, drawbacks using it ?
    oldRPATH = (env.has_key('RPATH') and deepcopy(env['RPATH'])) or []
    env.Append(LIBPATH = libpath)
    env.Append(LIBS = libs)
    env.Append(RPATH = libpath)

    # HACK: we add libpath and libs at the end of the source as a comment, to
    # add dependency of the check on those.
    src = '\n'.join(['#include <%s>' % h for h in headers] +\
                    [run_src, '#if 0', '%s' % libpath, 
                     '%s' % headers, '%s' % libs, '#endif'])
    ret = context.TryLink(src, '.c')
    if not ret:
        env.Replace(LIBS = oldLIBS)
        env.Replace(LIBPATH = oldLIBPATH)
        env.Replace(RPATH = oldRPATH)
        context.Result('Failed: %s test could not be linked and run' % name)
        return 0

    context.Result(ret)
    return ret
     
def CheckMKL(context, mkl_dir, nb):
    """mkl_lib is the root path of MKL (the one which contains include, lib,
    etc...). nb is 32, 64, emt, etc..."""

    libs = ['mkl']
    cpppath = os.path.join(mkl_dir, 'include')
    libpath = os.path.join(mkl_dir, 'lib', nb)

    return _check_include_and_run(context, 'MKL', cpppath, ['mkl.h'],
                                  cblas_src, libs, libpath, [], [])

def CheckATLAS(context, atl_dir):
    """atl_dir is the root path of ATLAS (the one which contains libatlas)."""

    libs = ['atlas', 'f77blas', 'cblas']
    libpath = atl_dir

    return _check_include_and_run(context, 'ATLAS', None, ['atlas_enum.h', 'cblas.h'],
                                  cblas_src, libs, libpath, [], [])

def CheckCBLAS(context):
    cflags = []
    libs = []
    headers = []
    if sys.platform == 'darwin':
        # According to
        # http://developer.apple.com/hardwaredrivers/ve/vector_libraries.html:
        #
        #   This page contains a continually expanding set of vector libraries
        #   that are available to the AltiVec programmer through the Accelerate
        #   framework on MacOS X.3, Panther. On earlier versions of MacOS X,
        #   these were available in vecLib.framework. The currently available
        #   libraries are described below.
        if get_platform()[-4:] == 'i386':
            is_intel = 1
            cflags.append('-msse3')
        else:
            is_intel = 0
            cflags.append('-faltivec')
        # TODO: we should have a small test code to test Accelerate vs veclib
        # XXX: This double append is not good, any other way ?
        cflags.append('-framework')
        cflags.append('Accelerate')
        headers.append('Accelerate/Accelerate.h')
    else:
        headers.append('cblas.h')
        libs.append('cblas')

    return _check_include_and_run(context, 'CBLAS', [], headers, cblas_src,
                                  libs, [], [], cflags)
