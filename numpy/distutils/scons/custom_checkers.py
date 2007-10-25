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

    #----------------------------
    # Check headers are available
    #----------------------------
    oldCPPPATH = (env.has_key('CPPPATH') and deepcopy(env['CPPPATH'])) or []
    env.Append(CPPPATH = cpppath)
    # XXX: handle context
    hcode = ['#include <%s>' % h for h in headers]
    # HACK: we add cpppath in the command of the source, to add dependency of
    # the check on the cpppath.
    hcode.extend(['#if 0', '%s' % cpppath, '#endif\n'])
    src = '\n'.join(hcode)

    ret = context.TryCompile(src, '.c')
    if not ret:
        env.Replace(CPPPATH = oldCPPPATH)
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
    src = '\n'.join([run_src, '#if 0', '%s' % libpath, '%s' % libs, '#endif'])
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
                                  cblas_src, libs, libpath)

def CheckATLAS(context, atl_dir):
    """atl_dir is the root path of ATLAS (the one which contains libatlas)."""

    libs = ['atlas', 'f77blas', 'cblas']
    libpath = atl_dir

    return _check_include_and_run(context, 'ATLAS', None, ['atlas_enum.h'],
                                  cblas_src, libs, libpath)

def CheckCBLAS(context):
    libs = ['cblas']

    return _check_include_and_run(context, 'CBLAS', [], [], cblas_src,
                                  libs, [])
