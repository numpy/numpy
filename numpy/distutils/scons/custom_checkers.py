#! /usr/bin/env python
# Last Change: Fri Oct 26 09:00 PM 2007 J

# Module for custom, common checkers for numpy (and scipy)
import sys
import os.path
from copy import deepcopy
from distutils.util import get_platform

from libinfo import get_config, get_config_from_section
from libinfo_scons import NumpyCheckLib
from testcode_snippets import cblas_sgemm as cblas_src, \
        c_sgemm as sunperf_src, lapack_sgesv

from fortran_scons import CheckF77Mangling, CheckF77Clib

def _check_include_and_run(context, name, cpppath, headers, run_src, libs,
                           libpath, linkflags, cflags, autoadd = 1):
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
    env.AppendUnique(CPPPATH = cpppath)
    env.AppendUnique(CFLAGS = cflags)
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
    oldLINKFLAGS = (env.has_key('LINKFLAGS') and deepcopy(env['LINKFLAGS'])) or []
    env.AppendUnique(LIBPATH = libpath)
    env.AppendUnique(LIBS = libs)
    env.AppendUnique(RPATH = libpath)
    env.AppendUnique(LINKFLAGS = linkflags)

    # HACK: we add libpath and libs at the end of the source as a comment, to
    # add dependency of the check on those.
    src = '\n'.join(['#include <%s>' % h for h in headers] +\
                    [run_src, '#if 0', '%s' % libpath, 
                     '%s' % headers, '%s' % libs, '#endif'])
    ret = context.TryLink(src, '.c')
    if (not ret or not autoadd):
        # If test failed or autoadd = 0, restore everything
        env.Replace(LIBS = oldLIBS)
        env.Replace(LIBPATH = oldLIBPATH)
        env.Replace(RPATH = oldRPATH)
        env.Replace(LINKFLAGS = oldLINKFLAGS)

    if not ret:
        context.Result('Failed: %s test could not be linked and run' % name)
        return 0

    context.Result(ret)
    return ret
     
#def CheckMKL(context, mkl_dir, nb):
#    """mkl_lib is the root path of MKL (the one which contains include, lib,
#    etc...). nb is 32, 64, emt, etc..."""
#
#    libs = ['mkl']
#    cpppath = os.path.join(mkl_dir, 'include')
#    libpath = os.path.join(mkl_dir, 'lib', nb)
#
#    return _check_include_and_run(context, 'MKL', cpppath, ['mkl.h'],
#                                  cblas_src, libs, libpath, [], [], autoadd)

def CheckMKL(context, autoadd = 1):
    """Check MKL is usable using a simple cblas example."""
    section = "mkl"
    siteconfig, cfgfiles = get_config()
    (cpppath, libs, libpath), found = get_config_from_section(siteconfig, section)
    if not found:
        # XXX: find exact options to use for the MKL
        libs.extend(['lapack', 'mkl', 'guide', 'm'])
    headers = ['mkl.h']

    return _check_include_and_run(context, 'MKL', cpppath, headers,
                                  cblas_src, libs, libpath, [], [], autoadd)

def CheckATLAS(context, autoadd = 1):
    """Check whether ATLAS is usable in C."""

    libs = ['atlas', 'f77blas', 'cblas']
    libpath = []

    return _check_include_and_run(context, 'ATLAS', None, ['atlas_enum.h', 'cblas.h'],
                                  cblas_src, libs, libpath, [], [], autoadd)

def CheckAccelerate(context, autoadd = 1):
    """Checker for Accelerate framework (on Mac OS X >= 10.3). Only test from
    C."""
    # According to
    # http://developer.apple.com/hardwaredrivers/ve/vector_libraries.html:
    #
    #   This page contains a continually expanding set of vector libraries
    #   that are available to the AltiVec programmer through the Accelerate
    #   framework on MacOS X.3, Panther. On earlier versions of MacOS X,
    #   these were available in vecLib.framework. The currently available
    #   libraries are described below.

    #XXX: get_platform does not seem to work...
    #if get_platform()[-4:] == 'i386':
    #    is_intel = 1
    #    cflags.append('-msse3')
    #else:
    #    is_intel = 0
    #    cflags.append('-faltivec')

    # XXX: This double append is not good, any other way ?
    linkflags = ['-framework', 'Accelerate']

    return _check_include_and_run(context, 'FRAMEWORK: Accelerate', None, 
                                  ['Accelerate/Accelerate.h'], cblas_src, [], 
                                  [], linkflags, [], autoadd)

def CheckVeclib(context, autoadd = 1):
    """Checker for Veclib framework (on Mac OS X < 10.3)."""
    # XXX: This double append is not good, any other way ?
    linkflags = ['-framework', 'vecLib']

    return _check_include_and_run(context, 'FRAMEWORK: veclib', None, 
                                  ['vecLib/vecLib.h'], cblas_src, [], 
                                  [], linkflags, [], autoadd)

def CheckSunperf(context, autoadd = 1):
    """Checker for sunperf using a simple sunperf example"""

    # XXX: Other options needed ?
    linkflags = ['-xlic_lib=sunperf']
    cflags = ['-dalign']

    return _check_include_and_run(context, 'sunperf', None, 
                                  ['sunperf.h'], sunperf_src, [], 
                                  [], linkflags, cflags, autoadd)

def CheckCBLAS(context, autoadd = 1):

    # If section cblas is in site.cfg, use those options. Otherwise, use default
    section = "cblas"
    siteconfig, cfgfiles = get_config()
    (cpppath, libs, libpath), found = get_config_from_section(siteconfig, section)
    if found:
        headers = ['cblas.h']
        linkflags = []
        cflags = []
        return _check_include_and_run(context, 'CBLAS', [], headers, cblas_src,
                                      libs, libpath, linkflags, cflags, autoadd)
    else:
        if sys.platform == 'darwin':
            st = CheckAccelerate(context, autoadd)
            if st:
                return st
            st = CheckVeclib(context, autoadd)
            return st
            
        else:
            # Check MKL, then ATLAS, then Sunperf
            st = CheckMKL(context, autoadd)
            if st:
                return st
            st = CheckATLAS(context, autoadd)
            if st:
                return st
            st = CheckSunperf(context, autoadd)
            return st

def CheckLAPACK(context, autoadd = 1):
    # XXX: this whole thing is ugly. Think more about how to combine checkers
    # in 'meta checker' like this.
    if sys.platform == 'nt':
        import warnings
        warning.warn('FIXME: LAPACK checks not implemented yet on win32')
        return 0
    else:
        # Get fortran stuff
        if not CheckF77Mangling(context):
            return 0
        if not CheckF77Clib(context):
            return 0

        env = context.env

        # Get the mangled name of our test function
        sgesv_string = env['F77_NAME_MANGLER']('sgesv')
        test_src = lapack_sgesv % sgesv_string

        # Check MKL
        st = CheckMKL(context, autoadd)
        if st:
            fdict = env.ParseFlags(context.env['F77_LDFLAGS'])
            fdict['LIBS'].append('lapack')
            if env.has_key('LIBS'):
                fdict['LIBS'].extend(context.env['LIBS'])
            if env.has_key('LIBPATH'):
                fdict['LIBPATH'].extend(context.env['LIBPATH'])
            st =_check_include_and_run(context, 'LAPACK (MKL)', [], [],
                    test_src, fdict['LIBS'], fdict['LIBPATH'], [], [], autoadd = 1)
            return st

        # Check ATLAS
        st = CheckATLAS(context, autoadd)
        if st:
            fdict = env.ParseFlags(context.env['F77_LDFLAGS'])
            fdict['LIBS'].append('lapack')
            if env.has_key('LIBS'):
                fdict['LIBS'].extend(context.env['LIBS'])
            if env.has_key('LIBPATH'):
                fdict['LIBPATH'].extend(context.env['LIBPATH'])
            st =_check_include_and_run(context, 'LAPACK (ATLAS)', [], [],
                    test_src, fdict['LIBS'], fdict['LIBPATH'], [], [], autoadd = 1)
            # XXX: Check complete LAPACK or not
            return st

    return 0
