#! /usr/bin/env python
# Last Change: Tue Oct 30 05:00 PM 2007 J

# Module for custom, common checkers for numpy (and scipy)
import sys
import os.path
from distutils.util import get_platform

from numpy.distutils.scons.libinfo import get_config_from_section, get_config
from numpy.distutils.scons.testcode_snippets import cblas_sgemm as cblas_src, \
        c_sgemm as sunperf_src, lapack_sgesv

from numpy.distutils.scons.fortran_scons import CheckF77Mangling, CheckF77Clib

from numpy.distutils.scons.configuration import opt_info, add_info

def CheckCBLAS(context, autoadd = 1):
    env = context.env

    # If section cblas is in site.cfg, use those options. Otherwise, use default
    section = "cblas"
    siteconfig, cfgfiles = get_config()
    (cpppath, libs, libpath), found = get_config_from_section(siteconfig, section)
    if found:
        headers = ['cblas.h']
        linkflags = []
        cflags = []
        st = _check_include_and_run(context, 'CBLAS', [], headers, cblas_src,
                                      libs, libpath, linkflags, cflags, autoadd)
        if st:
            add_info(env, 'cblas', opt_info('cblas', site = 1))
            return st
    else:
        if sys.platform == 'darwin':
            st = CheckAccelerate(context, autoadd)
            if st:
                add_info(env, 'cblas', opt_info('Accelerate'))
                return st
            st = CheckVeclib(context, autoadd)
            if st:
                add_info(env, 'cblas', opt_info('vecLib'))
                return st

            add_info(env, 'cblas', opt_info(''))
            return 0
            
        else:
            # Check MKL, then ATLAS, then Sunperf
            st = CheckMKL(context, autoadd)
            if st:
                add_info(env, 'cblas', opt_info('mkl'))
                return st
            st = CheckATLAS(context, autoadd)
            if st:
                add_info(env, 'cblas', opt_info('atlas'))
                return st
            st = CheckSunperf(context, autoadd)
            if st:
                add_info(env, 'cblas', opt_info('sunperf'))
                return st

            add_info(env, 'cblas', opt_info(''))
            return 0

def CheckLAPACK(context, autoadd = 1):
    # XXX: this whole thing is ugly. Think more about how to combine checkers
    # in 'meta checker' like this.
    if sys.platform == 'nt':
        import warnings
        warning.warn('FIXME: LAPACK checks not implemented yet on win32')
        return 0
    else:
        env = context.env

        # Get fortran stuff
        if not env.has_key('F77_NAME_MANGLER'):
            if not CheckF77Mangling(context):
                return 0
        if not env.has_key('F77_LDFLAGS'):
            if not CheckF77Clib(context):
                return 0

        # Get the mangled name of our test function
        sgesv_string = env['F77_NAME_MANGLER']('sgesv')
        test_src = lapack_sgesv % sgesv_string

        # Check MKL
        st = CheckMKL(context, autoadd = 1)
        if st:
            fdict = env.ParseFlags(context.env['F77_LDFLAGS'])
            fdict['LIBS'].append('lapack')
            if env.has_key('LIBS'):
                fdict['LIBS'].extend(context.env['LIBS'])
            if env.has_key('LIBPATH'):
                fdict['LIBPATH'].extend(context.env['LIBPATH'])
            st =_check_include_and_run(context, 'LAPACK (MKL)', [], [],
                    test_src, fdict['LIBS'], fdict['LIBPATH'], [], [], autoadd = 1)
            add_info(env, 'lapack', opt_info('mkl'))
            return st

        # Check ATLAS
        st = CheckATLAS(context, autoadd = 1)
        if st:
            fdict = env.ParseFlags(context.env['F77_LDFLAGS'])
            fdict['LIBS'].append('lapack')
            if env.has_key('LIBS'):
                fdict['LIBS'].extend(context.env['LIBS'])
            if env.has_key('LIBPATH'):
                fdict['LIBPATH'].extend(context.env['LIBPATH'])
            st =_check_include_and_run(context, 'LAPACK (ATLAS)', [], [],
                    test_src, fdict['LIBS'], fdict['LIBPATH'], [], [], autoadd = 1)
            add_info(env, 'lapack', opt_info('atlas'))
            # XXX: Check complete LAPACK or not
            return st

    return 0

def _my_try_link(context, src, libs, libpath, autoadd = 0):
    """Try to link the given text in src with libs and libpath."""
    env = context.env

    oldLIBS = (env.has_key('LIBS') and deepcopy(env['LIBS'])) or []
    oldLIBPATH = (env.has_key('LIBPATH') and deepcopy(env['LIBPATH'])) or []

    ret = 0
    try:
        env.AppendUnique(LIBS = libs, LIBPATH = libpath)
        ret = context.TryLink(src, '.c')
    finally:
        if not ret or not autoadd:
            env.Replace(LIBS = oldLIBS, LIBPATH = oldLIBPATH)

    return ret

def CheckGenericBLAS(context, autoadd = 1, section = 'blas'):
    """Check whether a BLAS library can be found.

    Use site.cfg if found (section given by section argument)."""
    siteconfig, cfgfiles = get_config()
    (cpppath, libs, libpath), found = get_config_from_section(siteconfig, section)
    if not found:
        libs.extend(['blas'])

    env = context.env
    # Get fortran mangling
    if not env.has_key('F77_NAME_MANGLER'):
        if not CheckF77Mangling(context):
            return 0

    test_func_name = env['F77_NAME_MANGLER']('dgemm')
    src = get_func_link_src(test_func_name)

    context.Message("Checking for Generic BLAS... ")

    st =  _my_try_link(context, src, libs, libpath, autoadd)
    if st:
        env['F77_BLAS_LIBS'] = libs
        env['F77_BLAS_LIBPATH'] = libpath

    context.Result(st)

    return st

def CheckGenericLAPACK(context, autoadd = 1, section = 'lapack'):
    """Check whether a LAPACK library can be found.

    Use site.cfg if found (section given by section argument)."""
    siteconfig, cfgfiles = get_config()
    (cpppath, libs, libpath), found = get_config_from_section(siteconfig, section)
    if not found:
        libs.extend(['lapack'])

    env = context.env
    # Get fortran mangling
    if not env.has_key('F77_NAME_MANGLER'):
        if not CheckF77Mangling(context):
            return 0

    test_func_name = env['F77_NAME_MANGLER']('dpotri')
    src = get_func_link_src(test_func_name)

    context.Message("Checking for Generic LAPACK... ")

    st =  _my_try_link(context, src, libs, libpath, autoadd)
    if st:
        env['F77_LAPACK_LIBS'] = libs
        env['F77_LAPACK_LIBPATH'] = libpath

    context.Result(st)

    return st
