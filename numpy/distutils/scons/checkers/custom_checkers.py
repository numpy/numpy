#! /usr/bin/env python
# Last Change: Tue Nov 06 11:00 PM 2007 J

# Module for custom, common checkers for numpy (and scipy)
import sys
import os.path
from distutils.util import get_platform

from numpy.distutils.scons.core.libinfo import get_config_from_section, get_config
from numpy.distutils.scons.testcode_snippets import cblas_sgemm as cblas_src, \
        c_sgemm as sunperf_src, lapack_sgesv

from numpy.distutils.scons.fortran_scons import CheckF77Mangling, CheckF77Clib

from numpy.distutils.scons.configuration import add_info

from perflib import CheckMKL, CheckATLAS, CheckSunperf, CheckAccelerate
from support import check_include_and_run, ConfigOpts, ConfigRes

# XXX: many perlib can be used from both C and F (Atlas being a notable
# exception for LAPACK). So shall we make the difference between BLAS, CBLAS,
# LAPACK and CLAPACK ? How to test for fortran ?

def CheckCBLAS(context, autoadd = 1):
    """This checker tries to find optimized library for cblas.

    This test is pretty strong: it first detects an optimized library, and then
    tests that a simple cblas program can be run using this lib.
    
    It looks for the following libs:
        - Mac OS X: Accelerate, and then vecLib.
        - Others: MKL, then ATLAS, then Sunperf."""
    # XXX: rpath vs LD_LIBRARY_PATH ?
    env = context.env

    # If section cblas is in site.cfg, use those options. Otherwise, use default
    section = "cblas"
    siteconfig, cfgfiles = get_config()
    (cpppath, libs, libpath), found = get_config_from_section(siteconfig, section)
    if found:
        cfg = ConfigOpts(cpppath = cpppath, libs = libs, libpath = libpath,
                         rpath = libpath)
        st = check_include_and_run(context, 'CBLAS (from site.cfg) ', cfg,
                                  [], cblas_src, autoadd)
        if st:
            add_info(env, 'cblas', ConfigRes('cblas', cfg, found))
        return st
    else:
        if sys.platform == 'darwin':
            st, res = CheckAccelerate(context, autoadd)
            if st:
                st = check_include_and_run(context, 'CBLAS (Accelerate Framework)', 
                                           res.cfgopts, [], cblas_src, autoadd)
                if st:
                    add_info(env, 'cblas', res)
                return st

            st, res = CheckVeclib(context, autoadd)
            if st:
                st = check_include_and_run(context, 'CBLAS (vecLib Framework)', 
                                           res.cfgopts, [], cblas_src, autoadd)
                if st:
                    add_info(env, 'cblas', res)
                return st

            add_info(env, 'cblas', 'Def numpy implementation used')
            return 0
            
        else:
            # XXX: think about how to share headers info between checkers ?
            # Check MKL
            st, res = CheckMKL(context, autoadd)
            if st:
                st = check_include_and_run(context, 'CBLAS (MKL)', res.cfgopts,
                                           [], cblas_src, autoadd)
                if st:
                    add_info(env, 'cblas', res)
                return st

            # Check ATLAS
            st, res = CheckATLAS(context, autoadd)
            if st:
                res.cfgopts['libs'].insert(0, 'blas')
                st = check_include_and_run(context, 'CBLAS (ATLAS)', res.cfgopts,
                                           [], cblas_src, autoadd)
                if st:
                    add_info(env, 'cblas', res)
                return st

            # Check Sunperf
            st, res = CheckSunperf(context, autoadd)
            if st:
                st = check_include_and_run(context, 'CBLAS (Sunperf)', res.cfgopts,
                                           [], cblas_src, autoadd)
                if st:
                    add_info(env, 'cblas', res)
                return st

            add_info(env, 'cblas', 'Def numpy implementation used')
            return 0

def CheckLAPACK(context, autoadd = 1):
    """This checker tries to find optimized library for lapack.

    This test is pretty strong: it first detects an optimized library, and then
    tests that a simple cblas program can be run using this lib.
    
    It looks for the following libs:
        - Mac OS X: Accelerate, and then vecLib.
        - Others: MKL, then ATLAS."""
    env = context.env

    # If section lapack is in site.cfg, use those options. Otherwise, use default
    section = "lapack"
    siteconfig, cfgfiles = get_config()
    (cpppath, libs, libpath), found = get_config_from_section(siteconfig, section)
    if found:
        cfg = ConfigOpts(cpppath = cpppath, libs = libs, libpath = libpath,
                         rpath = libpath)

        if not env.has_key('F77_NAME_MANGLER'):
            if not CheckF77Mangling(context):
                return 0
        if not env.has_key('F77_LDFLAGS'):
            if not CheckF77Clib(context):
                return 0

        # Get the mangled name of our test function
        sgesv_string = env['F77_NAME_MANGLER']('sgesv')
        test_src = lapack_sgesv % sgesv_string

        st = check_include_and_run(context, 'LAPACK (from site.cfg) ', cfg,
                                  [], test_src, autoadd)
        if st:
            add_info(env, 'lapack', ConfigRes('lapack', cfg, found))
        return st
    else:
        if sys.platform == 'nt':
            import warnings
            warning.warn('FIXME: LAPACK checks not implemented yet on win32')
            return 0

        if sys.platform == 'darwin':
            st, opts = CheckAccelerate(context, autoadd)
            if st:
                if st:
                    add_info(env, 'lapack: Accelerate', opts)
                return st
            st, opts = CheckAccelerate(context, autoadd)
            if st:
                if st:
                    add_info(env, 'lapack: vecLib', opts)
                return st

        else:
            # Get fortran stuff (See XXX at the top on F77 vs C)
            if not env.has_key('F77_NAME_MANGLER'):
                if not CheckF77Mangling(context):
		    add_info(env, 'lapack', 'Def numpy implementation used')
                    return 0
            if not env.has_key('F77_LDFLAGS'):
                if not CheckF77Clib(context):
		    add_info(env, 'lapack', 'Def numpy implementation used')
                    return 0

            # Get the mangled name of our test function
            sgesv_string = env['F77_NAME_MANGLER']('sgesv')
            test_src = lapack_sgesv % sgesv_string

            # Check MKL
            st, res = CheckMKL(context, autoadd)
            if st:
                # Intel recommends linking lapack before mkl, guide and co
                res.cfgopts['libs'].insert(0, 'lapack')
                st = check_include_and_run(context, 'LAPACK (MKL)', res.cfgopts,
                                           [], test_src, autoadd)
                if st:
                    add_info(env, 'lapack', res)
                return st

            # Check ATLAS
            st, res = CheckATLAS(context, autoadd = 1)
            if st:
                res.cfgopts['libs'].insert(0, 'lapack')
                st = check_include_and_run(context, 'LAPACK (ATLAS)', res.cfgopts,
                                           [], test_src, autoadd)
                if st:
                    add_info(env, 'lapack', res)
                # XXX: Check complete LAPACK or not. (Checking for not
                # implemented lapack symbols ?)
                return st

            # Check Sunperf
            st, res = CheckSunperf(context, autoadd)
            if st:
                st = check_include_and_run(context, 'LAPACK (Sunperf)', res.cfgopts,
                                           [], test_src, autoadd)
                if st:
                    add_info(env, 'lapack', res)
                return st

    add_info(env, 'lapack', 'Def numpy implementation used')
    return 0
