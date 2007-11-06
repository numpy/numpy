#! /usr/bin/env python
# Last Change: Tue Nov 06 01:00 PM 2007 J

# This module defines checkers for performances libs providing standard API,
# such as MKL (Intel), ATLAS, Sunperf (solaris and linux), Accelerate (Mac OS
# X), etc... Those checkers merely only check whether the library is found
# using a library specific check if possible, or other heuristics.
# Generally, you don't use those directly: they are used in 'meta' checkers,
# such as BLAS, CBLAS, LAPACK checkers.
import re
import os
from os.path import join as pjoin

from numpy.distutils.system_info import default_lib_dirs
from numpy.distutils.scons.libinfo import get_config_from_section, get_config
from numpy.distutils.scons.testcode_snippets import cblas_sgemm as cblas_src, \
        c_sgemm as sunperf_src, lapack_sgesv

from support import check_include_and_run, check_symbol
from support import save_and_set, restore, ConfigOpts, ConfigRes

def _check(context, name, section, defopts, headers_to_check, funcs_to_check, 
           check_version, version_checker, autoadd):
    """Generic implementation for perflib check.

    This checks for header (by compiling code including them) and symbols in
    libraries (by linking code calling for given symbols). Optionnaly, it can
    get the version using some specific function.
    
    See CheckATLAS or CheckMKL for examples."""
    context.Message("Checking %s ... " % name)

    try:
        value = os.environ[name]
        if value == 'None':
            return context.Result('Disabled from env through var %s !' % name), {}
    except KeyError:
        pass

    # Get site.cfg customization if any
    siteconfig, cfgfiles = get_config()
    (cpppath, libs, libpath), found = get_config_from_section(siteconfig, section)
    if found:
        opts = ConfigOpts(cpppath = cpppath, libpath = libpath, libs = libs)
    else:
        opts = defopts

    opts['rpath'] = opts['libpath']

    env = context.env

    # Check whether the header is available (CheckHeader-like checker)
    saved = save_and_set(env, opts)
    try:
        # XXX: add dep vars in code
        src = '\n'.join([r'#include <%s>' % h for h in headers_to_check])
        st = context.TryCompile(src, '.c')
    finally:
        restore(env, saved)

    if not st:
        context.Result('Failed (could not check header(s) : check config.log '\
                       'in %s for more details)' % env['build_dir'])
        return st, ConfigRes(name, opts, found)

    # Check whether the library is available (CheckLib-like checker)
    saved = save_and_set(env, opts)
    try:
        for sym in funcs_to_check:
            # XXX: add dep vars in code
            st = check_symbol(context, headers_to_check, sym)
            if not st:
                break
    finally:
        if st == 0 or autoadd == 0:
            restore(env, saved)
        
    if not st:
        context.Result('Failed (could not check symbol %s : check config.log '\
                       'in %s for more details))' % (sym, env['build_dir']))
        return st, ConfigRes(name, opts, found)
        
    context.Result(st)

    # Check version if requested
    if check_version:
        if version_checker:
            vst, v = version_checker(env, opts)
            if vst:
                version = v
            else:
                version = 'Unknown (checking version failed)'
        else:
            version = 'Unkown (not implemented)'
        cfgres = ConfigRes(name, opts, found, version)
    else:
        cfgres = ConfigRes(name, opts, found, version = 'Not checked')

    return st, cfgres

def _mkl_version_checker(env, opts):
    version_code = r"""
#include <stdio.h>
#include <mkl.h>

int main(void)
{
MKLVersion ver;
MKLGetVersion(&ver);

printf("Full version: %d.%d.%d\n", ver.MajorVersion,
       ver.MinorVersion,
       ver.BuildNumber);

return 0;
}
"""

    opts['rpath'] = opts['libpath']
    saved = save_and_set(env, opts)
    try:
        vst, out = context.TryRun(version_code, '.c')
    finally:
        restore(env, saved)

    if vst and re.search(r'Full version: (\d+[.]\d+[.]\d+)', out):
        version = m.group(1)
    else:
        version = ''

    return vst, version

def _atlas_version_checker(env, opts):
    version_code = """
void ATL_buildinfo(void);
int main(void) {
ATL_buildinfo();
return 0;
}
"""
    opts['rpath'] = opts['libpath']
    saved = save_and_set(env, opts)
    try:
        vst, out = context.TryRun(version_code, '.c')
    finally:
        restore(env, saved)

    if vst:
        m = re.search('ATLAS version (?P<version>\d+[.]\d+[.]\d+)', out)
        if m:
            version = m.group(1)
        else:
            version = ''
    else:
        version = ''

    return vst, version

def CheckMKL(context, autoadd = 1, check_version = 0):
    name    = 'MKL'
    section = 'mkl'
    defopts = ConfigOpts(libs = ['mkl', 'guide', 'm'])
    headers = ['mkl.h']
    funcs   = ['MKLGetVersion']

    return _check(context, name, section, defopts, headers, funcs,
                  check_version, _mkl_version_checker, autoadd)

def CheckATLAS(context, autoadd = 1, check_version = 0):
    """Check whether ATLAS is usable in C."""
    name    = 'ATLAS'
    section = 'atlas'
    defopts = ConfigOpts(libs = ['atlas'], 
                         libpath = [pjoin(i, 'atlas') for i in 
                                    default_lib_dirs])
    headers = ['atlas_enum.h']
    funcs   = ['ATL_sgemm']
    
    return _check(context, name, section, defopts, headers, funcs,
                  check_version, _atlas_version_checker, autoadd)

def CheckAccelerate(context, autoadd = 1, check_version = 0):
    """Checker for Accelerate framework (on Mac OS X >= 10.3). """
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

    name    = 'Framework: Accelerate'
    # XXX: does it make sense to customize mac os X frameworks ?
    section = 'accelerate'
    defopts = ConfigOpts(frameworks = ['Accelerate'])
    headers = ['Accelerate/Accelerate.h']
    funcs   = ['cblas_sgemm']
    
    return _check(context, name, section, defopts, headers, funcs,
                  check_version, None, autoadd)

def CheckVeclib(context, autoadd = 1, check_version = 0):
    """Checker for Veclib framework (on Mac OS X < 10.3)."""
    name    = 'Framework: Accelerate'
    # XXX: does it make sense to customize mac os X frameworks ?
    section = 'accelerate'
    defopts = ConfigOpts(frameworks = ['Accelerate'])
    headers = ['vecLib/vecLib.h']
    funcs   = ['cblas_sgemm']
    
    return _check(context, name, section, defopts, headers, funcs,
                  check_version, None, autoadd)


def CheckSunperf(context, autoadd = 1, check_version = 0):
    """Checker for sunperf."""
    name    = 'Sunperf'
    section = 'sunperf'
    # XXX: Other options needed ?
    defopts = ConfigOpts(cflags = ['-dalign'], linkflags = ['-xlic_lib=sunperf'])
    headers = ['sunperf.h']
    funcs   = ['cblas_sgemm']
    
    return _check(context, name, section, defopts, headers, funcs,
                  check_version, None, autoadd)
