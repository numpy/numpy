#! /usr/bin/env python
# Last Change: Tue Dec 04 03:00 PM 2007 J

# This module defines checkers for performances libs providing standard API,
# such as MKL (Intel), ATLAS, Sunperf (solaris and linux), Accelerate (Mac OS
# X), etc... Those checkers merely only check whether the library is found
# using a library specific check if possible, or other heuristics.
# Generally, you don't use those directly: they are used in 'meta' checkers,
# such as BLAS, CBLAS, LAPACK checkers.
import re
from copy import deepcopy
from os.path import join as pjoin

from numpy.distutils.scons.core.libinfo import get_config_from_section, get_config
from numpy.distutils.scons.testcode_snippets import cblas_sgemm as cblas_src, \
        c_sgemm as sunperf_src, lapack_sgesv

from support import check_include_and_run, check_symbol
from support import save_and_set, restore, ConfigOpts, ConfigRes
from support import check_code as _check
from perflib_config import PerflibConfig, IsFactory, GetVersionFactory, CONFIG

#--------------
# MKL checker
#--------------
def _mkl_version_checker(context, opts):
    env = context.env
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

    if vst:
        m = re.search(r'Full version: (\d+[.]\d+[.]\d+)', out)
        if m:
            version = m.group(1)
    else:
        version = ''

    return vst, version

def CheckMKL(context, autoadd = 1, check_version = 0):
    cfg = CONFIG['MKL']

    return _check(context, cfg.name, cfg.section, cfg.defopts, cfg.headers,
                  cfg.funcs, check_version, _mkl_version_checker, autoadd)

IsMKL = IsFactory('MKL').get_func()
GetMKLVersion = GetVersionFactory('MKL').get_func()

#---------------
# ATLAS Checker
#---------------
def _atlas_version_checker(context, opts):
    env = context.env
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

def CheckATLAS(context, autoadd = 1, check_version = 0):
    """Check whether ATLAS is usable in C."""
    cfg = CONFIG['ATLAS']

    return _check(context, cfg.name, cfg.section, cfg.defopts, cfg.headers,
                  cfg.funcs, check_version, _atlas_version_checker, autoadd)

IsATLAS = IsFactory('ATLAS').get_func()
GetATLASVersion = GetVersionFactory('ATLAS').get_func()

#------------------------------
# Mac OS X Frameworks checkers
#------------------------------
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

    cfg = CONFIG['Accelerate']

    return _check(context, cfg.name, cfg.section, cfg.defopts, cfg.headers,
                  cfg.funcs, check_version, None, autoadd)

IsAccelerate = IsFactory('Accelerate').get_func()

def CheckVeclib(context, autoadd = 1, check_version = 0):
    """Checker for Veclib framework (on Mac OS X < 10.3)."""
    cfg = CONFIG['vecLib']

    return _check(context, cfg.name, cfg.section, cfg.defopts, cfg.headers,
                  cfg.funcs, check_version, None, autoadd)

IsVeclib = IsFactory('vecLib').get_func()

#-----------------
# Sunperf checker
#-----------------
from os.path import basename, dirname
from copy import deepcopy
from numpy.distutils.scons.core.utils import popen_wrapper
from numpy.distutils.scons.testcode_snippets import cblas_sgemm

def CheckSunperf(context, autoadd = 1, check_version = 0):
    """Checker for sunperf."""
    cfg = CONFIG['Sunperf']
    
    st, res = _check(context, cfg.name, cfg.section, cfg.defopts, cfg.headers,
                     cfg.funcs, check_version, None, autoadd)
    if not st:
        return st, res

    # We are not done: the option -xlic_lib=sunperf is not used by the linked
    # for shared libraries, I have no idea why. So if the test is succesfull,
    # we need more work to get the link options necessary to make the damn
    # thing work.
    context.Message('Getting link options of sunperf ... ')

    opts = res.cfgopts
    test_code = cblas_sgemm
    env = context.env
    saved = save_and_set(env, opts)
    try:
        st = context.TryCompile(test_code, '.c')
    finally:
        restore(env, saved)

    if not res:
        return context.Result('Failed !'), res

    saved = save_and_set(env, opts)
    env.Append(LINKFLAGS = '-#')
    oldLINK = env['LINK']
    env['LINK'] = '$CC'
    try:
        # XXX: does this scheme to get the program name always work ? Can
        # we use Scons to get the target name from the object name ?
        slast = str(context.lastTarget)
        dir = dirname(slast)
        test_prog = pjoin(dir, basename(slast).split('.')[0])
    
        cmd = context.env.subst('$LINKCOM', 
            		    target = context.env.File(test_prog),
            		    source = context.lastTarget)
        st, out = popen_wrapper(cmd, merge = True)
    finally:
        restore(env, saved)
        env['LINK'] = oldLINK
    
    # If the verbose output succeeds, parse the output
    if not st:
        st = 1
        pa = floupi(out)
        for k, v in pa.items():
    	    opts[k].extend(deepcopy(v))
        res = ConfigRes(cfg.name, opts, res.is_customized())
	context.Result('Succeeded !')
    else:
        st = 0
	context.Result('Failed !')

    return st, res

haha = r"""
cc -o build/scons/numpy/scons_fake/checkers/.sconf/conftest_5 -xlic_lib=sunperf -# build/scons/numpy/scons_fake/checkers/.sconf/conftest_5.o
### Note: NLSPATH = /opt/SUNWspro/prod/bin/../lib/locale/%L/LC_MESSAGES/%N.cat:/opt/SUNWspro/prod/bin/../../lib/locale/%L/LC_MESSAGES/%N.cat
###     command line files and options (expanded):
	### -xlic_lib=sunperf build/scons/numpy/scons_fake/checkers/.sconf/conftest_5.o -o build/scons/numpy/scons_fake/checkers/.sconf/conftest_5
	### Note: LD_LIBRARY_PATH = <null>
	### Note: LD_RUN_PATH = <null>
	/usr/ccs/bin/ld /opt/SUNWspro/prod/lib/crti.o /opt/SUNWspro/prod/lib/crt1.o /opt/SUNWspro/prod/lib/values-xa.o -o build/scons/numpy/scons_fake/checkers/.sconf/conftest_5 -lsunperf -lfui -lfsu -lmtsk -lsunmath -lpicl -lm build/scons/numpy/scons_fake/checkers/.sconf/conftest_5.o -Y "P,/opt/SUNWspro/lib:/opt/SUNWspro/prod/lib:/usr/ccs/lib:/lib:/usr/lib" -Qy -R/opt/SUNWspro/lib -lc /opt/SUNWspro/prod/lib/crtn.o"""

def floupi(out):
    import shlex
    import os
    lexer = shlex.shlex(out, posix = True)
    lexer.whitespace_split = True

    accept_libs = ['sunperf', 'fui', 'fsu', 'mtsk', 'sunmath']
    keep = dict(zip(['libs', 'libpath', 'rpath'], [[], [], []]))
    t = lexer.get_token()
    while t:
        def parse(token):
            if token.startswith('-l'):
                n = token.split('-l')[1]
                if n in accept_libs:
                    keep['libs'].append(n)
                t = lexer.get_token()
            elif token.startswith('-Y'):
                n = token
                t = lexer.get_token()
		if t.startswith('P,'):
		    t = t[2:]
                nt = t.split(os.pathsep)
                keep['libpath'].extend(nt)
            elif token.startswith('-Qy'):
                n = token
                t = lexer.get_token()
                if t.startswith('-R'):
                    arg = t.split('-R', 1)[1]
                    nt = arg.split(os.pathsep)
                    keep['rpath'].extend(nt)
            else:
                t = lexer.get_token()
            return t
        t = parse(t)

    return keep

IsSunperf = IsFactory('Sunperf').get_func()

#--------------------
# FFT related perflib
#--------------------
def CheckFFTW3(context, autoadd = 1, check_version = 0):
    """This checker tries to find fftw3."""
    cfg = CONFIG['FFTW3']
    
    return _check(context, cfg.name, cfg.section, cfg.defopts, cfg.headers,
                  cfg.funcs, check_version, _mkl_version_checker, autoadd)

IsFFTW3 = IsFactory('FFTW3').get_func()

def CheckFFTW2(context, autoadd = 1, check_version = 0):
    """This checker tries to find fftw2."""
    cfg = CONFIG['FFTW2']
    
    return _check(context, cfg.name, cfg.section, cfg.defopts, cfg.headers,
                  cfg.funcs, check_version, _mkl_version_checker, autoadd)

IsFFTW2 = IsFactory('FFTW2').get_func()
