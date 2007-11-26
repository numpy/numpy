#! /usr/bin/env python
# Last Change: Sat Nov 24 12:00 PM 2007 J

# This module defines checkers for performances libs providing standard API,
# such as MKL (Intel), ATLAS, Sunperf (solaris and linux), Accelerate (Mac OS
# X), etc... Those checkers merely only check whether the library is found
# using a library specific check if possible, or other heuristics.
# Generally, you don't use those directly: they are used in 'meta' checkers,
# such as BLAS, CBLAS, LAPACK checkers.
import re
import os
from os.path import join as pjoin
from copy import deepcopy

from numpy.distutils.system_info import default_lib_dirs
from numpy.distutils.scons.core.libinfo import get_config_from_section, get_config
from numpy.distutils.scons.testcode_snippets import cblas_sgemm as cblas_src, \
        c_sgemm as sunperf_src, lapack_sgesv

from support import check_include_and_run, check_symbol
from support import save_and_set, restore, ConfigOpts, ConfigRes

#-------------------------------------------
# Perflib specific configuration and helpers
#-------------------------------------------
class PerflibConfig:
    def __init__(self, name, section, defopts, headers, funcs, version_checker = None):
        """Initialize the configuration.

        Args:
            - name : str
                the name of the perflib
            - section : str
                the name of the section used in site.cfg for customization
            - defopts : ConfigOpts
                the compilation configuration for the checker
            - headers : list
                the list of headers to test in the checker
            - funcs : list
                the list of functions to test in the checker.
            - version_checker : callable
                optional function to check version of the perflib. Its
                arguments should be env and opts, where env is a scons
                environment and opts a ConfigOpts instance. It should return an
                integer (1 if successfull) and a version string."""
                
        self.name = name
        self.section = section
        self.defopts = defopts
        self.headers = headers
        self.funcs = funcs
        self.version_checker = version_checker

_CONFIG = {
        'MKL': PerflibConfig('MKL', 'mkl', ConfigOpts(libs = ['mkl', 'guide', 'm']),
                             ['mkl.h'], ['MKLGetVersion']),
        'ATLAS': PerflibConfig('ATLAS', 'atlas', 
                               ConfigOpts(libs = ['atlas'], 
                               libpath = [pjoin(i, 'atlas') for i in 
                                          default_lib_dirs]),
                               ['atlas_enum.h'],
                               ['ATL_sgemm']),
        'Accelerate' : PerflibConfig('Framework: Accelerate', 'accelerate', 
                                      ConfigOpts(frameworks = ['Accelerate']),
                                      ['Accelerate/Accelerate.h'],
                                      ['cblas_sgemm']),
        'vecLib' : PerflibConfig('Framework: vecLib', 'vecLib', 
                                 ConfigOpts(frameworks = ['vecLib']),
                                 ['vecLib/vecLib.h'],
                                 ['cblas_sgemm']),
        'Sunperf' : PerflibConfig('Sunperf', 'sunperf', 
                                  ConfigOpts(cflags = ['-dalign'], 
                                             linkflags = ['-xlic_lib=sunperf']),
                                  ['sunperf.h'],
                                  ['cblas_sgemm'])}

class IsFactory:
    def __init__(self, name):
        """Name should be one key of _CONFIG."""
        try:
            _CONFIG[name]
        except KeyError, e:
            raise RuntimeError("name %s is unknown")

        def f(env, libname):
            if env['NUMPY_PKG_CONFIG'][libname] is None:
                return 0 == 1
            else:
                return env['NUMPY_PKG_CONFIG'][libname].name == \
                       _CONFIG[name].name
        self.func = f

    def get_func(self):
        return self.func

class GetVersionFactory:
    def __init__(self, name):
        """Name should be one key of _CONFIG."""
        try:
            _CONFIG[name]
        except KeyError, e:
            raise RuntimeError("name %s is unknown")

        def f(env, libname):
            if env['NUMPY_PKG_CONFIG'][libname] is None or \
               not env['NUMPY_PKG_CONFIG'][libname].name == _CONFIG[name].name:
                return 'No version info'
            else:
                return env['NUMPY_PKG_CONFIG'][libname].version
        self.func = f

    def get_func(self):
        return self.func

#------------------------
# Generic functionalities
#------------------------
def _check(context, name, section, defopts, headers_to_check, funcs_to_check, 
           check_version, version_checker, autoadd, rpath_is_libpath = True):
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
        if len(libs) == 1 and len(libs[0]) == 0:
            opts['libs'] = defopts['libs']
    else:
        opts = defopts

    if rpath_is_libpath:
	opts['rpath'] = deepcopy(opts['libpath'])

    env = context.env

    # Check whether the header is available (CheckHeader-like checker)
    saved = save_and_set(env, opts)
    try:
        src_code = [r'#include <%s>' % h for h in headers_to_check]
        src_code.extend([r'#if 0', str(opts), r'#endif', '\n'])
        src = '\n'.join(src_code)
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
            extra = [r'#if 0', str(opts), r'#endif', '\n']
            st = check_symbol(context, headers_to_check, sym, '\n'.join(extra))
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
            vst, v = version_checker(context, opts)
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
    cfg = _CONFIG['MKL']

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
    cfg = _CONFIG['ATLAS']

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

    cfg = _CONFIG['Accelerate']

    return _check(context, cfg.name, cfg.section, cfg.defopts, cfg.headers,
                  cfg.funcs, check_version, None, autoadd)

IsAccelerate = IsFactory('Accelerate').get_func()

def CheckVeclib(context, autoadd = 1, check_version = 0):
    """Checker for Veclib framework (on Mac OS X < 10.3)."""
    cfg = _CONFIG['vecLib']

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
    cfg = _CONFIG['Sunperf']
    
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
