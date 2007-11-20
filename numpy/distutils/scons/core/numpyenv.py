# Last Changed: .
import os.path
from os.path import join as pjoin, dirname as pdirname
import sys

import re
from distutils.sysconfig import get_config_vars

from numpy.distutils.misc_util import get_scons_build_dir, get_scons_configres_dir,\
    get_scons_configres_filename

from default import tool_list, get_cc_config
from custom_builders import NumpySharedLibrary, NumpyCtypes, NumpyPythonExtension
from libinfo import get_config
from extension_scons import PythonExtension, built_with_mstools
from utils import pkg_to_path

import numpy.distutils.scons.tools
from numpy.distutils.scons.tools.substinfile import TOOL_SUBST

__all__ = ['GetNumpyEnvironment']

def pyplat2sconsplat():
    # XXX: should see how env['PLATFORM'] is defined, make this a dictionary 
    if sys.platform[:5] == 'linux':
        return 'posix'
    elif sys.platform[:5] == 'sunos':
        return 'sunos'
    else:
        return sys.platform

DEF_LINKERS, DEF_C_COMPILERS, DEF_CXX_COMPILERS, DEF_ASSEMBLERS, \
DEF_FORTRAN_COMPILERS, DEF_ARS, DEF_OTHER_TOOLS = tool_list(pyplat2sconsplat())

def is_cc_suncc(fullpath):
    """Return true if the compiler is suncc."""
    # I wish there was a better way: we launch suncc -V, read the output, and
    # returns true if Sun is found in the output. We cannot check the status
    # code, because the compiler does not seem to have a way to do nothing
    # while returning success (0).
    
    import os
    suncc = re.compile('Sun C')
    # Redirect stderr to stdout
    cmd = fullpath + ' -V 2>&1'
    out = os.popen(cmd)
    cnt = out.read()
    st = out.close()

    return suncc.search(cnt)

def get_vs_version(env):
    try:
         version = env['MSVS']['VERSION']
	 m = re.compile("([0-9]).([0-9])").match(version)
	 if m:
	     major = int(m.group(1))
	     minor = int(m.group(2))
	     return (major, minor)
	 else:
 	     raise RuntimeError("FIXME: failed to parse VS version")
    except KeyError:
	 raise RuntimeError("Could not get VS version !")

def GetNumpyOptions(args):
    """Call this with args=ARGUMENTS to take into account command line args."""
    from SCons.Options import Options

    opts = Options(None, args)
    # Add directories related info
    opts.Add('pkg_name', 'name of the package (including parent package if any)', '')
    opts.Add('src_dir', 'src dir relative to top called', '.')
    opts.Add('build_prefix', 'build prefix (NOT including the package name)', 
             get_scons_build_dir())
    opts.Add('distutils_libdir', 
             'build dir for libraries of distutils (NOT including the package name)', 
             pjoin('build', 'lib'))
    opts.Add('include_bootstrap', 
             "include directories for boostraping numpy (if you do not know" \
             " what that means, you don't need it)" ,
             '')

    # Add compiler related info
    opts.Add('cc_opt', 'name of C compiler', '')
    opts.Add('cc_opt_path', 'path of the C compiler set in cc_opt', '')

    opts.Add('f77_opt', 'name of F77 compiler', '')
    opts.Add('f77_opt_path', 'path of the F77 compiler set in cc_opt', '')

    return opts

def customize_cc(name, env):
    """Customize env options related to the given tool."""
    cfg = get_cc_config(name)
    env.AppendUnique(**cfg.get_flags_dict())

def finalize_env(env):
    if built_with_mstools(env):
        major, minor = get_vs_version(env)
        # For VS 8 and above (VS 2005), use manifest for DLL
        if major >= 8:
            env['LINKCOM'] = [env['LINKCOM'], 
                      'mt.exe -nologo -manifest ${TARGET}.manifest '\
                      '-outputresource:$TARGET;1']
            env['SHLINKCOM'] = [env['SHLINKCOM'], 
                        'mt.exe -nologo -manifest ${TARGET}.manifest '\
                        '-outputresource:$TARGET;2']
            env['LDMODULECOM'] = [env['LDMODULECOM'], 
                        'mt.exe -nologo -manifest ${TARGET}.manifest '\
                        '-outputresource:$TARGET;2']

def GetNumpyEnvironment(args):
    env = _GetNumpyEnvironment(args)
    env.AppendUnique(CFLAGS  = env['NUMPY_WARN_CFLAGS'] + env['NUMPY_OPTIM_CFLAGS'] +\
                               env['NUMPY_DEBUG_SYMBOL_CFLAGS'] +\
                               env['NUMPY_EXTRA_CFLAGS'] +\
                               env['NUMPY_THREAD_CFLAGS'])
    env.AppendUnique(LINKFLAGS = env['NUMPY_OPTIM_LDFLAGS'])
    return env

def initialize_cc(env, path_list):
    # XXX: how to handle tools which are not in standard location ? Is adding
    # the full path of the compiler enough ? (I am sure some compilers also
    # need LD_LIBRARY_SHARED and other variables to be set, too....)
    from SCons.Tool import Tool, FindTool

    if len(env['cc_opt']) > 0:
        try:
            if len(env['cc_opt_path']) > 0:
                if env['cc_opt'] == 'intelc':
                    # Intel Compiler SCons.Tool has a special way to set the
                    # path, so we use this one instead of changing
                    # env['ENV']['PATH'].
                    t = Tool(env['cc_opt'], 
                             topdir = os.path.split(env['cc_opt_path'])[0])
                    t(env) 
                    customize_cc(t.name, env)
                else:
                    if is_cc_suncc(pjoin(env['cc_opt_path'], env['cc_opt'])):
                        env['cc_opt'] = 'suncc'
                    t = Tool(env['cc_opt'])
                    t(env) 
                    customize_cc(t.name, env)
                    path_list.append(env['cc_opt_path'])
            else:
                # Do not care about PATH info because none given from scons
                # distutils command
                t = Tool(env['cc_opt'])
                t(env) 
                customize_cc(t.name, env)
        except EnvironmentError, e:
            # scons could not understand cc_opt (bad name ?)
            raise AssertionError("SCONS: Could not initialize tool ? Error is %s" % \
                                 str(e))
    else:
        t = Tool(FindTool(DEF_C_COMPILERS, env))
        t(env)
        customize_cc(t.name, env)

def initialize_f77(env, path_list):
    from SCons.Tool import Tool, FindTool

    if len(env['f77_opt']) > 0:
        try:
            if len(env['f77_opt_path']) > 0:
                t = Tool(env['f77_opt'], toolpath = ['numpy/distutils/scons/tools'])
                t(env) 
                path_list.append(env['f77_opt_path'])
        except EnvironmentError, e:
            # scons could not understand cc_opt (bad name ?)
            raise AssertionError("SCONS: Could not initialize tool ? Error is %s" % \
                                 str(e))
        # XXX: really have to understand how fortran compilers work in scons...
        env['F77'] = env['_FORTRAND']
    else:
        def_fcompiler =  FindTool(DEF_FORTRAN_COMPILERS, env)
        if def_fcompiler:
            t = Tool(def_fcompiler)
            t(env)
        else:
            print "========== NO FORTRAN COMPILER FOUND ==========="

def _GetNumpyEnvironment(args):
    """Call this with args = ARGUMENTS."""
    from SCons.Environment import Environment
    from SCons.Tool import Tool, FindTool, FindAllTools
    from SCons.Script import BuildDir, Help
    from SCons.Errors import EnvironmentError

    # XXX: I would prefer subclassing Environment, because we really expect
    # some different behaviour than just Environment instances...
    opts = GetNumpyOptions(args)

    # Get the python extension suffix
    # XXX this should be defined somewhere else. Is there a way to reliably get
    # all the necessary informations specific to python extensions (linkflags,
    # etc...) dynamically ?
    pyextsuffix = get_config_vars('SO')

    # We set tools to an empty list, to be sure that the custom options are
    # given first. We have to 
    env = Environment(options = opts, tools = [], PYEXTSUFFIX = pyextsuffix)

    # Add the file substitution tool
    TOOL_SUBST(env)

    # Setting dirs according to command line options
    env.AppendUnique(build_dir = pjoin(env['build_prefix'], env['src_dir']))
    env.AppendUnique(distutils_installdir = pjoin(env['distutils_libdir'], 
                                                  pkg_to_path(env['pkg_name'])))

    #------------------------------------------------
    # Setting tools according to command line options
    #------------------------------------------------

    # List of supplemental paths to take into account
    path_list = []

    # Initialize CC tool from distutils info
    initialize_cc(env, path_list)

    # Initialize F77 tool from distutils info
    initialize_f77(env, path_list)

    # Adding default tools for the one we do not customize: mingw is special
    # according to scons, don't ask me why, but this does not work as expected
    # for this tool.
    if not env['cc_opt'] == 'mingw':
        for i in [DEF_LINKERS, DEF_CXX_COMPILERS, DEF_ASSEMBLERS, DEF_ARS]:
            t = FindTool(i, env) or i[0]
            Tool(t)(env)
    else:
        try:
            t = FindTool(['g++'], env)
            env['LINK'] = t
        except EnvironmentError:
            raise RuntimeError('g++ not found: this is necessary with mingw32 '\
                               'to build numpy !') 
        # XXX: is this really the right place ?
        env.AppendUnique(CFLAGS = '-mno-cygwin')
			
    for t in FindAllTools(DEF_OTHER_TOOLS, env):
        Tool(t)(env)

    t = Tool('f2py', toolpath = [os.path.dirname(numpy.distutils.scons.tools.__file__)])
    try:
        t(env)
    except Exception, e:
        pass
        #print "===== BOOTSTRAPPING, f2py scons tool not available (%s) =====" % e
    t = Tool('npytpl', 
             toolpath = [os.path.dirname(numpy.distutils.scons.tools.__file__)])
    try:
        t(env)
    except Exception, e:
        pass
        #print "===== BOOTSTRAPPING, f2py scons tool not available (%s) =====" % e

    finalize_env(env)

    # Add the tool paths in the environment
    if not env['ENV'].has_key('PATH'):
        env['ENV']['PATH'] = os.pathsep.join(path_list)
    else:
        env['ENV']['PATH'] = os.pathsep.join(path_list + env['ENV']['PATH'].split(os.pathsep))

    # XXX: Really, we should use our own subclass of Environment, instead of
    # adding Numpy* functions !

    #---------------
    #     Misc
    #---------------

    # Put config code and log in separate dir for each subpackage
    from utils import curry
    NumpyConfigure = curry(env.Configure, 
                           conf_dir = pjoin(env['build_dir'], '.sconf'), 
                           log_file = pjoin(env['build_dir'], 'config.log'))
    env.NumpyConfigure = NumpyConfigure

    # XXX: Huge, ugly hack ! SConsign needs an absolute path or a path relative
    # to where the SConstruct file is. We have to find the path of the build
    # dir relative to the src_dir: we add n .., where n is the number of
    # occurances of the path separator in the src dir.
    def get_build_relative_src(srcdir, builddir):
        n = srcdir.count(os.sep)
        if len(srcdir) > 0 and not srcdir == '.':
            n += 1
        return pjoin(os.sep.join([os.pardir for i in range(n)]), builddir)

    sconsign = pjoin(get_build_relative_src(env['src_dir'], 
                                            env['build_dir']),
                     '.sconsign.dblite')
    env.SConsignFile(sconsign)

    # Add HOME in the environment: some tools seem to require it (Intel
    # compiler, for licenses stuff)
    try:
        env['ENV']['HOME'] = os.environ['HOME']
    except KeyError:
        pass

    # Adding custom builder
    # XXX: Put them into tools ?
    env['BUILDERS']['NumpySharedLibrary'] = NumpySharedLibrary
    env['BUILDERS']['NumpyCtypes'] = NumpyCtypes
    env['BUILDERS']['PythonExtension'] = PythonExtension
    env['BUILDERS']['NumpyPythonExtension'] = NumpyPythonExtension

    # Setting build directory according to command line option
    if len(env['src_dir']) > 0:
        BuildDir(env['build_dir'], env['src_dir'])
    else:
        BuildDir(env['build_dir'], '.')

    # Generate help (if calling scons directly during debugging, this could be useful)
    Help(opts.GenerateHelpText(env))

    # Getting the config options from *.cfg files
    config = get_config()
    env['NUMPY_SITE_CONFIG'] = config

    # This will be used to keep configuration information on a per package basis
    env['NUMPY_PKG_CONFIG'] = {}
    env['NUMPY_PKG_CONFIG_FILE'] = pjoin(get_scons_configres_dir(), env['src_dir'], 
                                         get_scons_configres_filename())

    return env
