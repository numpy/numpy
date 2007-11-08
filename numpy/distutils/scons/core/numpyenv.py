# Last Changed: .
import os.path
from os.path import join as pjoin, dirname as pdirname
import sys

from distutils.sysconfig import get_config_vars

from numpy.distutils.misc_util import get_scons_build_dir, get_scons_configres_dir,\
    get_scons_configres_filename

from default import tool_list, get_cc_config
from custom_builders import NumpySharedLibrary, NumpyCtypes, NumpyPythonExtension
from libinfo import get_config
from extension_scons import PythonExtension

from numpy.distutils.scons.tools.substinfile import TOOL_SUBST

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
    import re
    suncc = re.compile('Sun C')
    # Redirect stderr to stdout
    cmd = fullpath + ' -V 2>&1'
    out = os.popen(cmd)
    cnt = out.read()
    st = out.close()

    return suncc.search(cnt)

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

def GetNumpyEnvironment(args):
    env = _GetNumpyEnvironment(args)
    env.AppendUnique(CFLAGS  = env['NUMPY_WARN_CFLAGS'] + env['NUMPY_OPTIM_CFLAGS'] +\
                               env['NUMPY_DEBUG_SYMBOL_CFLAGS'] +\
                               env['NUMPY_EXTRA_CFLAGS'] +\
                               env['NUMPY_THREAD_CFLAGS'])
    return env

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
                                                  env['pkg_name']))

    # ===============================================
    # Setting tools according to command line options

    # XXX: how to handle tools which are not in standard location ? Is adding
    # the full path of the compiler enough ? (I am sure some compilers also
    # need LD_LIBRARY_SHARED and other variables to be set, too....)
    if len(env['cc_opt']) > 0:
        try:
            if len(env['cc_opt_path']) > 0:
                if env['cc_opt'] == 'intelc':
                    # Intel Compiler SCons.Tool has a special way to set the
                    # path, o we use this one instead of changing
                    # env['ENV']['PATH'].
                    t = Tool(env['cc_opt'], 
                             topdir = os.path.split(env['cc_opt_path'])[0])
                    t(env) 
                    customize_cc(t.name, env)
                else:
                    if is_cc_suncc(pjoin(env['cc_opt_path'], env['cc_opt'])):
                        env['cc_opt'] = 'suncc'
                    # XXX: what is the right way to add one directory in the
                    # PATH ? (may not work on windows).
                    t = Tool(env['cc_opt'])
                    t(env) 
                    customize_cc(t.name, env)
                    if sys.platform == 'win32':
                        env['ENV']['PATH'] += ';%s' % env['cc_opt_path']
                    else:
                        env['ENV']['PATH'] += ':%s' % env['cc_opt_path']
        except EnvironmentError, e:
            # scons could not understand cc_opt (bad name ?)
            raise AssertionError("SCONS: Could not initialize tool ? Error is %s" % \
                                 str(e))
    else:
        t = Tool(FindTool(DEF_C_COMPILERS))
        t(env)
        customize_cc(t.name, env)

    # F77 compiler
    if len(env['f77_opt']) > 0:
        try:
            if len(env['f77_opt_path']) > 0:
                # XXX: what is the right way to add one directory in the
                # PATH ? (may not work on windows).
                t = Tool(env['f77_opt'], toolpath = ['numpy/distutils/scons/tools'])
                t(env) 
                if sys.platform == 'win32':
                    env['ENV']['PATH'] += ';%s' % env['f77_opt_path']
                else:
                    env['ENV']['PATH'] += ':%s' % env['f77_opt_path']
        except EnvironmentError, e:
            # scons could not understand cc_opt (bad name ?)
            raise AssertionError("SCONS: Could not initialize tool ? Error is %s" % \
                                 str(e))
        # XXX: really have to understand how fortran compilers work in scons...
        env['F77'] = env['_FORTRAND']
    else:
	raise NotImplementedError('FIXME: Support for env wo fcompiler not tested yet !')
        #t = Tool(FindTool(DEF_FORTRAN_COMPILERS))
        #t(env)

    # XXX: Really, we should use our own subclass of Environment, instead of
    # adding Numpy* functions !

    # Put config code and log in separate dir for each subpackage
    from utils import curry
    NumpyConfigure = curry(env.Configure, 
                           conf_dir = pjoin(env['build_dir'], '.sconf'), 
                           log_file = pjoin(env['build_dir'], 'config.log'))
    env.NumpyConfigure = NumpyConfigure

    # XXX: Huge, ugly hack ! SConsign needs an absolute path or a path
    # relative to where the SConstruct file is. We have to find the path of
    # the build dir relative to the src_dir: we add n .., where n is the number
    # of occureant of the path separator in the src dir.
    def get_build_relative_src(srcdir, builddir):
        n = srcdir.count(os.sep) + 1
        return pjoin(os.sep.join([os.pardir for i in range(n)]), builddir)
    sconsign = pjoin(get_build_relative_src(env['src_dir'], 
                                            env['build_dir']),
                     '.sconsign.dblite')
    env.SConsignFile(sconsign)

    # ========================================================================
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
