from os.path import join as pjoin
import os.path

from SCons.Options import Options
from SCons.Tool import Tool
from SCons.Environment import Environment
from SCons.Script import BuildDir, Help

# XXX: all this should be put in another files eventually once it is getting in
# shape

def NumpySharedLibrary(env, target, source, *args, **kw):
    """This builder is the same than SharedLibrary, except for the fact that it
    takes into account build dir info passed by distutils, and put the target at
    the right location in distutils build directory for correct installation."""
    source = [pjoin(env['build_dir'], i) for i in source]
    # XXX: why target is a list ? It is always true ?
    lib = env.SharedLibrary("$build_dir/%s" % target[0], source, *args, **kw)

    inst_lib = env.Install("$distutils_installdir", lib)
    return lib, inst_lib
	
	
def NumpyCTypes(env, target, source, *args, **kw):
    """This builder is essentially the same than SharedLibrary, but should be
    used for libraries which will only be used through ctypes.

    In particular, it does not install .exp/.lib files on windows. """
    source = [pjoin(env['build_dir'], i) for i in source]
    # XXX: why target is a list ? It is always true ?
    # XXX: handle cases where SHLIBPREFIX is in args
    lib = env.SharedLibrary("$build_dir/%s" % target[0], 
                            source, 
                            SHLIBPREFIX = '', 
                            *args, 
                            **kw)
    lib = [i for i in lib if not (str(i).endswith('.exp') or str(i).endswith('.lib')) ]
    inst_lib = env.Install("$distutils_installdir", lib)
    return lib, inst_lib

def GetNumpyOptions(args):
    """Call this with args=ARGUMENTS to take into account command line args."""
    opts = Options(None, args)
    # Add directories related info
    opts.Add('pkg_name', 'name of the package (including parent package if any)', '')
    opts.Add('src_dir', 'src dir relative to top called', '.')
    opts.Add('build_prefix', 'build prefix (NOT including the package name)', 
             pjoin('build', 'scons'))
    opts.Add('distutils_libdir', 
             'build dir for libraries of distutils (NOT including the package name)', 
             pjoin('build', 'lib'))

    # Add compiler related info
    opts.Add('cc_opt', 'name of C compiler', '')
    opts.Add('cc_opt_path', 'path of the C compiler set in cc_opt', '')
    return opts

def GetNumpyEnvironment(args):
    """Call this with args = ARGUMENTS."""
    opts = GetNumpyOptions(args)
    env = Environment(options = opts)

    # Setting dirs according to command line options
    env.AppendUnique(build_dir = pjoin(env['build_prefix']))
    env.AppendUnique(distutils_installdir = pjoin(env['distutils_libdir'], 
                                                  env['pkg_name']))

    # Setting tools according to command line options
    if len(env['cc_opt']) > 0:
        try:
            t = Tool(env['cc_opt'])
            t(env) 
            if len(env['cc_opt_path']) > 0:
                # XXX: what is the right way to add one directory in the PATH ?
                env['ENV']['PATH'] += ':%s' % env['cc_opt_path']
        except EnvironmentError, e:
            # scons could not understand cc_opt (bad name ?)
            raise AssertionError("SCONS: Could not initialize tool ? Error is %s" % \
                                 str(e))

    # Adding custom builder
    env['BUILDERS']['NumpySharedLibrary'] = NumpySharedLibrary
    env['BUILDERS']['NumpyCTypes'] = NumpyCTypes

    # Setting build directory according to command line option
    if len(env['src_dir']) > 0:
        BuildDir(env['build_dir'], env['src_dir'])
    else:
        BuildDir(env['build_dir'], '.')

    # Generate help (if calling scons directly during debugging, this could be useful)
    Help(opts.GenerateHelpText(env))

    #print "setup.cfg abspath is %s" % pjoin(env['src_dir'], 'setup.cfg')
    find_config_file()
    return env

def find_config_file(name = 'site.cfg'):
    import ConfigParser

    from numpy.distutils.system_info import default_lib_dirs
    from numpy.distutils.system_info import default_include_dirs
    from numpy.distutils.system_info import default_src_dirs
    from numpy.distutils.system_info import get_standard_file

    section = 'DEFAULT'
    defaults = {}
    defaults['libraries'] = ''
    defaults['library_dirs'] = os.pathsep.join(default_lib_dirs)
    defaults['include_dirs'] = os.pathsep.join(default_include_dirs)
    defaults['src_dirs'] = os.pathsep.join(default_src_dirs)
    cp = ConfigParser.ConfigParser(defaults)
    files = []
    files.extend(get_standard_file('.numpy-site.cfg'))
    files.extend(get_standard_file('site.cfg'))

    def parse_config_files():
        cp.read(files)
        if not cp.has_section(section):
            cp.add_section(section)

    parse_config_files()
    #print cp.sections()
    #print dir(cp)
    #print cp.get.__doc__
    print cp.get('sndfile', 'include_dirs')
