from os.path import join as pjoin
import os.path
import ConfigParser

from SCons.Options import Options
from SCons.Tool import Tool
from SCons.Environment import Environment
from SCons.Script import BuildDir, Help
from SCons.Util import is_List

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
    # XXX: I would prefer subclassing Environment, because we really expect
    # some different behaviour than just Environment instances...
    opts = GetNumpyOptions(args)
    env = Environment(options = opts)

    # Setting dirs according to command line options
    env.AppendUnique(build_dir = pjoin(env['build_prefix']))
    env.AppendUnique(distutils_installdir = pjoin(env['distutils_libdir'], 
                                                  env['pkg_name']))

    # Setting tools according to command line options
    # XXX: how to handle tools which are not in standard location ? Is adding
    # the full path of the compiler enough ? (I am sure some compilers also
    # need LD_LIBRARY_SHARED and other variables to be set, too....)
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
                else:
                    # XXX: what is the right way to add one directory in the
                    # PATH ? (may not work on windows).
                    env['ENV']['PATH'] += ':%s' % env['cc_opt_path']
            else:
                t = Tool(env['cc_opt'])
                t(env) 
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

    # Getting the config options from *.cfg files
    config = get_config()
    env['NUMPYCONFIG'] = config

    return env

def _get_empty(dict, key):
    print "++++++ Deprecated, do not use _get_empty +++++++++"
    try:
        return dict[key]
    except KeyError, e:
        return []

def cfgentry2list(entry):
    """This convert one entry in a section of .cfg file to something usable in
    scons."""
    pass

def NumpyCheckLib(context, section, libs, symbol = None):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # This is really preliminary, and needs a lot of love before being in good
    # shape !!!!!
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # XXX: handle extension in arg list
    extension = '.c'
    # XXX: handle symbol
    src = """
int main(int argc, char** argv)
{
    return 0;
}"""

    # Make sure libs is a list of libs:
    if not is_List(libs):
        libs = [libs]

    config = context.env['NUMPYCONFIG']
    context.Message('Checking for library %s...' % libs)
    if config.has_section(section):
        #print "Checking %s from section %s" % (library, section)
        try:
            # XXX: handle list of directories here
            # XXX: factorize this away
            newLIBPATH = config.get(section, 'library_dirs') 
            newCPPPATH = config.get(section, 'include_dirs') 
            newLIBS = config.get(section, 'libraries') 
            lastLIBPATH = _get_empty(context.env,'LIBPATH')
            lastLIBS = _get_empty(context.env,'LIBS')
            lastCPPPATH = _get_empty(context.env,'CPPPATH')
            res = context.TryLink(src, extension)
            if not res:
                context.env.Replace(LIBS = lastLIBS, 
                                    LIBPATH = lastLIBPATH, 
                                    CPPPATH = lastCPPPATH)
        except ConfigParser.NoOptionError, e:
            print "+++++++++++++++++++++++++++++++"
            print e
            print "+++++++++++++++++++++++++++++++"
            res = 0
    else:
        lastLIBS = context.AppendLIBS(libs)
        res = context.TryLink(src, extension)
        if not res:
            context.env.Replace(LIBS = lastLIBS) 
    return context.Result(res)

def get_config():
    """ This tries to read .cfg files in several locations, and merge its
    information into a ConfigParser object for the first found file.
    
    Returns the ConfigParser instance. This copies the logic in system_info
    from numpy.distutils."""
    # Below is the feature we are copying from numpy.distutils:
    # 
    # The file 'site.cfg' is looked for in

    # 1) Directory of main setup.py file being run.
    # 2) Home directory of user running the setup.py file as ~/.numpy-site.cfg
    # 3) System wide directory (location of this file...)

    # The first one found is used to get system configuration options The
    # format is that used by ConfigParser (i.e., Windows .INI style). The
    # section DEFAULT has options that are the default for each section. The
    # available sections are fftw, atlas, and x11. Appropiate defaults are
    # used if nothing is specified.

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
    return cp
