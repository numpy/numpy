from os.path import join as pjoin
import os.path

from SCons.Options import Options
from SCons.Environment import Environment
from SCons.Script import BuildDir


def NumpySharedLibrary(env, target, source, *args, **kw):
    source = [pjoin(env['build_dir'], i) for i in source]
    # XXX: why target is a list ? It is always true ?
    lib = env.SharedLibrary("$build_dir/%s" % target[0], source, *args, **kw)

    inst_lib = env.Install("$distutils_installdir", lib)
    return lib, inst_lib
	
	
def NumpyCTypes(env, target, source, *args, **kw):
    source = [pjoin(env['build_dir'], i) for i in source]
    # XXX: why target is a list ? It is always true ?
    # XXX: handle cases where SHLIBPREFIX is in args
    lib = env.SharedLibrary("$build_dir/%s" % target[0], source, SHLIBPREFIX = '', *args, **kw)
    lib = [i for i in lib if not (str(i).endswith('.exp') or str(i).endswith('.lib')) ]
    inst_lib = env.Install("$distutils_installdir", lib)
    return lib, inst_lib

def GetNumpyOptions(args):
    """Call this with args=ARGUMENTS to take into account command line args."""
    opts = Options(None, args)
    opts.Add('pkg_name', 'name of the package (including parent package if any)', '')
    opts.Add('src_dir', 'src dir relative to top called', '.')
    opts.Add('build_prefix', 'build prefix (NOT including the package name)', 
             pjoin('build', 'scons'))
    opts.Add('distutils_libdir', 
             'build dir for libraries of distutils (NOT including the package name)', 
             pjoin('build', 'lib'))

    return opts

def GetNumpyEnvironment(args):
    """Call this with args = ARGUMENTS."""
    opts = GetNumpyOptions(args)
    env = Environment(options = opts)
    env.AppendUnique(build_dir = pjoin(env['build_prefix']))
    env.AppendUnique(distutils_installdir = pjoin(env['distutils_libdir'], 
                                                  env['pkg_name']))

    env['BUILDERS']['NumpySharedLibrary'] = NumpySharedLibrary
    env['BUILDERS']['NumpyCTypes'] = NumpyCTypes
    print env['src_dir']
    if len(env['src_dir']) > 0:
        BuildDir(env['build_dir'], env['src_dir'])
    else:
        BuildDir(env['build_dir'], '.')

    return env
