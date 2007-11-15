"""f2py Tool

Tool-specific initialization for f2py.

There normally shouldn't be any need to import this module directly.
It will usually be imported through the generic SCons.Tool.Tool()
selection method.

"""

import os.path
import re

import SCons.Action
import SCons.Defaults
import SCons.Scanner
import SCons.Tool
import SCons.Util
import SCons.Node

def _f2pySuffixEmitter(env, source):
    return '$F2PYCFILESUFFIX'

#_reModule = re.compile(r'%module\s+(.+)')

def _f2pyEmitter(target, source, env):
    build_dir = os.path.dirname(str(target[0]))
    target.append(SCons.Node.FS.default_fs.Entry(os.path.join(build_dir, 'fortranobject.c')))
    target.append(SCons.Node.FS.default_fs.Entry(os.path.join(build_dir, 'fortranobject.h')))
    return (target, source)

def _pyf2c(target, source, env):
    import numpy.f2py
    import shutil

    # We need filenames from source/target for path handling
    target_file_names = [str(i) for i in target]
    source_file_names = [str(i) for i in source]

    # Get source files necessary for f2py generated modules
    d = os.path.dirname(numpy.f2py.__file__)
    source_c = os.path.join(d,'src','fortranobject.c')
    source_h = os.path.join(d,'src','fortranobject.h')

    # XXX: scons has a way to force buidler to only use one source file
    if len(source_file_names) > 1:
        raise "YATA"

    # Copy source files for f2py generated modules in the build dir
    build_dir = os.path.dirname(target_file_names[0])
    shutil.copy(source_c, build_dir)
    shutil.copy(source_h, build_dir)

    # Generate the source file from pyf description
    haha = numpy.f2py.run_main(['--build-dir', build_dir,
                                source_file_names[0]])
    return 0

def generate(env):
    """Add Builders and construction variables for swig to an Environment."""
    c_file, cxx_file = SCons.Tool.createCFileBuilders(env)

    c_file.suffix['.pyf'] = _f2pySuffixEmitter

    c_file.add_action('.pyf', SCons.Action.Action(_pyf2c))
    c_file.add_emitter('.pyf', _f2pyEmitter)

    env['F2PYOPTIONS']        = SCons.Util.CLVar('')
    env['F2PYBUILDDIR']      = ''
    env['F2PYCFILESUFFIX']   = 'module$CFILESUFFIX'

    # XXX: scanner ?

    #expr = '^[ \t]*%[ \t]*(?:include|import|extern)[ \t]*(<|"?)([^>\s"]+)(?:>|"?)'
    #scanner = SCons.Scanner.ClassicCPP("SWIGScan", ".i", "SWIGPATH", expr)
    #env.Append(SCANNERS = scanner)

def exists(env):
    try:
        import numpy.f2py
        st = 1
    except ImportError, e:
        print "Warning : f2py tool not found, error was %s" % e
        st = 0

    return st
