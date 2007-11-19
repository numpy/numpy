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

# XXX: this whole thing needs cleaning !

def _f2pySuffixEmitter(env, source):
    return '$F2PYCFILESUFFIX'

#_reModule = re.compile(r'%module\s+(.+)')

def _mangle_fortranobject(targetname, filename):
    basename = os.path.splitext(os.path.basename(targetname))[0]
    return '%s_%s' % (basename, filename)
    
def _f2pyEmitter(target, source, env):
    build_dir = os.path.dirname(str(target[0]))
    target.append(SCons.Node.FS.default_fs.Entry(
        os.path.join(build_dir, _mangle_fortranobject(str(target[0]), 'fortranobject.c'))))
    basename = os.path.splitext(os.path.basename(str(target[0])))
    basename = basename[0]
    basename = basename.split('module')[0]
    target.append(SCons.Node.FS.default_fs.Entry(
        os.path.join(build_dir, '%s-f2pywrappers.f' % basename)))
    return (target, source)

def _pyf2c(target, source, env):
    from SCons.Script import Touch
    from threading import Lock
    import numpy.f2py
    import shutil

    # We need filenames from source/target for path handling
    target_file_names = [str(i) for i in target]
    source_file_names = [str(i) for i in source]

    # Get source files necessary for f2py generated modules
    d = os.path.dirname(numpy.f2py.__file__)
    source_c = os.path.join(d,'src','fortranobject.c')

    # XXX: scons has a way to force buidler to only use one source file
    if len(source_file_names) > 1:
        raise NotImplementedError("FIXME: multiple source files")

    # Copy source files for f2py generated modules in the build dir
    build_dir = os.path.dirname(target_file_names[0])

    # XXX: blah
    if build_dir == '':
        build_dir = '.'

    try:
        shutil.copy(source_c, os.path.join(build_dir, 
               _mangle_fortranobject(target_file_names[0], 'fortranobject.c')))
    except IOError, e:
        msg = "Error while copying fortran source files (error was %s)" % str(e)
        raise IOError(msg)

    basename = os.path.basename(str(target[0]).split('module')[0])
    wrapper = os.path.join(build_dir, '%s-f2pywrappers.f' % basename)

    # Generate the source file from pyf description
    # XXX: lock does not work...
    #l = Lock()
    #st = l.acquire()
    #print "ST is %s" % st
    try:
        #print " STARTING %s" % basename
        st = numpy.f2py.run_main([source_file_names[0], '--build-dir', build_dir])
        if not os.path.exists(wrapper):
            #print "++++++++++++++++++++++++++++++++"
            f = open(wrapper, 'w')
            f.close()
        else:
            pass
            #print "--------------------------------"
        #print " FINISHED %s" % basename
    finally:
        #l.release()
        pass

    return 0

def generate(env):
    """Add Builders and construction variables for swig to an Environment."""
    import numpy.f2py
    d = os.path.dirname(numpy.f2py.__file__)

    c_file, cxx_file = SCons.Tool.createCFileBuilders(env)

    c_file.suffix['.pyf'] = _f2pySuffixEmitter

    c_file.add_action('.pyf', SCons.Action.Action(_pyf2c))
    c_file.add_emitter('.pyf', _f2pyEmitter)

    env['F2PYOPTIONS']      = SCons.Util.CLVar('')
    env['F2PYBUILDDIR']     = ''
    env['F2PYCFILESUFFIX']  = 'module$CFILESUFFIX'
    env['F2PYINCLUDEDIR']   = os.path.join(d, 'src')

    # XXX: adding a scanner using c_file.add_scanner does not work...
    expr = '(<)include_file=(\S+)>'
    scanner = SCons.Scanner.ClassicCPP("F2PYScan", ".pyf", "F2PYPATH", expr)
    env.Append(SCANNERS = scanner)

_MINC = re.compile(r'<include_file=(\S+)>')                                              
def _pyf_scanner(node, env, path):
    print "================== SCANNING ===================="
    cnt = node.get_contents()
    return _parse(cnt)

def _parse(lines):                                                                        
    """Return list of included files in .pyf from include_file directive."""
    dep = []                                                                             
    for line in lines:                                                                   
        m = _MINC.search(line)                                                           
        if m:                                                                            
            dep.append(m.group(1))                                                       
    return dep  

def exists(env):
    try:
        import numpy.f2py
        st = 1
    except ImportError, e:
        print "Warning : f2py tool not found, error was %s" % e
        st = 0

    return st
