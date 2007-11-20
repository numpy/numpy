"""npytpl Tool

Tool-specific initialization for npyctpl, a tool to generate C source file from
.c.src files.

There normally shouldn't be any need to import this module directly.
It will usually be imported through the generic SCons.Tool.Tool()
selection method.

"""

from os.path import basename as pbasename, splitext, join as pjoin, dirname as pdirname
#import re

import SCons.Action
#import SCons.Defaults
import SCons.Scanner
import SCons.Tool

from numpy.distutils.conv_template import process_str

# XXX: this is general and can be used outside numpy.core.
def _do_generate_from_template(targetfile, sourcefile, env):
    t = open(targetfile, 'w')
    s = open(sourcefile, 'r')
    allstr = s.read()
    s.close()
    writestr = process_str(allstr)
    t.write(writestr)
    t.close()
    return 0

def _generate_from_template(target, source, env):
    for t, s in zip(target, source):
        _do_generate_from_template(str(t), str(s), env)
    return 0

def _generate_from_template_emitter(target, source, env):
    base, ext = splitext(pbasename(str(source[0])))
    t = pjoin(pdirname(str(target[0])), base)
    return ([t], source)
    
def generate(env):
    """Add Builders and construction variables for npytpl to an Environment."""
    c_file, cxx_file = SCons.Tool.createCFileBuilders(env)

    #c_file.suffix['.src'] = _generate_from_template_emitter

    c_file.add_action('.c.src', SCons.Action.Action(_generate_from_template))
    c_file.add_emitter('.c.src', _generate_from_template_emitter)

    env['NPYTPLOPTIONS']      = SCons.Util.CLVar('')
    #env['NPYTPLBUILDDIR']     = ''
    #env['NPYTPLCFILESUFFIX']  = 'module$CFILESUFFIX'
    #env['NPYTPLINCLUDEDIR']   = os.path.join(d, 'src')

    # # XXX: adding a scanner using c_file.add_scanner does not work...
    # expr = '(<)include_file=(\S+)>'
    # scanner = SCons.Scanner.ClassicCPP("F2PYScan", ".pyf", "F2PYPATH", expr)
    # env.Append(SCANNERS = scanner)

def exists(env):
    try:
        import numpy.distutils.conv_template
        st = 1
    except ImportError, e:
        print "Warning : npytpl tool not found, error was %s" % e
        st = 0

    return st
