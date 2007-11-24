"""npyftpl Tool

Tool-specific initialization for npyftpl, a tool to generate fortran/f2py
source file from .xxx.src where xxx is f, f90 or pyf.

There normally shouldn't be any need to import this module directly.
It will usually be imported through the generic SCons.Tool.Tool()
selection method.

"""

from os.path import basename as pbasename, splitext, join as pjoin, \
                    dirname as pdirname

import SCons.Action
#import SCons.Defaults
import SCons.Scanner
import SCons.Tool

from numpy.distutils.from_template import process_file

# XXX: this is general and can be used outside numpy.core.
def _do_generate_from_template(targetfile, sourcefile, env):
    t = open(targetfile, 'w')
    s = open(sourcefile, 'r')
    allstr = s.read()
    s.close()
    writestr = process_file(allstr)
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
    f_file = SCons.Builder.Builder(action = {}, emitter = {}, suffix = {None: ['.f']})

    f_file.add_action('.f.src', SCons.Action.Action(_generate_from_template))
    f_file.add_emitter('.f.src', _generate_from_template_emitter)

    env['NPYFTPLOPTIONS']     = SCons.Util.CLVar('')

def exists(env):
    try:
        import numpy.distutils.from_template
        st = 1
    except ImportError, e:
        print "Warning : npyftpl tool not found, error was %s" % e
        st = 0

    return st
