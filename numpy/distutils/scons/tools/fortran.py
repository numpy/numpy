"""SCons.Tool.fortran

Tool-specific initialization for a generic Posix f77/f90 Fortran compiler.

There normally shouldn't be any need to import this module directly.
It will usually be imported through the generic SCons.Tool.Tool()
selection method.

"""

#
# __COPYRIGHT__
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY
# KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

__revision__ = "__FILE__ __REVISION__ __DATE__ __DEVELOPER__"

import re
import string

import SCons.Action
import SCons.Defaults
import SCons.Scanner.Fortran
import SCons.Tool
import SCons.Util

compilers = ['f95', 'f90', 'f77']

#
#  Not yet sure how to deal with fortran pre-processor functions.
#  Different compilers do this differently in modern fortran.  Some still
#  rely on the c pre-processor, some (like cvf, ivf) have their own
#  pre-processor technology and use intermediary suffixes (.i90)
#
FortranSuffixes = [".for", ".ftn", ]
FortranPPSuffixes = ['.fpp', '.FPP']
upper_case = [".FOR", ".FTN"]
if SCons.Util.case_sensitive_suffixes('.f', '.F'):
    FortranPPSuffixes.extend(upper_case)
else:
    FortranSuffixes.extend(upper_case)

#
FortranScan = SCons.Scanner.Fortran.FortranScan("FORTRANPATH")

for suffix in FortranSuffixes + FortranPPSuffixes:
    SCons.Tool.SourceFileScanner.add_scanner(suffix, FortranScan)
del suffix

#
def _fortranEmitter(target, source, env):
    node = source[0].rfile()
    if not node.exists() and not node.is_derived():
       print "Could not locate " + str(node.name)
       return ([], [])
    mod_regex = """(?i)^\s*MODULE\s+(?!PROCEDURE)(\w+)"""
    cre = re.compile(mod_regex,re.M)
    # Retrieve all USE'd module names
    modules = cre.findall(node.get_contents())
    # Remove unique items from the list
    modules = SCons.Util.unique(modules)
    # Convert module name to a .mod filename
    suffix = env.subst('$FORTRANMODSUFFIX', target=target, source=source)
    moddir = env.subst('$FORTRANMODDIR', target=target, source=source)
    modules = map(lambda x, s=suffix: string.lower(x) + s, modules)
    for m in modules:
       target.append(env.fs.File(m, moddir))
    return (target, source)

def FortranEmitter(target, source, env):
    target, source = _fortranEmitter(target, source, env)
    return SCons.Defaults.StaticObjectEmitter(target, source, env)

def ShFortranEmitter(target, source, env):
    target, source = _fortranEmitter(target, source, env)
    return SCons.Defaults.SharedObjectEmitter(target, source, env)

class VariableListGenerator:
    def __init__(self, *variablelist):
        self.variablelist = variablelist
    def __call__(self, env, target, source, for_signature=0):
        for v in self.variablelist:
            try: return env[v]
            except KeyError: pass
        return ''

#
FortranGenerator = VariableListGenerator('FORTRAN', 'F77', '_FORTRAND')
FortranFlagsGenerator = VariableListGenerator('FORTRANFLAGS', 'F77FLAGS')
FortranCommandGenerator = VariableListGenerator('FORTRANCOM', 'F77COM', '_FORTRANCOMD')
FortranCommandStrGenerator = VariableListGenerator('FORTRANCOMSTR', 'F77COMSTR', '_FORTRANCOMSTRD')
FortranPPCommandGenerator = VariableListGenerator('FORTRANPPCOM', 'F77PPCOM', '_FORTRANPPCOMD')
FortranPPCommandStrGenerator = VariableListGenerator('FORTRANPPCOMSTR', 'F77PPCOMSTR', '_FORTRANPPCOMSTRD')
ShFortranGenerator = VariableListGenerator('SHFORTRAN', 'SHF77', 'FORTRAN', 'F77', '_FORTRAND')
ShFortranFlagsGenerator = VariableListGenerator('SHFORTRANFLAGS', 'SHF77FLAGS')
ShFortranCommandGenerator = VariableListGenerator('SHFORTRANCOM', 'SHF77COM', '_SHFORTRANCOMD')
ShFortranCommandStrGenerator = VariableListGenerator('SHFORTRANCOMSTR', 'SHF77COMSTR', '_SHFORTRANCOMSTRD')
ShFortranPPCommandGenerator = VariableListGenerator('SHFORTRANPPCOM', 'SHF77PPCOM', '_SHFORTRANPPCOMD')
ShFortranPPCommandStrGenerator = VariableListGenerator('SHFORTRANPPCOMSTR', 'SHF77PPCOMSTR', '_SHFORTRANPPCOMSTRD')

#
FortranAction = SCons.Action.Action('$_FORTRANCOMG ', '$_FORTRANCOMSTRG')
FortranPPAction = SCons.Action.Action('$_FORTRANPPCOMG ', '$_FORTRANPPCOMSTRG')
ShFortranAction = SCons.Action.Action('$_SHFORTRANCOMG ', '$_SHFORTRANCOMSTRG')
ShFortranPPAction = SCons.Action.Action('$_SHFORTRANPPCOMG ', '$_SHFORTRANPPCOMSTRG')

def add_to_env(env):
    """Add Builders and construction variables for Fortran to an Environment."""

    env['_FORTRANG']            = FortranGenerator
    env['_FORTRANFLAGSG']       = FortranFlagsGenerator
    env['_FORTRANCOMG']         = FortranCommandGenerator
    env['_FORTRANCOMSTRG']      = FortranCommandStrGenerator
    env['_FORTRANPPCOMG']       = FortranPPCommandGenerator
    env['_FORTRANPPCOMSTRG']    = FortranPPCommandStrGenerator

    env['_SHFORTRANG']          = ShFortranGenerator
    env['_SHFORTRANFLAGSG']     = ShFortranFlagsGenerator
    env['_SHFORTRANCOMG']       = ShFortranCommandGenerator
    env['_SHFORTRANCOMSTRG']    = ShFortranCommandStrGenerator
    env['_SHFORTRANPPCOMG']     = ShFortranPPCommandGenerator
    env['_SHFORTRANPPCOMSTRG']  = ShFortranPPCommandStrGenerator

    env['_FORTRANINCFLAGS'] = '$( ${_concat(INCPREFIX, FORTRANPATH, INCSUFFIX, __env__, RDirs, TARGET, SOURCE)} $)'

    env['FORTRANMODPREFIX'] = ''     # like $LIBPREFIX
    env['FORTRANMODSUFFIX'] = '.mod' # like $LIBSUFFIX

    env['FORTRANMODDIR'] = ''          # where the compiler should place .mod files
    env['FORTRANMODDIRPREFIX'] = ''    # some prefix to $FORTRANMODDIR - similar to $INCPREFIX
    env['FORTRANMODDIRSUFFIX'] = ''    # some suffix to $FORTRANMODDIR - similar to $INCSUFFIX
    env['_FORTRANMODFLAG'] = '$( ${_concat(FORTRANMODDIRPREFIX, FORTRANMODDIR, FORTRANMODDIRSUFFIX, __env__, RDirs)} $)'

    env.AppendUnique(FORTRANSUFFIXES = FortranSuffixes + FortranPPSuffixes)

    static_obj, shared_obj = SCons.Tool.createObjBuilders(env)

    for suffix in FortranSuffixes:
        static_obj.add_action(suffix, FortranAction)
        shared_obj.add_action(suffix, ShFortranAction)
        static_obj.add_emitter(suffix, FortranEmitter)
        shared_obj.add_emitter(suffix, ShFortranEmitter)

    for suffix in FortranPPSuffixes:
        static_obj.add_action(suffix, FortranPPAction)
        shared_obj.add_action(suffix, ShFortranPPAction)
        static_obj.add_emitter(suffix, FortranEmitter)
        shared_obj.add_emitter(suffix, ShFortranEmitter)

    env['_FORTRANCOMD']     = '$_FORTRANG -o $TARGET -c $_FORTRANFLAGSG $_FORTRANINCFLAGS $_FORTRANMODFLAG $SOURCES'
    env['_FORTRANPPCOMD']   = '$_FORTRANG -o $TARGET -c $_FORTRANFLAGSG $CPPFLAGS $_CPPDEFFLAGS $_FORTRANINCFLAGS $_FORTRANMODFLAG $SOURCES'
    env['_SHFORTRANCOMD']   = '$_SHFORTRANG -o $TARGET -c $_SHFORTRANFLAGSG $_FORTRANINCFLAGS $_FORTRANMODFLAG $SOURCES'
    env['_SHFORTRANPPCOMD'] = '$_SHFORTRANG -o $TARGET -c $_SHFORTRANFLAGSG $CPPFLAGS $_CPPDEFFLAGS $_FORTRANINCFLAGS $_FORTRANMODFLAG $SOURCES'

import f77
import f90
import f95

def generate(env):
    f77.add_to_env(env)
    f90.add_to_env(env)
    f95.add_to_env(env)

    add_to_env(env)

    env['_FORTRAND'] = env.Detect(compilers) or 'f77'

def exists(env):
    return env.Detect(compilers)
