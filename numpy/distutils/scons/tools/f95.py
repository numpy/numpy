"""engine.SCons.Tool.f95

Tool-specific initialization for the generic Posix f95 Fortran compiler.

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

import SCons.Defaults
import SCons.Tool
import SCons.Util
import fortran

compilers = ['f95']

#
F95Suffixes = ['.f95']
F95PPSuffixes = []
if SCons.Util.case_sensitive_suffixes('.f95', '.F95'):
    F95PPSuffixes.append('.F95')
else:
    F95Suffixes.append('.F95')

#
F95Scan = SCons.Scanner.Fortran.FortranScan("F95PATH")

for suffix in F95Suffixes + F95PPSuffixes:
    SCons.Tool.SourceFileScanner.add_scanner(suffix, F95Scan)
del suffix

#
fVLG = fortran.VariableListGenerator

F95Generator = fVLG('F95', 'FORTRAN', '_FORTRAND')
F95FlagsGenerator = fVLG('F95FLAGS', 'FORTRANFLAGS')
F95CommandGenerator = fVLG('F95COM', 'FORTRANCOM', '_F95COMD')
F95CommandStrGenerator = fVLG('F95COMSTR', 'FORTRANCOMSTR', '_F95COMSTRD')
F95PPCommandGenerator = fVLG('F95PPCOM', 'FORTRANPPCOM', '_F95PPCOMD')
F95PPCommandStrGenerator = fVLG('F95PPCOMSTR', 'FORTRANPPCOMSTR', '_F95PPCOMSTRD')
ShF95Generator = fVLG('SHF95', 'SHFORTRAN', 'F95', 'FORTRAN', '_FORTRAND')
ShF95FlagsGenerator = fVLG('SHF95FLAGS', 'SHFORTRANFLAGS')
ShF95CommandGenerator = fVLG('SHF95COM', 'SHFORTRANCOM', '_SHF95COMD')
ShF95CommandStrGenerator = fVLG('SHF95COMSTR', 'SHFORTRANCOMSTR', '_SHF95COMSTRD')
ShF95PPCommandGenerator = fVLG('SHF95PPCOM', 'SHFORTRANPPCOM', '_SHF95PPCOMD')
ShF95PPCommandStrGenerator = fVLG('SHF95PPCOMSTR', 'SHFORTRANPPCOMSTR', '_SHF95PPCOMSTRD')

del fVLG

#
F95Action = SCons.Action.Action('$_F95COMG ', '$_F95COMSTRG')
F95PPAction = SCons.Action.Action('$_F95PPCOMG ', '$_F95PPCOMSTRG')
ShF95Action = SCons.Action.Action('$_SHF95COMG ', '$_SHF95COMSTRG')
ShF95PPAction = SCons.Action.Action('$_SHF95PPCOMG ', '$_SHF95PPCOMSTRG')

def add_to_env(env):
    """Add Builders and construction variables for f95 to an Environment."""
    env.AppendUnique(FORTRANSUFFIXES = F95Suffixes + F95PPSuffixes)

    static_obj, shared_obj = SCons.Tool.createObjBuilders(env)

    for suffix in F95Suffixes:
        static_obj.add_action(suffix, F95Action)
        shared_obj.add_action(suffix, ShF95Action)
        static_obj.add_emitter(suffix, fortran.FortranEmitter)
        shared_obj.add_emitter(suffix, fortran.ShFortranEmitter)

    for suffix in F95PPSuffixes:
        static_obj.add_action(suffix, F95PPAction)
        shared_obj.add_action(suffix, ShF95PPAction)
        static_obj.add_emitter(suffix, fortran.FortranEmitter)
        shared_obj.add_emitter(suffix, fortran.ShFortranEmitter)

    env['_F95G']           = F95Generator
    env['_F95FLAGSG']      = F95FlagsGenerator
    env['_F95COMG']        = F95CommandGenerator
    env['_F95COMSTRG']     = F95CommandStrGenerator
    env['_F95PPCOMG']      = F95PPCommandGenerator
    env['_F95PPCOMSTRG']   = F95PPCommandStrGenerator

    env['_SHF95G']         = ShF95Generator
    env['_SHF95FLAGSG']    = ShF95FlagsGenerator
    env['_SHF95COMG']      = ShF95CommandGenerator
    env['_SHF95COMSTRG']   = ShF95CommandStrGenerator
    env['_SHF95PPCOMG']    = ShF95PPCommandGenerator
    env['_SHF95PPCOMSTRG'] = ShF95PPCommandStrGenerator

    env['_F95INCFLAGS'] = '$( ${_concat(INCPREFIX, F95PATH, INCSUFFIX, __env__, RDirs, TARGET, SOURCE)} $)'

    env['_F95COMD']     = '$_F95G -o $TARGET -c $_F95FLAGSG $_F95INCFLAGS $_FORTRANMODFLAG $SOURCES'
    env['_F95PPCOMD']   = '$_F95G -o $TARGET -c $_F95FLAGSG $CPPFLAGS $_CPPDEFFLAGS $_F95INCFLAGS $_FORTRANMODFLAG $SOURCES'
    env['_SHF95COMD']   = '$_SHF95G -o $TARGET -c $_SHF95FLAGSG $_F95INCFLAGS $_FORTRANMODFLAG $SOURCES'
    env['_SHF95PPCOMD'] = '$_SHF95G -o $TARGET -c $_SHF95FLAGSG $CPPFLAGS $_CPPDEFFLAGS $_F95INCFLAGS $_FORTRANMODFLAG $SOURCES'

def generate(env):
    fortran.add_to_env(env)

    import f77
    f77.add_to_env(env)

    import f90
    f90.add_to_env(env)

    add_to_env(env)

    env['_FORTRAND']        = env.Detect(compilers) or 'f95'

def exists(env):
    return env.Detect(compilers)
