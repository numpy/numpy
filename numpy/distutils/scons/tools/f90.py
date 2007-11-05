"""engine.SCons.Tool.f90

Tool-specific initialization for the generic Posix f90 Fortran compiler.

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
import SCons.Scanner.Fortran
import SCons.Tool
import SCons.Util
import fortran

compilers = ['f90']

#
F90Suffixes = ['.f90']
F90PPSuffixes = []
if SCons.Util.case_sensitive_suffixes('.f90', '.F90'):
    F90PPSuffixes.append('.F90')
else:
    F90Suffixes.append('.F90')

#
F90Scan = SCons.Scanner.Fortran.FortranScan("F90PATH")

for suffix in F90Suffixes + F90PPSuffixes:
    SCons.Tool.SourceFileScanner.add_scanner(suffix, F90Scan)
del suffix

#
fVLG = fortran.VariableListGenerator

F90Generator = fVLG('F90', 'FORTRAN', '_FORTRAND')
F90FlagsGenerator = fVLG('F90FLAGS', 'FORTRANFLAGS')
F90CommandGenerator = fVLG('F90COM', 'FORTRANCOM', '_F90COMD')
F90CommandStrGenerator = fVLG('F90COMSTR', 'FORTRANCOMSTR', '_F90COMSTRD')
F90PPCommandGenerator = fVLG('F90PPCOM', 'FORTRANPPCOM', '_F90PPCOMD')
F90PPCommandStrGenerator = fVLG('F90PPCOMSTR', 'FORTRANPPCOMSTR', '_F90PPCOMSTRD')
ShF90Generator = fVLG('SHF90', 'SHFORTRAN', 'F90', 'FORTRAN', '_FORTRAND')
ShF90FlagsGenerator = fVLG('SHF90FLAGS', 'SHFORTRANFLAGS')
ShF90CommandGenerator = fVLG('SHF90COM', 'SHFORTRANCOM', '_SHF90COMD')
ShF90CommandStrGenerator = fVLG('SHF90COMSTR', 'SHFORTRANCOMSTR', '_SHF90COMSTRD')
ShF90PPCommandGenerator = fVLG('SHF90PPCOM', 'SHFORTRANPPCOM', '_SHF90PPCOMD')
ShF90PPCommandStrGenerator = fVLG('SHF90PPCOMSTR', 'SHFORTRANPPCOMSTR', '_SHF90PPCOMSTRD')

del fVLG

#
F90Action = SCons.Action.Action('$_F90COMG ', '$_F90COMSTRG')
F90PPAction = SCons.Action.Action('$_F90PPCOMG ', '$_F90PPCOMSTRG')
ShF90Action = SCons.Action.Action('$_SHF90COMG ', '$_SHF90COMSTRG')
ShF90PPAction = SCons.Action.Action('$_SHF90PPCOMG ', '$_SHF90PPCOMSTRG')

def add_to_env(env):
    """Add Builders and construction variables for f90 to an Environment."""
    env.AppendUnique(FORTRANSUFFIXES = F90Suffixes + F90PPSuffixes)

    static_obj, shared_obj = SCons.Tool.createObjBuilders(env)

    for suffix in F90Suffixes:
        static_obj.add_action(suffix, F90Action)
        shared_obj.add_action(suffix, ShF90Action)
        static_obj.add_emitter(suffix, fortran.FortranEmitter)
        shared_obj.add_emitter(suffix, fortran.ShFortranEmitter)

    for suffix in F90PPSuffixes:
        static_obj.add_action(suffix, F90PPAction)
        shared_obj.add_action(suffix, ShF90PPAction)
        static_obj.add_emitter(suffix, fortran.FortranEmitter)
        shared_obj.add_emitter(suffix, fortran.ShFortranEmitter)
  
    env['_F90G']            = F90Generator
    env['_F90FLAGSG']       = F90FlagsGenerator
    env['_F90COMG']         = F90CommandGenerator
    env['_F90COMSTRG']      = F90CommandStrGenerator
    env['_F90PPCOMG']       = F90PPCommandGenerator
    env['_F90PPCOMSTRG']    = F90PPCommandStrGenerator

    env['_SHF90G']          = ShF90Generator
    env['_SHF90FLAGSG']     = ShF90FlagsGenerator
    env['_SHF90COMG']       = ShF90CommandGenerator
    env['_SHF90COMSTRG']    = ShF90CommandStrGenerator
    env['_SHF90PPCOMG']     = ShF90PPCommandGenerator
    env['_SHF90PPCOMSTRG']  = ShF90PPCommandStrGenerator

    env['_F90INCFLAGS'] = '$( ${_concat(INCPREFIX, F90PATH, INCSUFFIX, __env__, RDirs, TARGET, SOURCE)} $)'
    env['_F90COMD']     = '$_F90G -o $TARGET -c $_F90FLAGSG $_F90INCFLAGS $_FORTRANMODFLAG $SOURCES'
    env['_F90PPCOMD']   = '$_F90G -o $TARGET -c $_F90FLAGSG $CPPFLAGS $_CPPDEFFLAGS $_F90INCFLAGS $_FORTRANMODFLAG $SOURCES'
    env['_SHF90COMD']   = '$_SHF90G -o $TARGET -c $_SHF90FLAGSG $_F90INCFLAGS $_FORTRANMODFLAG $SOURCES'
    env['_SHF90PPCOMD'] = '$_SHF90G -o $TARGET -c $_SHF90FLAGSG $CPPFLAGS $_CPPDEFFLAGS $_F90INCFLAGS $_FORTRANMODFLAG $SOURCES'

def generate(env):
    fortran.add_to_env(env)

    import f77
    f77.add_to_env(env)

    add_to_env(env)

    env['_FORTRAND']        = env.Detect(compilers) or 'f90'

def exists(env):
    return env.Detect(compilers)
