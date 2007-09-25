"""engine.SCons.Tool.f77

Tool-specific initialization for the generic Posix f77 Fortran compiler.

There normally shouldn't be any need to import this module directly.
It will usually be imported through the generic SCons.Tool.Tool()
selection method.

"""

#
# Copyright (c) 2001, 2002, 2003, 2004, 2005, 2006, 2007 The SCons Foundation
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

__revision__ = "src/engine/SCons/Tool/f77.py 2446 2007/09/18 11:41:57 knight"

import SCons.Defaults
import SCons.Scanner.Fortran
import SCons.Tool
import SCons.Util
import fortran

compilers = ['f77']

#
F77Suffixes = ['.f77']
F77PPSuffixes = []
if SCons.Util.case_sensitive_suffixes('.f77', '.F77'):
    F77PPSuffixes.append('.F77')
else:
    F77Suffixes.append('.F77')

#
F77Scan = SCons.Scanner.Fortran.FortranScan("F77PATH")

for suffix in F77Suffixes + F77PPSuffixes:
    SCons.Tool.SourceFileScanner.add_scanner(suffix, F77Scan)
del suffix

#
fVLG = fortran.VariableListGenerator

F77Generator = fVLG('F77', 'FORTRAN', '_FORTRAND')
F77FlagsGenerator = fVLG('F77FLAGS', 'FORTRANFLAGS')
F77CommandGenerator = fVLG('F77COM', 'FORTRANCOM', '_F77COMD')
F77CommandStrGenerator = fVLG('F77COMSTR', 'FORTRANCOMSTR', '_F77COMSTRD')
F77PPCommandGenerator = fVLG('F77PPCOM', 'FORTRANPPCOM', '_F77PPCOMD')
F77PPCommandStrGenerator = fVLG('F77PPCOMSTR', 'FORTRANPPCOMSTR', '_F77PPCOMSTRD')
ShF77Generator = fVLG('SHF77', 'SHFORTRAN', 'F77', 'FORTRAN', '_FORTRAND')
ShF77FlagsGenerator = fVLG('SHF77FLAGS', 'SHFORTRANFLAGS')
ShF77CommandGenerator = fVLG('SHF77COM', 'SHFORTRANCOM', '_SHF77COMD')
ShF77CommandStrGenerator = fVLG('SHF77COMSTR', 'SHFORTRANCOMSTR', '_SHF77COMSTRD')
ShF77PPCommandGenerator = fVLG('SHF77PPCOM', 'SHFORTRANPPCOM', '_SHF77PPCOMD')
ShF77PPCommandStrGenerator = fVLG('SHF77PPCOMSTR', 'SHFORTRANPPCOMSTR', '_SHF77PPCOMSTRD')

del fVLG

#
F77Action = SCons.Action.Action('$_F77COMG ', '$_F77COMSTRG')
F77PPAction = SCons.Action.Action('$_F77PPCOMG ', '$_F77PPCOMSTRG')
ShF77Action = SCons.Action.Action('$_SHF77COMG ', '$_SHF77COMSTRG')
ShF77PPAction = SCons.Action.Action('$_SHF77PPCOMG ', '$_SHF77PPCOMSTRG')

def add_to_env(env):
    """Add Builders and construction variables for f77 to an Environment."""
    env.AppendUnique(FORTRANSUFFIXES = F77Suffixes + F77PPSuffixes)

    static_obj, shared_obj = SCons.Tool.createObjBuilders(env)

    for suffix in F77Suffixes:
        static_obj.add_action(suffix, F77Action)
        shared_obj.add_action(suffix, ShF77Action)
        static_obj.add_emitter(suffix, fortran.FortranEmitter)
        shared_obj.add_emitter(suffix, fortran.ShFortranEmitter)

    for suffix in F77PPSuffixes:
        static_obj.add_action(suffix, F77PPAction)
        shared_obj.add_action(suffix, ShF77PPAction)
        static_obj.add_emitter(suffix, fortran.FortranEmitter)
        shared_obj.add_emitter(suffix, fortran.ShFortranEmitter)

    env['_F77G']            = F77Generator
    env['_F77FLAGSG']       = F77FlagsGenerator
    env['_F77COMG']         = F77CommandGenerator
    env['_F77PPCOMG']       = F77PPCommandGenerator
    env['_F77COMSTRG']      = F77CommandStrGenerator
    env['_F77PPCOMSTRG']    = F77PPCommandStrGenerator

    env['_SHF77G']          = ShF77Generator
    env['_SHF77FLAGSG']     = ShF77FlagsGenerator
    env['_SHF77COMG']       = ShF77CommandGenerator
    env['_SHF77PPCOMG']     = ShF77PPCommandGenerator
    env['_SHF77COMSTRG']    = ShF77CommandStrGenerator
    env['_SHF77PPCOMSTRG']  = ShF77PPCommandStrGenerator

    env['_F77INCFLAGS'] = '$( ${_concat(INCPREFIX, F77PATH, INCSUFFIX, __env__, RDirs, TARGET, SOURCE)} $)'

    env['_F77COMD']     = '$_F77G -o $TARGET -c $_F77FLAGSG $_F77INCFLAGS $SOURCES'
    env['_F77PPCOMD']   = '$_F77G -o $TARGET -c $_F77FLAGSG $CPPFLAGS $_CPPDEFFLAGS $_F77INCFLAGS $SOURCES'
    env['_SHF77COMD']   = '$_SHF77G -o $TARGET -c $_SHF77FLAGSG $_F77INCFLAGS $SOURCES'
    env['_SHF77PPCOMD'] = '$_SHF77G -o $TARGET -c $_SHF77FLAGSG $CPPFLAGS $_CPPDEFFLAGS $_F77INCFLAGS $SOURCES'

def generate(env):
    fortran.add_to_env(env)

    import f90
    import f95
    f90.add_to_env(env)
    f95.add_to_env(env)

    add_to_env(env)

    env['_FORTRAND']        = env.Detect(compilers) or 'f77'

def exists(env):
    return env.Detect(compilers)
