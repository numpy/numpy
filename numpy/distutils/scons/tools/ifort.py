"""SCons.Tool.ifort

Tool-specific initialization for newer versions of the Intel Fortran Compiler
for Linux. 

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

__revision__ = "src/engine/SCons/Tool/ifort.py 2446 2007/09/18 11:41:57 knight"

import string

import SCons.Defaults

import fortran

def generate(env):
    """Add Builders and construction variables for ifort to an Environment."""
    # ifort supports Fortran 90 and Fortran 95
    # Additionally, ifort recognizes more file extensions.
    SCons.Tool.SourceFileScanner.add_scanner('.i', fortran.FortranScan)
    SCons.Tool.SourceFileScanner.add_scanner('.i90', fortran.FortranScan)
    fortran.FortranSuffixes.extend(['.i', '.i90'])
    fortran.generate(env)

    env['_FORTRAND'] = 'ifort'

    # Additionally, no symbols can be defined in an archive file; to use
    # Intel Fortran to create shared libraries, all external symbols must
    # be in shared libraries.
    env['SHLINKFLAGS'] = '-shared -no_archive'

    #
    if env['PLATFORM'] == 'win32':
        # On Windows, the ifort compiler specifies the object on the
        # command line with -object:, not -o.  Massage the necessary
        # command-line construction variables.
        for var in ['_FORTRANCOMD', '_FORTRANPPCOMD',
                    '_SHFORTRANCOMD', '_SHFORTRANPPCOMD']:
            env[var] = string.replace(env[var], '-o $TARGET', '-object:$TARGET')

    if env['PLATFORM'] in ['cygwin', 'win32']:
        env['SHFORTRANFLAGS'] = SCons.Util.CLVar('$FORTRANFLAGS')
    else:
        env['SHFORTRANFLAGS'] = SCons.Util.CLVar('$FORTRANFLAGS -fPIC')

def exists(env):
    return env.Detect('ifort')
