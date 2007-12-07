"""engine.SCons.Tool.g77

Tool-specific initialization for g77.

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

import f77

compilers = ['g77', 'f77']

def generate(env):
    """Add Builders and construction variables for g77 to an Environment."""
    f77.generate(env)

    g77exec = env.Detect(compilers) or 'g77'
    env['F77'] = g77exec
    env['SHF77'] = g77exec
    if env['PLATFORM'] in ['cygwin', 'win32']:
        env['SHF77FLAGS'] = SCons.Util.CLVar('$F77FLAGS')
    else:
        env['SHF77FLAGS'] = SCons.Util.CLVar('$F77FLAGS -fPIC')

    env['FORTRAN'] = g77exec
    env['SHFORTRAN'] = g77exec
    if env['PLATFORM'] in ['cygwin', 'win32']:
        env['SHFORTRANFLAGS'] = SCons.Util.CLVar('$FORTRANFLAGS')
    else:
        env['SHFORTRANFLAGS'] = SCons.Util.CLVar('$FORTRANFLAGS -fPIC')

def exists(env):
    return env.Detect(compilers)
