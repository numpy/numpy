"""SCons.Tool.g++

Tool-specific initialization for g++.

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

__revision__ = "src/engine/SCons/Tool/g++.py 2446 2007/09/18 11:41:57 knight"

import os.path
import re

import SCons.Defaults
import SCons.Tool
import SCons.Util

cplusplus = __import__('c++', globals(), locals(), [])

compilers = ['g++']

def generate(env):
    """Add Builders and construction variables for g++ to an Environment."""
    static_obj, shared_obj = SCons.Tool.createObjBuilders(env)

    cplusplus.generate(env)

    env['CXX']        = env.Detect(compilers)

    # platform specific settings
    if env['PLATFORM'] == 'cygwin':
        env['SHCXXFLAGS'] = SCons.Util.CLVar('$CXXFLAGS')
    elif env['PLATFORM'] == 'aix':
        # Original line from Christian Engel added -DPIC:
        #env['SHCXXFLAGS'] = SCons.Util.CLVar('$CXXFLAGS -DPIC -mminimal-toc')
        env['SHCXXFLAGS'] = SCons.Util.CLVar('$CXXFLAGS -mminimal-toc')
        env['STATIC_AND_SHARED_OBJECTS_ARE_THE_SAME'] = 1
        env['SHOBJSUFFIX'] = '$OBJSUFFIX'
    elif env['PLATFORM'] == 'hpux':
        # Original line from Christian Engel added -DPIC:
        #env['SHCXXFLAGS'] = SCons.Util.CLVar('$CXXFLAGS -fPIC -DPIC')
        env['SHCXXFLAGS'] = SCons.Util.CLVar('$CXXFLAGS -fPIC')
        env['SHOBJSUFFIX'] = '.pic.o'
    elif env['PLATFORM'] == 'sunos':
        # Original line from Christian Engel added -DPIC:
        #env['SHCXXFLAGS'] = SCons.Util.CLVar('$CXXFLAGS -fPIC -DPIC')
        env['SHCXXFLAGS'] = SCons.Util.CLVar('$CXXFLAGS -fPIC')
        env['SHOBJSUFFIX'] = '.pic.o'
    else:
        # Original line from Christian Engel added -DPIC:
        #env['SHCXXFLAGS'] = SCons.Util.CLVar('$CXXFLAGS -fPIC -DPIC')
        env['SHCXXFLAGS'] = SCons.Util.CLVar('$CXXFLAGS -fPIC')
    # determine compiler version
    if env['CXX']:
        line = os.popen(env['CXX'] + ' --version').readline()
        match = re.search(r'[0-9]+(\.[0-9]+)+', line)
        if match:
            env['CXXVERSION'] = match.group(0)


def exists(env):
    return env.Detect(compilers)
