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

"""SCons.Warnings

This file implements the warnings framework for SCons.

"""

__revision__ = "src/engine/SCons/Warnings.py 2446 2007/09/18 11:41:57 knight"

import SCons.Errors

class Warning(SCons.Errors.UserError):
    pass


# NOTE:  If you add a new warning class, add it to the man page, too!

class CacheWriteErrorWarning(Warning):
    pass

class CorruptSConsignWarning(Warning):
    pass

class DependencyWarning(Warning):
    pass

class DeprecatedWarning(Warning):
    pass

class DuplicateEnvironmentWarning(Warning):
    pass

class MisleadingKeywordsWarning(Warning):
    pass

class MissingSConscriptWarning(Warning):
    pass

class NoMD5ModuleWarning(Warning):
    pass

class NoMetaclassSupportWarning(Warning):
    pass

class NoObjectCountWarning(Warning):
    pass

class NoParallelSupportWarning(Warning):
    pass

class ReservedVariableWarning(Warning):
    pass

_warningAsException = 0

# The below is a list of 2-tuples.  The first element is a class object.
# The second element is true if that class is enabled, false if it is disabled.
_enabled = []

_warningOut = None

def suppressWarningClass(clazz):
    """Suppresses all warnings that are of type clazz or
    derived from clazz."""
    _enabled.insert(0, (clazz, 0))
    
def enableWarningClass(clazz):
    """Suppresses all warnings that are of type clazz or
    derived from clazz."""
    _enabled.insert(0, (clazz, 1))

def warningAsException(flag=1):
    """Turn warnings into exceptions.  Returns the old value of the flag."""
    global _warningAsException
    old = _warningAsException
    _warningAsException = flag
    return old

def warn(clazz, *args):
    global _enabled, _warningAsException, _warningOut

    warning = clazz(args)
    for clazz, flag in _enabled:
        if isinstance(warning, clazz):
            if flag:
                if _warningAsException:
                    raise warning
            
                if _warningOut:
                    _warningOut(warning)
            break
