"""SCons.Tool.dvipdf

Tool-specific initialization for dvipdf.

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

__revision__ = "src/engine/SCons/Tool/dvipdf.py 2446 2007/09/18 11:41:57 knight"

import SCons.Action
import SCons.Defaults
import SCons.Tool.pdf
import SCons.Util

PDFAction = None

def PDFEmitter(target, source, env):
    """Strips any .aux or .log files from the input source list.
    These are created by the TeX Builder that in all likelihood was
    used to generate the .dvi file we're using as input, and we only
    care about the .dvi file.
    """
    def strip_suffixes(n):
        return not SCons.Util.splitext(str(n))[1] in ['.aux', '.log']
    source = filter(strip_suffixes, source)
    return (target, source)

def generate(env):
    """Add Builders and construction variables for dvipdf to an Environment."""
    global PDFAction
    if PDFAction is None:
        PDFAction = SCons.Action.Action('$DVIPDFCOM', '$DVIPDFCOMSTR')

    import pdf
    pdf.generate(env)

    bld = env['BUILDERS']['PDF']
    bld.add_action('.dvi', PDFAction)
    bld.add_emitter('.dvi', PDFEmitter)

    env['DVIPDF']      = 'dvipdf'
    env['DVIPDFFLAGS'] = SCons.Util.CLVar('')
    env['DVIPDFCOM']   = 'cd ${TARGET.dir} && $DVIPDF $DVIPDFFLAGS ${SOURCE.file} ${TARGET.file}'

    # Deprecated synonym.
    env['PDFCOM']      = ['$DVIPDFCOM']

def exists(env):
    return env.Detect('dvipdf')
