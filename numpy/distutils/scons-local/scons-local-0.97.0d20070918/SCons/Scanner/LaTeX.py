"""SCons.Scanner.LaTeX

This module implements the dependency scanner for LaTeX code.

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

__revision__ = "src/engine/SCons/Scanner/LaTeX.py 2446 2007/09/18 11:41:57 knight"

import os.path
import string

import SCons.Scanner

def LaTeXScanner():
    """Return a prototype Scanner instance for scanning LaTeX source files"""
    ds = LaTeX(name = "LaTeXScanner",
               suffixes =  '$LATEXSUFFIXES',
               path_variable = 'TEXINPUTS',
               regex = '\\\\(include|includegraphics(?:\[[^\]]+\])?|input|bibliography|usepackage){([^}]*)}',
               recursive = 0)
    return ds

class LaTeX(SCons.Scanner.Classic):
    """Class for scanning LaTeX files for included files.

    Unlike most scanners, which use regular expressions that just
    return the included file name, this returns a tuple consisting
    of the keyword for the inclusion ("include", "includegraphics",
    "input", or "bibliography"), and then the file name itself.  
    Based on a quick look at LaTeX documentation, it seems that we 
    need a should append .tex suffix for the "include" keywords, 
    append .tex if there is no extension for the "input" keyword, 
    but leave the file name untouched for "includegraphics." For
    the "bibliography" keyword we need to add .bib if there is
    no extension. (This need to be revisited since if there
    is no extension for an :includegraphics" keyword latex will 
    append .ps or .eps to find the file; while pdftex will use 
    other extensions.)
    """
    def latex_name(self, include):
        filename = include[1]
        if include[0] == 'input':
            base, ext = os.path.splitext( filename )
            if ext == "":
                filename = filename + '.tex'
        if (include[0] == 'include'):
            filename = filename + '.tex'
        if include[0] == 'bibliography':
            base, ext = os.path.splitext( filename )
            if ext == "":
                filename = filename + '.bib'
        if include[0] == 'usepackage':
            base, ext = os.path.splitext( filename )
            if ext == "":
                filename = filename + '.sty'
        return filename
    def sort_key(self, include):
        return SCons.Node.FS._my_normcase(self.latex_name(include))
    def find_include(self, include, source_dir, path):
        i = SCons.Node.FS.find_file(self.latex_name(include),
                                    (source_dir,) + path)
        return i, include

    def scan(self, node, path=()):
        #
        # Modify the default scan function to allow for the regular
        # expression to return a comma separated list of file names
        # as can be the case with the bibliography keyword.
        #
        # cache the includes list in node so we only scan it once:
        if node.includes != None:
            includes = node.includes
        else:
            includes = self.cre.findall(node.get_contents())
            node.includes = includes

        # This is a hand-coded DSU (decorate-sort-undecorate, or
        # Schwartzian transform) pattern.  The sort key is the raw name
        # of the file as specifed on the #include line (including the
        # " or <, since that may affect what file is found), which lets
        # us keep the sort order constant regardless of whether the file
        # is actually found in a Repository or locally.
        nodes = []
        source_dir = node.get_dir()
        for include in includes:
            #
            # Handle multiple filenames in include[1]
            #
            inc_list = string.split(include[1],',')
            for j in range(len(inc_list)):
                include_local = [include[0],inc_list[j]]
                n, i = self.find_include(include_local, source_dir, path)

            if n is None:
                SCons.Warnings.warn(SCons.Warnings.DependencyWarning,
                                    "No dependency generated for file: %s (included from: %s) -- file not found" % (i, node))
            else:
                sortkey = self.sort_key(include)
                nodes.append((sortkey, n))

        nodes.sort()
        nodes = map(lambda pair: pair[1], nodes)
        return nodes
