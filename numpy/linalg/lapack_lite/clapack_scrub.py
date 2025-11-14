#!/usr/bin/env python2.7
# WARNING! This a Python 2 script. Read README.rst for rationale.
import os
import re
import sys

from plex import IGNORE, TEXT, AnyChar, Bol, Lexicon, Opt, Scanner, State, Str
from plex.traditional import re as Re

try:
    from io import BytesIO as UStringIO  # Python 2
except ImportError:
    from io import StringIO as UStringIO  # Python 3


class MyScanner(Scanner):
    def __init__(self, info, name='<default>'):
        Scanner.__init__(self, self.lexicon, info, name)

    def begin(self, state_name):
        Scanner.begin(self, state_name)

def sep_seq(sequence, sep):
    pat = Str(sequence[0])
    for s in sequence[1:]:
        pat += sep + Str(s)
    return pat

def runScanner(data, scanner_class, lexicon=None):
    info = UStringIO(data)
    outfo = UStringIO()
    if lexicon is not None:
        scanner = scanner_class(lexicon, info)
    else:
        scanner = scanner_class(info)
    while True:
        value, text = scanner.read()
        if value is None:
            break
        elif value is IGNORE:
            pass
        else:
            outfo.write(value)
    return outfo.getvalue(), scanner

class LenSubsScanner(MyScanner):
    """Following clapack, we remove ftnlen arguments, which f2c puts after
    a char * argument to hold the length of the passed string. This is just
    a nuisance in C.
    """
    def __init__(self, info, name='<ftnlen>'):
        MyScanner.__init__(self, info, name)
        self.paren_count = 0

    def beginArgs(self, text):
        if self.paren_count == 0:
            self.begin('args')
        self.paren_count += 1
        return text

    def endArgs(self, text):
        self.paren_count -= 1
        if self.paren_count == 0:
            self.begin('')
        return text

    digits = Re('[0-9]+')
    iofun = Re(r'\([^;]*;')
    decl = Re(r'\([^)]*\)[,;' + '\n]')
    any = Re('[.]*')
    S = Re('[ \t\n]*')
    cS = Str(',') + S
    len_ = Re('[a-z][a-z0-9]*_len')

    iofunctions = Str("s_cat", "s_copy", "s_stop", "s_cmp",
                      "i_len", "do_fio", "do_lio") + iofun

    # Routines to not scrub the ftnlen argument from
    keep_ftnlen = (Str('ilaenv_') | Str('iparmq_') | Str('s_rnge')) + Str('(')

    lexicon = Lexicon([
        (iofunctions,                                        TEXT),
        (keep_ftnlen,                                        beginArgs),
        State('args', [
            (Str(')'),   endArgs),
            (Str('('),   beginArgs),
            (AnyChar,    TEXT),
        ]),
        (cS + Re(r'[1-9][0-9]*L'),                           IGNORE),
        (cS + Str('ftnlen') + Opt(S + len_),                 IGNORE),
        (cS + sep_seq(['(', 'ftnlen', ')'], S) + S + digits, IGNORE),
        (Bol + Str('ftnlen ') + len_ + Str(';\n'),           IGNORE),
        (cS + len_,                                          TEXT),
        (AnyChar,                                            TEXT),
    ])

def scrubFtnlen(source):
    return runScanner(source, LenSubsScanner)[0]

def cleanSource(source):
    # remove whitespace at end of lines
    source = re.sub(r'[\t ]+\n', '\n', source)
    # remove comments like .. Scalar Arguments ..
    source = re.sub(r'(?m)^[\t ]*/\* *\.\. .*?\n', '', source)
    # collapse blanks of more than two in-a-row to two
    source = re.sub(r'\n\n\n\n+', r'\n\n\n', source)
    return source

class LineQueue:
    def __init__(self):
        object.__init__(self)
        self._queue = []

    def add(self, line):
        self._queue.append(line)

    def clear(self):
        self._queue = []

    def flushTo(self, other_queue):
        for line in self._queue:
            other_queue.add(line)
        self.clear()

    def getValue(self):
        q = LineQueue()
        self.flushTo(q)
        s = ''.join(q._queue)
        self.clear()
        return s

class CommentQueue(LineQueue):
    def __init__(self):
        LineQueue.__init__(self)

    def add(self, line):
        if line.strip() == '':
            LineQueue.add(self, '\n')
        else:
            line = '  ' + line[2:-3].rstrip() + '\n'
            LineQueue.add(self, line)

    def flushTo(self, other_queue):
        if len(self._queue) == 0:
            pass
        elif len(self._queue) == 1:
            other_queue.add('/*' + self._queue[0][2:].rstrip() + ' */\n')
        else:
            other_queue.add('/*\n')
            LineQueue.flushTo(self, other_queue)
            other_queue.add('*/\n')
        self.clear()

# This really seems to be about 4x longer than it needs to be
def cleanComments(source):
    lines = LineQueue()
    comments = CommentQueue()

    def isCommentLine(line):
        return line.startswith('/*') and line.endswith('*/\n')

    blanks = LineQueue()

    def isBlank(line):
        return line.strip() == ''

    def SourceLines(line):
        if isCommentLine(line):
            comments.add(line)
            return HaveCommentLines
        else:
            lines.add(line)
            return SourceLines

    def HaveCommentLines(line):
        if isBlank(line):
            blanks.add('\n')
            return HaveBlankLines
        elif isCommentLine(line):
            comments.add(line)
            return HaveCommentLines
        else:
            comments.flushTo(lines)
            lines.add(line)
            return SourceLines

    def HaveBlankLines(line):
        if isBlank(line):
            blanks.add('\n')
            return HaveBlankLines
        elif isCommentLine(line):
            blanks.flushTo(comments)
            comments.add(line)
            return HaveCommentLines
        else:
            comments.flushTo(lines)
            blanks.flushTo(lines)
            lines.add(line)
            return SourceLines

    state = SourceLines
    for line in UStringIO(source):
        state = state(line)
    comments.flushTo(lines)
    return lines.getValue()

def removeHeader(source):
    lines = LineQueue()

    def LookingForHeader(line):
        m = re.match(r'/\*[^\n]*-- translated', line)
        if m:
            return InHeader
        else:
            lines.add(line)
            return LookingForHeader

    def InHeader(line):
        if line.startswith('*/'):
            return OutOfHeader
        else:
            return InHeader

    def OutOfHeader(line):
        if line.startswith('#include "f2c.h"'):
            pass
        else:
            lines.add(line)
        return OutOfHeader

    state = LookingForHeader
    for line in UStringIO(source):
        state = state(line)
    return lines.getValue()

def removeSubroutinePrototypes(source):
    # This function has never worked as advertised by its name:
    # - "/* Subroutine */" declarations may span multiple lines and
    #   cannot be matched by a line by line approach.
    # - The caret in the initial regex would prevent any match, even
    #   of single line "/* Subroutine */" declarations.
    #
    # While we could "fix" this function to do what the name implies
    # it should do, we have no hint of what it should really do.
    #
    # Therefore we keep the existing (non-)functionality, documenting
    # this function as doing nothing at all.
    return source

def removeBuiltinFunctions(source):
    lines = LineQueue()

    def LookingForBuiltinFunctions(line):
        if line.strip() == '/* Builtin functions */':
            return InBuiltInFunctions
        else:
            lines.add(line)
            return LookingForBuiltinFunctions

    def InBuiltInFunctions(line):
        if line.strip() == '':
            return LookingForBuiltinFunctions
        else:
            return InBuiltInFunctions

    state = LookingForBuiltinFunctions
    for line in UStringIO(source):
        state = state(line)
    return lines.getValue()

def replaceDlamch(source):
    """Replace dlamch_ calls with appropriate macros"""
    def repl(m):
        s = m.group(1)
        return {'E': 'EPSILON', 'P': 'PRECISION', 'S': 'SAFEMINIMUM',
                    'B': 'BASE'}[s[0]]
    source = re.sub(r'dlamch_\("(.*?)"\)', repl, source)
    source = re.sub(r'^\s+extern.*? dlamch_.*?;$(?m)', '', source)
    return source

# do it

def scrubSource(source, nsteps=None, verbose=False):
    steps = [
             ('scrubbing ftnlen', scrubFtnlen),
             ('remove header', removeHeader),
             ('clean source', cleanSource),
             ('clean comments', cleanComments),
             ('replace dlamch_() calls', replaceDlamch),
             ('remove prototypes', removeSubroutinePrototypes),
             ('remove builtin function prototypes', removeBuiltinFunctions),
            ]

    if nsteps is not None:
        steps = steps[:nsteps]

    for msg, step in steps:
        if verbose:
            print(msg)
        source = step(source)

    return source


if __name__ == '__main__':
    filename = sys.argv[1]
    outfilename = os.path.join(sys.argv[2], os.path.basename(filename))
    with open(filename) as fo:
        source = fo.read()

    if len(sys.argv) > 3:
        nsteps = int(sys.argv[3])
    else:
        nsteps = None

    source = scrubSource(source, nsteps, verbose=True)

    with open(outfilename, 'w') as writefo:
        writefo.write(source)
