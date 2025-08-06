# WARNING! This a Python 2 script. Read README.rst for rationale.
import itertools
import re


def isBlank(line):
    return not line
def isLabel(line):
    return line[0].isdigit()
def isComment(line):
    return line[0] != ' '
def isContinuation(line):
    return line[5] != ' '


COMMENT, STATEMENT, CONTINUATION = 0, 1, 2
def lineType(line):
    """Return the type of a line of Fortran code."""
    if isBlank(line):
        return COMMENT
    elif isLabel(line):
        return STATEMENT
    elif isComment(line):
        return COMMENT
    elif isContinuation(line):
        return CONTINUATION
    else:
        return STATEMENT

class LineIterator:
    """LineIterator(iterable)

    Return rstrip()'d lines from iterable, while keeping a count of the
    line number in the .lineno attribute.
    """
    def __init__(self, iterable):
        object.__init__(self)
        self.iterable = iter(iterable)
        self.lineno = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.lineno += 1
        line = next(self.iterable)
        line = line.rstrip()
        return line

    next = __next__


class PushbackIterator:
    """PushbackIterator(iterable)

    Return an iterator for which items can be pushed back into.
    Call the .pushback(item) method to have item returned as the next
    value of next().
    """
    def __init__(self, iterable):
        object.__init__(self)
        self.iterable = iter(iterable)
        self.buffer = []

    def __iter__(self):
        return self

    def __next__(self):
        if self.buffer:
            return self.buffer.pop()
        else:
            return next(self.iterable)

    def pushback(self, item):
        self.buffer.append(item)

    next = __next__


def fortranSourceLines(fo):
    """Return an iterator over statement lines of a Fortran source file.

    Comment and blank lines are stripped out, and continuation lines are
    merged.
    """
    numberingiter = LineIterator(fo)
    # add an extra '' at the end
    with_extra = itertools.chain(numberingiter, [''])
    pushbackiter = PushbackIterator(with_extra)
    for line in pushbackiter:
        t = lineType(line)
        if t == COMMENT:
            continue
        elif t == STATEMENT:
            lines = [line]
            # this is where we need the extra '', so we don't finish reading
            # the iterator when we don't want to handle that
            for next_line in pushbackiter:
                t = lineType(next_line)
                if t == CONTINUATION:
                    lines.append(next_line[6:])
                else:
                    pushbackiter.pushback(next_line)
                    break
            yield numberingiter.lineno, ''.join(lines)
        else:
            raise ValueError("jammed: continuation line not expected: %s:%d" %
                             (fo.name, numberingiter.lineno))

def getDependencies(filename):
    """For a Fortran source file, return a list of routines declared as EXTERNAL
    in it.
    """
    external_pat = re.compile(r'^\s*EXTERNAL\s', re.I)
    routines = []
    with open(filename) as fo:
        for lineno, line in fortranSourceLines(fo):
            m = external_pat.match(line)
            if m:
                names = line[m.end():].strip().split(',')
                names = [n.strip().lower() for n in names]
                names = [n for n in names if n]
                routines.extend(names)
    return routines
