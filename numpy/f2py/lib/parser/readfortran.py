#!/usr/bin/env python
"""
Defines FortranReader classes for reading Fortran codes from
files and strings. FortranReader handles comments and line continuations
of both fix and free format Fortran codes.

-----
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License. See http://scipy.org.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
Author: Pearu Peterson <pearu@cens.ioc.ee>
Created: May 2006
-----
"""

__all__ = ['FortranFileReader',
           'FortranStringReader',
           'FortranReaderError',
           'Line', 'SyntaxErrorLine',
           'Comment',
           'MultiLine','SyntaxErrorMultiLine',
           ]

import re
import os
import sys
import tempfile
import traceback
from cStringIO import StringIO
from numpy.distutils.misc_util import yellow_text, red_text, blue_text

from sourceinfo import get_source_info
from splitline import String, string_replace_map, splitquote

_spacedigits=' 0123456789'
_cf2py_re = re.compile(r'(?P<indent>\s*)!f2py(?P<rest>.*)',re.I)
_is_fix_cont = lambda line: line and len(line)>5 and line[5]!=' ' and line[:5]==5*' '
_is_f90_cont = lambda line: line and '&' in line and line.rstrip()[-1]=='&'
_f90label_re = re.compile(r'\s*(?P<label>(\w+\s*:|\d+))\s*(\b|(?=&)|\Z)',re.I)
_is_include_line = re.compile(r'\s*include\s*("[^"]+"|\'[^\']+\')\s*\Z',re.I).match
_is_fix_comment = lambda line: line and line[0] in '*cC!'
_hollerith_start_search = re.compile(r'(?P<pre>\A|,\s*)(?P<num>\d+)h',re.I).search
_is_call_stmt = re.compile(r'call\b', re.I).match

class FortranReaderError: # TODO: may be derive it from Exception
    def __init__(self, message):
        self.message = message
        print >> sys.stderr,message
        sys.stderr.flush()

class Line:
    """ Holds a Fortran source line.
    """
    
    f2py_strmap_findall = re.compile(r'(_F2PY_STRING_CONSTANT_\d+_|F2PY_EXPR_TUPLE_\d+)').findall
    
    def __init__(self, line, linenospan, label, reader):
        self.line = line.strip()
        self.span = linenospan
        self.label = label
        self.reader = reader
        self.strline = None
        self.is_f2py_directive = linenospan[0] in reader.f2py_comment_lines

    def has_map(self):
        return not not (hasattr(self,'strlinemap') and self.strlinemap)

    def apply_map(self, line):
        if not hasattr(self,'strlinemap') or not self.strlinemap:
            return line
        findall = self.f2py_strmap_findall
        str_map = self.strlinemap
        keys = findall(line)
        for k in keys:
            line = line.replace(k, str_map[k])
        return line

    def copy(self, line = None, apply_map = False):
        if line is None:
            line = self.line
        if apply_map:
            line = self.apply_map(line)
        return Line(line, self.span, self.label, self.reader)

    def clone(self, line):
        self.line = self.apply_map(line)
        self.strline = None
        return
    
    def __repr__(self):
        return self.__class__.__name__+'(%r,%s,%r)' \
               % (self.line, self.span, self.label)

    def isempty(self, ignore_comments=False):
        return not (self.line.strip() or self.label)

    def get_line(self):
        if self.strline is not None:
            return self.strline
        line = self.line
        if self.reader.isfix77:
            # Handle Hollerith constants by replacing them
            # with char-literal-constants.
            # H constants may appear only in DATA statements and
            # in the argument list of CALL statement.
            # Holleriht constants were removed from the Fortran 77 standard.
            # The following handling is not perfect but works for simple
            # usage cases.
            # todo: Handle hollerith constants in DATA statement
            if _is_call_stmt(line):
                l2 = self.line[4:].lstrip()
                i = l2.find('(')
                if i != -1 and l2[-1]==')':
                    substrings = ['call '+l2[:i+1]]
                    start_search = _hollerith_start_search
                    l2 = l2[i+1:-1].strip()
                    m = start_search(l2)
                    while m:
                        substrings.append(l2[:m.start()])
                        substrings.append(m.group('pre'))
                        num = int(m.group('num'))
                        substrings.append("'"+l2[m.end():m.end()+num]+"'")
                        l2 = l2[m.end()+num:]
                        m = start_search(l2)
                    substrings.append(l2)
                    substrings.append(')')
                    line = ''.join(substrings)

        line, str_map = string_replace_map(line, lower=not self.reader.ispyf)
        self.strline = line
        self.strlinemap = str_map
        return line

class SyntaxErrorLine(Line, FortranReaderError):
    def __init__(self, line, linenospan, label, reader, message):
        Line.__init__(self, line, linenospan, label, reader)
        FortranReaderError.__init__(self, message)

class Comment:
    """ Holds Fortran comment.
    """
    def __init__(self, comment, linenospan, reader):
        self.comment = comment
        self.span = linenospan
        self.reader = reader
    def __repr__(self):
        return self.__class__.__name__+'(%r,%s)' \
               % (self.comment, self.span)
    def isempty(self, ignore_comments=False):
        return ignore_comments or len(self.comment)<2

class MultiLine:
    """ Holds (prefix, line list, suffix) representing multiline
    syntax in .pyf files:
      prefix+'''+lines+'''+suffix.
    """
    def __init__(self, prefix, block, suffix, linenospan, reader):
        self.prefix = prefix
        self.block  = block
        self.suffix = suffix
        self.span = linenospan
        self.reader = reader
    def __repr__(self):
        return self.__class__.__name__+'(%r,%r,%r,%s)' \
               % (self.prefix,self.block,self.suffix,
                  self.span)
    def isempty(self, ignore_comments=False):
        return not (self.prefix or self.block or self.suffix)

class SyntaxErrorMultiLine(MultiLine, FortranReaderError):
    def __init__(self, prefix, block, suffix, linenospan, reader, message):
        MultiLine.__init__(self, prefix, block, suffix, linenospan, reader)
        FortranReaderError.__init__(self, message)


class FortranReaderBase:

    def __init__(self, source, isfree, isstrict):
        """
        source - file-like object with .next() method
                 used to retrive a line.
        source may contain
          - Fortran 77 code
          - fixed format Fortran 90 code          
          - free format Fortran 90 code
          - .pyf signatures - extended free format Fortran 90 syntax
        """

        self.linecount = 0
        self.source = source
        self.isclosed = False

        self.filo_line = []
        self.fifo_item = []
        self.source_lines = []

        self.f2py_comment_lines = [] # line numbers that contain f2py directives

        self.reader = None
        self.include_dirs = ['.']

        self.set_mode(isfree, isstrict)
        return

    def set_mode(self, isfree, isstrict):
        self.isfree90 = isfree and not isstrict
        self.isfix90 = not isfree and not isstrict
        self.isfix77 = not isfree and isstrict
        self.ispyf   = isfree and isstrict
        self.isfree  = isfree
        self.isfix   = not isfree
        self.isstrict = isstrict

        if self.isfree90: mode = 'free90'
        elif self.isfix90: mode = 'fix90'
        elif self.isfix77: mode = 'fix77'
        else: mode = 'pyf'
        self.mode = mode
        self.name = '%s mode=%s' % (self.source, mode)
        return

    def close_source(self):
        # called when self.source.next() raises StopIteration.
        pass

    # For handling raw source lines:

    def put_single_line(self, line):
        self.filo_line.append(line)
        self.linecount -= 1
        return

    def get_single_line(self):
        try:
            line = self.filo_line.pop()
            self.linecount += 1
            return line
        except IndexError:
            pass
        if self.isclosed:
            return None
        try:
            line = self.source.next()
        except StopIteration:
            self.isclosed = True
            self.close_source()
            return None
        self.linecount += 1
        # expand tabs, replace special symbols, get rid of nl characters
        line = line.expandtabs().replace('\xa0',' ').rstrip()
        self.source_lines.append(line)
        if not line:
            return self.get_single_line()
        return line

    def get_next_line(self):
        line = self.get_single_line()
        if line is None: return
        self.put_single_line(line)
        return line

    # Parser methods:
    def get_item(self):
        try:
            return self.next(ignore_comments = True)
        except StopIteration:
            pass
        return

    def put_item(self, item):
        self.fifo_item.insert(0, item)
        return
    # Iterator methods:

    def __iter__(self):
        return self

    def next(self, ignore_comments = False):
            
        try:
            if self.reader is not None:
                try:
                    return self.reader.next()
                except StopIteration:
                    self.reader = None
            item = self._next(ignore_comments)
            if isinstance(item, Line) and _is_include_line(item.line):
                reader = item.reader
                filename = item.line.strip()[7:].lstrip()[1:-1]
                include_dirs = self.include_dirs[:]
                path = filename
                for incl_dir in include_dirs:
                    path = os.path.join(incl_dir, filename)
                    if os.path.exists(path):
                        break
                if not os.path.isfile(path):
                    dirs = os.pathsep.join(include_dirs)
                    message = reader.format_message(\
                        'WARNING',
                        'include file %r not found in %r,'\
                        ' ignoring.' % (filename, dirs),
                        item.span[0], item.span[1])
                    reader.show_message(message, sys.stdout)
                    return self.next(ignore_comments = ignore_comments)
                message = reader.format_message('INFORMATION',
                                              'found file %r' % (path),
                                              item.span[0], item.span[1])
                reader.show_message(message, sys.stdout)
                self.reader = FortranFileReader(path, include_dirs = include_dirs)
                return self.reader.next(ignore_comments = ignore_comments)
            return item
        except StopIteration:
            raise
        except:
            message = self.format_message('FATAL ERROR',
                                          'while processing line',
                                          self.linecount, self.linecount)
            self.show_message(message, sys.stdout)
            traceback.print_exc(file=sys.stdout)
            self.show_message(red_text('STOPPED READING'), sys.stdout)
            raise StopIteration

    def _next(self, ignore_comments = False):
        fifo_item_pop = self.fifo_item.pop
        while 1:
            try:
                item = fifo_item_pop(0)
            except IndexError:
                item = self.get_source_item()
                if item is None:
                    raise StopIteration
            if not item.isempty(ignore_comments):
                break
            # else ignore empty lines and comments
        if not isinstance(item, Comment):
            if not self.ispyf and isinstance(item, Line) \
                   and not item.is_f2py_directive \
                   and ';' in item.get_line():
                # ;-separator not recognized in pyf-mode
                items = []
                for line in item.get_line().split(';'):
                    line = line.strip()
                    items.append(item.copy(line, apply_map=True))
                items.reverse()
                for newitem in items:
                    self.fifo_item.insert(0, newitem)
                return fifo_item_pop(0)
            return item
        # collect subsequent comments to one comment instance
        comments = []
        start = item.span[0]
        while isinstance(item, Comment):
            comments.append(item.comment)
            end = item.span[1]
            while 1:
                try:
                    item = fifo_item_pop(0)
                except IndexError:
                    item = self.get_source_item()
                if item is None or not item.isempty(ignore_comments):
                    break
            if item is None:
                break # hold raising StopIteration for the next call.
        if item is not None:
            self.fifo_item.insert(0,item)
        return self.comment_item('\n'.join(comments), start, end)

    # Interface to returned items:

    def line_item(self, line, startlineno, endlineno, label, errmessage=None):
        if errmessage is None:
            return  Line(line, (startlineno, endlineno), label, self)
        return SyntaxErrorLine(line, (startlineno, endlineno),
                               label, self, errmessage)

    def multiline_item(self, prefix, lines, suffix,
                       startlineno, endlineno, errmessage=None):
        if errmessage is None:
            return MultiLine(prefix, lines, suffix, (startlineno, endlineno), self)
        return SyntaxErrorMultiLine(prefix, lines, suffix,
                                    (startlineno, endlineno), self, errmessage)

    def comment_item(self, comment, startlineno, endlineno):
        return Comment(comment, (startlineno, endlineno), self)

    # For handling messages:

    def show_message(self, message, stream = sys.stdout):
        stream.write(message+'\n')
        stream.flush()
        return

    def format_message(self, kind, message, startlineno, endlineno,
                       startcolno=0, endcolno=-1):
        back_index = {'warning':2,'error':3,'info':0}.get(kind.lower(),3)
        r = ['%s while processing %r (mode=%r)..' % (kind, self.id, self.mode)]
        for i in range(max(1,startlineno-back_index),startlineno):
            r.append('%5d:%s' % (i,self.source_lines[i-1]))
        for i in range(startlineno,min(endlineno+back_index,len(self.source_lines))+1):
            if i==0 and not self.source_lines:
                break
            linenostr = '%5d:' % (i)
            if i==endlineno:
                sourceline = self.source_lines[i-1] 
                l0 = linenostr+sourceline[:startcolno]
                if endcolno==-1:
                    l1 = sourceline[startcolno:]
                    l2 = ''
                else:
                    l1 = sourceline[startcolno:endcolno]
                    l2 = sourceline[endcolno:]
                r.append('%s%s%s <== %s' % (l0,yellow_text(l1),l2,red_text(message)))
            else:
                r.append(linenostr+ self.source_lines[i-1])
        return '\n'.join(r)
    
    def format_error_message(self, message, startlineno, endlineno,
                             startcolno=0, endcolno=-1):
        return self.format_message('ERROR',message, startlineno,
                                   endlineno, startcolno, endcolno)

    def format_warning_message(self, message, startlineno, endlineno,
                               startcolno=0, endcolno=-1):
        return self.format_message('WARNING',message, startlineno,
                                   endlineno, startcolno, endcolno)

    def error(self, message, item=None):
        if item is None:
            m = self.format_error_message(message, len(self.source_lines)-2, len(self.source_lines))
        else:
            m = self.format_error_message(message, item.span[0], item.span[1])
        self.show_message(m)
        return

    def warning(self, message, item=None):
        if item is None:
            m = self.format_warning_message(message, len(self.source_lines)-2, len(self.source_lines))
        else:
            m = self.format_warning_message(message, item.span[0], item.span[1])
        self.show_message(m)
        return

    # Auxiliary methods for processing raw source lines:

    def handle_cf2py_start(self, line):
        """
        f2py directives can be used only in Fortran codes.
        They are ignored when used inside .pyf files.
        """
        if not line or self.ispyf: return line
        if self.isfix:
            if line[0] in '*cC!#':
                if line[1:5].lower() == 'f2py':
                    line = 5*' ' + line[5:]
                    self.f2py_comment_lines.append(self.linecount)
            if self.isfix77:
                return line
        m = _cf2py_re.match(line)
        if m:
            newline = m.group('indent')+5*' '+m.group('rest')
            self.f2py_comment_lines.append(self.linecount)
            assert len(newline)==len(line),`newlinel,line`
            return newline
        return line

    def handle_inline_comment(self, line, lineno, quotechar=None):
        if quotechar is None and '!' not in line and \
           '"' not in line and "'" not in line:
            return line, quotechar
        i = line.find('!')
        put_item = self.fifo_item.append
        if quotechar is None and i!=-1:
            # first try a quick method
            newline = line[:i]
            if '"' not in newline and '\'' not in newline:
                if self.isfix77 or not line[i:].startswith('!f2py'):
                    put_item(self.comment_item(line[i:], lineno, lineno))
                    return newline, quotechar
        # handle cases where comment char may be a part of a character content
        #splitter = LineSplitter(line, quotechar)
        #items = [item for item in splitter]
        #newquotechar = splitter.quotechar
        items, newquotechar = splitquote(line, quotechar)

        noncomment_items = []
        noncomment_items_append = noncomment_items.append
        n = len(items)
        commentline = None
        for k in range(n):
            item = items[k]
            if isinstance(item, String) or '!' not in item:
                noncomment_items_append(item)
                continue
            j = item.find('!')
            noncomment_items_append(item[:j])
            items[k] = item[j:]
            commentline = ''.join(items[k:])
            break
        if commentline is not None:
            if commentline.startswith('!f2py'):
                # go to next iteration:
                newline = ''.join(noncomment_items) + commentline[5:]
                self.f2py_comment_lines.append(lineno)
                return self.handle_inline_comment(newline, lineno, quotechar)
            put_item(self.comment_item(commentline, lineno, lineno))
        return ''.join(noncomment_items), newquotechar

    def handle_multilines(self, line, startlineno, mlstr):
        i = line.find(mlstr)
        if i != -1:
            prefix = line[:i]
            # skip fake multiline starts
            p,k = prefix,0
            while p.endswith('\\'):
                p,k = p[:-1],k+1
            if k % 2: return
        if i != -1 and '!' not in prefix:
            # Note character constans like 'abc"""123',
            # so multiline prefix should better not contain `'' or `"' not `!'.
            for quote in '"\'':
                if prefix.count(quote) % 2:
                    message = self.format_warning_message(\
                            'multiline prefix contains odd number of %r characters' \
                            % (quote), startlineno, startlineno,
                            0, len(prefix))
                    self.show_message(message, sys.stderr)

            suffix = None
            multilines = []
            line = line[i+3:]
            while line is not None:
                j = line.find(mlstr)
                if j != -1 and '!' not in line[:j]:
                    multilines.append(line[:j])
                    suffix = line[j+3:]
                    break
                multilines.append(line)
                line = self.get_single_line()
            if line is None:
                message = self.format_error_message(\
                            'multiline block never ends', startlineno,
                            startlineno, i)
                return self.multiline_item(\
                            prefix,multilines,suffix,\
                            startlineno, self.linecount, message)
            suffix,qc = self.handle_inline_comment(suffix, self.linecount)
            # no line continuation allowed in multiline suffix
            if qc is not None:
                message = self.format_message(\
                            'ASSERTION FAILURE(pyf)',
                        'following character continuation: %r, expected None.' % (qc),
                            startlineno, self.linecount)
                self.show_message(message, sys.stderr)
            # XXX: should we do line.replace('\\'+mlstr[0],mlstr[0])
            #      for line in multilines?
            return self.multiline_item(prefix,multilines,suffix,
                                       startlineno, self.linecount)        

    # The main method of interpreting raw source lines within
    # the following contexts: f77, fixed f90, free f90, pyf.

    def get_source_item(self):
        """
        a source item is ..
        - a fortran line
        - a list of continued fortran lines
        - a multiline - lines inside triple-qoutes, only when in ispyf mode        
        """
        get_single_line = self.get_single_line
        line = get_single_line()
        if line is None: return
        startlineno = self.linecount
        line = self.handle_cf2py_start(line)
        is_f2py_directive = startlineno in self.f2py_comment_lines

        label = None
        if self.ispyf:
            # handle multilines
            for mlstr in ['"""',"'''"]:
                r = self.handle_multilines(line, startlineno, mlstr)
                if r: return r

        if self.isfix:
            label = line[:5].strip().lower()
            if label.endswith(':'): label = label[:-1].strip()
            if not line.strip():
                # empty line
                return self.line_item(line[6:],startlineno,self.linecount,label)
            if _is_fix_comment(line):
                return self.comment_item(line, startlineno, startlineno)
            for i in range(5):
                if line[i] not in _spacedigits:
                    message =  'non-space/digit char %r found in column %i'\
                              ' of fixed Fortran code' % (line[i],i+1)
                    if self.isfix90:
                        message = message + ', switching to free format mode'
                        message = self.format_warning_message(\
                            message,startlineno, self.linecount)
                        self.show_message(message, sys.stderr)
                        self.set_mode(True, False)
                    else:
                        return self.line_item(line[6:], startlineno, self.linecount,
                                           label, self.format_error_message(\
                            message, startlineno, self.linecount))

        if self.isfix77 and not is_f2py_directive:
            lines = [line[6:72]]
            while _is_fix_cont(self.get_next_line()):
                # handle fix format line continuations for F77 code
                line = get_single_line()
                lines.append(line[6:72])
            return self.line_item(''.join(lines),startlineno,self.linecount,label)

        handle_inline_comment = self.handle_inline_comment
        
        if self.isfix90 and not is_f2py_directive:
            # handle inline comment
            newline,qc = handle_inline_comment(line[6:], startlineno)
            lines = [newline]
            next_line = self.get_next_line()
            while _is_fix_cont(next_line) or _is_fix_comment(next_line):
                # handle fix format line continuations for F90 code.
                # mixing fix format and f90 line continuations is not allowed
                # nor detected, just eject warnings.
                line2 = get_single_line()
                if _is_fix_comment(line2):
                    # handle fix format comments inside line continuations
                    citem = self.comment_item(line2,self.linecount,self.linecount)
                    self.fifo_item.append(citem)
                else:
                    newline, qc = self.handle_inline_comment(line2[6:],
                                                             self.linecount, qc)
                    lines.append(newline)
                next_line = self.get_next_line()
            # no character continuation should follows now
            if qc is not None:
                message = self.format_message(\
                            'ASSERTION FAILURE(fix90)',
                            'following character continuation: %r, expected None.'\
                            % (qc), startlineno, self.linecount)
                self.show_message(message, sys.stderr)
            if len(lines)>1:
                for i in range(len(lines)):
                    l = lines[i]
                    if l.rstrip().endswith('&'):
                        message = self.format_warning_message(\
                        'f90 line continuation character `&\' detected'\
                        ' in fix format code',
                        startlineno + i, startlineno + i, l.rfind('&')+5)
                        self.show_message(message, sys.stderr)
                return self.line_item(''.join(lines),startlineno,
                                      self.linecount,label)
        start_index = 0
        if self.isfix90:
            start_index = 6

        lines = []
        lines_append = lines.append
        put_item = self.fifo_item.append
        qc = None
        while line is not None:
            if start_index: # fix format code
                line,qc = handle_inline_comment(line[start_index:],
                                                self.linecount,qc)
                is_f2py_directive = self.linecount in self.f2py_comment_lines
            else:
                line_lstrip = line.lstrip()
                if lines:
                    if line_lstrip.startswith('!'):
                        # check for comment line within line continuation
                        put_item(self.comment_item(line_lstrip,
                                                   self.linecount, self.linecount))
                        line = get_single_line()
                        continue
                else:
                    # first line, check for a f90 label
                    m = _f90label_re.match(line)
                    if m:
                        assert not label,`label,m.group('label')`
                        label = m.group('label').strip()
                        if label.endswith(':'): label = label[:-1].strip()
                        if not self.ispyf: label = label.lower()
                        line = line[m.end():]
                line,qc = handle_inline_comment(line, self.linecount, qc)
                is_f2py_directive = self.linecount in self.f2py_comment_lines

            i = line.rfind('&')
            if i!=-1:
                line_i1_rstrip = line[i+1:].rstrip()
            if not lines:
                # first line
                if i == -1 or line_i1_rstrip:
                    lines_append(line)
                    break
                lines_append(line[:i])
                line = get_single_line()
                continue
            if i == -1 or line_i1_rstrip:
                # no line continuation follows
                i = len(line)
            k = -1
            if i != -1:
                # handle the beggining of continued line
                k = line[:i].find('&')
                if k != 1 and line[:k].lstrip():
                    k = -1
            lines_append(line[k+1:i])
            if i==len(line):
                break
            line = get_single_line()

        if qc is not None:
            message = self.format_message('ASSERTION FAILURE(free)',
                'following character continuation: %r, expected None.' % (qc),
                startlineno, self.linecount)
            self.show_message(message, sys.stderr)
        return self.line_item(''.join(lines),startlineno,self.linecount,label)

    ##  FortranReaderBase

# Fortran file and string readers:

class FortranFileReader(FortranReaderBase):

    def __init__(self, filename,
                 include_dirs = None):
        isfree, isstrict = get_source_info(filename)
        self.id = filename
        self.file = open(filename,'r')
        FortranReaderBase.__init__(self, self.file, isfree, isstrict)
        if include_dirs is None:
            self.include_dirs.insert(0, os.path.dirname(filename))
        else:
            self.include_dirs = include_dirs[:]
        return

    def close_source(self):
        self.file.close()

class FortranStringReader(FortranReaderBase):
    
    def __init__(self, string, isfree, isstrict, include_dirs = None):
        self.id = 'string-'+str(id(string))
        source = StringIO(string)
        FortranReaderBase.__init__(self, source, isfree, isstrict)
        if include_dirs is not None:
            self.include_dirs = include_dirs[:]
        return

# Testing:

def test_f77():
    string_f77 = """
c12346 comment
      subroutine foo
      call foo
     'bar
a    'g
      abc=2
cf2py call me ! hey
      call you ! hi
      end
     '"""
    reader = FortranStringReader(string_f77,False,True)
    for item in reader:
        print item
    
    filename = tempfile.mktemp()+'.f'
    f = open(filename,'w')
    f.write(string_f77)
    f.close()

    reader = FortranFileReader(filename)
    for item in reader:
        print item

def test_pyf():
    string_pyf = """\
python module foo
  interface
  beginml '''1st line
  2nd line
  end line'''endml='tere!fake comment'!should be a comment
  a = 2
  'charc\"onstant' ''' single line mline '''a='hi!fake comment'!should be a comment
  a=\\\\\\\\\\'''not a multiline'''
  !blah='''never ending multiline
  b=3! hey, fake line continuation:&
  c=4& !line cont
  &45
  thisis_label_2 : c = 3
   xxif_isotropic_2 :     if ( string_upper_compare ( o%opt_aniso, 'ISOTROPIC' ) ) then
   g=3
   endif
  end interface
  if ( pc_get_lun() .ne. 6) &

    write ( pc_get_lun(), '( &
    & /, a, /, " p=", i4, " stopping c_flag=", a, &
    & /, " print unit=", i8)') &
    trim(title), pcpsx_i_pel(), trim(c_flag), pc_get_lun()
end python module foo
! end of file
"""
    reader = FortranStringReader(string_pyf,True, True)
    for item in reader:
        print item

def test_fix90():
    string_fix90 = """\
      subroutine foo
cComment
 1234 a = 3 !inline comment
      b = 3
!
     !4!line cont. with comment symbol
     &5
      a = 3!f2py.14 ! pi!
!   KDMO
      write (obj%print_lun, *) ' KDMO : '
      write (obj%print_lun, *) '  COORD = ',coord, '  BIN_WID = ',             &
       obj%bin_wid,'  VEL_DMO = ', obj%vel_dmo
      end subroutine foo
      subroutine
 
     & foo
      end
"""
    reader = FortranStringReader(string_fix90,False, False)
    for item in reader:
        print item

def simple_main():
    for filename in sys.argv[1:]:
        print 'Processing',filename
        reader = FortranFileReader(filename)
        for item in reader:
            print >> sys.stdout, item
            sys.stdout.flush()
            pass

def profile_main():
    import hotshot, hotshot.stats
    prof = hotshot.Profile("readfortran.prof")
    prof.runcall(simple_main)
    prof.close()
    stats = hotshot.stats.load("readfortran.prof")
    stats.strip_dirs()
    stats.sort_stats('time', 'calls')
    stats.print_stats(30)

if __name__ == "__main__":
    #test_pyf()
    #test_fix90()
    #profile_main()
    simple_main()
