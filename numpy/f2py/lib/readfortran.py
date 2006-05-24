#!/usr/bin/env python
"""
Defines FortranReader classes for reading Fortran codes from
files and strings. FortranReader handles comments and line continuations
of both fix and free format Fortran codes.

Copyright 2006 Pearu Peterson all rights reserved,
Pearu Peterson <pearu@cens.ioc.ee>
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License. See http://scipy.org.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
Pearu Peterson
"""

import re
import sys
import tempfile
from cStringIO import StringIO

from numpy.distutils.misc_util import yellow_text, red_text, blue_text

from sourceinfo import get_source_info
from splitline import LineSplitter, String

_spacedigits=' 0123456789'
_cf2py_re = re.compile(r'(?P<indent>\s*)!f2py(?P<rest>.*)',re.I)
_is_fix_cont = lambda line: line and len(line)>5 and line[5]!=' ' and line[0]==' '
_is_f90_cont = lambda line: line and '&' in line and line.rstrip()[-1]=='&'


class FortranReaderError: # TODO: may be derive it from Exception
    def __init__(self, message):
        self.message = message
        print >> sys.stderr,message

class Line:
    """ Holds a Fortran source line.
    """
    def __init__(self, line, linenospan, label, reader):
        self.line = line
        self.span = linenospan
        self.label = label
        self.reader = reader
    def __repr__(self):
        return self.__class__.__name__+'(%r,%s,%r)' \
               % (self.line, self.span, self.label)
    def isempty(self):
        return not (self.line.strip() or (self.label and self.label.strip()))

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
    def isempty(self):
        return len(self.comment)<2 # comment includes comment character

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
    def isempty(self):
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
        self.isfree90 = isfree and not isstrict
        self.isfix90 = not isfree and not isstrict
        self.isfix77 = not isfree and isstrict
        self.ispyf   = isfree and isstrict
        self.isfree  = isfree
        self.isfix   = not isfree

        self.linecount = 0
        self.source = source
        self.isclosed = False

        self.filo_line = []
        self.fifo_item = []
        self.source_lines = []

    def close_source(self):
        pass

    # For handling raw source lines:

    def put_single_line(self, line):
        self.filo_line.append(line)
        self.linecount -= 1

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
        line = line.expandtabs().replace('\xa0',' ').rstrip('\n\r\f')
        self.source_lines.append(line)            
        return line

    def get_next_line(self):
        line = self.get_single_line()
        if line is None: return
        self.put_single_line(line)
        return line

    # Iterator methods:

    def __iter__(self):
        return self

    def next(self):
        fifo_item_pop = self.fifo_item.pop
        while 1:
            try:
                item = fifo_item_pop(0)
            except IndexError:
                item = self.get_source_item()
                if item is None:
                    raise StopIteration
            if not item.isempty():
                break
            # else ignore empty lines and comments
        if not isinstance(item, Comment):
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
                if item is None or not item.isempty():
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

    def format_message(self, kind, message, startlineno, endlineno,
                       startcolno=0, endcolno=-1):
        r = ['%s while processing %s..' % (kind, self.source)]
        for i in range(max(1,startlineno-3),startlineno):
            r.append('%5d:%s' % (i,self.source_lines[i-1]))
        for i in range(startlineno,min(endlineno+3,len(self.source_lines))+1):
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
            if self.isfix77:
                return line
        m = _cf2py_re.match(line)
        if m:
            newline = m.group('indent')+5*' '+m.group('rest')
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
                put_item(self.comment_item(line[i:], lineno, lineno))
                return newline, quotechar
        # handle cases where comment char may be a part of a character content
        splitter = LineSplitter(line, quotechar)
        items = [item for item in splitter]
        newquotechar = splitter.quotechar
        noncomment_items = []
        noncomment_items_append = noncomment_items.append
        n = len(items)
        for i in range(n):
            item = items[i]
            if isinstance(item, String) or '!' not in item:
                noncomment_items_append(item)
                continue
            j = item.find('!')
            noncomment_items_append(item[:j])
            items[i] = item[j:]
            put_item(self.comment_item(''.join(items[i:]), lineno, lineno))
            break
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
                    print >> sys.stderr, message

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
            # no line continuation allowed in mulitline suffix
            if qc is not None:
                message = self.format_message(\
                            'ASSERTION FAILURE(pyf)',
                        'following character continuation: %r, expected None.' % (qc),
                            startlineno, self.linecount)
                print >> sys.stderr, message
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

        if self.ispyf:
            # handle multilines
            for mlstr in ['"""',"'''"]:
                r = self.handle_multilines(line, startlineno, mlstr)
                if r: return r

        if self.isfix:
            label = line[:5]
            if not line.strip():
                # empty line
                return self.line_item(line[6:],startlineno,self.linecount,label)
            if line[0] in '*cC!':
                return self.comment_item(line, startlineno, startlineno)
            for i in range(5):
                if line[i] not in _spacedigits:
                    message =  'non-space/digit char %r found in column %i'\
                              ' of fixed Fortran code' % (line[i],i+1)
                    if self.isfix90:
                        message = message + ', switching to free format mode'
                        message = self.format_warning_message(\
                            message,startlineno, self.linecount)
                        print >> sys.stderr, message
                        self.isfree = True
                        self.isfix90 = False
                        self.isfree90 = True
                    else:
                        return self.line_item(line[6:], startlineno, self.linecount,
                                           label, self.format_error_message(\
                            message, startlineno, self.linecount))
        if self.isfix77:
            lines = [line[6:72]]
            while _is_fix_cont(self.get_next_line()):
                # handle fix format line continuations for F77 code
                line = get_single_line()
                lines.append(line[6:72])
            return self.line_item(''.join(lines),startlineno,self.linecount,label)

        handle_inline_comment = self.handle_inline_comment

        if self.isfix90 and _is_fix_cont(self.get_next_line()):
            # handle inline comment
            newline,qc = handle_inline_comment(line[6:], startlineno)
            lines = [newline]
            while _is_fix_cont(self.get_next_line()):
                # handle fix format line continuations for F90 code.
                # mixing fix format and f90 line continuations is not allowed
                # nor detected, just eject warnings.
                line = get_single_line()
                newline,qc = self.handle_inline_comment(line[6:], self.linecount, qc)
                lines.append(newline)
            # no character continuation should follows now
            if qc is not None:
                message = self.format_message(\
                            'ASSERTION FAILURE(fix90)',
                            'following character continuation: %r, expected None.' % (qc),
                            startlineno, self.linecount)
                print >> sys.stderr, message
            for i in range(len(lines)):
                l = lines[i]
                if l.rstrip().endswith('&'):
                    message = self.format_warning_message(\
                        'f90 line continuation character `&\' detected'\
                        ' in fix format code',
                        startlineno + i, startlineno + i, l.rfind('&')+5)
                    print >> sys.stderr, message
            return self.line_item(''.join(lines),startlineno,self.linecount,label)

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
            else:
                line_lstrip = line.lstrip()
                if lines and line_lstrip.startswith('!'):
                    # check for comment line within line continuation
                    put_item(self.comment_item(line_lstrip,
                                               self.linecount, self.linecount))
                    line = get_single_line()
                    continue
                line,qc = handle_inline_comment(line, self.linecount, qc)

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
            print >> sys.stderr, message
        return self.line_item(''.join(lines),startlineno,self.linecount,None)

    ##  FortranReaderBase

class FortranFileReader(FortranReaderBase):

    def __init__(self, filename):
        isfree, isstrict = get_source_info(filename)
        self.file = open(filename,'r')
        FortranReaderBase.__init__(self, self.file, isfree, isstrict)

    def close_source(self):
        self.file.close()

class FortranStringReader(FortranReaderBase):
    
    def __init__(self, string, isfree, isstrict):
        source = StringIO(string)
        FortranReaderBase.__init__(self, source, isfree, isstrict)

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
  end interface
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

      end subroutine foo
"""
    reader = FortranStringReader(string_fix90,False, False)
    for item in reader:
        print item

def simple_main():
    for filename in sys.argv[1:]:
        print 'Processing',filename
        reader = FortranFileReader(filename)
        for item in reader:
            #print item
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
