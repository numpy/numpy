#!/usr/bin/env python
"""
Defines FortranParser.

Permission to use, modify, and distribute this software is given under the
terms of the NumPy License. See http://scipy.org.
NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.

Author: Pearu Peterson <pearu@cens.ioc.ee>
Created: May 2006
"""

__all__ = ['FortranParser']

import re
import sys
import traceback
from numpy.distutils.misc_util import yellow_text, red_text

from readfortran import FortranFileReader, FortranStringReader
from block_statements import BeginSource
from utils import AnalyzeError

class FortranParser:

    cache = {}

    def __init__(self, reader):
        """
        Parser of FortranReader structure.
        Use .parse() method for parsing, parsing result is saved in .block attribute.
        """
        self.reader = reader
        if reader.id in self.cache:
            parser = self.cache[reader.id]
            self.block = parser.block
            self.is_analyzed = parser.is_analyzed
            self.block.show_message('using cached %s' % (reader.id))
        else:
            self.cache[reader.id] = self
            self.block = None
            self.is_analyzed = False
        return

    def get_item(self):
        try:
            return self.reader.next(ignore_comments = True)
        except StopIteration:
            pass
        return

    def put_item(self, item):
        self.reader.fifo_item.insert(0, item)
        return

    def parse(self):
        if self.block is not None:
            return
        try:
            block = self.block = BeginSource(self)
        except KeyboardInterrupt:
            raise
        except:
            reader = self.reader
            while reader is not None:
                message = reader.format_message('FATAL ERROR',
                                                'while processing line',
                                                reader.linecount, reader.linecount)
                reader.show_message(message, sys.stderr)
                reader = reader.reader
            traceback.print_exc(file=sys.stderr)
            self.reader.show_message(red_text('STOPPED PARSING'), sys.stderr)
            return
        return

    def analyze(self):
        if self.is_analyzed:
            return
        if self.block is None:
            self.reader.show_message('Nothing to analyze.')
            return

        try:
            self.block.analyze()
        except AnalyzeError:
            pass
        except Exception, msg:
            if str(msg) != '123454321':
                traceback.print_exc(file=sys.stderr)
                self.reader.show_message(red_text('FATAL ERROR: STOPPED ANALYSING %r CONTENT' % (self.reader.source) ), sys.stderr)
                sys.exit(123454321)
            return
        self.is_analyzed = True
        return

def test_pyf():
    string = """
python module foo
  interface tere
    subroutine bar
    real r
    end subroutine bar
  end interface tere
end python module foo
"""
    reader = FortranStringReader(string, True, True)
    parser = FortranParser(reader)
    block = parser.parse()
    print block

def test_free90():
    string = """
module foo

   subroutine bar
    real r
    if ( pc_get_lun() .ne. 6) &
    write ( pc_get_lun(), '( &
    & /, a, /, " p=", i4, " stopping c_flag=", a, &
    & /, " print unit=", i8)') &
    trim(title), pcpsx_i_pel(), trim(c_flag), pc_get_lun()
    if (.true.) then
      call smth
    end if
    aaa : if (.false.) then
    else if (a) then aaa
    else aaa
    end if aaa
    hey = 1
    end subroutine bar
    abstract interface

    end interface

end module foo
"""
    reader = FortranStringReader(string, True, False)
    parser = FortranParser(reader)
    block = parser.parse()
    print block

def test_f77():
    string = """\
      program foo
      a = 3
      end
      subroutine bar
      end
      pure function foo(a)
      end
      pure real*4 recursive function bar()
      end
"""
    reader = FortranStringReader(string, False, True)
    parser = FortranParser(reader)
    block = parser.parse()
    print block

def simple_main():
    import sys
    if not sys.argv[1:]:
        return parse_all_f()
    for filename in sys.argv[1:]:
        reader = FortranFileReader(filename)
        print yellow_text('Processing '+filename+' (mode=%r)' % (reader.mode))
        parser = FortranParser(reader)
        parser.parse()
        parser.analyze()
        print parser.block.torepr(4)
        #print parser.block

def profile_main():
    import hotshot, hotshot.stats
    prof = hotshot.Profile("_parsefortran.prof")
    prof.runcall(simple_main)
    prof.close()
    stats = hotshot.stats.load("_parsefortran.prof")
    stats.strip_dirs()
    stats.sort_stats('time', 'calls')
    stats.print_stats(30)

def parse_all_f():
    for filename in open('opt_all_f.txt'):
        filename = filename.strip()
        reader = FortranFileReader(filename)
        print yellow_text('Processing '+filename+' (mode=%r)' % (reader.mode))
        parser = FortranParser(reader)
        block = parser.parse()
        print block

if __name__ == "__main__":
    #test_f77()
    #test_free90()
    #test_pyf()
    simple_main()
    #profile_main()
    #parse_all_f()
