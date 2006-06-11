#!/usr/bin/env python
"""
Defines FortranParser.

Permission to use, modify, and distribute this software is given under the
terms of the NumPy License. See http://scipy.org.
NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.

Author: Pearu Peterson <pearu@cens.ioc.ee>
Created: May 2006
"""

import re
import sys
import traceback
from numpy.distutils.misc_util import yellow_text, red_text

from readfortran import FortranFileReader, FortranStringReader
from block import Block

class FortranParser:

    def __init__(self, reader):
        self.reader = reader
        self.isfix77 = reader.isfix77

    def get_item(self):
        try:
            item = self.reader.next(ignore_comments = True)
            return item
        except StopIteration:
            pass

    def put_item(self, item):
        self.reader.fifo_item.insert(0, item)

    def parse(self):
        import init
        try:
            main = Block(self)
            main.fill()
            return main
        except KeyboardInterrupt:
            raise
        except:
            message = self.reader.format_message('FATAL ERROR',
                                                 'while processing line',
                                                 self.reader.linecount, self.reader.linecount)
            self.reader.show_message(message, sys.stdout)
            traceback.print_exc(file=sys.stdout)
            self.reader.show_message(red_text('STOPPED PARSING'), sys.stdout)

def test_pyf():
    string = """
python module foo
  interface
    subroutine bar
    real r
    end subroutine bar
  end interface
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
    end subroutine bar

end module foo
"""
    reader = FortranStringReader(string, True, False)
    parser = FortranParser(reader)
    block = parser.parse()
    print block

def test_f77():
    string = """\
c      program foo
      a = 3
      end
      subroutine bar
      end
"""
    reader = FortranStringReader(string, False, True)
    parser = FortranParser(reader)
    block = parser.parse()
    print block

def simple_main():
    import sys
    for filename in sys.argv[1:]:
        reader = FortranFileReader(filename)
        print yellow_text('Processing '+filename+' (mode=%r)' % (reader.mode))

        parser = FortranParser(reader)
        block = parser.parse()
        print block

def profile_main():
    import hotshot, hotshot.stats
    prof = hotshot.Profile("_parsefortran.prof")
    prof.runcall(simple_main)
    prof.close()
    stats = hotshot.stats.load("_parsefortran.prof")
    stats.strip_dirs()
    stats.sort_stats('time', 'calls')
    stats.print_stats(30)

if __name__ == "__main__":
    #test_f77()
    #test_free90()
    #test_pyf()
    simple_main()
    #profile_main()
