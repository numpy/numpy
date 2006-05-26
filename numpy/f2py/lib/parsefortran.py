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

from numpy.distutils.misc_util import yellow_text

from readfortran import FortranFileReader, FortranStringReader
from block import Block

class FortranParser:

    def __init__(self, reader):
        self.reader = reader
        self.isfix77 = reader.isfix77

    def get_item(self):
        try:
            return self.reader.next(ignore_comments = True)
        except StopIteration:
            pass

    def put_item(self, item):
        self.reader.fifo_item.insert(0, item)

    def parse(self):
        main = Block(self)
        main.fill()
        return main

def test_pyf():
    string = """
python module foo
  interface
    subroutine bar
    real r
    end subroutine bar
  end interface
end python module
"""
    reader = FortranStringReader(string, True, True)
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
        print yellow_text('Processing '+filename)
        reader = FortranFileReader(filename)
        parser = FortranParser(reader)
        block = parser.parse()
        #print block
        
if __name__ == "__main__":
    #test_f77()
    #test_pyf()
    simple_main()
