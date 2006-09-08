#!/usr/bin/env python
"""
Defines FortranAnalyzer.

Permission to use, modify, and distribute this software is given under the
terms of the NumPy License. See http://scipy.org.
NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.

Author: Pearu Peterson <pearu@cens.ioc.ee>
Created: June 2006
"""

from numpy.distutils.misc_util import yellow_text, red_text

class FortranAnalyzer:

    def __init__(self, block):
        """
        block is a BeginSource instance with relevant attributes:
          name - reader name
          content - a list of statements
        Statements are either block statements or simple statements.
        Block statements have the following relevant attributes:
          name - block name
          blocktype - statement name (equal to lowered statement class name)
          content - a list of statements
        Block statements may have additional attributes:
          BeginSource: top
          Module:
          PythonModule:
          Program:
          BlockData:
          Interface: isabstract, generic_spec
          Subroutine: prefix, args, suffix
          Function: prefix, typedecl, args, suffix
          Select: expr
          Where: expr
          Forall: specs
          IfThen: expr
          If: expr
          Do: endlabel, loopcontrol
          Associate: associations
          Type: specs, params
          Enum:
        Simple statements have various attributes:
          Assignment: variable, expr
          PointerAssignment: variable, expr
          Assign: items
          Call: designator, items
          Goto: label
          ComputedGoto: items, expr
          AssignedGoto: varname, items
          Continue: label
          Return: expr
          Stop: code
          Print: format, items
          Read0: specs, items
          Read1: format, items
          Write: specs, items
          Flush: specs
          Wait: specs
          Contains:
          Allocate: spec, items
          Deallocate: items
          ModuleProcedure: items
          Public | Private: items
          Close: specs
          Cycle: name
          Rewind | Backspace | Endfile: specs
          Open: specs
          Format: specs
          Save: items
          Data: stmts
          Nullify: items
          Use: nature, name, isonly, items
          Exit: name
          Parameter: items
          Equivalence: items
          Dimension: items
          Target: items
          Pointer: items
          Protected | Volatile | Value | Intrinsic | External | Optional: items
          ArithmeticIf: expr, labels
          Inquire: specs, items
          Sequence:
          Common: items
          Intent: specs, items
          Entry: name, items, result, binds
          Import: items
          Forall: specs, content
          SpecificBinding: iname, attrs, name, bname
          GenericBinding: aspec, spec, items
          FinalBinding: items
          Allocatable: items
          Asynchronous: items
          Bind: specs, items
          Else: name
          ElseIf: name, expr
          Case: name, items
          Else: name
          ElseIf: name, expr
          Case: name, items
          Where: name, expr
          ElseWhere: name, expr
          Enumerator: items
          FortranName: value
          Threadsafe:
          Depend: depends, items
          Check: expr, value
          CallStatement: expr
          CallProtoArgument: specs
          Pause: value
        """
        self.block = block
        print block.item
    def analyze(self):
        
        pass

def simple_main():
    import sys
    from parsefortran import FortranParser
    from readfortran import FortranFileReader
    for filename in sys.argv[1:]:
        reader = FortranFileReader(filename)
        print yellow_text('Processing '+filename+' (mode=%r)' % (reader.mode))
        parser = FortranParser(reader)
        block = parser.parse()
        analyzer = FortranAnalyzer(block)
        r = analyzer.analyze()
        print r

if __name__ == "__main__":
    simple_main()
