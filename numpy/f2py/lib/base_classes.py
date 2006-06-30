
__all__ = ['Statement','BeginStatement','EndStatement']

import re
import sys
from readfortran import Line


class Statement:
    """
    Statement instance has attributes:
      parent  - Parent BeginStatement or FortranParser instance
      item    - Line instance containing the statement line
      isvalid - boolean, when False, the Statement instance will be ignored
    """
    modes = ['free90','fix90','fix77','pyf']

    def __init__(self, parent, item):
        self.parent = parent
        self.reader = parent.reader
        self.item = item

        # when a statement instance is constructed by error, set isvalid to False
        self.isvalid = True
        # when a statement should be ignored, set ignore to True
        self.ignore = False

        self.process_item()

    def get_indent_tab(self,colon=None,deindent=False):
        if self.reader.isfix:
            tab = ' '*6
        else:
            tab = ''
        p = self.parent
        while isinstance(p, Statement):
            tab += '  '
            p = p.parent
        if deindent:
            tab = tab[:-2]
        if self.item is None:
            return tab
        s = self.item.label
        if colon is None:
            if self.reader.isfix:
                colon = ''
            else:
                colon = ':'
        if s:
            c = ''
            if self.reader.isfix:
                c = ' '
            tab = tab[len(c+s)+len(colon):]
            if not tab: tab = ' '
            tab = c + s + colon + tab
        return tab

    def show_message(self, message):
        print >> sys.stderr, message
        sys.stderr.flush()
        return

class BeginStatement(Statement):
    """ <blocktype> <name>

    BeginStatement instances have additional attributes:
      name
      blocktype

    Block instance has attributes:
      content - list of Line or Statement instances
      name    - name of the block, unnamed blocks are named
                with the line label
      parent  - Block or FortranParser instance
      item    - Line instance containing the block start statement
      get_item, put_item - methods to retrive/submit Line instaces
                from/to Fortran reader.
      isvalid - boolean, when False, the Block instance will be ignored.

      stmt_cls, end_stmt_cls

    """
    def __init__(self, parent, item=None):

        self.content = []
        self.get_item = parent.get_item # get line function
        self.put_item = parent.put_item # put line function
        if not hasattr(self, 'blocktype'):
            self.blocktype = self.__class__.__name__.lower()
        if not hasattr(self, 'name'):
            # process_item may change this
            self.name = '__'+self.blocktype.upper()+'__' 

        Statement.__init__(self, parent, item)
        return

    def tostr(self):
        return self.blocktype.upper() + ' '+ self.name
    
    def __str__(self):
        l=[self.get_indent_tab(colon=':') + self.tostr()]
        for c in self.content:
            l.append(str(c))
        return '\n'.join(l)

    def process_item(self):
        """ Process the line
        """
        item = self.item
        if item is None: return
        self.fill()
        return

    def fill(self, end_flag = False):
        """
        Fills blocks content until the end of block statement.
        """

        mode = self.reader.mode
        classes = self.get_classes()
        self.classes = [cls for cls in classes if mode in cls.modes]
        self.pyf_classes = [cls for cls in classes if 'pyf' in cls.modes]

        item = self.get_item()
        while item is not None:
            if isinstance(item, Line):
                if self.process_subitem(item):
                    end_flag = True
                    break
            item = self.get_item()

        if not end_flag:
            message = self.item.reader.format_message(\
                        'WARNING',
                        'failed to find the end of block for %s'\
                        % (self.__class__.__name__),
                        self.item.span[0],self.item.span[1])
            self.show_message(message)
        return

    def process_subitem(self, item):
        """
        Check is item is blocks start statement, if it is, read the block.

        Return True to stop adding items to given block.
        """
        line = item.get_line()

        # First check for the end of block
        cls = self.end_stmt_cls
        if cls.match(line):
            stmt = cls(self, item)
            if stmt.isvalid:
                self.content.append(stmt)
                return True

        if item.is_f2py_directive:
            classes = self.pyf_classes
        else:
            classes = self.classes

        # Look for statement match
        for cls in classes:
            if cls.match(line):
                stmt = cls(self, item)
                if stmt.isvalid:
                    if not stmt.ignore:
                        self.content.append(stmt)
                    return False
                # item may be cloned that changes the items line:
                line = item.get_line()
                
        # Check if f77 code contains inline comments or other f90
        # constructs that got undetected by get_source_info.
        if item.reader.isfix77:
            i = line.find('!')
            if i != -1:
                message = item.reader.format_message(\
                        'WARNING',
                        'no parse pattern found for "%s" in %r block'\
                        ' maybe due to inline comment.'\
                        ' Trying to remove the comment.'\
                        % (item.get_line(),self.__class__.__name__),
                        item.span[0], item.span[1])
                # .. but at the expense of loosing the comment.
                self.show_message(message)
                newitem = item.copy(line[:i].rstrip())
                return self.process_subitem(newitem)

            # try fix90 statement classes
            f77_classes = self.classes
            classes = []
            for cls in self.get_classes():
                if 'fix90' in cls.modes and cls not in f77_classes:
                    classes.append(cls)
            if classes:
                message = item.reader.format_message(\
                        'WARNING',
                        'no parse pattern found for "%s" in %r block'\
                        ' maybe due to strict f77 mode.'\
                        ' Trying f90 fix mode patterns..'\
                        % (item.get_line(),self.__class__.__name__),
                        item.span[0], item.span[1])
                self.show_message(message)
    
                item.reader.set_mode(False, False)
                self.classes = classes
            
                r = BeginStatement.process_subitem(self, item)
                if r is None:
                    # restore f77 fix mode
                    self.classes = f77_classes
                    item.reader.set_mode(False, True)
                else:
                    message = item.reader.format_message(\
                        'INFORMATION',
                        'The f90 fix mode resolved the parse pattern issue.'\
                        ' Setting reader to f90 fix mode.',
                        item.span[0], item.span[1])
                    self.show_message(message)
                    # set f90 fix mode
                    self.classes = f77_classes + classes
                    self.reader.set_mode(False, False)
                return r

        self.handle_unknown_item(item)
        return

    def handle_unknown_item(self, item):
        message = item.reader.format_message(\
                        'WARNING',
                        'no parse pattern found for "%s" in %r block.'\
                        % (item.get_line(),self.__class__.__name__),
                        item.span[0], item.span[1])
        self.show_message(message)
        self.content.append(item)
        #sys.exit()
        return


class EndStatement(Statement):
    """
    END [<blocktype> [<name>]]

    EndStatement instances have additional attributes:
      name
      blocktype
    """

    def __init__(self, parent, item):
        if not hasattr(self, 'blocktype'):
            self.blocktype = self.__class__.__name__.lower()[3:]
        Statement.__init__(self, parent, item)

    def process_item(self):
        item = self.item
        line = item.get_line().replace(' ','')[3:]
        blocktype = self.blocktype
        if line.startswith(blocktype):
            line = line[len(blocktype):].strip()
        else:
            if line:
                # not the end of expected block
                line = ''
                self.isvalid = False
        if line:
            if not line==self.parent.name:
                message = item.reader.format_message(\
                        'WARNING',
                        'expected the end of %r block but got end of %r, skipping.'\
                        % (self.parent.name, line),
                        item.span[0],item.span[1])
                self.show_message(message)
                self.isvalid = False
        self.name = self.parent.name

    def __str__(self):
        return self.get_indent_tab()[:-2] + 'END %s %s'\
               % (self.blocktype.upper(),self.name or '')

