
__all__ = ['CLine', 'Keyword', 'CTypeSpec', 'CDeclarator', 'CDeclaration',
           'CArgument', 'CCode', 'CFunction', 'CSource', 'CHeader', 'CStdHeader']

from base import Component
from utils import Line, Code, FileSource

class CLine(Line):
    pass

class Keyword(CLine):
    pass

class CInitExpr(CLine):
    pass

class CTypeSpec(CLine):

    """
    >>> i = CTypeSpec('int')
    >>> print i.generate()
    int
    >>> print i.as_ptr().generate()
    int*
    """
    def as_ptr(self): return self.__class__(self.generate()+'*')


class CDeclarator(Component):

    """

    >>> CDeclarator('name').generate()
    'name'
    >>> CDeclarator('name','0').generate()
    'name = 0'
    """
    container_options = dict(
        Initializer = dict(default='',prefix=' = ', skip_prefix_when_empty=True,
                                 ignore_empty_content = True
                                 ),
        ScalarInitializer = dict(default='',prefix=' = ', skip_prefix_when_empty=True,
                                 ignore_empty_content = True
                                 ),
        SequenceInitializer = dict(default='',prefix=' = {\n', skip_prefix_when_empty=True,
                                   suffix='}', skip_suffix_when_empty=True,
                                   ignore_empty_content = True,
                                   separator = ',\n', use_indent=True,
                                   ),
        StringInitializer = dict(default='',prefix=' = "', skip_prefix_when_empty=True,
                                 suffix='"', skip_suffix_when_empty=True,
                                 ignore_empty_content = True,
                                 separator='\\n"\n"', replace_map = {'\n':'\\n'},
                                 use_firstline_indent = True,
                                 ),
        )

    default_component_class_name = 'CInitExpr'

    component_container_map = dict(
        CInitExpr = 'Initializer'
        )

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, ', '.join(map(repr,[self.name]+[c for (c,l) in self.components])))

    def initialize(self, name, *initvalues, **options):
        self.name = name
        self.is_string = options.get('is_string', None)
        if self.is_string:
            assert not options.get('is_scalar', None)
            self.is_scalar = False
        else:
            if name.endswith(']'):
                self.is_scalar = False
            else:
                self.is_scalar = options.get('is_scalar', True)

        map(self.add, initvalues)
        return self

    def update_containers(self):
        if self.is_scalar:
            self.container_ScalarInitializer += self.container_Initializer
            self.template = '%(name)s%(ScalarInitializer)s'
        elif self.is_string:
            self.container_StringInitializer += self.container_Initializer
            self.template = '%(name)s%(StringInitializer)s'
        elif len(self.containers)>1 or not self.is_scalar:
            self.container_SequenceInitializer += self.container_Initializer
            self.template = '%(name)s%(SequenceInitializer)s'
        else:
            self.container_ScalarInitializer += self.container_Initializer
            self.template = '%(name)s%(ScalarInitializer)s'

class CDeclaration(Component):

    """
    >>> d = CDeclaration('int', 'a')
    >>> print d.generate()
    int a
    >>> d += 'b'
    >>> print d.generate()
    int a, b
    >>> d += CDeclarator('c',1)
    >>> print d.generate()
    int a, b, c = 1
    """

    template = '%(CTypeSpec)s %(CDeclarator)s'

    container_options = dict(
        CTypeSpec = dict(default='int', separator=' '),
        CDeclarator = dict(default='<KILLLINE>', separator=', '),
        )

    component_container_map = dict(
        CTypeSpec = 'CTypeSpec',
        CDeclarator = 'CDeclarator',
        )

    default_component_class_name = 'CDeclarator'

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, ', '.join(map(repr,[c for (c,l) in self.components])))

    def initialize(self, ctype, *declarators, **options):
        ctype = CTypeSpec(ctype)
        self.ctype = ctype
        self.add(ctype)
        map(self.add, declarators)
        return self

class CArgument(CDeclaration):

    def initialize(self, name, ctype, **options):
        return CDeclaration.initialize(self, ctype, name, **options)


class CCode(Code):
    parent_container_options = dict(default='<KILLLINE>', use_indent=True, ignore_empty_content=True)

class CFunction(Component):

    """
    >>> f = CFunction('foo')
    >>> print f.generate()
    int
    foo(void) {
    }
    >>> f += Keyword('static')
    >>> f += CArgument('a', 'int')
    >>> f += 'a = 2;'
    >>> print f.generate()
    static
    int
    foo(int a) {
      a = 2;
    }
    >>> f += CArgument('b', 'float')
    >>> f += CDeclaration('float', 'c')
    >>> f += CDeclaration('float', CDeclarator('d','3.0'))
    >>> print f.generate()
    static
    int
    foo(int a, float b) {
      float c;
      float d = 3.0;
      a = 2;
    }
    """

    template = '''\
%(CSpecifier)s
%(CTypeSpec)s
%(name)s(%(CArgument)s) {
  %(CDeclaration)s
  %(CBody)s
}'''
    
    container_options = dict(
        CArgument = dict(separator=', ', default='void'),
        CDeclaration = dict(default='<KILLLINE>', use_indent=True, ignore_empty_content=True,
                            separator = ';\n', suffix=';', skip_suffix_when_empty=True),
        CBody = dict(default='<KILLLINE>', use_indent=True, ignore_empty_content=True),
        CTypeSpec = dict(default='int', separator = ' ', ignore_empty_content=True),
        CSpecifier = dict(default='<KILLLINE>', separator = ' ', ignore_empty_content = True)
        )

    component_container_map = dict(
        CArgument = 'CArgument',
        CDeclaration = 'CDeclaration',
        CCode = 'CBody',
        CTypeSpec = 'CTypeSpec',
        Keyword = 'CSpecifier',
        )

    default_component_class_name = 'CCode'

    def initialize(self, name, rctype='int', *components, **options):
        self.name = name
        rctype = CTypeSpec(rctype)
        self.rctype = rctype
        self.add(rctype)
        map(self.add, components)
        if options: self.warning('%s unused options: %s\n' % (self.__class__.__name__, options))
        return self

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, ', '.join(map(repr,[self.name, self.rctype]+[c for (c,l) in self.components])))

class CHeader(CLine):

    """
    >>> h = CHeader('noddy.h')
    >>> print h.generate()
    #include "noddy.h"
    
    """
    template = '#include "%(line)s"'

class CStdHeader(CHeader):
    template = '#include <%(line)s>'

class CSource(FileSource):

    """
    >>> s = CSource('foo.c')
    >>> print s.generate() #doctest: +ELLIPSIS
    /* -*- c -*- */
    /* This file 'foo.c' is generated using ExtGen tool
       from NumPy version ...
       ExtGen is developed by Pearu Peterson <pearu.peterson@gmail.com>.
       For more information see http://www.scipy.org/ExtGen/ .
    */
    #ifdef __cplusplus
    extern "C" {
    #endif
    #ifdef __cplusplus
    }
    #endif
    <BLANKLINE>
    """

    container_options = dict(
        CHeader = dict(default='<KILLLINE>', prefix='\n/* CHeader */\n', skip_prefix_when_empty=True),
        CTypeDef = dict(default='<KILLLINE>', prefix='\n/* CTypeDef */\n', skip_prefix_when_empty=True),
        CProto = dict(default='<KILLLINE>', prefix='\n/* CProto */\n', skip_prefix_when_empty=True),
        CDefinition = dict(default='<KILLLINE>', prefix='\n/* CDefinition */\n', skip_prefix_when_empty=True),
        CDeclaration = dict(default='<KILLLINE>', separator=';\n', suffix=';',
                            prefix='\n/* CDeclaration */\n', skip_prefix_when_empty=True),
        CMainProgram = dict(default='<KILLLINE>', prefix='\n/* CMainProgram */\n', skip_prefix_when_empty=True),
        )

    template_c_header = '''\
/* -*- c -*- */
/* This file %(path)r is generated using ExtGen tool
   from NumPy version %(numpy_version)s.
   ExtGen is developed by Pearu Peterson <pearu.peterson@gmail.com>.
   For more information see http://www.scipy.org/ExtGen/ .
*/'''


    template = template_c_header + '''
#ifdef __cplusplus
extern \"C\" {
#endif
%(CHeader)s
%(CTypeDef)s
%(CProto)s
%(CDefinition)s
%(CDeclaration)s
%(CMainProgram)s
#ifdef __cplusplus
}
#endif
'''

    component_container_map = dict(
      CHeader = 'CHeader',
      CFunction = 'CDefinition',
      CDeclaration = 'CDeclaration',
    )




def _test():
    import doctest
    doctest.testmod()
    
if __name__ == "__main__":
    _test()
