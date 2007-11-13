
__all__ = ['Word', 'Line', 'Code', 'FileSource']

from base import Component

class Word(Component):
    template = '%(word)s'

    def initialize(self, word):
        if not word: return None
        self.word = word
        return self

    def add(self, component, container_label=None):
        raise ValueError('%s does not take components' % (self.__class__.__name__))

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, ', '.join(map(repr,[self.word]+[c for (c,l) in self.components])))


class Line(Component):

    """
    >>> l = Line('hey')
    >>> l += ' you '
    >>> l += 2
    >>> print l
    Line('hey you 2')
    >>> print l.generate()
    hey you 2
    >>> l += l
    >>> print l.generate()
    hey you 2hey you 2
    """

    template = '%(line)s'

    def initialize(self, *strings):
        self.line = ''
        map(self.add, strings)
        return self

    def add(self, component, container_label=None):
        if isinstance(component, Line):
            self.line += component.line
        elif isinstance(component, str):
            self.line += component
        elif component is None:
            pass
        else:
            self.line += str(component)

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, ', '.join(map(repr,[self.line]+[c for (c,l) in self.components])))


class Code(Component):

    """
    >>> c = Code('start')
    >>> c += 2
    >>> c += 'end'
    >>> c
    Code(Line('start'), Line('2'), Line('end'))
    >>> print c.generate()
    start
    2
    end
    """

    template = '%(Line)s'

    container_options = dict(
        Line = dict(default = '<KILLLINE>', ignore_empty_content=True)
        )
    component_container_map = dict(
        Line = 'Line'
        )
    default_component_class_name = 'Line'

    def initialize(self, *lines):
        map(self.add, lines)
        return self

    def add(self, component, label=None):
        if isinstance(component, Code):
            assert label is None,`label`
            self.components += component.components
        else:
            Component.add(self, component, label)


class FileSource(Component):

    container_options = dict(
        Content = dict(default='<KILLLINE>')
        )

    template = '%(Content)s'

    default_component_class_name = 'Code'

    component_container_map = dict(
      Line = 'Content',
      Code = 'Content',
    )

    def initialize(self, path, *components, **options):
        self.path = path
        map(self.add, components)
        self._provides = options.pop('provides', path)
        if options: self.warning('%s unused options: %s\n' % (self.__class__.__name__, options))
        return self

    def finalize(self):
        self._provides = self.get_path() or self._provides

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, ', '.join(map(repr,[self.path]+[c for (c,l) in self.components])))

def _test():
    import doctest
    doctest.testmod()

if __name__ == "__main__":
    _test()
