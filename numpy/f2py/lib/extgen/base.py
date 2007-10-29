"""
ExtGen --- Python Extension module Generator.

Defines Component and Container classes.
"""

import os
import re
import sys
import time

class ComponentMetaClass(type):

    classnamespace = {}

    def __new__(mcls, *args, **kws):
        cls = type.__new__(mcls, *args, **kws)
        n = cls.__name__
        c = ComponentMetaClass.classnamespace.get(n)
        if c is None:
            ComponentMetaClass.classnamespace[n] = cls
        else:
            if not c.__module__=='__main__':
                sys.stderr.write('ComponentMetaClass: returning %s as %s\n'\
                                 % (cls, c))
            ComponentMetaClass.classnamespace[n] = c
            cls = c
        return cls

    def __getattr__(cls, name):
        try: return ComponentMetaClass.classnamespace[name]
        except KeyError: pass
        raise AttributeError("'%s' object has no attribute '%s'"%
                             (cls.__name__, name))

class Component(object):

    __metaclass__ = ComponentMetaClass

    container_options = dict()
    component_container_map = dict()
    default_container_label = None
    default_component_class_name = 'Code'
    template = ''

    def __new__(cls, *args, **kws):
        obj = object.__new__(cls)
        obj._provides = kws.get('provides', None)
        obj.parent = None
        obj.containers = {} # holds containers for named string lists
        obj._components = [] # holds pairs (<Component subclass instance>, <container name or None>)
        obj._generate_components = {} # temporary copy of components used for finalize and generate methods.
        obj = obj.initialize(*args, **kws)    # initialize from constructor arguments
        return obj

    def components(self):
        if Component._running_generate:
            try:
                return self._generate_components[Component._running_generate_id]
            except KeyError:
                pass
            while self._generate_components: # clean up old cache
                self._generate_components.popitem()
            self._generate_components[Component._running_generate_id] = l = list(self._components)
            return l
        return self._components
    components = property(components)

    def initialize(self, *components, **options):
        """
        Set additional attributes, add components to instance, etc.
        """
        # self.myattr = ..
        # map(self.add, components)
        return self

    def finalize(self):
        """
        Set components after all components are added.
        """
        return

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, ', '.join([repr(c) for (c,l) in self.components]))

    def provides(self):
        """
        Return a code idiom name that the current class defines.

        Used in avoiding redefinitions of functions and variables.
        """
        if self._provides is None:
            return '%s_%s' % (self.__class__.__name__, id(self))
        return self._provides
    provides = property(provides)

    def warning(message):
        #raise RuntimeError('extgen:' + message)
        print >> sys.stderr, 'extgen:',message
    warning = staticmethod(warning)

    def info(message):
        print >> sys.stderr, message
    info = staticmethod(info)

    def __getattr__(self, attr):
        if attr.startswith('container_'): # convenience feature
            return self.get_container(attr[10:])
        if attr.startswith('component_'): # convenience feature
            return self.get_component(attr[10:])
        raise AttributeError('%s instance has no attribute %r' % (self.__class__.__name__, attr))

    def __add__(self, other): # convenience method
        self.add(other)
        return self
    __iadd__ = __add__

    def _get_class_names(cls):
        if not issubclass(cls, Component):
            return [cls]
        r = [cls]
        for b in cls.__bases__:
            r += Component._get_class_names(b)
        return r
    _get_class_names = staticmethod(_get_class_names)

    def add(self, component, container_label=None):
        """
        Append component and its target container label to components list.
        """
        if isinstance(component, tuple) and len(component)==2 and isinstance(component[0], Component):
            assert container_label is None, `container_label`
            component, container_label = component
        if not isinstance(component, Component) and self.default_component_class_name!=component.__class__.__name__:
            clsname = self.default_component_class_name
            if clsname is not None:
                component = getattr(Component, clsname)(component)
            else:
                raise ValueError('%s.add requires Component instance but got %r' \
                                 % (self.__class__.__name__, component.__class__.__name__))
        if container_label is None:
            container_label = self.default_container_label
            for n in self._get_class_names(component.__class__):
                try:
                    container_label = self.component_container_map[n.__name__]
                    break
                except KeyError:
                    pass
        if container_label is None:
            container_label = component.__class__.__name__
        self.components.append((component, container_label))
        component.update_parent(self)
        return

    def update_parent(self, parent):
        pass

    def get_path(self, *paths):
        if not hasattr(self, 'path'):
            if paths:
                return os.path.join(*paths)
            return ''
        if not self.parent:
            return os.path.join(*((self.path,) + paths))
        return os.path.join(*((self.parent.get_path(), self.path)+paths))

    def get_component(self, cls):
        if isinstance(cls, str):
            cls = getattr(Component, cls)
        if isinstance(self, cls):
            return self
        if self.parent:
            return self.parent.get_component(cls)
        self.warning('could not find %r parent component %s, returning self'\
                  % (self.__class__.__name__, cls.__name__))
        return self

    _running_generate = False
    _running_generate_id = 0
    _generate_dry_run = True

    def generate(self, dry_run=True):
        old_dry_run = Component._generate_dry_run
        Component._generate_dry_run = dry_run
        Component._running_generate_id += 1
        Component._running_generate = True
        self._finalize()
        result = self._generate()
        Component._running_generate = False
        Component._generate_dry_run = old_dry_run
        return result

    def _finalize(self):
        # recursively finalize all components.
        for component, container_key in self.components:
            old_parent = component.parent
            component.parent = self
            component._finalize()
            component.parent = old_parent
        self.finalize()

    def _generate(self):
        """
        Generate code idioms (saved in containers) and
        return evaluated template strings.
        """
        #self.finalize()

        # clean up containers
        self.containers = {}
        for n in dir(self):
            if n.startswith('container_') and isinstance(getattr(self, n), Container):
                delattr(self, n)

        # create containers
        for k,kwargs in self.container_options.items():
            self.containers[k] = Container(**kwargs)

        # initialize code idioms
        self.init_containers()

        # generate component code idioms
        for component, container_key in self.components:
            if not isinstance(component, Component):
                result = str(component)
                if container_key == '<IGNORE>':
                    pass
                elif container_key is not None:
                    self.get_container(container_key).add(result)
                else:
                    self.warning('%s: no container label specified for component %r'\
                                 % (self.__class__.__name__,component))
                continue
            old_parent = component.parent
            component.parent = self
            result = component._generate()
            if container_key == '<IGNORE>':
                pass
            elif container_key is not None:
                if isinstance(container_key, tuple):
                    assert len(result)==len(container_key),`len(result),container_key`
                    results = result
                    keys = container_key
                else:
                    assert isinstance(result, str) and isinstance(container_key, str), `result, container_key`
                    results = result,
                    keys = container_key,
                for r,k in zip(results, keys):
                    container = component.get_container(k)
                    container.add(r, component.provides)
            else:

                self.warning('%s: no container label specified for component providing %r'\
                                 % (self.__class__.__name__,component.provides))
            component.parent = old_parent

        # update code idioms
        self.update_containers()

        # fill templates with code idioms
        templates = self.get_templates()
        if isinstance(templates, str):
            result = self.evaluate(templates)
        else:
            assert isinstance(templates, (tuple, list)),`type(templates)`
            result = tuple(map(self.evaluate, templates))
        return result

    def init_containers(self):
        """
        Update containers before processing components.
        """
        # container = self.get_container(<key>)
        # container.add(<string>, label=None)
        return

    def update_containers(self):
        """
        Update containers after processing components.
        """
        # container = self.get_container(<key>)
        # container.add(<string>, label=None)
        return

    def get_container(self, name):
        """ Return named container.

        Rules for returning containers:
        (1) return local container if exists
        (2) return parent container if exists
        (3) create local container and return it with warning
        """
        # local container
        try:
            return self.containers[name]
        except KeyError:
            pass

        # parent container
        parent = self.parent
        while parent is not None:
            try:
                return parent.containers[name]
            except KeyError:
                parent = parent.parent
                continue

        # create local container
        self.warning('Created container for %r with name %r, define it in'\
                     ' parent .container_options mapping to get rid of this warning' \
                     % (self.__class__.__name__, name))
        c = self.containers[name] = Container()
        return c

    def get_templates(self):
        """
        Return instance templates.
        """
        return self.template

    def evaluate(self, template, **attrs):
        """
        Evaluate template using instance attributes and code
        idioms from containers.
        """
        d = self.containers.copy()
        for n in dir(self):
            if n in ['show', 'build'] or n.startswith('_'):
                continue
            v = getattr(self, n)
            if isinstance(v, str):
                d[n] = v
        d.update(attrs)
        for label, container in self.containers.items():
            if not container.use_indent:
                continue
            replace_list = set(re.findall(r'[ ]*%\('+label+r'\)s', template))
            for s in replace_list:
                old_indent = container.indent_offset
                container.indent_offset = old_indent + len(s) - len(s.lstrip())
                i = template.index(s)
                template = template[:i] + str(container) + template[i+len(s):]
                container.indent_offset = old_indent
        try:
            template = template % d
        except KeyError, msg:
            raise KeyError('%s.container_options needs %s item' % (self.__class__.__name__, msg))
        return re.sub(r'.*[<]KILLLINE[>].*(\n|$)','', template)


    _registered_components_map = {}

    def register(*components):
        """
        Register components so that component classes can use
        predefined components via `.get(<provides>)` method.
        """
        d = Component._registered_components_map
        for component in components:
            provides = component.provides
            if d.has_key(provides):
                Component.warning('component that provides %r is already registered, ignoring.' % (provides))
            else:
                d[provides] = component
        return
    register = staticmethod(register)

    def get(provides):
        """
        Return predefined component with given provides property..
        """
        try:
            return Component._registered_components_map[provides]
        except KeyError:
            pass
        raise KeyError('no registered component provides %r' % (provides))
    get = staticmethod(get)

    def numpy_version(self):
        import numpy
        return numpy.__version__
    numpy_version = property(numpy_version)

class Container(object):
    """
    Container of a list of named strings.

    >>> c = Container(separator=', ', prefix='"', suffix='"')
    >>> c.add('hey',1)
    >>> c.add('hoo',2)
    >>> print c
    "hey, hoo"
    >>> c.add('hey',1)
    >>> c.add('hey2',1)
    Traceback (most recent call last):
    ...
    ValueError: Container item 1 exists with different value

    >>> c2 = Container()
    >>> c2.add('bar')
    >>> c += c2
    >>> print c
    "hey, hoo, bar"

    """
    __metaclass__ = ComponentMetaClass

    def __init__(self,
                 separator='\n', prefix='', suffix='',
                 skip_prefix_when_empty=False,
                 skip_suffix_when_empty=False,
                 default = '', reverse=False,
                 user_defined_str = None,
                 use_indent = False,
                 indent_offset = 0,
                 use_firstline_indent = False, # implies use_indent
                 replace_map = {},
                 ignore_empty_content = False,
                 skip_prefix_suffix_when_single = False
                 ):
        self.list = []
        self.label_map = {}

        self.separator = separator
        self.prefix = prefix
        self.suffix = suffix
        self.skip_prefix = skip_prefix_when_empty
        self.skip_suffix = skip_suffix_when_empty
        self.default = default
        self.reverse = reverse
        self.user_str = user_defined_str
        self.use_indent = use_indent or use_firstline_indent
        self.indent_offset = indent_offset
        self.use_firstline_indent = use_firstline_indent
        self.replace_map = replace_map
        self.ignore_empty_content = ignore_empty_content
        self.skip_prefix_suffix_when_single = skip_prefix_suffix_when_single

    def __nonzero__(self):
        return bool(self.list)

    def has(self, label):
        return self.label_map.has_key(label)

    def get(self, label):
        return self.list[self.label_map[label]]

    def __add__(self, other):
        if isinstance(other, Container):
            lst = [(i,l) for (l,i) in other.label_map.items()]
            lst.sort()
            for i,l in lst:
                self.add(other.list[i], l)
        else:
            self.add(other)
        return self
    __iadd__ = __add__

    def add(self, content, label=None):
        """ Add content to container using label.
        If label is None, an unique label will be generated using time.time().
        """
        if content is None:
            return
        if content=='' and self.ignore_empty_content:
            return
        assert isinstance(content, str),`type(content)`
        if label is None:
            label = time.time()
        if self.has(label):
            d = self.get(label)
            if d!=content:
                raise ValueError("Container item %r exists with different value" % (label))
            return
        for old, new in self.replace_map.items():
            content = content.replace(old, new)
        self.list.append(content)
        self.label_map[label] = len(self.list)-1
        return

    def __str__(self):
        if self.user_str is not None:
            return self.user_str(self)
        if self.list:
            l = self.list
            if self.reverse:
                l = l[:]
                l.reverse()
            if self.use_firstline_indent:
                new_l = []
                for l1 in l:
                    lines = l1.split('\\n')
                    i = len(lines[0]) - len(lines[0].lstrip())
                    indent = i * ' '
                    new_l.append(lines[0])
                    new_l.extend([indent + l2 for l2 in lines[1:]])
                l = new_l
            r = self.separator.join(l)
            if not (len(self.list)==1 and self.skip_prefix_suffix_when_single):
                r = self.prefix + r
                r = r + self.suffix
        else:
            r = self.default
            if not self.skip_prefix:
                r = self.prefix + r
            if not self.skip_suffix:
                r = r + self.suffix
        if r and self.use_indent:
            lines = r.splitlines(True)
            indent = self.indent_offset * ' '
            r = ''.join([indent + line for line in lines])
        return r

    def copy(self, mapping=None, **extra_options):
        options = dict(separator=self.separator, prefix=self.prefix, suffix=self.suffix,
                       skip_prefix_when_empty=self.skip_prefix,
                       skip_suffix_when_empty=self.skip_suffix,
                       default = self.default, reverse=self.reverse,
                       user_defined_str = self.user_str,
                       use_indent = self.use_indent,
                       indent_offset = self.indent_offset,
                       use_firstline_indent = self.use_firstline_indent,
                       replace_map = self.replace_map,
                       ignore_empty_content = self.ignore_empty_content,
                       skip_prefix_suffix_when_single = self.skip_prefix_suffix_when_single
                       )
        options.update(extra_options)
        cpy = Container(**options)
        if mapping is None:
            cpy += self
        else:
            lst = [(i,l) for (l,i) in self.label_map.items()]
            lst.sort()
            for i,l in lst:
                cpy.add(mapping(other.list[i]), l)
        return cpy

def _test():
    import doctest
    doctest.testmod()

if __name__ == "__main__":
    _test()
