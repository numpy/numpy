
import os
import sys
import re

__all__ = ['WrapperBase','WrapperCPPMacro','WrapperCCode']

class WrapperBase:

    def __init__(self):
        self.srcdir = os.path.join(os.path.dirname(__file__),'src')
        return
    def warning(self, message):
        print >> sys.stderr, message
    def info(self, message):
        print >> sys.stderr, message

    def get_resource_content(self, name, ext):
        if name.startswith('pyobj_to_'):
            try:
                return self.generate_pyobj_to_ctype_c(name[9:])
            except NotImplementedError:
                pass
        elif name.startswith('pyobj_from_'):
            try:
                return self.generate_pyobj_from_ctype_c(name[11:])
            except NotImplementedError:
                pass
        generator_mth_name = 'generate_' + name + ext.replace('.','_')
        generator_mth = getattr(self, generator_mth_name, lambda : None)
        body = generator_mth()
        if body is not None:
            return body
        fn = os.path.join(self.srcdir,name+ext)
        if os.path.isfile(fn):
            f = open(fn,'r')
            body = f.read()
            f.close()
            return body
        self.warning('No such file: %r' % (fn))
        return

    def get_dependencies(self, code):
        l = []
        for uses in re.findall(r'(?<=depends:)([,\w\s.]+)', code, re.I):
            for use in uses.split(','):
                use = use.strip()
                if not use: continue
                l.append(use)
        return l

    def resolve_dependencies(self, parent, body):
        assert isinstance(body, str),type(body)
        for d in self.get_dependencies(body):
            if d.endswith('.cpp'):
                WrapperCPPMacro(parent, d[:-4])
            elif d.endswith('.c'):
                WrapperCCode(parent, d[:-2])
            else:
                self.warning('Unknown dependence: %r.' % (d))        
        return

    def apply_attributes(self, template):
        """
        Apply instance attributes to template string.

        Replace rules for attributes:
        _list  - will be joined with newline
        _clist - _list will be joined with comma
        _elist - _list will be joined
        ..+.. - attributes will be added
        [..]  - will be evaluated
        """
        replace_names = set(re.findall(r'[ ]*%\(.*?\)s', template))
        d = {}
        for name in replace_names:
            tab = ' ' * (len(name)-len(name.lstrip()))
            name = name.lstrip()[2:-2]
            names = name.split('+')
            joinsymbol = '\n'
            attrs = None
            for n in names:
                realname = n.strip()
                if n.endswith('_clist'):
                    joinsymbol = ', '
                    realname = realname[:-6] + '_list'
                elif n.endswith('_elist'):
                    joinsymbol = ''
                    realname = realname[:-6] + '_list'
                if hasattr(self, realname):
                    attr = getattr(self, realname)
                elif realname.startswith('['):
                    attr = eval(realname)
                else:
                    self.warning('Undefined %r attribute: %r' % (self.__class__.__name__, realname))
                    continue
                if attrs is None:
                    attrs = attr
                else:
                    attrs += attr
            if isinstance(attrs, list):
                attrs = joinsymbol.join(attrs)
            d[name] = str(attrs).replace('\n','\n'+tab)
        return template % d

    def apply_templates(self, child):
        for n in parent.list_names:
            l = getattr(parent,n + '_list')
            l.append(child.apply_attributes(getattr(child, n+'_template','')))
        return

class WrapperCPPMacro(WrapperBase):
    """
    CPP macros
    """
    _defined_macros = []
    def __init__(self, parent, name):
        WrapperBase.__init__(self)
        if name in self._defined_macros:
            return
        self._defined_macros.append(name)

        body = self.get_resource_content(name,'.cpp')
        if body is None:
            self.warning('Failed to get CPP macro %r content.' % (name))
            return
        self.resolve_dependencies(parent, body)
        parent.header_list.append(body)
        return

class WrapperCCode(WrapperBase):
    """
    C code
    """
    _defined_codes = []
    def __init__(self, parent, name):
        WrapperBase.__init__(self)
        if name in self._defined_codes:
            return
        self._defined_codes.append(name)

        body = self.get_resource_content(name,'.c')
        if body is None:
            self.warning('Failed to get C code %r content.' % (name))
            return
        if isinstance(body, dict):
            for k,v in body.items():
                self.resolve_dependencies(parent, v)
            for k,v in body.items():
                l = getattr(parent,k+'_list')
                l.append(v)
        else:
            self.resolve_dependencies(parent, body)
            parent.c_code_list.append(body)
        return

    def generate_pyobj_to_ctype_c(self, ctype):
        from generate_pyobj_tofrom_funcs import pyobj_to_npy_scalar, pyobj_to_f2py_string
        if ctype.startswith('npy_'):
            return pyobj_to_npy_scalar(ctype)
        elif ctype.startswith('f2py_string'):
            return pyobj_to_f2py_string(ctype)
        raise NotImplementedError,`ctype`

    def generate_pyobj_from_ctype_c(self, ctype):
        from generate_pyobj_tofrom_funcs import pyobj_from_npy_scalar
        if ctype.startswith('npy_'):
            return pyobj_from_npy_scalar(ctype)
        raise NotImplementedError,`ctype`
