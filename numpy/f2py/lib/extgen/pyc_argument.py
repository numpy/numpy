
from base import Component

class PyCArgument(Component):

    """
    PyCArgument(<name>, ctype, *components, provides=..,
                input_intent = 'required' | 'optional' | 'extra' | 'hide',
                output_intent = 'hide' | 'return',
                input_title = None,
                output_title = None,
                input_description = None,
                output_description = None,
                depends = []
               )

    """

    template = '%(name)s'

    component_container_map = dict(CDecl = 'Decl')

    def initialize(self, name, ctype=None, *components, **options):
        self.name = name
        self._provides = options.pop('provides',
                                     '%s_%s' % (self.__class__.__name__, name))
        self.input_intent = options.pop('input_intent','required') # 'optional', 'extra', 'hide'
        self.output_intent = options.pop('output_intent','hide')   # 'return'
        self.input_title = options.pop('input_title', None)
        self.output_title = options.pop('output_title', None)
        self.input_description = options.pop('input_description', None)
        self.output_description = options.pop('output_description', None)
        self.depends = options.pop('depends', [])

        if options: self.warning('%s unused options: %s\n' % (self.__class__.__name__, options))
        
        map(self.add, components)

        self.cvar = name
        self.pycvar = None
        self.retpycvar = None

        if ctype is None:
            ctype = object
        if isinstance(ctype,  Component.CTypeBase):
            pass
        elif isinstance(ctype, type) or Component.CTypePython.typeinfo_map.has_key(ctype):
            ctype = Component.CTypePython(ctype)
        else:
            ctype = Component.CType(ctype)
        self.ctype = ctype

        retfmt = ctype.get_pyret_fmt(self)
        if isinstance(ctype, Component.CTypePython):
            if retfmt and retfmt in 'SON':
                if self.output_intent == 'return':
                    if self.input_intent=='hide':
                        self.retpycvar = name
                    else:
                        self.pycvar = name
                        self.retpycvar = name + '_return'
                elif self.input_intent!='hide':
                    self.pycvar = name
            else:
                self.pycvar = name
                self.retpycvar = name
        else:
            self.pycvar = name + '_pyc'
            self.retpycvar = name + '_pyc_r'

        ctype.set_titles(self)
        ctype.set_Decl(self)

        return self
            
    def update_containers(self):
        evaluate = self.evaluate
        ctype = self.ctype

        # update PyCFunction containers
        input_doc_title = '%s - %s' % (self.name, self.input_title)
        output_doc_title = '%s - %s' % (self.name, self.output_title)
        if self.input_description is not None:
            input_doc_descr = '  %s' % (self.input_description)
        else:
            input_doc_descr = None
        if self.output_description is not None:
            output_doc_descr = '  %s' % (self.output_description)
        else:
            output_doc_descr = None

        if self.input_intent=='required':
            self.container_ReqArgs += self.name
            self.container_ReqKWList += '"' + self.name + '"'
            self.container_ReqArgFmt += ctype.get_pyarg_fmt(self)
            self.container_ReqArgObj += ctype.get_pyarg_obj(self)
            self.container_ReqArgsDoc += input_doc_title
            self.container_ReqArgsDoc += input_doc_descr
        elif self.input_intent=='optional':
            self.container_OptArgs += self.name
            self.container_OptKWList += '"' + self.name + '"'
            self.container_OptArgFmt += ctype.get_pyarg_fmt(self)
            self.container_OptArgObj += ctype.get_pyarg_obj(self)
            self.container_OptArgsDoc += input_doc_title
            self.container_OptArgsDoc += input_doc_descr
        elif self.input_intent=='extra':
            self.container_ExtArgs += self.name
            self.container_ExtKWList += '"' + self.name + '"'
            self.container_ExtArgFmt += ctype.get_pyarg_fmt(self)
            self.container_ExtArgObj += ctype.get_pyarg_obj(self)
            self.container_ExtArgsDoc += input_doc_title
            self.container_ExtArgsDoc += input_doc_descr
        elif self.input_intent=='hide':
            pass
        else:
            raise NotImplementedError('input_intent=%r' % (self.input_intent))
            
        if self.output_intent=='return':
            self.container_RetArgs += self.name
            self.container_RetFmt += ctype.get_pyret_fmt(self)
            self.container_RetObj += ctype.get_pyret_obj(self)
            self.container_RetDoc += output_doc_title
            self.container_RetDoc += output_doc_descr
        elif self.output_intent=='hide':
            pass
        else:
            raise NotImplementedError('output_intent=%r' % (self.output_intent))

        return
        
