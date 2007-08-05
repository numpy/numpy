
from base import Component

class PyCArgument(Component):

    """
    PyCArgument(<name>, ctype, *components, provides=..,
                input_intent = 'required' | 'optional' | 'extra' | 'hide',
                output_intent = 'hide' | 'return',
                input_title = None,
                output_title = None,
                input_description = None,
                output_description = None
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

        if options: self.warning('%s unused options: %s\n' % (self.__class__.__name__, options))
        
        map(self.add, components)
        self.cvar = name
        self.pycvar = name + '_pyc'

        if ctype is None:
            ctype = Component.get('PyObject*')
        self.ctype = ctype
        ctype.set_pyarg_decl(self)
        #self.add(ctype)

        return self
            
    def update_containers(self):
        evaluate = self.evaluate
        ctype = self.ctype

        # get containers
        ReqArgs = self.container_ReqArgs
        OptArgs = self.container_OptArgs
        ExtArgs = self.container_ExtArgs
        RetArgs = self.container_RetArgs

        ReqArgsDoc = self.container_ReqArgsDoc
        OptArgsDoc = self.container_OptArgsDoc
        ExtArgsDoc = self.container_ExtArgsDoc

        ReqKWList = self.container_ReqKWList
        OptKWList = self.container_OptKWList
        ExtKWList = self.container_ExtKWList

        ReqPyArgFmt = self.container_ReqPyArgFmt
        OptPyArgFmt = self.container_OptPyArgFmt
        ExtPyArgFmt = self.container_ExtPyArgFmt

        ReqPyArgObj = self.container_ReqPyArgObj
        OptPyArgObj = self.container_OptPyArgObj
        ExtPyArgObj = self.container_ExtPyArgObj

        RetDoc = self.container_RetDoc
        RetFmt = self.container_RetFmt
        RetObj = self.container_RetObj

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
            ReqArgs += self.name
            ReqKWList += '"' + self.name + '"'
            ReqPyArgFmt += ctype.get_pyarg_fmt(self)
            ReqPyArgObj += ctype.get_pyarg_obj(self)
            ReqArgsDoc += input_doc_title
            ReqArgsDoc += input_doc_descr
        elif self.input_intent=='optional':
            OptArgs += self.name
            OptKWList += '"' + self.name + '"'
            OptPyArgFmt += ctype.get_pyarg_fmt(self)
            OptPyArgObj += ctype.get_pyarg_obj(self)
            OptArgsDoc += input_doc_title
            OptArgsDoc += input_doc_descr
        elif self.input_intent=='extra':
            ExtArgs += self.name
            ExtKWList += '"' + self.name + '"'
            ExtPyArgFmt += ctype.get_pyarg_fmt(self)
            ExtPyArgObj += ctype.get_pyarg_obj(self)
            ExtArgsDoc += input_doc_title
            ExtArgsDoc += input_doc_descr
        elif self.input_intent=='hide':
            pass
        else:
            raise NotImplementedError('input_intent=%r' % (self.input_intent))
            
        if self.output_intent=='return':
            RetArgs += self.name
            RetFmt += ctype.get_pyret_fmt(self)
            RetObj += ctype.get_pyret_obj(self)
            RetDoc += output_doc_title
            RetDoc += output_doc_descr
        elif self.output_intent=='hide':
            pass
        else:
            raise NotImplementedError('output_intent=%r' % (self.output_intent))

        return
        
