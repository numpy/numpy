
from base import Component

class PyCArgument(Component):

    """
    PyCArgument(<name>, *components, provides=..,
                input_intent = 'required' | 'optional' | 'extra' | 'hide',
                output_intent = 'hide' | 'return',
                input_title = None,
                output_title = None,
                input_description = None,
                output_description = None
               )

    """

    template = '%(name)s'

    def initialize(self, name, *components, **options):
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

    def get_ctype(self):
        for component, container_label in self.components:
            if isinstance(component, Component.CTypeBase):
                return component
        return

    def init_containers(self):
        ctype = self.get_ctype()
        if ctype is None:
            self.cname = self.name
        else:
            self.cname = self.provides

    def update_containers(self):
        evaluate = self.evaluate
        ctype = self.get_ctype()

        # get containers
        ReqArgs = self.container_ReqArgs
        OptArgs = self.container_OptArgs
        ExtArgs = self.container_ExtArgs
        RetArgs = self.container_RetArgs

        ReqArgsDoc = self.container_ReqArgsDoc
        OptArgsDoc = self.container_OptArgsDoc
        ExtArgsDoc = self.container_ExtArgsDoc

        Decl = self.container_Decl

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
        Decl.add('PyObject* %s = NULL;' % (self.cname))
        if ctype is not None:
            Decl.add(ctype.declare(self.name))
        
        if ctype is None:
            input_doc_title = '%s - %s' % (self.name, self.input_title)
            output_doc_title = '%s - %s' % (self.name, self.output_title)
            if self.input_description is not None:
                input_doc_descr = '  %s' % (self.input_description.replace('\n','\\n'))
            else:
                input_doc_descr = None
            if self.output_description is not None:
                output_doc_descr = '  %s' % (self.output_description.replace('\n','\\n'))
            else:
                output_doc_descr = None
            iopt = (self.name, '"%s"' % (self.name), 'O', '&%s' % (self.cname), input_doc_title, input_doc_descr)
            ropt = (self.name, 'N', self.cname, output_doc_title, output_doc_descr)
        else:
            raise NotImplementedError('ctype=%r' % (ctype))
            
        if self.input_intent=='required':
            ReqArgs.add(iopt[0])
            ReqKWList.add(iopt[1])
            ReqPyArgFmt.add(iopt[2])
            ReqPyArgObj.add(iopt[3])
            ReqArgsDoc.add(iopt[4])
            ReqArgsDoc.add(iopt[5])
        elif self.input_intent=='optional':
            OptArgs.add(iopt[0])
            OptKWList.add(iopt[1])
            OptPyArgFmt.add(iopt[2])
            OptPyArgObj.add(iopt[3])
            OptArgsDoc.add(iopt[4])
            OptArgsDoc.add(iopt[5])
        elif self.input_intent=='extra':
            ExtArgs.add(iopt[0])
            ExtKWList.add(iopt[1])
            ExtPyArgFmt.add(iopt[2])
            ExtPyArgObj.add(iopt[3])
            ExtArgsDoc.add(iopt[4])
            ExtArgsDoc.add(iopt[5])
        elif self.input_intent=='hide':
            ropt = (self.name, 'O', self.cname, output_doc_title, output_doc_descr)
        else:
            raise NotImplementedError('input_intent=%r' % (self.input_intent))
            
        if self.output_intent=='return':
            RetArgs.add(ropt[0])
            RetFmt.add(ropt[1])
            RetObj.add(ropt[2])
            RetDoc.add(ropt[3])
            RetDoc.add(ropt[4])
        elif self.output_intent=='hide':
            pass
        else:
            raise NotImplementedError('output_intent=%r' % (self.output_intent))

        return
        
