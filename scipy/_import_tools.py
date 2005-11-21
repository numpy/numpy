
import os
import sys
import imp
from glob import glob

class PackageImport:
    """ Import packages from the current directory that implement
    info.py. See scipy/doc/DISTUTILS.txt for more info.
    """

    imported_packages = []

    def __init__(self):
        self.frame = frame = sys._getframe(1)
        self.parent_name = eval('__name__',frame.f_globals,frame.f_locals)
        self.parent_path = eval('__path__[0]',frame.f_globals,frame.f_locals)

    def get_info_modules(self,packages=None):
        """
        Return info modules of packages or all packages in parent path.
        """
        if packages is None:
            info_files = glob(os.path.join(self.parent_path,'*','info.py'))
        else:
            info_files = [os.path.join(self.parent_path,package,'info.py') \
                          for package in packages]
        info_modules = {}
        for info_file in info_files:
            package_name = os.path.basename(os.path.dirname(info_file))
            fullname = self.parent_name +'.'+ package_name
            try:
                info_module = imp.load_module(fullname+'.info',
                                              open(info_file,'U'),
                                                info_file,
                                              ('.py','U',1))
            except Exception,msg:
                print msg
                info_module = None

            if info_module is None:
                continue
            if getattr(info_module,'ignore',False):
                continue

            info_modules[fullname] = info_module

        return info_modules

    def _sort_info_modules(self, info_modules):
        """
        Return package names sorted in the order as they should be
        imported due to dependence relations between packages. 
        """
        depend_dict = {}
        for fullname,info_module in info_modules.items():
            depend_dict[fullname] = getattr(info_module,'depends',[])
        package_names = []

        for name in depend_dict.keys():
            if not depend_dict[name]:
                package_names.append(name)
                del depend_dict[name]

        while depend_dict:
            for name, lst in depend_dict.items():
                new_lst = [n for n in lst if depend_dict.has_key(n)]
                if not new_lst:
                    package_names.append(name)
                    del depend_dict[name]
                else:
                    depend_dict[name] = new_lst

        return package_names

    def _get_doc_title(self, info_module):
        """ Get the title from a package info.py file.
        """
        title = getattr(info_module,'__doc_title__',None)
        if title is not None:
            return title
        title = getattr(info_module,'__doc__',None)
        if title is not None:
            title = title.lstrip().split('\n',1)[0]
            return title
        return '* Not Available *'

    def _format_titles(self,titles):
        lengths = [len(name)-name.find('.')-1 for (name,title) in titles]
        max_length = max(lengths)
        lines = []
        for (name,title) in titles:
            name = name[name.find('.')+1:]
            w = max_length - len(name)
            lines.append('%s%s --- %s' % (name, w*' ', title))
        return '\n'.join(lines)

    def import_packages(self, packages=None):
        """
        Import packages that implement info.py.
        Return a list of documentation strings info.__doc__ of succesfully
        imported packages.
        """
        info_modules = self.get_info_modules(packages)
        package_names = self._sort_info_modules(info_modules)
        frame = self.frame

        titles = []

        for fullname in package_names:
            if fullname in self.imported_packages:
                continue
            package_name = fullname.split('.')[-1]
            info_module = info_modules[fullname]
            global_symbols = getattr(info_module,'global_symbols',[])
            postpone_import = getattr(info_module,'postpone_import',True)
        
            try:
                print 'Importing',package_name,'to',self.parent_name
                exec ('import '+package_name, frame.f_globals,frame.f_locals)
            except Exception,msg:
                print 'Failed to import',package_name
                print msg
                continue

            self.imported_packages.append(fullname)

            for symbol in global_symbols:
                try:
                    exec ('from '+package_name+' import '+symbol,
                          frame.f_globals,frame.f_locals)
                except Exception,msg:
                    print 'Failed to import',symbol,'from',package_name
                    print msg
                    continue

            titles.append((fullname,self._get_doc_title(info_module)))

            try:
                exec ('\n%s.test = ScipyTest(%s).test' \
                      % (package_name,package_name),
                      frame.f_globals,frame.f_locals)
            except Exception,msg:
                print 'Failed to set test function for',package_name
                print msg

        return self._format_titles(titles)
