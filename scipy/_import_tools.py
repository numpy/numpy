
import os
import sys
import imp
from glob import glob

def import_packages():
    """ Import packages in the current directory that implement
    info.py. See DEVELOPERS.txt for more info.
    """
    frame = sys._getframe(1)
    parent_name = eval('__name__',frame.f_globals,frame.f_locals)
    parent_path = eval('__path__[0]',frame.f_globals,frame.f_locals)
    info_modules = {}
    depend_dict = {}
    for info_file in glob(os.path.join(parent_path,'*','info.py')):
         package_name = os.path.basename(os.path.dirname(info_file))
         fullname = parent_name +'.'+ package_name
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
    
    for fullname in package_names:
        package_name = fullname.split('.')[-1]
        info_module = info_modules[fullname]
        global_symbols = getattr(info_module,'global_symbols',[])
        postpone_import = getattr(info_module,'postpone_import',True)
        
        try:
            print 'Importing',package_name,'to',parent_name
            exec ('import '+package_name, frame.f_globals,frame.f_locals)
        except Exception,msg:
            print 'Failed to import',package_name
            print msg
            continue

        for symbol in global_symbols:
            try:
                exec ('from '+package_name+' import '+symbol,
                      frame.f_globals,frame.f_locals)
            except Exception,msg:
                print 'Failed to import',symbol,'from',package_name
                print msg
                continue

        try:
            exec ('\n%s.test = ScipyTest(%s).test' % (package_name,package_name),
                  frame.f_globals,frame.f_locals)
        except Exception,msg:
            print 'Failed to set test function for',package_name
            print msg        
