import inspect
import types
import sys, os

# NOTE:  pydoc defines a help function which works simliarly to this
#  except it uses a pager to take over the screen.

# combine name and arguments and split to multiple lines of
#  width characters.  End lines on a comma and begin argument list
#  indented with the rest of the arguments.
def split_line(name, arguments, width):
    firstwidth = len(name)
    k = firstwidth
    newstr = name
    sepstr = ", "
    arglist = arguments.split(sepstr)
    for argument in arglist:
        if k == firstwidth:
            addstr = ""
        else:
            addstr = sepstr
        k = k + len(argument) + len(addstr)
        if k > width:
            k = firstwidth + 1 + len(argument)
            newstr = newstr + ",\n" + " "*(firstwidth+2) + argument
        else:
            newstr = newstr + addstr + argument
    return newstr

_namedict = None
_dictlist = None

# Traverse all module directories underneath globals to see if something is defined
def makenamedict():
    import scipy
    thedict = {'scipy':scipy.__dict__}
    dictlist = ['scipy']
    totraverse = [scipy.__dict__]
    while 1:
        if len(totraverse) == 0:
            break
        thisdict = totraverse.pop(0)
        for x in thisdict.keys():
            if isinstance(thisdict[x],types.ModuleType):
                modname = thisdict[x].__name__
                if modname not in dictlist:
                    moddict = thisdict[x].__dict__
                    dictlist.append(modname)
                    totraverse.append(moddict)
                    thedict[modname] = moddict
    return thedict, dictlist


def help(object=None,maxwidth=76,output=sys.stdout,):
    """Get help information for a function, class, or module.
    
       Example:
          >>> from scipy import * 
          >>> help(polyval)
          polyval(p, x)
        
            Evaluate the polymnomial p at x.
            
            Description:        
                If p is of length N, this function returns the value:
                p[0]*(x**N-1) + p[1]*(x**N-2) + ... + p[N-2]*x + p[N-1]
    """
    global _namedict, _dictlist
    if object is None:        
        help(help)
    elif isinstance(object, types.StringType):
        if _namedict is None:
            _namedict, _dictlist = makenamedict()
        numfound = 0
        objlist = []
        for namestr in _dictlist:
            try:
                obj = _namedict[namestr][object]
                if id(obj) in objlist:
                    print >> output, "\n     *** Repeat reference found in %s *** " % namestr
                else:
                    objlist.append(id(obj))
                    print >> output, "     *** Found in %s ***" % namestr
                    help(obj)
                    print >> output, "-"*maxwidth
                numfound += 1
            except KeyError:
                pass
        if numfound == 0:
            print >> output, "Help for %s not found." % object
        else:
            print >> output, "\n     *** Total of %d references found. ***" % numfound
                    
    elif inspect.isfunction(object):
        name = object.func_name
        arguments = apply(inspect.formatargspec, inspect.getargspec(object))

        if len(name+arguments) > maxwidth:
            argstr = split_line(name, arguments, maxwidth)
        else:
            argstr = name + arguments

        print >> output, " " + argstr + "\n"
        print >> output, inspect.getdoc(object)

    elif inspect.isclass(object):  
        name = object.__name__
        if hasattr(object, '__init__'):
            arguments = apply(inspect.formatargspec, inspect.getargspec(object.__init__.im_func))        
            arglist = arguments.split(', ')
            if len(arglist) > 1:
                arglist[1] = "("+arglist[1]
                arguments = ", ".join(arglist[1:])
            else:
                arguments = "()"
        else:
            arguments = "()"

        if len(name+arguments) > maxwidth:
            argstr = split_line(name, arguments, maxwidth)
        else:
            argstr = name + arguments

        print >> output, " " + argstr + "\n"
        doc1 = inspect.getdoc(object) 
        if doc1 is None:
            if hasattr(object,'__init__'):
                print >> output, inspect.getdoc(object.__init__)
        else:
            print >> output, inspect.getdoc(object)

    elif type(object) is types.InstanceType: ## check for __call__ method
        print >> output, "Instance of class: ", object.__class__.__name__
        print >> output
        if hasattr(object, '__call__'):
            arguments = apply(inspect.formatargspec, inspect.getargspec(object.__call__.im_func))
            arglist = arguments.split(', ')
            if len(arglist) > 1:
                arglist[1] = "("+arglist[1]
                arguments = ", ".join(arglist[1:])
            else:
                arguments = "()"

            name = "<name>"
            if len(name+arguments) > maxwidth:
                argstr = split_line(name, arguments, maxwidth)
            else:
                argstr = name + arguments

            print >> output, " " + argstr + "\n"
            doc = inspect.getdoc(object.__call__)
            if doc is not None:
                print >> output, inspect.getdoc(object.__call__)
            print >> output, inspect.getdoc(object)
            
        else:
            print >> output, inspect.getdoc(object)
                                            
    elif inspect.ismethod(object):
        name = object.__name__
        arguments = apply(inspect.formatargspec, inspect.getargspec(object.im_func))
        arglist = arguments.split(', ')
        if len(arglist) > 1:
            arglist[1] = "("+arglist[1]
            arguments = ", ".join(arglist[1:])
        else:
            arguments = "()"

        if len(name+arguments) > maxwidth:
            argstr = split_line(name, arguments, maxwidth)
        else:
            argstr = name + arguments

        print >> output, " " + argstr + "\n"
        print >> output, inspect.getdoc(object)
                
    elif hasattr(object, '__doc__'):
        print >> output, inspect.getdoc(object)
        

def source(object, output=sys.stdout):
    if inspect.isroutine(object):
        print >> output,  "In file: %s\n" % inspect.getsourcefile(object)
        print >> output,  inspect.getsource(object)
    else:
        print >> output,  "Not available for this object."
