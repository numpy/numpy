# These classes implement a doctest runner plugin for nose.
# Because this module imports nose directly, it should not
# be used except by nosetester.py to avoid a general NumPy
# dependency on nose.

import os
import doctest

from nose.plugins import doctests as npd
from nose.plugins.base import Plugin
from nose.util import src, tolist
import numpy
from nosetester import get_package_name
import inspect

_doctest_ignore = ['generate_numpy_api.py', 'scons_support.py',
                   'setupscons.py', 'setup.py']

# All the classes in this module begin with 'numpy' to clearly distinguish them
# from the plethora of very similar names from nose/unittest/doctest


#-----------------------------------------------------------------------------
# Modified version of the one in the stdlib, that fixes a python bug (doctests
# not found in extension modules, http://bugs.python.org/issue3158)
class numpyDocTestFinder(doctest.DocTestFinder):

    def _from_module(self, module, object):
        """
        Return true if the given object is defined in the given
        module.
        """
        if module is None:
            #print '_fm C1'  # dbg
            return True
        elif inspect.isfunction(object):
            #print '_fm C2'  # dbg
            return module.__dict__ is object.func_globals
        elif inspect.isbuiltin(object):
            #print '_fm C2-1'  # dbg
            return module.__name__ == object.__module__
        elif inspect.isclass(object):
            #print '_fm C3'  # dbg
            return module.__name__ == object.__module__
        elif inspect.ismethod(object):
            # This one may be a bug in cython that fails to correctly set the
            # __module__ attribute of methods, but since the same error is easy
            # to make by extension code writers, having this safety in place
            # isn't such a bad idea
            #print '_fm C3-1'  # dbg
            return module.__name__ == object.im_class.__module__
        elif inspect.getmodule(object) is not None:
            #print '_fm C4'  # dbg
            #print 'C4 mod',module,'obj',object # dbg
            return module is inspect.getmodule(object)
        elif hasattr(object, '__module__'):
            #print '_fm C5'  # dbg
            return module.__name__ == object.__module__
        elif isinstance(object, property):
            #print '_fm C6'  # dbg
            return True # [XX] no way not be sure.
        else:
            raise ValueError("object must be a class or function")



    def _find(self, tests, obj, name, module, source_lines, globs, seen):
        """
        Find tests for the given object and any contained objects, and
        add them to `tests`.
        """

        doctest.DocTestFinder._find(self,tests, obj, name, module,
                                    source_lines, globs, seen)

        # Below we re-run pieces of the above method with manual modifications,
        # because the original code is buggy and fails to correctly identify
        # doctests in extension modules.

        # Local shorthands
        from inspect import isroutine, isclass, ismodule

        # Look for tests in a module's contained objects.
        if inspect.ismodule(obj) and self._recurse:
            for valname, val in obj.__dict__.items():
                valname1 = '%s.%s' % (name, valname)
                if ( (isroutine(val) or isclass(val))
                     and self._from_module(module, val) ):

                    self._find(tests, val, valname1, module, source_lines,
                               globs, seen)


        # Look for tests in a class's contained objects.
        if inspect.isclass(obj) and self._recurse:
            #print 'RECURSE into class:',obj  # dbg
            for valname, val in obj.__dict__.items():
                #valname1 = '%s.%s' % (name, valname)  # dbg
                #print 'N',name,'VN:',valname,'val:',str(val)[:77] # dbg
                # Special handling for staticmethod/classmethod.
                if isinstance(val, staticmethod):
                    val = getattr(obj, valname)
                if isinstance(val, classmethod):
                    val = getattr(obj, valname).im_func

                # Recurse to methods, properties, and nested classes.
                if ((inspect.isfunction(val) or inspect.isclass(val) or
                     inspect.ismethod(val) or
                      isinstance(val, property)) and
                      self._from_module(module, val)):
                    valname = '%s.%s' % (name, valname)
                    self._find(tests, val, valname, module, source_lines,
                               globs, seen)


class numpyDocTestCase(npd.DocTestCase):
    """Proxy for DocTestCase: provides an address() method that
    returns the correct address for the doctest case. Otherwise
    acts as a proxy to the test case. To provide hints for address(),
    an obj may also be passed -- this will be used as the test object
    for purposes of determining the test address, if it is provided.
    """

    # doctests loaded via find(obj) omit the module name
    # so we need to override id, __repr__ and shortDescription
    # bonus: this will squash a 2.3 vs 2.4 incompatiblity
    def id(self):
        name = self._dt_test.name
        filename = self._dt_test.filename
        if filename is not None:
            pk = getpackage(filename)
            if pk is not None and not name.startswith(pk):
                name = "%s.%s" % (pk, name)
        return name


# second-chance checker; if the default comparison doesn't 
# pass, then see if the expected output string contains flags that
# tell us to ignore the output
class numpyOutputChecker(doctest.OutputChecker):
    def check_output(self, want, got, optionflags):
        ret = doctest.OutputChecker.check_output(self, want, got, 
                                                 optionflags)
        if not ret:
            if "#random" in want:
                return True

        return ret


# Subclass nose.plugins.doctests.DocTestCase to work around a bug in 
# its constructor that blocks non-default arguments from being passed
# down into doctest.DocTestCase
class numpyDocTestCase(npd.DocTestCase):
    def __init__(self, test, optionflags=0, setUp=None, tearDown=None,
                 checker=None, obj=None, result_var='_'):
        self._result_var = result_var
        self._nose_obj = obj
        doctest.DocTestCase.__init__(self, test, 
                                     optionflags=optionflags,
                                     setUp=setUp, tearDown=tearDown, 
                                     checker=checker)


print_state = numpy.get_printoptions()        

class numpyDoctest(npd.Doctest):
    name = 'numpydoctest'   # call nosetests with --with-numpydoctest
    enabled = True

    def options(self, parser, env=os.environ):
        Plugin.options(self, parser, env)

    def configure(self, options, config):
        Plugin.configure(self, options, config)
        self.doctest_tests = True
        self.extension = tolist(options.doctestExtension)
        self.finder = numpyDocTestFinder()
        self.parser = doctest.DocTestParser()

    # Turns on whitespace normalization, set a minimal execution context
    # for doctests, implement a "#random" directive to allow executing a
    # command while ignoring its output.
    def loadTestsFromModule(self, module):
        if not self.matches(module.__name__):
            npd.log.debug("Doctest doesn't want module %s", module)
            return
        try:
            tests = self.finder.find(module)
        except AttributeError:
            # nose allows module.__test__ = False; doctest does not and 
            # throws AttributeError
            return
        if not tests:
            return
        tests.sort()
        module_file = src(module.__file__)
        for test in tests:
            if not test.examples:
                continue
            if not test.filename:
                test.filename = module_file

            pkg_name = get_package_name(os.path.dirname(test.filename))

            # Each doctest should execute in an environment equivalent to
            # starting Python and executing "import numpy as np", and,
            # for SciPy packages, an additional import of the local 
            # package (so that scipy.linalg.basic.py's doctests have an
            # implicit "from scipy import linalg" as well.
            #
            # Note: __file__ allows the doctest in NoseTester to run
            # without producing an error
            test.globs = {'__builtins__':__builtins__,
                          '__file__':'__main__', 
                          '__name__':'__main__', 
                          'np':numpy}
            
            # add appropriate scipy import for SciPy tests
            if 'scipy' in pkg_name:
                p = pkg_name.split('.')
                p1 = '.'.join(p[:-1])
                p2 = p[-1]
                test.globs[p2] = __import__(pkg_name, fromlist=[p2])
                    
                print 'additional import for %s: from %s import %s' % (test.filename, p1, p2)
                print '    (%s): %r' % (pkg_name, test.globs[p2])

            # always use whitespace and ellipsis options
            optionflags = doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS

            yield numpyDocTestCase(test, 
                                   optionflags=optionflags,
                                   checker=numpyOutputChecker())


    # Add an afterContext method to nose.plugins.doctests.Doctest in order
    # to restore print options to the original state after each doctest
    def afterContext(self):
        numpy.set_printoptions(**print_state)


    # Implement a wantFile method so that we can ignore NumPy-specific 
    # build files that shouldn't be searched for tests
    def wantFile(self, file):
        bn = os.path.basename(file)
        if bn in _doctest_ignore:
            return False
        return npd.Doctest.wantFile(self, file)
