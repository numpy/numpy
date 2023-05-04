#!/usr/bin/env python3
"""
refguide_check.py [OPTIONS] [-- ARGS]

- Check for a NumPy submodule whether the objects in its __all__ dict
  correspond to the objects included in the reference guide.
- Check docstring examples
- Check example blocks in RST files

Example of usage::

    $ python refguide_check.py optimize

Note that this is a helper script to be able to check if things are missing;
the output of this script does need to be checked manually.  In some cases
objects are left out of the refguide for a good reason (it's an alias of
another function, or deprecated, or ...)

Another use of this helper script is to check validity of code samples
in docstrings::

    $ python tools/refguide_check.py --doctests ma

or in RST-based documentations::

    $ python tools/refguide_check.py --rst doc/source

"""
import copy
import doctest
import inspect
import io
import os
import re
import shutil
import sys
import tempfile
import warnings
import docutils.core
from argparse import ArgumentParser
from contextlib import contextmanager, redirect_stderr
from doctest import NORMALIZE_WHITESPACE, ELLIPSIS, IGNORE_EXCEPTION_DETAIL

from docutils.parsers.rst import directives

import sphinx
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'doc', 'sphinxext'))
from numpydoc.docscrape_sphinx import get_doc_object

SKIPBLOCK = doctest.register_optionflag('SKIPBLOCK')

# Enable specific Sphinx directives
from sphinx.directives.other import SeeAlso, Only
directives.register_directive('seealso', SeeAlso)
directives.register_directive('only', Only)


BASE_MODULE = "numpy"

PUBLIC_SUBMODULES = [
    'core',
    'f2py',
    'linalg',
    'lib',
    'lib.recfunctions',
    'fft',
    'ma',
    'polynomial',
    'matrixlib',
    'random',
    'testing',
]

# Docs for these modules are included in the parent module
OTHER_MODULE_DOCS = {
    'fftpack.convolve': 'fftpack',
    'io.wavfile': 'io',
    'io.arff': 'io',
}

# these names are known to fail doctesting and we like to keep it that way
# e.g. sometimes pseudocode is acceptable etc
#
# Optionally, a subset of methods can be skipped by setting dict-values
# to a container of method-names
DOCTEST_SKIPDICT = {
    # cases where NumPy docstrings import things from SciPy:
    'numpy.lib.vectorize': None,
    'numpy.random.standard_gamma': None,
    'numpy.random.gamma': None,
    'numpy.random.vonmises': None,
    'numpy.random.power': None,
    'numpy.random.zipf': None,
    # cases where NumPy docstrings import things from other 3'rd party libs:
    'numpy.core.from_dlpack': None,
    # remote / local file IO with DataSource is problematic in doctest:
    'numpy.lib.DataSource': None,
    'numpy.lib.Repository': None,
}

# Skip non-numpy RST files, historical release notes
# Any single-directory exact match will skip the directory and all subdirs.
# Any exact match (like 'doc/release') will scan subdirs but skip files in
# the matched directory.
# Any filename will skip that file
RST_SKIPLIST = [
    'scipy-sphinx-theme',
    'sphinxext',
    'neps',
    'changelog',
    'doc/release',
    'doc/source/release',
    'doc/release/upcoming_changes',
    'c-info.ufunc-tutorial.rst',
    'c-info.python-as-glue.rst',
    'f2py.getting-started.rst',
    'f2py-examples.rst',
    'arrays.nditer.cython.rst',
    'how-to-verify-bug.rst',
    # See PR 17222, these should be fixed
    'basics.dispatch.rst',
    'basics.subclassing.rst',
    'basics.interoperability.rst',
    'misc.rst',
    'TESTS.rst'
]

# these names are not required to be present in ALL despite being in
# autosummary:: listing
REFGUIDE_ALL_SKIPLIST = [
    r'scipy\.sparse\.linalg',
    r'scipy\.spatial\.distance',
    r'scipy\.linalg\.blas\.[sdczi].*',
    r'scipy\.linalg\.lapack\.[sdczi].*',
]

# these names are not required to be in an autosummary:: listing
# despite being in ALL
REFGUIDE_AUTOSUMMARY_SKIPLIST = [
    # NOTE: should NumPy have a better match between autosummary
    # listings and __all__? For now, TR isn't convinced this is a
    # priority -- focus on just getting docstrings executed / correct
    r'numpy\.*',
]
# deprecated windows in scipy.signal namespace
for name in ('barthann', 'bartlett', 'blackmanharris', 'blackman', 'bohman',
             'boxcar', 'chebwin', 'cosine', 'exponential', 'flattop',
             'gaussian', 'general_gaussian', 'hamming', 'hann', 'hanning',
             'kaiser', 'nuttall', 'parzen', 'slepian', 'triang', 'tukey'):
    REFGUIDE_AUTOSUMMARY_SKIPLIST.append(r'scipy\.signal\.' + name)

HAVE_MATPLOTLIB = False


def short_path(path, cwd=None):
    """
    Return relative or absolute path name, whichever is shortest.

    Parameters
    ----------
    path : str or None
    cwd : str or None

    Returns
    -------
    str
        Relative path or absolute path based on current working directory
    """
    if not isinstance(path, str):
        return path
    if cwd is None:
        cwd = os.getcwd()
    abspath = os.path.abspath(path)
    relpath = os.path.relpath(path, cwd)
    if len(abspath) <= len(relpath):
        return abspath
    return relpath


def find_names(module, names_dict):
    """
    Finds the occurrences of function names, special directives like data
    and functions and scipy constants in the docstrings of `module`. The
    following patterns are searched for:

    * 3 spaces followed by function name, and maybe some spaces, some
      dashes, and an explanation; only function names listed in
      refguide are formatted like this (mostly, there may be some false
      positives
    * special directives, such as data and function
    * (scipy.constants only): quoted list

    The `names_dict` is updated by reference and accessible in calling method

    Parameters
    ----------
    module : ModuleType
        The module, whose docstrings is to be searched
    names_dict : dict
        Dictionary which contains module name as key and a set of found
        function names and directives as value

    Returns
    -------
    None
    """
    patterns = [
        r"^\s\s\s([a-z_0-9A-Z]+)(\s+-+.*)?$",
        r"^\.\. (?:data|function)::\s*([a-z_0-9A-Z]+)\s*$"
    ]

    if module.__name__ == 'scipy.constants':
        patterns += ["^``([a-z_0-9A-Z]+)``"]

    patterns = [re.compile(pattern) for pattern in patterns]
    module_name = module.__name__

    for line in module.__doc__.splitlines():
        res = re.search(r"^\s*\.\. (?:currentmodule|module):: ([a-z0-9A-Z_.]+)\s*$", line)
        if res:
            module_name = res.group(1)
            continue

        for pattern in patterns:
            res = re.match(pattern, line)
            if res is not None:
                name = res.group(1)
                entry = '.'.join([module_name, name])
                names_dict.setdefault(module_name, set()).add(name)
                break


def get_all_dict(module):
    """
    Return a copy of the __all__ dict with irrelevant items removed.

    Parameters
    ----------
    module : ModuleType
        The module whose __all__ dict has to be processed

    Returns
    -------
    deprecated : list
        List of callable and deprecated sub modules
    not_deprecated : list
        List of non callable or non deprecated sub modules
    others : list
        List of remaining types of sub modules
    """
    if hasattr(module, "__all__"):
        all_dict = copy.deepcopy(module.__all__)
    else:
        all_dict = copy.deepcopy(dir(module))
        all_dict = [name for name in all_dict
                    if not name.startswith("_")]
    for name in ['absolute_import', 'division', 'print_function']:
        try:
            all_dict.remove(name)
        except ValueError:
            pass
    if not all_dict:
        # Must be a pure documentation module
        all_dict.append('__doc__')

    # Modules are almost always private; real submodules need a separate
    # run of refguide_check.
    all_dict = [name for name in all_dict
                if not inspect.ismodule(getattr(module, name, None))]

    deprecated = []
    not_deprecated = []
    for name in all_dict:
        f = getattr(module, name, None)
        if callable(f) and is_deprecated(f):
            deprecated.append(name)
        else:
            not_deprecated.append(name)

    others = set(dir(module)).difference(set(deprecated)).difference(set(not_deprecated))

    return not_deprecated, deprecated, others


def compare(all_dict, others, names, module_name):
    """
    Return sets of objects from all_dict.
    Will return three sets:
     {in module_name.__all__},
     {in REFGUIDE*},
     and {missing from others}

    Parameters
    ----------
    all_dict : list
        List of non deprecated sub modules for module_name
    others : list
        List of sub modules for module_name
    names : set
        Set of function names or special directives present in
        docstring of module_name
    module_name : ModuleType

    Returns
    -------
    only_all : set
    only_ref : set
    missing : set
    """
    only_all = set()
    for name in all_dict:
        if name not in names:
            for pat in REFGUIDE_AUTOSUMMARY_SKIPLIST:
                if re.match(pat, module_name + '.' + name):
                    break
            else:
                only_all.add(name)

    only_ref = set()
    missing = set()
    for name in names:
        if name not in all_dict:
            for pat in REFGUIDE_ALL_SKIPLIST:
                if re.match(pat, module_name + '.' + name):
                    if name not in others:
                        missing.add(name)
                    break
            else:
                only_ref.add(name)

    return only_all, only_ref, missing


def is_deprecated(f):
    """
    Check if module `f` is deprecated

    Parameters
    ----------
    f : ModuleType

    Returns
    -------
    bool
    """
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("error")
        try:
            f(**{"not a kwarg":None})
        except DeprecationWarning:
            return True
        except Exception:
            pass
        return False


def check_items(all_dict, names, deprecated, others, module_name, dots=True):
    """
    Check that `all_dict` is consistent with the `names` in `module_name`
    For instance, that there are no deprecated or extra objects.

    Parameters
    ----------
    all_dict : list

    names : set

    deprecated : list

    others : list

    module_name : ModuleType

    dots : bool
        Whether to print a dot for each check

    Returns
    -------
    list
        List of [(name, success_flag, output)...]
    """
    num_all = len(all_dict)
    num_ref = len(names)

    output = ""

    output += "Non-deprecated objects in __all__: %i\n" % num_all
    output += "Objects in refguide: %i\n\n" % num_ref

    only_all, only_ref, missing = compare(all_dict, others, names, module_name)
    dep_in_ref = only_ref.intersection(deprecated)
    only_ref = only_ref.difference(deprecated)

    if len(dep_in_ref) > 0:
        output += "Deprecated objects in refguide::\n\n"
        for name in sorted(deprecated):
            output += "    " + name + "\n"

    if len(only_all) == len(only_ref) == len(missing) == 0:
        if dots:
            output_dot('.')
        return [(None, True, output)]
    else:
        if len(only_all) > 0:
            output += "ERROR: objects in %s.__all__ but not in refguide::\n\n" % module_name
            for name in sorted(only_all):
                output += "    " + name + "\n"

            output += "\nThis issue can be fixed by adding these objects to\n"
            output += "the function listing in __init__.py for this module\n"

        if len(only_ref) > 0:
            output += "ERROR: objects in refguide but not in %s.__all__::\n\n" % module_name
            for name in sorted(only_ref):
                output += "    " + name + "\n"

            output += "\nThis issue should likely be fixed by removing these objects\n"
            output += "from the function listing in __init__.py for this module\n"
            output += "or adding them to __all__.\n"

        if len(missing) > 0:
            output += "ERROR: missing objects::\n\n"
            for name in sorted(missing):
                output += "    " + name + "\n"

        if dots:
            output_dot('F')
        return [(None, False, output)]


def validate_rst_syntax(text, name, dots=True):
    """
    Validates the doc string in a snippet of documentation
    `text` from file `name`
    Parameters
    ----------
    text : str
        Docstring text
    name : str
        File name for which the doc string is to be validated
    dots : bool
        Whether to print a dot symbol for each check
    Returns
    -------
    (bool, str)
    """
    if text is None:
        if dots:
            output_dot('E')
        return False, "ERROR: %s: no documentation" % (name,)

    ok_unknown_items = set([
        'mod', 'doc', 'currentmodule', 'autosummary', 'data', 'attr',
        'obj', 'versionadded', 'versionchanged', 'module', 'class',
        'ref', 'func', 'toctree', 'moduleauthor', 'term', 'c:member',
        'sectionauthor', 'codeauthor', 'eq', 'doi', 'DOI', 'arXiv', 'arxiv'
    ])

    # Run through docutils
    error_stream = io.StringIO()

    def resolve(name, is_label=False):
        return ("http://foo", name)

    token = '<RST-VALIDATE-SYNTAX-CHECK>'

    docutils.core.publish_doctree(
        text, token,
        settings_overrides = dict(halt_level=5,
                                  traceback=True,
                                  default_reference_context='title-reference',
                                  default_role='emphasis',
                                  link_base='',
                                  resolve_name=resolve,
                                  stylesheet_path='',
                                  raw_enabled=0,
                                  file_insertion_enabled=0,
                                  warning_stream=error_stream))

    # Print errors, disregarding unimportant ones
    error_msg = error_stream.getvalue()
    errors = error_msg.split(token)
    success = True
    output = ""

    for error in errors:
        lines = error.splitlines()
        if not lines:
            continue

        m = re.match(r'.*Unknown (?:interpreted text role|directive type) "(.*)".*$', lines[0])
        if m:
            if m.group(1) in ok_unknown_items:
                continue

        m = re.match(r'.*Error in "math" directive:.*unknown option: "label"', " ".join(lines), re.S)
        if m:
            continue

        output += name + lines[0] + "::\n    " + "\n    ".join(lines[1:]).rstrip() + "\n"
        success = False

    if not success:
        output += "    " + "-"*72 + "\n"
        for lineno, line in enumerate(text.splitlines()):
            output += "    %-4d    %s\n" % (lineno+1, line)
        output += "    " + "-"*72 + "\n\n"

    if dots:
        output_dot('.' if success else 'F')
    return success, output


def output_dot(msg='.', stream=sys.stderr):
    stream.write(msg)
    stream.flush()


def check_rest(module, names, dots=True):
    """
    Check reStructuredText formatting of docstrings

    Parameters
    ----------
    module : ModuleType

    names : set

    Returns
    -------
    result : list
        List of [(module_name, success_flag, output),...]
    """

    try:
        skip_types = (dict, str, unicode, float, int)
    except NameError:
        # python 3
        skip_types = (dict, str, float, int)


    results = []

    if module.__name__[6:] not in OTHER_MODULE_DOCS:
        results += [(module.__name__,) +
                    validate_rst_syntax(inspect.getdoc(module),
                                        module.__name__, dots=dots)]

    for name in names:
        full_name = module.__name__ + '.' + name
        obj = getattr(module, name, None)

        if obj is None:
            results.append((full_name, False, "%s has no docstring" % (full_name,)))
            continue
        elif isinstance(obj, skip_types):
            continue

        if inspect.ismodule(obj):
            text = inspect.getdoc(obj)
        else:
            try:
                text = str(get_doc_object(obj))
            except Exception:
                import traceback
                results.append((full_name, False,
                                "Error in docstring format!\n" +
                                traceback.format_exc()))
                continue

        m = re.search("([\x00-\x09\x0b-\x1f])", text)
        if m:
            msg = ("Docstring contains a non-printable character %r! "
                   "Maybe forgot r\"\"\"?" % (m.group(1),))
            results.append((full_name, False, msg))
            continue

        try:
            src_file = short_path(inspect.getsourcefile(obj))
        except TypeError:
            src_file = None

        if src_file:
            file_full_name = src_file + ':' + full_name
        else:
            file_full_name = full_name

        results.append((full_name,) + validate_rst_syntax(text, file_full_name, dots=dots))

    return results


### Doctest helpers ####

# the namespace to run examples in
DEFAULT_NAMESPACE = {'np': np}

# the namespace to do checks in
CHECK_NAMESPACE = {
      'np': np,
      'numpy': np,
      'assert_allclose': np.testing.assert_allclose,
      'assert_equal': np.testing.assert_equal,
      # recognize numpy repr's
      'array': np.array,
      'matrix': np.matrix,
      'int64': np.int64,
      'uint64': np.uint64,
      'int8': np.int8,
      'int32': np.int32,
      'float32': np.float32,
      'float64': np.float64,
      'dtype': np.dtype,
      'nan': np.nan,
      'NaN': np.nan,
      'inf': np.inf,
      'Inf': np.inf,
      'StringIO': io.StringIO,
}


class DTRunner(doctest.DocTestRunner):
    """
    The doctest runner
    """
    DIVIDER = "\n"

    def __init__(self, item_name, checker=None, verbose=None, optionflags=0):
        self._item_name = item_name
        doctest.DocTestRunner.__init__(self, checker=checker, verbose=verbose,
                                       optionflags=optionflags)

    def _report_item_name(self, out, new_line=False):
        if self._item_name is not None:
            if new_line:
                out("\n")
            self._item_name = None

    def report_start(self, out, test, example):
        self._checker._source = example.source
        return doctest.DocTestRunner.report_start(self, out, test, example)

    def report_success(self, out, test, example, got):
        if self._verbose:
            self._report_item_name(out, new_line=True)
        return doctest.DocTestRunner.report_success(self, out, test, example, got)

    def report_unexpected_exception(self, out, test, example, exc_info):
        self._report_item_name(out)
        return doctest.DocTestRunner.report_unexpected_exception(
            self, out, test, example, exc_info)

    def report_failure(self, out, test, example, got):
        self._report_item_name(out)
        return doctest.DocTestRunner.report_failure(self, out, test,
                                                    example, got)

class Checker(doctest.OutputChecker):
    """
    Check the docstrings
    """
    obj_pattern = re.compile('at 0x[0-9a-fA-F]+>')
    vanilla = doctest.OutputChecker()
    rndm_markers = {'# random', '# Random', '#random', '#Random', "# may vary",
                    "# uninitialized", "#uninitialized"}
    stopwords = {'plt.', '.hist', '.show', '.ylim', '.subplot(',
                 'set_title', 'imshow', 'plt.show', '.axis(', '.plot(',
                 '.bar(', '.title', '.ylabel', '.xlabel', 'set_ylim', 'set_xlim',
                 '# reformatted', '.set_xlabel(', '.set_ylabel(', '.set_zlabel(',
                 '.set(xlim=', '.set(ylim=', '.set(xlabel=', '.set(ylabel='}

    def __init__(self, parse_namedtuples=True, ns=None, atol=1e-8, rtol=1e-2):
        self.parse_namedtuples = parse_namedtuples
        self.atol, self.rtol = atol, rtol
        if ns is None:
            self.ns = CHECK_NAMESPACE
        else:
            self.ns = ns

    def check_output(self, want, got, optionflags):
        # cut it short if they are equal
        if want == got:
            return True

        # skip stopwords in source
        if any(word in self._source for word in self.stopwords):
            return True

        # skip random stuff
        if any(word in want for word in self.rndm_markers):
            return True

        # skip function/object addresses
        if self.obj_pattern.search(got):
            return True

        # ignore comments (e.g. signal.freqresp)
        if want.lstrip().startswith("#"):
            return True

        # try the standard doctest
        try:
            if self.vanilla.check_output(want, got, optionflags):
                return True
        except Exception:
            pass

        # OK then, convert strings to objects
        try:
            a_want = eval(want, dict(self.ns))
            a_got = eval(got, dict(self.ns))
        except Exception:
            # Maybe we're printing a numpy array? This produces invalid python
            # code: `print(np.arange(3))` produces "[0 1 2]" w/o commas between
            # values. So, reinsert commas and retry.
            # TODO: handle (1) abberivation (`print(np.arange(10000))`), and
            #              (2) n-dim arrays with n > 1
            s_want = want.strip()
            s_got = got.strip()
            cond = (s_want.startswith("[") and s_want.endswith("]") and
                    s_got.startswith("[") and s_got.endswith("]"))
            if cond:
                s_want = ", ".join(s_want[1:-1].split())
                s_got = ", ".join(s_got[1:-1].split())
                return self.check_output(s_want, s_got, optionflags)

            if not self.parse_namedtuples:
                return False
            # suppose that "want"  is a tuple, and "got" is smth like
            # MoodResult(statistic=10, pvalue=0.1).
            # Then convert the latter to the tuple (10, 0.1),
            # and then compare the tuples.
            try:
                num = len(a_want)
                regex = (r'[\w\d_]+\(' +
                         ', '.join([r'[\w\d_]+=(.+)']*num) +
                         r'\)')
                grp = re.findall(regex, got.replace('\n', ' '))
                if len(grp) > 1:  # no more than one for now
                    return False
                # fold it back to a tuple
                got_again = '(' + ', '.join(grp[0]) + ')'
                return self.check_output(want, got_again, optionflags)
            except Exception:
                return False

        # ... and defer to numpy
        try:
            return self._do_check(a_want, a_got)
        except Exception:
            # heterog tuple, eg (1, np.array([1., 2.]))
           try:
                return all(self._do_check(w, g) for w, g in zip(a_want, a_got))
           except (TypeError, ValueError):
                return False

    def _do_check(self, want, got):
        # This should be done exactly as written to correctly handle all of
        # numpy-comparable objects, strings, and heterogeneous tuples
        try:
            if want == got:
                return True
        except Exception:
            pass
        return np.allclose(want, got, atol=self.atol, rtol=self.rtol)


def _run_doctests(tests, full_name, verbose, doctest_warnings):
    """
    Run modified doctests for the set of `tests`.

    Parameters
    ----------
    tests : list

    full_name : str

    verbose : bool
    doctest_warnings : bool

    Returns
    -------
    tuple(bool, list)
        Tuple of (success, output)
    """
    flags = NORMALIZE_WHITESPACE | ELLIPSIS
    runner = DTRunner(full_name, checker=Checker(), optionflags=flags,
                      verbose=verbose)

    output = io.StringIO(newline='')
    success = True

    # Redirect stderr to the stdout or output
    tmp_stderr = sys.stdout if doctest_warnings else output

    @contextmanager
    def temp_cwd():
        cwd = os.getcwd()
        tmpdir = tempfile.mkdtemp()
        try:
            os.chdir(tmpdir)
            yield tmpdir
        finally:
            os.chdir(cwd)
            shutil.rmtree(tmpdir)

    # Run tests, trying to restore global state afterward
    cwd = os.getcwd()
    with np.errstate(), np.printoptions(), temp_cwd() as tmpdir, \
            redirect_stderr(tmp_stderr):
        # try to ensure random seed is NOT reproducible
        np.random.seed(None)

        ns = {}
        for t in tests:
            # We broke the tests up into chunks to try to avoid PSEUDOCODE
            # This has the unfortunate side effect of restarting the global
            # namespace for each test chunk, so variables will be "lost" after
            # a chunk. Chain the globals to avoid this
            t.globs.update(ns)
            t.filename = short_path(t.filename, cwd)
            # Process our options
            if any([SKIPBLOCK in ex.options for ex in t.examples]):
                continue
            fails, successes = runner.run(t, out=output.write, clear_globs=False)
            if fails > 0:
                success = False
            ns = t.globs

    output.seek(0)
    return success, output.read()


def check_doctests(module, verbose, ns=None,
                   dots=True, doctest_warnings=False):
    """
    Check code in docstrings of the module's public symbols.

    Parameters
    ----------
    module : ModuleType
        Name of module
    verbose : bool
        Should the result be verbose
    ns : dict
        Name space of module
    dots : bool

    doctest_warnings : bool

    Returns
    -------
    results : list
        List of [(item_name, success_flag, output), ...]
    """
    if ns is None:
        ns = dict(DEFAULT_NAMESPACE)

    # Loop over non-deprecated items
    results = []

    for name in get_all_dict(module)[0]:
        full_name = module.__name__ + '.' + name

        if full_name in DOCTEST_SKIPDICT:
            skip_methods = DOCTEST_SKIPDICT[full_name]
            if skip_methods is None:
                continue
        else:
            skip_methods = None

        try:
            obj = getattr(module, name)
        except AttributeError:
            import traceback
            results.append((full_name, False,
                            "Missing item!\n" +
                            traceback.format_exc()))
            continue

        finder = doctest.DocTestFinder()
        try:
            tests = finder.find(obj, name, globs=dict(ns))
        except Exception:
            import traceback
            results.append((full_name, False,
                            "Failed to get doctests!\n" +
                            traceback.format_exc()))
            continue

        if skip_methods is not None:
            tests = [i for i in tests if
                     i.name.partition(".")[2] not in skip_methods]

        success, output = _run_doctests(tests, full_name, verbose,
                                        doctest_warnings)

        if dots:
            output_dot('.' if success else 'F')

        results.append((full_name, success, output))

        if HAVE_MATPLOTLIB:
            import matplotlib.pyplot as plt
            plt.close('all')

    return results


def check_doctests_testfile(fname, verbose, ns=None,
                   dots=True, doctest_warnings=False):
    """
    Check code in a text file.

    Mimic `check_doctests` above, differing mostly in test discovery.
    (which is borrowed from stdlib's doctest.testfile here,
     https://github.com/python-git/python/blob/master/Lib/doctest.py)

    Parameters
    ----------
    fname : str
        File name
    verbose : bool

    ns : dict
        Name space

    dots : bool

    doctest_warnings : bool

    Returns
    -------
    list
        List of [(item_name, success_flag, output), ...]

    Notes
    -----

    refguide can be signalled to skip testing code by adding
    ``#doctest: +SKIP`` to the end of the line. If the output varies or is
    random, add ``# may vary`` or ``# random`` to the comment. for example

    >>> plt.plot(...)  # doctest: +SKIP
    >>> random.randint(0,10)
    5 # random

    We also try to weed out pseudocode:
    * We maintain a list of exceptions which signal pseudocode,
    * We split the text file into "blocks" of code separated by empty lines
      and/or intervening text.
    * If a block contains a marker, the whole block is then assumed to be
      pseudocode. It is then not being doctested.

    The rationale is that typically, the text looks like this:

    blah
    <BLANKLINE>
    >>> from numpy import some_module   # pseudocode!
    >>> func = some_module.some_function
    >>> func(42)                  # still pseudocode
    146
    <BLANKLINE>
    blah
    <BLANKLINE>
    >>> 2 + 3        # real code, doctest it
    5

    """
    if ns is None:
        ns = CHECK_NAMESPACE
    results = []

    _, short_name = os.path.split(fname)
    if short_name in DOCTEST_SKIPDICT:
        return results

    full_name = fname
    with open(fname, encoding='utf-8') as f:
        text = f.read()

    PSEUDOCODE = set(['some_function', 'some_module', 'import example',
                      'ctypes.CDLL',     # likely need compiling, skip it
                      'integrate.nquad(func,'  # ctypes integrate tutotial
    ])

    # split the text into "blocks" and try to detect and omit pseudocode blocks.
    parser = doctest.DocTestParser()
    good_parts = []
    base_line_no = 0
    for part in text.split('\n\n'):
        try:
            tests = parser.get_doctest(part, ns, fname, fname, base_line_no)
        except ValueError as e:
            if e.args[0].startswith('line '):
                # fix line number since `parser.get_doctest` does not increment
                # the reported line number by base_line_no in the error message
                parts = e.args[0].split()
                parts[1] = str(int(parts[1]) + base_line_no)
                e.args = (' '.join(parts),) + e.args[1:]
            raise
        if any(word in ex.source for word in PSEUDOCODE
                                 for ex in tests.examples):
            # omit it
            pass
        else:
            # `part` looks like a good code, let's doctest it
            good_parts.append((part, base_line_no))
        base_line_no += part.count('\n') + 2

    # Reassemble the good bits and doctest them:
    tests = []
    for good_text, line_no in good_parts:
        tests.append(parser.get_doctest(good_text, ns, fname, fname, line_no))
    success, output = _run_doctests(tests, full_name, verbose,
                                    doctest_warnings)

    if dots:
        output_dot('.' if success else 'F')

    results.append((full_name, success, output))

    if HAVE_MATPLOTLIB:
        import matplotlib.pyplot as plt
        plt.close('all')

    return results


def iter_included_files(base_path, verbose=0, suffixes=('.rst',)):
    """
    Generator function to walk `base_path` and its subdirectories, skipping
    files or directories in RST_SKIPLIST, and yield each file with a suffix in
    `suffixes`

    Parameters
    ----------
    base_path : str
        Base path of the directory to be processed
    verbose : int

    suffixes : tuple

    Yields
    ------
    path
        Path of the directory and its sub directories
    """
    if os.path.exists(base_path) and os.path.isfile(base_path):
        yield base_path
    for dir_name, subdirs, files in os.walk(base_path, topdown=True):
        if dir_name in RST_SKIPLIST:
            if verbose > 0:
                sys.stderr.write('skipping files in %s' % dir_name)
            files = []
        for p in RST_SKIPLIST:
            if p in subdirs:
                if verbose > 0:
                    sys.stderr.write('skipping %s and subdirs' % p)
                subdirs.remove(p)
        for f in files:
            if (os.path.splitext(f)[1] in suffixes and
                    f not in RST_SKIPLIST):
                yield os.path.join(dir_name, f)


def check_documentation(base_path, results, args, dots):
    """
    Check examples in any *.rst located inside `base_path`.
    Add the output to `results`.

    See Also
    --------
    check_doctests_testfile
    """
    for filename in iter_included_files(base_path, args.verbose):
        if dots:
            sys.stderr.write(filename + ' ')
            sys.stderr.flush()

        tut_results = check_doctests_testfile(
            filename,
            (args.verbose >= 2), dots=dots,
            doctest_warnings=args.doctest_warnings)

        # stub out a "module" which is needed when reporting the result
        def scratch():
            pass
        scratch.__name__ = filename
        results.append((scratch, tut_results))
        if dots:
            sys.stderr.write('\n')
            sys.stderr.flush()


def init_matplotlib():
    """
    Check feasibility of matplotlib initialization.
    """
    global HAVE_MATPLOTLIB

    try:
        import matplotlib
        matplotlib.use('Agg')
        HAVE_MATPLOTLIB = True
    except ImportError:
        HAVE_MATPLOTLIB = False


def main(argv):
    """
    Validates the docstrings of all the pre decided set of
    modules for errors and docstring standards.
    """
    parser = ArgumentParser(usage=__doc__.lstrip())
    parser.add_argument("module_names", metavar="SUBMODULES", default=[],
                        nargs='*', help="Submodules to check (default: all public)")
    parser.add_argument("--doctests", action="store_true",
                        help="Run also doctests on ")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    parser.add_argument("--doctest-warnings", action="store_true",
                        help="Enforce warning checking for doctests")
    parser.add_argument("--rst", nargs='?', const='doc', default=None,
                        help=("Run also examples from *rst files "
                              "discovered walking the directory(s) specified, "
                              "defaults to 'doc'"))
    args = parser.parse_args(argv)

    modules = []
    names_dict = {}

    if not args.module_names:
        args.module_names = list(PUBLIC_SUBMODULES)

    os.environ['SCIPY_PIL_IMAGE_VIEWER'] = 'true'

    module_names = list(args.module_names)
    for name in module_names:
        if name in OTHER_MODULE_DOCS:
            name = OTHER_MODULE_DOCS[name]
            if name not in module_names:
                module_names.append(name)

    dots = True
    success = True
    results = []
    errormsgs = []


    if args.doctests or args.rst:
        init_matplotlib()

    for submodule_name in module_names:
        prefix = BASE_MODULE + '.'
        if not submodule_name.startswith(prefix):
            module_name = prefix + submodule_name
        else:
            module_name = submodule_name

        __import__(module_name)
        module = sys.modules[module_name]

        if submodule_name not in OTHER_MODULE_DOCS:
            find_names(module, names_dict)

        if submodule_name in args.module_names:
            modules.append(module)


    if args.doctests or not args.rst:
        print("Running checks for %d modules:" % (len(modules),))
        for module in modules:
            if dots:
                sys.stderr.write(module.__name__ + ' ')
                sys.stderr.flush()

            all_dict, deprecated, others = get_all_dict(module)
            names = names_dict.get(module.__name__, set())

            mod_results = []
            mod_results += check_items(all_dict, names, deprecated, others,
                                       module.__name__)
            mod_results += check_rest(module, set(names).difference(deprecated),
                                      dots=dots)
            if args.doctests:
                mod_results += check_doctests(module, (args.verbose >= 2), dots=dots,
                                              doctest_warnings=args.doctest_warnings)

            for v in mod_results:
                assert isinstance(v, tuple), v

            results.append((module, mod_results))

            if dots:
                sys.stderr.write('\n')
                sys.stderr.flush()

    if args.rst:
        base_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
        rst_path = os.path.relpath(os.path.join(base_dir, args.rst))
        if os.path.exists(rst_path):
            print('\nChecking files in %s:' % rst_path)
            check_documentation(rst_path, results, args, dots)
        else:
            sys.stderr.write(f'\ninvalid --rst argument "{args.rst}"')
            errormsgs.append('invalid directory argument to --rst')
        if dots:
            sys.stderr.write("\n")
            sys.stderr.flush()

    # Report results
    for module, mod_results in results:
        success = all(x[1] for x in mod_results)
        if not success:
            errormsgs.append(f'failed checking {module.__name__}')

        if success and args.verbose == 0:
            continue

        print("")
        print("=" * len(module.__name__))
        print(module.__name__)
        print("=" * len(module.__name__))
        print("")

        for name, success, output in mod_results:
            if name is None:
                if not success or args.verbose >= 1:
                    print(output.strip())
                    print("")
            elif not success or (args.verbose >= 2 and output.strip()):
                print(name)
                print("-"*len(name))
                print("")
                print(output.strip())
                print("")

    if len(errormsgs) == 0:
        print("\nOK: all checks passed!")
        sys.exit(0)
    else:
        print('\nERROR: ', '\n        '.join(errormsgs))
        sys.exit(1)


if __name__ == '__main__':
    main(argv=sys.argv[1:])
