#!/usr/bin/env python3
"""
refguide_check.py [OPTIONS] [-- ARGS]

- Check for a NumPy submodule whether the objects in its __all__ dict
  correspond to the objects included in the reference guide.
- Check docstring examples
- Check example blocks in RST files

Example of usage::

    $ python tools/refguide_check.py

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
import inspect
import io
import os
import re
import sys
import warnings
import docutils.core
from argparse import ArgumentParser

from docutils.parsers.rst import directives


sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'doc', 'sphinxext'))
from numpydoc.docscrape_sphinx import get_doc_object

# Enable specific Sphinx directives
from sphinx.directives.other import SeeAlso, Only
directives.register_directive('seealso', SeeAlso)
directives.register_directive('only', Only)


BASE_MODULE = "numpy"

PUBLIC_SUBMODULES = [
    "f2py",
    "linalg",
    "lib",
    "lib.format",
    "lib.mixins",
    "lib.recfunctions",
    "lib.scimath",
    "lib.stride_tricks",
    "lib.npyio",
    "lib.introspect",
    "lib.array_utils",
    "fft",
    "char",
    "rec",
    "ma",
    "ma.extras",
    "ma.mrecords",
    "polynomial",
    "polynomial.chebyshev",
    "polynomial.hermite",
    "polynomial.hermite_e",
    "polynomial.laguerre",
    "polynomial.legendre",
    "polynomial.polynomial",
    "matrixlib",
    "random",
    "strings",
    "testing",
]

# Docs for these modules are included in the parent module
OTHER_MODULE_DOCS = {
    'fftpack.convolve': 'fftpack',
    'io.wavfile': 'io',
    'io.arff': 'io',
}


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
        res = re.search(r"^\s*\.\. (?:currentmodule|module):: ([a-z0-9A-Z_.]+)\s*$",
                        line)
        if res:
            module_name = res.group(1)
            continue

        for pattern in patterns:
            res = re.match(pattern, line)
            if res is not None:
                name = res.group(1)
                entry = f'{module_name}.{name}'
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

    others = set(dir(module)).difference(set(deprecated)).difference(set(not_deprecated))  # noqa: E501

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
            f(**{"not a kwarg": None})
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

    output += f"Non-deprecated objects in __all__: {num_all}\n"
    output += f"Objects in refguide: {num_ref}\n\n"

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
            output += f"ERROR: objects in {module_name}.__all__ but not in refguide::\n\n"  # noqa: E501
            for name in sorted(only_all):
                output += "    " + name + "\n"

            output += "\nThis issue can be fixed by adding these objects to\n"
            output += "the function listing in __init__.py for this module\n"

        if len(only_ref) > 0:
            output += f"ERROR: objects in refguide but not in {module_name}.__all__::\n\n"  # noqa: E501
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
        return False, f"ERROR: {name}: no documentation"

    ok_unknown_items = {
        'mod', 'doc', 'currentmodule', 'autosummary', 'data', 'attr',
        'obj', 'versionadded', 'versionchanged', 'module', 'class',
        'ref', 'func', 'toctree', 'moduleauthor', 'term', 'c:member',
        'sectionauthor', 'codeauthor', 'eq', 'doi', 'DOI', 'arXiv', 'arxiv'
    }

    # Run through docutils
    error_stream = io.StringIO()

    def resolve(name, is_label=False):
        return ("http://foo", name)

    token = '<RST-VALIDATE-SYNTAX-CHECK>'

    docutils.core.publish_doctree(
        text, token,
        settings_overrides = {'halt_level': 5,
                                  'traceback': True,
                                  'default_reference_context': 'title-reference',
                                  'default_role': 'emphasis',
                                  'link_base': '',
                                  'resolve_name': resolve,
                                  'stylesheet_path': '',
                                  'raw_enabled': 0,
                                  'file_insertion_enabled': 0,
                                  'warning_stream': error_stream})

    # Print errors, disregarding unimportant ones
    error_msg = error_stream.getvalue()
    errors = error_msg.split(token)
    success = True
    output = ""

    for error in errors:
        lines = error.splitlines()
        if not lines:
            continue

        m = re.match(r'.*Unknown (?:interpreted text role|directive type) "(.*)".*$', lines[0])  # noqa: E501
        if m:
            if m.group(1) in ok_unknown_items:
                continue

        m = re.match(r'.*Error in "math" directive:.*unknown option: "label"', " ".join(lines), re.S)  # noqa: E501
        if m:
            continue

        output += name + lines[0] + "::\n    " + "\n    ".join(lines[1:]).rstrip() + "\n"  # noqa: E501
        success = False

    if not success:
        output += "    " + "-" * 72 + "\n"
        for lineno, line in enumerate(text.splitlines()):
            output += "    %-4d    %s\n" % (lineno + 1, line)
        output += "    " + "-" * 72 + "\n\n"

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
            results.append((full_name, False, f"{full_name} has no docstring"))
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

        m = re.search("([\x00-\x09\x0b-\x1f])", text)  # noqa: RUF039
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

        results.append((full_name,) +
                       validate_rst_syntax(text, file_full_name, dots=dots))

    return results


def main(argv):
    """
    Validates the docstrings of all the pre decided set of
    modules for errors and docstring standards.
    """
    parser = ArgumentParser(usage=__doc__.lstrip())
    parser.add_argument("module_names", metavar="SUBMODULES", default=[],
                        nargs='*', help="Submodules to check (default: all public)")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args(argv)

    modules = []
    names_dict = {}

    if not args.module_names:
        args.module_names = list(PUBLIC_SUBMODULES) + [BASE_MODULE]

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

    for submodule_name in module_names:
        prefix = BASE_MODULE + '.'
        if not (
            submodule_name.startswith(prefix) or
            submodule_name == BASE_MODULE
        ):
            module_name = prefix + submodule_name
        else:
            module_name = submodule_name

        __import__(module_name)
        module = sys.modules[module_name]

        if submodule_name not in OTHER_MODULE_DOCS:
            find_names(module, names_dict)

        if submodule_name in args.module_names:
            modules.append(module)

    if modules:
        print(f"Running checks for {len(modules)} modules:")
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

            for v in mod_results:
                assert isinstance(v, tuple), v

            results.append((module, mod_results))

            if dots:
                sys.stderr.write('\n')
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
                print("-" * len(name))
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
