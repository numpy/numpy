import sys
import os
import logging
import re

from pathlib import Path
from typing import Any, Dict, List, Tuple
from argparse import Namespace

# Distutil dependencies
from numpy.distutils.misc_util import dict_append
from numpy.distutils.system_info import get_info
from numpy.distutils.core import setup, Extension

from . import crackfortran
from . import capi_maps
from . import rules
from . import auxfuncs
from . import cfuncs
from . import cb_rules

from .utils import get_f2py_dir

outmess = auxfuncs.outmess

logger = logging.getLogger("f2py_cli")
logger.setLevel(logging.WARNING)

_f2py_module_name_match = re.compile(r'\s*python\s*module\s*(?P<name>[\w_]+)',
                                     re.I).match
_f2py_user_module_name_match = re.compile(r'\s*python\s*module\s*(?P<name>[\w_]*?'
                                          r'__user__[\w_]*)', re.I).match

def check_fortran(fname: str):
    """Function which checks <fortran files>

    This is meant as a sanity check, but will not raise an error, just a
    warning.  It is called with ``type``

    Parameters
    ----------
    fname : str
        The name of the file

    Returns
    -------
    pathlib.Path
        This is the string as a path, irrespective of the suffix
    """
    fpname = Path(fname)
    if fpname.suffix.lower() not in [".f90", ".f", ".f77"]:
        logger.warning(
            """Does not look like a standard fortran file ending in *.f90, *.f or
            *.f77, continuing against better judgement"""
        )
    return fpname


def check_dir(dname: str):
    """Function which checks the build directory

    This is meant to ensure no odd directories are passed, it will fail if a
    file is passed. Creates directory if not present.

    Parameters
    ----------
    dname : str
        The name of the directory, by default it will be a temporary one

    Returns
    -------
    pathlib.Path
        This is the string as a path
    """
    if dname:
        dpname = Path(dname)
        dpname.mkdir(parents=True, exist_ok=True)
        return dpname


def check_dccomp(opt: str):
    """Function which checks for an np.distutils compliant c compiler

    Meant to enforce sanity checks, note that this just checks against distutils.show_compilers()

    Parameters
    ----------
    opt: str
        The compiler name, must be a distutils option

    Returns
    -------
    str
        This is the option as a string
    """
    cchoices = ["bcpp", "cygwin", "mingw32", "msvc", "unix"]
    if opt in cchoices:
        return opt
    else:
        raise RuntimeError(f"{opt} is not an distutils supported C compiler, choose from {cchoices}")


def check_npfcomp(opt: str):
    """Function which checks for an np.distutils compliant fortran compiler

    Meant to enforce sanity checks

    Parameters
    ----------
    opt: str
        The compiler name, must be a np.distutils option

    Returns
    -------
    str
        This is the option as a string
    """
    from numpy.distutils import fcompiler
    fcompiler.load_all_fcompiler_classes()
    fchoices = list(fcompiler.fcompiler_class.keys())
    if opt in fchoices[0]:
        return opt
    else:
        raise RuntimeError(f"{opt} is not an np.distutils supported compiler, choose from {fchoices}")


def _set_additional_headers(headers: List[str]):
    for header in headers:
        cfuncs.outneeds['userincludes'].append(header[1:-1])
        cfuncs.userincludes[header[1:-1]] = f"#include {header}"

def _set_crackfortran(crackfortran_setts):
    crackfortran.reset_global_f2py_vars()
    crackfortran.f77modulename = crackfortran_setts["module"]
    crackfortran.include_paths[:] = crackfortran_setts["include_paths"]
    crackfortran.debug = crackfortran_setts["debug"]
    crackfortran.verbose = crackfortran_setts["verbose"]
    crackfortran.skipfuncs = crackfortran_setts["skipfuncs"]
    crackfortran.onlyfuncs = crackfortran_setts["onlyfuncs"]
    crackfortran.dolowercase = crackfortran_setts["do-lower"]

def _set_rules(rules_setts):
    rules.options = rules_setts

def _set_capi_maps(capi_maps_setts):
    capi_maps.load_f2cmap_file(capi_maps_setts["f2cmap"])
    _set_additional_headers(capi_maps_setts["headers"])

def _set_auxfuncs(aux_funcs_setts):
    auxfuncs.options = {'verbose': aux_funcs_setts['verbose']}
    auxfuncs.debugoptions = aux_funcs_setts["debug"]
    auxfuncs.wrapfuncs = aux_funcs_setts['wrapfuncs']

def _dict_append(d_out, d_in):
    for (k, v) in d_in.items():
        if k not in d_out:
            d_out[k] = []
        if isinstance(v, list):
            d_out[k] = d_out[k] + v
        else:
            d_out[k].append(v)


def _buildmodules(lst):
    cfuncs.buildcfuncs()
    outmess('Building modules...\n')
    modules, mnames, isusedby = [], [], {}
    for item in lst:
        if '__user__' in item['name']:
            cb_rules.buildcallbacks(item)
        else:
            if 'use' in item:
                for u in item['use'].keys():
                    if u not in isusedby:
                        isusedby[u] = []
                    isusedby[u].append(item['name'])
            modules.append(item)
            mnames.append(item['name'])
    ret = {}
    for module, name in zip(modules, mnames):
        if name in isusedby:
            outmess('\tSkipping module "%s" which is used by %s.\n' % (
                name, ','.join('"%s"' % s for s in isusedby[name])))
        else:
            um = []
            if 'use' in module:
                for u in module['use'].keys():
                    if u in isusedby and u in mnames:
                        um.append(modules[mnames.index(u)])
                    else:
                        outmess(
                            f'\tModule "{name}" uses nonexisting "{u}" '
                            'which will be ignored.\n')
            ret[name] = {}
            _dict_append(ret[name], rules.buildmodule(module, um))
    return ret


def _generate_signature(postlist, sign_file: str):
    outmess(f"Saving signatures to file {sign_file}" + "\n")
    pyf = crackfortran.crack2fortran(postlist)
    if sign_file in {"-", "stdout"}:
        sys.stdout.write(pyf)
    else:
        with open(sign_file, "w") as f:
            f.write(pyf)

def _check_postlist(postlist, sign_file: str, verbose: bool):
    isusedby = {}
    for plist in postlist:
        if 'use' in plist:
            for u in plist['use'].keys():
                if u not in isusedby:
                    isusedby[u] = []
                isusedby[u].append(plist['name'])
    for plist in postlist:
        if plist['block'] == 'python module' and '__user__' in plist['name'] and plist['name'] in isusedby:
            # if not quiet:
            outmess(
                f'Skipping Makefile build for module "{plist["name"]}" '
                'which is used by {}\n'.format(
                    ','.join(f'"{s}"' for s in isusedby[plist['name']])))
    if(sign_file):
        if verbose:
            outmess(
                'Stopping. Edit the signature file and then run f2py on the signature file: ')
            outmess('%s %s\n' %
                    (os.path.basename(sys.argv[0]), sign_file))
        return
    for plist in postlist:
        if plist['block'] != 'python module':
            # if 'python module' not in options:
                outmess(
                    'Tip: If your original code is Fortran source then you must use -m option.\n')
            # raise TypeError('All blocks must be python module blocks but got %s' % (
            #     repr(postlist[i]['block'])))

def _callcrackfortran(files: List[Path], module_name: str):
    postlist = crackfortran.crackfortran([str(file) for file in files])
    for mod in postlist:
        mod["coutput"] = f"{module_name}module.c"
        mod["f2py_wrapper_output"] = f"{module_name}-f2pywrappers.f"
    return postlist

def _set_dependencies_dist(ext_args, link_resource):
    for dep in link_resource:
        info = get_info(dep)
        if not info:
                outmess('No %s resources found in system'
                        ' (try `f2py --help-link`)\n' % (repr(n)))
        dict_append(ext_args, **link_resource)

def get_f2py_modulename(source):
    name = None
    _f2py_module_name_match = re.compile(r'\s*python\s*module\s*(?P<name>[\w_]+)',
                                        re.I).match
    _f2py_user_module_name_match = re.compile(r'\s*python\s*module\s*(?P<name>[\w_]*?'
                                            r'__user__[\w_]*)', re.I).match
    with open(source) as f:
        for line in f:
            if m := _f2py_module_name_match(line):
                if _f2py_user_module_name_match(line): # skip *__user__* names
                    continue
                name = m.group('name')
                break
    return name

def wrapper_settings(rules_setts, crackfortran_setts, capi_maps_setts, auxfuncs_setts):
    _set_rules(rules_setts)
    _set_crackfortran(crackfortran_setts)
    _set_capi_maps(capi_maps_setts)
    _set_auxfuncs(auxfuncs_setts)

def generate_files(files: List[Path], module_name: str, sign_file: str):
    postlist = _callcrackfortran(files, module_name)
    _check_postlist(postlist, sign_file)
    if(sign_file):
        _generate_signature(postlist, sign_file)
    if(module_name):
        _buildmodules(postlist)

def compile_dist(ext_args, link_resources, build_dir, fc_flags, flib_flags, quiet_build, remove_build_dir=True):
    _set_dependencies_dist(ext_args, link_resources)
    f2py_dir = get_f2py_dir()
    ext = Extension(**ext_args)
    f2py_build_flags = ['--quiet'] if quiet_build else ['--verbose']
    f2py_build_flags.extend( ['build', '--build-temp', str(build_dir),
                              '--build-base', str(build_dir),
                              '--build-platlib', '.',
                              '--disable-optimization'])
    if fc_flags:
        f2py_build_flags.extend(['config_fc'] + fc_flags)
    if flib_flags:
        f2py_build_flags.extend(['build_ext'] + flib_flags)
    setup(ext_modules=[ext], script_name=f2py_dir, script_args=f2py_build_flags)
    if(remove_build_dir and os.path.exists(build_dir)):
        import shutil
        outmess('Removing build directory %s\n' % (build_dir))
        shutil.rmtree(build_dir)

def segregate_files(files: List[str]) -> Tuple[List[Path], List[Path], List[Path], List[Path]]:
	"""
	Segregate files into three groups:
	* .f files
	* .o files
	* others
	"""
	f77_ext = ('.f', '.for', '.ftn', '.f77')
	f90_ext = ('.f90', '.f95', '.f03', '.f08')
	pyf_ext = ('.pyf',)
	out_ext = ('.o', '.out', '.so', '.a')

	f77_files = []
	f90_files = []
	out_files = []
	pyf_files = []
	other_files = []

	for f in files:
		f = Path(f)
		ext = os.path.splitext(f)[1]
		if ext in f77_ext:
			f77_files.append(f)
		elif ext in f90_ext:
			f90_files.append(f)
		elif ext in out_ext:
			out_files.append(f)
		elif ext in pyf_ext:
			pyf_files.append(f)
		else:
			other_files.append(f)

	return f77_files, f90_files, pyf_files, out_files, other_files