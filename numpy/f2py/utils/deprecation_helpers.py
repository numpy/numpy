"""
These helpers allow for backwards compatible ways to change module structures.
The main reason this is required is because throwing warnings on module imports
is not viable otherwise. The design is **strongly inspired** by the Apache
licensed `Cirq project's similar machinery`_.

.. _`Cirq project's similar machinery`: https://github.com/quantumlib/Cirq/pull/3917
"""

from types import ModuleType
import sys, importlib
from importlib.abc import MetaPathFinder, Loader

def _validate_deadline(deadline, old_module_name,
                       new_module_name):
    """Validates the deadline for the deprecation

    Parameters
    ----------
    deadline : str
        Version deadline for the deprecation.
    old_module_name : str
        Name of the older module.
    new_module_name : str
        Name of the new module.

    Returns
    -------
    None
    """
    import numpy as np
    import re
    major_version = deadline.split('.')[0]
    minor_version = deadline.split('.')[1]
    np_major_version = np.__version__.split('.')[0]
    np_minor_version = np.__version__.split('.')[1]
    DEADLINE_REGEX = r"^(\d)+\.(\d)+$"
    assert re.match(DEADLINE_REGEX, deadline),\
           "deadline should match X.Y"
    assert (float(f"{major_version}.{minor_version}") >
            float(f"{np_major_version}.{np_minor_version}")), \
            f"Deadline was {major_version}.{minor_version} "\
            f"for replacing {old_module_name} with {new_module_name},"\
            f" got {np_major_version}.{np_minor_version}"

def deprecated_submodule(*, new_module_name, old_parent, old_child, deadline):
    """Creates a recursively defined deprecated module reference
    For any `new_name`, an alias is created in the module cache. By recursively
    checking imported submodules and generating maliases for them as well, it
    supports backwards compatible calls like `from numpy.f2py import crackfortran`

    .. note::

       `new_module_name` will be executed to populate the module cache

    Parameters
    ----------
    new_module_name : str
        Absolute import for the new module.
    old_parent : str
        Name of the older import parent.
    old_child: str
        Submodule being relocated.
    deadline: str
        Version deadline for the deprecation.

    Returns
    -------
    None
        Will raise otherwise.
    """
    old_module_name = f"{old_parent}.{old_child}"
    _validate_deadline(deadline, old_module_name, new_module_name)
    try:
        new_module = importlib.import_module(new_module_name)
        _setup_deprecated_submodule_attribute(
            new_module_name, old_parent, old_child, deadline, new_module
        )
    except ImportError as ex:
        raise ModuleNotFoundError(f"{old_module_name} does not map to a new module")
    finder = DeprecatedModuleFinder(new_module_name, old_module_name, deadline)
    sys.meta_path.append(finder)

def _setup_deprecated_submodule_attribute(new_module_name, old_parent,
                                          old_child, deadline, new_module):
    parent_module = sys.modules[old_parent]
    setattr(parent_module, old_child, new_module)
    class Wrapped(ModuleType):
        __dict__ = parent_module.__dict__
        __spec__ = parent_module.__spec__
        def __getattr__(self, name):
            if name == old_child:
                _module_warn(
                    f"{old_parent}.{old_child}", new_module_name, deadline
                )
            return getattr(parent_module, name)
    wrapped_parent_module = Wrapped(parent_module.__name__, parent_module.__doc__)
    if '.' in old_parent:
        grandpa_name, parent_tail = old_parent.rsplit('.', 1)
        grandpa_module = sys.modules[grandpa_name]
        setattr(grandpa_module, parent_tail, wrapped_parent_module)
    sys.modules[old_parent] = wrapped_parent_module

def _module_warn(old_module_name, new_module_name, deadline):
    import warnings
    msg = f"""
    {old_module_name} was used but is deprecated.
    it will be removed in numpy {deadline}.
    Use {new_module_name} instead.
    """
    stack_level = 3
    warnings.warn(
        msg,
        DeprecationWarning,
        stacklevel=stack_level,
    )


class DeprecatedModuleFinder(MetaPathFinder):
    """A module finder to handle deprecated module references.
    Sends a deprecation warning when a deprecated module is asked to be found.
    Used as a wrapper around existing MetaPathFinder instances.

    Parameters
    ----------
    new_module_name: str
        The new module's fully qualified name.
    old_module_name: str
        The deprecated module's fully qualified name.
    deadline: str
        The deprecation deadline. Will raise beyond this.
    """

    def __init__(
        self,
        new_module_name,
        old_module_name,
        deadline,
    ):
        """An aliasing module finder that uses existing module finders to find a python
        module spec and intercept the execution of matching modules.
        """
        self.new_module_name = new_module_name
        self.old_module_name = old_module_name
        self.deadline = deadline

    def find_spec(self, fullname, path, target):
        """Finds the specification of a module.
        This is an implementation of the importlib.abc.MetaPathFinder.find_spec method.
        See https://docs.python.org/3/library/importlib.html#importlib.abc.MetaPathFinder.

        Parameters
        ----------
        fullname: str
            name of the module.
        path: str (optional)
            if present, this is the parent module's submodule search path.
        target: str (optional)
            if present, used to guess the spec. Passed to the wrapped finder
            unused.
        """
        if fullname != self.old_module_name and not fullname.startswith(self.old_module_name + "."):
            return None

        # warn for deprecation
        _module_warn(self.old_module_name, self.new_module_name, self.deadline)

        new_fullname = self.new_module_name + fullname[len(self.old_module_name) :]

        # use normal import mechanism for the new module specs
        spec = importlib.util.find_spec(new_fullname)

        # if the spec exists, return the DeprecatedModuleLoader that will do the loading as well
        # as set the alias(es) in sys.modules as necessary
        if spec is not None:
            # change back the name to the deprecated module name
            spec.name = fullname
            # some loaders do a check to ensure the module's name is the same
            # as the loader was created for
            if getattr(spec.loader, "name", None) == new_fullname:
                setattr(spec.loader, "name", fullname)
            spec.loader = DeprecatedModuleLoader(spec.loader, fullname, new_fullname)
        return spec

class DeprecatedModuleLoader(Loader):
    """A Loader for deprecated modules.
    It wraps an existing Loader instance, to which it delegates the loading. On top of that
    it ensures that the sys.modules cache has both the deprecated module's name and the
    new module's name pointing to the same exact ModuleType instance.

    Parameters
    ----------
    loader: str
        the loader to be wrapped
    old_module_name: str
        the deprecated module's fully qualified name
    new_module_name: str
        the new module's fully qualified name
    """

    def __init__(self, loader, old_module_name, new_module_name):
        """A module loader that uses an existing module loader and intercepts
        the execution of a module.
        """
        self.loader = loader
        if hasattr(loader, 'exec_module'):
            self.exec_module = self._wrap_exec_module(loader.exec_module)
        if hasattr(loader, 'create_module'):
            self.create_module = loader.create_module
        self.old_module_name = old_module_name
        self.new_module_name = new_module_name

    def module_repr(self, module: ModuleType):
        return self.loader.module_repr(module)

    def _wrap_exec_module(self, method):
        def exec_module(module):
            assert module.__name__ == self.old_module_name, (
                f"DeprecatedModuleLoader for {self.old_module_name} was asked to "
                f"load {module.__name__}"
            )
            # check for new_module whether it was loaded
            if self.new_module_name in sys.modules:
                # found it - no need to load the module again
                sys.modules[self.old_module_name] = sys.modules[self.new_module_name]
                return

            # now we know we have to initialize the module
            sys.modules[self.old_module_name] = module
            sys.modules[self.new_module_name] = module

            try:
                return method(module)
            except BaseException:
                # if there's an error, we atomically remove both
                del sys.modules[self.new_module_name]
                del sys.modules[self.old_module_name]
                raise

        return exec_module
