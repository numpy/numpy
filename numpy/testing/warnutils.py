""" Context manager for dealing with warnings
"""

from warnings import catch_warnings


class catch_clear_warnings(catch_warnings):
    """ ``catch_warnings`` context manager that resets warning registry

    Warnings can be slippery, because, whenever a warning is triggered, Python
    adds a ``__warningregistry__`` member to the *calling* module.  This makes
    it impossible to retrigger the warning in this module, whatever you put in
    the warnings filters.  The ``catch_clear_warnings`` context manager accepts
    a sequence of `modules` as a keyword argument to its constructor and:

    * stores and removes any ``__warningregistry__`` entries in given `modules`
      on entry;
    * resets ``__warningregistry__`` to its previous state on exit.

    This makes it possible to trigger any warning afresh inside the context
    manager without disturbing the state of warnings outside.
    """
    class_modules = ()

    def __init__(self, *args, **kwargs):
        modules = kwargs.pop('modules', set())
        self.modules = set(modules).union(self.class_modules)
        self._warnreg_copies = {}
        super(catch_clear_warnings, self).__init__(*args, **kwargs)

    def __enter__(self):
        for mod in self.modules:
            if hasattr(mod, '__warningregistry__'):
                mod_reg = mod.__warningregistry__
                self._warnreg_copies[mod] = mod_reg.copy()
                mod_reg.clear()
        return super(catch_clear_warnings, self).__enter__()

    def __exit__(self, *exc_info):
        super(catch_clear_warnings, self).__exit__(*exc_info)
        for mod in self.modules:
            if hasattr(mod, '__warningregistry__'):
                mod.__warningregistry__.clear()
            if mod in self._warnreg_copies:
                mod.__warningregistry__.update(self._warnreg_copies[mod])
