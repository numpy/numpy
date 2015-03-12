""" Context manager for dealing with warnings
"""

from warnings import catch_warnings


class catch_warn_reset(catch_warnings):
    """ ``catch_warnings`` context manager that resets warning registry

    Warnings can be slippery, because, whenever a warning is triggered, Python
    adds a ``__warningregistry__`` member to the *calling* module.  This makes
    it impossible to retrigger the warning in this module, whatever you put in
    the warnings filters.  The ``catch_warn_reset`` decorator removes the
    ``__warningregistry__`` member as the context manager exits, making it
    possible to retrigger the warning.
    """
    def __init__(self, *args, **kwargs):
        self.modules = kwargs.pop('modules', [])
        self._warnreg_copies = {}
        super(catch_warn_reset, self).__init__(*args, **kwargs)

    def __enter__(self):
        for mod in self.modules:
            if hasattr(mod, '__warningregistry__'):
                mod_reg = mod.__warningregistry__
                self._warnreg_copies[mod] = mod_reg.copy()
                mod_reg.clear()
        return super(catch_warn_reset, self).__enter__()

    def __exit__(self, *exc_info):
        super(catch_warn_reset, self).__exit__(*exc_info)
        for mod in self.modules:
            if hasattr(mod, '__warningregistry__'):
                mod.__warningregistry__.clear()
            if mod in self._warnreg_copies:
                mod.__warningregistry__.update(self._warnreg_copies[mod])
