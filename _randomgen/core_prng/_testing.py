"""
Shim for NumPy's suppress_warnings
"""


try:
    from numpy.testing import suppress_warnings
except ImportError:

    # The following two classes are copied from python 2.6 warnings module
    # (context manager)
    class WarningMessage(object):

        """
        Holds the result of a single showwarning() call.
        Deprecated in 1.8.0
        Notes
        -----
        `WarningMessage` is copied from the Python 2.6 warnings module,
        so it can be used in NumPy with older Python versions.
        """

        _WARNING_DETAILS = ("message", "category", "filename", "lineno",
                            "file", "line")

        def __init__(self, message, category, filename, lineno, file=None,
                     line=None):
            local_values = locals()
            for attr in self._WARNING_DETAILS:
                setattr(self, attr, local_values[attr])
            if category:
                self._category_name = category.__name__
            else:
                self._category_name = None

        def __str__(self):
            return ("{message : %r, category : %r, "
                    "filename : %r, lineno : %s, "
                    "line : %r}" % (self.message, self._category_name,
                                    self.filename, self.lineno, self.line))

    import re
    import warnings
    from functools import wraps

    class suppress_warnings(object):
        """
        Context manager and decorator doing much the same as
        ``warnings.catch_warnings``.
        However, it also provides a filter mechanism to work around
        http://bugs.python.org/issue4180.
        This bug causes Python before 3.4 to not reliably show warnings again
        after they have been ignored once (even within catch_warnings). It
        means that no "ignore" filter can be used easily, since following
        tests might need to see the warning. Additionally it allows easier
        specificity for testing warnings and can be nested.
        Parameters
        ----------
        forwarding_rule : str, optional
            One of "always", "once", "module", or "location". Analogous to
            the usual warnings module filter mode, it is useful to reduce
            noise mostly on the outmost level. Unsuppressed and unrecorded
            warnings will be forwarded based on this rule. Defaults to
            "always". "location" is equivalent to the warnings "default", match
            by exact location the warning warning originated from.
        Notes
        -----
        Filters added inside the context manager will be discarded again
        when leaving it. Upon entering all filters defined outside a
        context will be applied automatically.
        When a recording filter is added, matching warnings are stored in the
        ``log`` attribute as well as in the list returned by ``record``.
        If filters are added and the ``module`` keyword is given, the
        warning registry of this module will additionally be cleared when
        applying it, entering the context, or exiting it. This could cause
        warnings to appear a second time after leaving the context if they
        were configured to be printed once (default) and were already
        printed before the context was entered.
        Nesting this context manager will work as expected when the
        forwarding rule is "always" (default). Unfiltered and unrecorded
        warnings will be passed out and be matched by the outer level.
        On the outmost level they will be printed (or caught by another
        warnings context). The forwarding rule argument can modify this
        behaviour.
        Like ``catch_warnings`` this context manager is not threadsafe.
        Examples
        --------
        >>> with suppress_warnings() as sup:
        ...     sup.filter(DeprecationWarning, "Some text")
        ...     sup.filter(module=np.ma.core)
        ...     log = sup.record(FutureWarning, "Does this occur?")
        ...     command_giving_warnings()
        ...     # The FutureWarning was given once, the filtered warnings were
        ...     # ignored. All other warnings abide outside settings (may be
        ...     # printed/error)
        ...     assert_(len(log) == 1)
        ...     assert_(len(sup.log) == 1)  # also stored in log attribute
        Or as a decorator:
        >>> sup = suppress_warnings()
        >>> sup.filter(module=np.ma.core)  # module must match exact
        >>> @sup
        >>> def some_function():
        ...     # do something which causes a warning in np.ma.core
        ...     pass
        """
        def __init__(self, forwarding_rule="always"):
            self._entered = False

            # Suppressions are instance or defined inside one with block:
            self._suppressions = []

            if forwarding_rule not in {"always", "module", "once", "location"}:
                raise ValueError("unsupported forwarding rule.")
            self._forwarding_rule = forwarding_rule

        def _clear_registries(self):
            if hasattr(warnings, "_filters_mutated"):
                # clearing the registry should not be necessary on new pythons,
                # instead the filters should be mutated.
                warnings._filters_mutated()
                return
            # Simply clear the registry, this should normally be harmless,
            # note that on new pythons it would be invalidated anyway.
            for module in self._tmp_modules:
                if hasattr(module, "__warningregistry__"):
                    module.__warningregistry__.clear()

        def _filter(self, category=Warning, message="", module=None,
                    record=False):
            if record:
                record = []  # The log where to store warnings
            else:
                record = None
            if self._entered:
                if module is None:
                    warnings.filterwarnings(
                        "always", category=category, message=message)
                else:
                    module_regex = module.__name__.replace('.', '\.') + '$'
                    warnings.filterwarnings(
                        "always", category=category, message=message,
                        module=module_regex)
                    self._tmp_modules.add(module)
                    self._clear_registries()

                self._tmp_suppressions.append(
                    (category, message, re.compile(message, re.I), module,
                     record))
            else:
                self._suppressions.append(
                    (category, message, re.compile(message, re.I), module,
                     record))

            return record

        def filter(self, category=Warning, message="", module=None):
            """
            Add a new suppressing filter or apply it if the state is entered.
            Parameters
            ----------
            category : class, optional
                Warning class to filter
            message : string, optional
                Regular expression matching the warning message.
            module : module, optional
                Module to filter for. Note that the module (and its file)
                must match exactly and cannot be a submodule. This may make
                it unreliable for external modules.
            Notes
            -----
            When added within a context, filters are only added inside
            the context and will be forgotten when the context is exited.
            """
            self._filter(category=category, message=message, module=module,
                         record=False)

        def record(self, category=Warning, message="", module=None):
            """
            Append a new recording filter or apply it if the state is entered.
            All warnings matching will be appended to the ``log`` attribute.
            Parameters
            ----------
            category : class, optional
                Warning class to filter
            message : string, optional
                Regular expression matching the warning message.
            module : module, optional
                Module to filter for. Note that the module (and its file)
                must match exactly and cannot be a submodule. This may make
                it unreliable for external modules.
            Returns
            -------
            log : list
                A list which will be filled with all matched warnings.
            Notes
            -----
            When added within a context, filters are only added inside
            the context and will be forgotten when the context is exited.
            """
            return self._filter(category=category, message=message,
                                module=module, record=True)

        def __enter__(self):
            if self._entered:
                raise RuntimeError("cannot enter suppress_warnings twice.")

            self._orig_show = warnings.showwarning
            if hasattr(warnings, "_showwarnmsg"):
                self._orig_showmsg = warnings._showwarnmsg
            self._filters = warnings.filters
            warnings.filters = self._filters[:]

            self._entered = True
            self._tmp_suppressions = []
            self._tmp_modules = set()
            self._forwarded = set()

            self.log = []  # reset global log (no need to keep same list)

            for cat, mess, _, mod, log in self._suppressions:
                if log is not None:
                    del log[:]  # clear the log
                if mod is None:
                    warnings.filterwarnings(
                        "always", category=cat, message=mess)
                else:
                    module_regex = mod.__name__.replace('.', '\.') + '$'
                    warnings.filterwarnings(
                        "always", category=cat, message=mess,
                        module=module_regex)
                    self._tmp_modules.add(mod)
            warnings.showwarning = self._showwarning
            if hasattr(warnings, "_showwarnmsg"):
                warnings._showwarnmsg = self._showwarnmsg
            self._clear_registries()

            return self

        def __exit__(self, *exc_info):
            warnings.showwarning = self._orig_show
            if hasattr(warnings, "_showwarnmsg"):
                warnings._showwarnmsg = self._orig_showmsg
            warnings.filters = self._filters
            self._clear_registries()
            self._entered = False
            del self._orig_show
            del self._filters

        def _showwarnmsg(self, msg):
            self._showwarning(msg.message, msg.category, msg.filename,
                              msg.lineno, msg.file, msg.line, use_warnmsg=msg)

        def _showwarning(self, message, category, filename, lineno,
                         *args, **kwargs):
            use_warnmsg = kwargs.pop("use_warnmsg", None)
            for cat, _, pattern, mod, rec in (
                    self._suppressions + self._tmp_suppressions)[::-1]:
                if (issubclass(category, cat) and
                        pattern.match(message.args[0]) is not None):
                    if mod is None:
                        # Message and category match, recorded or ignored
                        if rec is not None:
                            msg = WarningMessage(message, category, filename,
                                                 lineno, **kwargs)
                            self.log.append(msg)
                            rec.append(msg)
                        return
                    # Use startswith, because warnings strips the c or o from
                    # .pyc/.pyo files.
                    elif mod.__file__.startswith(filename):
                        # The message and module (filename) match
                        if rec is not None:
                            msg = WarningMessage(message, category, filename,
                                                 lineno, **kwargs)
                            self.log.append(msg)
                            rec.append(msg)
                        return

            # There is no filter in place, so pass to the outside handler
            # unless we should only pass it once
            if self._forwarding_rule == "always":
                if use_warnmsg is None:
                    self._orig_show(message, category, filename, lineno,
                                    *args, **kwargs)
                else:
                    self._orig_showmsg(use_warnmsg)
                return

            if self._forwarding_rule == "once":
                signature = (message.args, category)
            elif self._forwarding_rule == "module":
                signature = (message.args, category, filename)
            elif self._forwarding_rule == "location":
                signature = (message.args, category, filename, lineno)

            if signature in self._forwarded:
                return
            self._forwarded.add(signature)
            if use_warnmsg is None:
                self._orig_show(message, category, filename, lineno, *args,
                                **kwargs)
            else:
                self._orig_showmsg(use_warnmsg)

        def __call__(self, func):
            """
            Function decorator to apply certain suppressions to a whole
            function.
            """
            @wraps(func)
            def new_func(*args, **kwargs):
                with self:
                    return func(*args, **kwargs)

            return new_func
