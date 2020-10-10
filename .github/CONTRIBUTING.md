# Contributing to numpy

## Reporting issues

When reporting issues please include as much detail as possible about your
operating system, numpy version and python version. Whenever possible, please
also include a brief, self-contained code example that demonstrates the problem.

If you are reporting a segfault please include a GDB traceback, which you can
generate by following
[these instructions.](https://github.com/numpy/numpy/blob/master/doc/source/dev/development_environment.rst#debugging)

## Contributing code

Thanks for your interest in contributing code to numpy!

+ If this is your first time contributing to a project on GitHub, please read
through our
[guide to contributing to numpy](https://numpy.org/devdocs/dev/index.html)
+ If you have contributed to other projects on GitHub you can go straight to our
[development workflow](https://numpy.org/devdocs/dev/development_workflow.html)

Either way, please be sure to follow our
[convention for commit messages](https://numpy.org/devdocs/dev/development_workflow.html#writing-the-commit-message).

If you are writing new C code, please follow the style described in
``doc/C_STYLE_GUIDE``.

Suggested ways to work on your development version (compile and run
the tests without interfering with system packages) are described in
``doc/source/dev/development_environment.rst``.

### A note on feature enhancements/API changes

If you are interested in adding a new feature to NumPy, consider
submitting your feature proposal to the [mailing list][mail], 
which is the preferred forum for discussing new features and
API changes.

[mail]: https://mail.python.org/mailman/listinfo/numpy-discussion
