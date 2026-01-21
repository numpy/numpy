===============================
NumPy's Contributing guidelines
===============================

Welcome to the NumPy community! We're excited to have you here.
Whether you're new to open source or experienced, your contributions
help us grow.

Pull requests (PRs) are always welcome, but making a PR is just the
start. Please respond to comments and requests for changes to help move the process forward.
Skip asking for an issue to be assigned to you on GitHub—send in your PR, explain what you did and ask for a review. It makes collaboration and support much easier.
Please follow our
`Code of Conduct <https://numpy.org/code-of-conduct/>`__, which applies
to all interactions, including issues and PRs.

For more, please read https://www.numpy.org/devdocs/dev/index.html

Thank you for contributing, and happy coding!

.. _spin_tool:

Spin: NumPy’s developer tool
----------------------------

NumPy uses a command-line tool called ``spin`` to support common development
tasks such as building from source, running tests, building documentation, 
and managing other
developer workflows.

The ``spin`` tool provides a consistent interface for contributors working on
NumPy itself, wrapping multiple underlying tools and configurations into a
single command that follows NumPy’s development conventions.

Most contributors will interact with ``spin`` when running tests locally,
building the documentation, or preparing changes for review.
