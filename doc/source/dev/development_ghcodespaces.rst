.. _development_ghcodespaces:


Using GitHub Codespaces for NumPy development
=============================================

This section of the documentation will guide you through:

*  using GitHub Codespaces for your NumPy development environment
*  creating a personal fork of the NumPy repository on GitHub
*  a quick tour of GitHub Codespaces and VSCode desktop application
*  working on the NumPy documentation in GitHub Codespaces

GitHub Codespaces
-----------------

`GitHub Codespaces`_ is a service that provides cloud based 
development environments so that you don't have to install anything
on your local machine or worry about configuration.

What is a codespace?
--------------------

A codespace is an instance of Codespaces - and thus a development environment
that is hosted in the cloud. Each codespace runs on a virtual machine hosted by
GitHub. You can choose the type of machine you want to use, depending on the
resources you need. Various types of machine are available, starting with a
2-core processor, 4 GB of RAM, and 32 GB of storage.  You can connect to a
codespace from your browser, from Visual Studio Code, from the JetBrains
Gateway application, or by using GitHub CLI.

Forking the NumPy repository
----------------------------

The best way to work on the NumPy codebase as a contributor is by making a fork
of the repository first.

#. Browse to the `NumPy repository on GitHub`_ and `create your own fork`_.
#. Browse to your fork. Your fork will have a URL like 
   https://github.com/inessapawson/numpy, except with your GitHub username in place of ``inessapawson``.
     
Starting GitHub Codespaces
--------------------------

You can create a codespace from the green "<> Code" button on the repository
home page and choose "Codespaces", or click this link `open`_.

Quick workspace tour
--------------------

You can develop code in a codespace using your choice of tool:

* a command shell, via an SSH connection initiated using GitHub CLI._
* one of the JetBrains IDEs, via the JetBrains Gateway._
* the Visual Studio Code desktop application._
* a browser-based version of Visual Studio Code._

In this quickstart, we will be using the VSCode desktop application as the
editor.  If you have not used it before, see the Getting started `VSCode docs`_
to familiarize yourself with this tool.

Your workspace will look similar to the image below:

Development workflow with GitHub Codespaces
-------------------------------------------

The  :ref:`development-workflow` section of this documentation contains
information regarding the NumPy development workflow. Make sure to check this
before you start working on your contributions.

Rendering the NumPy documentation
---------------------------------

You can find the detailed documentation on how the rendering of the
documentation with Sphinx works in the :ref:`howto-build-docs` section.

The documentation is pre-built during your codespace initialization. So once
this task is completed, you have two main options to render the documentation
in GitHub Codespaces.

FAQs and troubleshooting
------------------------

**How long does my codespace stay active if I'm not using it?**
If you leave your codespace running without interaction, or if you exit your
codespace without explicitly stopping it, by default the codespace will timeout
after 30 minutes of inactivity. You can customize the duration of the timeout
period for new codespaces that you create.

**Can I come back to a previous codespace?**
The lifecycle of a codespace begins when you create a codespace and ends when
you delete it. You can disconnect and reconnect to an active codespace without
affecting its running processes. You may stop and restart a codespace without
losing changes that you have made to your project.

.. _GitHub Codespaces: https://github.com/features/codespaces
.. _NumPy repository on GitHub: https://github.com/NumPy/NumPy
.. _create your own fork: https://help.github.com/en/articles/fork-a-repo
.. _open: https://github.com/codespaces/new?hide_repo_select=true&ref=main&repo=908607
.. _VSCode docs: https://code.visualstudio.com/docs/getstarted/tips-and-tricks
.. _command shell, via an SSH connection initiated using GitHub CLI: https://docs.github.com/en/authentication/connecting-to-github-with-ssh
.. _one of the JetBrains IDEs, via the JetBrains Gateway: https://docs.github.com/en/codespaces/developing-in-codespaces/using-github-codespaces-in-your-jetbrains-ide
.. _the Visual Studio Code desktop application: https://docs.github.com/en/codespaces/developing-in-codespaces/using-github-codespaces-in-visual-studio-code
.. _a browser-based version of Visual Studio Code: https://docs.github.com/en/codespaces/developing-in-codespaces/developing-in-a-codespace
