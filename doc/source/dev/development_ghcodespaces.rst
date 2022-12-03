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
`GitHub Codespaces`_ is a service that provides cloud based development environments
so that you don't have to install anything on your local machine or worry about configuration.

.. _https://github.com/features/codespaces

TO DO


What is a codespace?
--------------------
A codespace is an instance of Codespaces - and thus a development environment that is hosted in the cloud.
Each codespace runs on a virtual machine hosted by GitHub. You can choose 
the type of machine you want to use, depending on the resources you need. Various 
types of machine are available, starting with a 2-core processor, 4 GB of RAM, 
and 32 GB of storage.
You can connect to a codespace from your browser, from Visual Studio Code, from 
the JetBrains Gateway application, or by using GitHub CLI.


Forking the NumPy repository
----------------------------
The best way to work on NumPy as a contributor is by making a fork of the 
repository first.

#. Browse to the `NumPy repository on GitHub`_ and `create your own fork`_.
#. Browse to your fork. Your fork will have a URL like 
   https://github.com/inessapawson/NumPy, except with your GitHub username in place of ``inessapawson``.
   
   
Starting GitHub Codespaces
--------------------------
TO DO


Quick workspace tour
--------------------
You can develop code in a codespace using your choice of tool:

* a command shell, via an SSH connection initiated using GitHub CLI.
* one of the JetBrains IDEs, via the JetBrains Gateway.
* the Visual Studio Code desktop application.
* a browser-based version of Visual Studio Code.

In this quickstart, we will be using the VSCode desktop application as the editor. If you have not used it before, see the Getting started `VSCode docs`_ to familiarize yourself with this tool.

Your workspace will look similar to the image below:

TO DO


Development workflow with GitHub Codespaces
--------------------------------
The  :ref:`development-workflow` section of this documentation contains 
information regarding the NumPy development workflow. Make sure to check this 
before working on your contributions.

TO DO


Rendering the NumPy documentation
---------------------------------
You can find the detailed documentation on how the rendering of the documentation with 
Sphinx works in the :ref:`howto-build-docs` section.

The documentation is pre-built during your codespace initialization. So once 
this task is completed, you have two main options to render the documentation 
in GitHub Codespaces.

Option 1: Using Liveserve
~~~~~~~~~~~~~~~~~~~~~~~~~

#. View the documentation in ``NumPy/doc/build/html``. You can start with 
   ``index.html`` and browse, or you can jump straight to the file you're 
   interested in.
#. To see the rendered version of a page, you can right-click on the ``.html`` 
   file and click on **Open with Live Serve**. Alternatively, you can open the 
   file in the editor and click on the **Go live** button on the status bar.

    .. image:: 
        :alt: 

#. A simple browser will open to the right-hand side of the editor. We recommend 
   closing it and click on the **Open in browser** button in the pop-up.
#. To stop the server click on the **Port: 5500** button on the status bar.

Option 2: Using the rst extension
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A quick and easy way to see live changes in a ``.rst`` file as you work on it 
uses the rst extension with docutils.

.. note:: This will generate a simple live preview of the document without the 
    ``html`` theme, and some backlinks might not be added correctly. But it is an 
    easy and lightweight way to get instant feedback on your work.

#. Open any of the source documentation files located in ``doc/source`` in the 
   editor.
#. Open VSCode Command Palette with :kbd:`Cmd-Shift-P` in Mac or 
   :kbd:`Ctrl-Shift-P` in Linux and Windows. Start typing "restructured" 
   and choose either "Open preview" or "Open preview to the Side".

    .. image:: 
        :alt: 

#. As you work on the document, you will see a live rendering of it on the editor.

    .. image:: 
        :alt: 

To see the final output with the ``html`` theme, you need to 
rebuild the docs with ``make html`` and use Live Serve as described in option 1.


FAQs and troubleshooting
-------------------------
TO DO

