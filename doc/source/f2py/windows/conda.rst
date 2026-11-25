.. _f2py-win-conda:

=========================
F2PY and Conda on Windows
=========================

As a convenience measure, we will additionally assume the
existence of ``scoop``, which can be used to install tools without
administrative access.

.. code-block:: powershell

  Invoke-Expression (New-Object System.Net.WebClient).DownloadString('https://get.scoop.sh')

Now we will setup a ``conda`` environment.

.. code-block:: powershell

	scoop install miniconda3
	# For conda activate / deactivate in powershell
	conda install -n root -c pscondaenvs pscondaenvs
	Powershell -c Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
	conda init powershell
	# Open a new shell for the rest

``conda`` pulls packages from ``msys2``, however, the UX is sufficiently different enough to warrant a separate discussion.


