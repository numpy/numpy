import subprocess as sp
import pytest


def test_no_untracked_files():
    fd_1 = sp.run(["git", "status", "-u", "--porcelain"], stdout=sp.PIPE,
                  stderr=sp.PIPE, universal_newlines=True, check=True)
    fd_1 = fd_1.stdout.split('\n')

    untracked_files = [output[3:] for output in fd_1 if
                       output.startswith('??')]

    msg = f"git detected these untracked files:\n{untracked_files}"
    assert len(untracked_files) == 0, msg
