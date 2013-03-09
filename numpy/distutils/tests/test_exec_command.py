import sys
import StringIO

from numpy.distutils import exec_command


class redirect_stdout(object):
    """Context manager to redirect stdout for exec_command test."""
    def __init__(self, stdout=None):
        self._stdout = stdout or sys.stdout

    def __enter__(self):
        self.old_stdout = sys.stdout
        sys.stdout = self._stdout

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush()
        sys.stdout = self.old_stdout


def test_exec_command():
    # Regression test for gh-2999 and gh-2915.
    # There are several packages (nose, scipy.weave.inline, Sage inline
    # Fortran) that replace stdout, in which case it doesn't have a fileno
    # method.  This is tested here, with a do-nothing command that fails if the
    # presence of fileno() is assumed in exec_command.
    with redirect_stdout(StringIO.StringIO()):
        exec_command.exec_command("cd '.'")

