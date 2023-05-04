import os
import sys
import subprocess
from argparse import ArgumentParser
from git import Repo, exc

CONFIG = os.path.join(
         os.path.abspath(os.path.dirname(__file__)),
         'lint_diff.ini',
)

# NOTE: The `diff` and `exclude` options of pycodestyle seem to be
# incompatible, so instead just exclude the necessary files when
# computing the diff itself.
EXCLUDE = (
    "numpy/typing/tests/data/",
    "numpy/typing/_char_codes.py",
    "numpy/__config__.py",
    "numpy/f2py",
)


class DiffLinter:
    def __init__(self, branch):
        self.branch = branch
        self.repo = Repo('.')
        self.head = self.repo.head.commit

    def get_branch_diff(self, uncommitted = False):
        """
            Determine the first common ancestor commit.
            Find diff between branch and FCA commit.
            Note: if `uncommitted` is set, check only
                  uncommitted changes
        """
        try:
            commit = self.repo.merge_base(self.branch, self.head)[0]
        except exc.GitCommandError:
            print(f"Branch with name `{self.branch}` does not exist")
            sys.exit(1)

        exclude = [f':(exclude){i}' for i in EXCLUDE]
        if uncommitted:
            diff = self.repo.git.diff(
                self.head, '--unified=0', '***.py', *exclude
            )
        else:
            diff = self.repo.git.diff(
                commit, self.head, '--unified=0', '***.py', *exclude
            )
        return diff

    def run_pycodestyle(self, diff):
        """
            Original Author: Josh Wilson (@person142)
            Source:
              https://github.com/scipy/scipy/blob/main/tools/lint_diff.py
            Run pycodestyle on the given diff.
        """
        res = subprocess.run(
            ['pycodestyle', '--diff', '--config', CONFIG],
            input=diff,
            stdout=subprocess.PIPE,
            encoding='utf-8',
        )
        return res.returncode, res.stdout

    def run_lint(self, uncommitted):
        diff = self.get_branch_diff(uncommitted)
        retcode, errors = self.run_pycodestyle(diff)

        errors and print(errors)

        sys.exit(retcode)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--branch", type=str, default='main',
                        help="The branch to diff against")
    parser.add_argument("--uncommitted", action='store_true',
                        help="Check only uncommitted changes")
    args = parser.parse_args()

    DiffLinter(args.branch).run_lint(args.uncommitted)
