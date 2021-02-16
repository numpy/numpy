import os
import sys
import subprocess
from argparse import ArgumentParser
from git import Repo

BASE_BRANCH = 'master'
CONFIG = os.path.join(
         os.path.abspath(os.path.dirname(__file__)),
         'lint_diff.ini',
)


class DiffLinter:
    def __init__(self, branch):
        self.branch = branch
        self.repo = Repo('.')

    def get_branch_diff(self, uncommitted):
        commit = self.repo.merge_base(BASE_BRANCH, self.branch)[0]

        if uncommitted:
            diff = self.repo.git.diff(self.branch, '***.py')
        else:
            diff = self.repo.git.diff(commit, self.branch, '***.py')
        return diff

    def run_pycodestyle(self, diff):
        """
            Original Author: Josh Wilson (@person142)
            Source:
              https://github.com/scipy/scipy/blob/master/tools/lint_diff.py
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
    parser.add_argument("--branch", type=str, default='master',
                        help="The branch to diff against")
    parser.add_argument("--uncommitted", action='store_true',
                        help="Check only uncommitted changes")
    args = parser.parse_args()

    DiffLinter(args.branch).run_lint(args.uncommitted)
