import os
import sys
import subprocess
from argparse import ArgumentParser
from git import Repo, exc

CWD = os.path.abspath(os.path.dirname(__file__))
CONFIG = os.path.join(CWD, '..', 'ruff.toml')


class DiffLinter:
    def __init__(self):
        self.repo = Repo(os.path.join(CWD, '..'))
        self.head = self.repo.head.commit

    def run_ruff(self):
        """
            Original Author: Josh Wilson (@person142)
            Source:
              https://github.com/scipy/scipy/blob/main/tools/lint_diff.py
            Unlike pycodestyle, ruff by itself is not capable of limiting
            its output to the given diff.
        """
        res = subprocess.run(
            ['ruff', 'check', '--statistics', '--config', CONFIG],
            stdout=subprocess.PIPE,
            encoding='utf-8',
        )
        return res.returncode, res.stdout

    def run_lint(self):
        retcode, errors = self.run_ruff()

        errors and print(errors)

        sys.exit(retcode)


if __name__ == '__main__':
    parser = ArgumentParser()
    args = parser.parse_args()

    DiffLinter().run_lint()
