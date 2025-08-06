import os
import subprocess
import sys
from argparse import ArgumentParser

CWD = os.path.abspath(os.path.dirname(__file__))


class DiffLinter:
    def __init__(self) -> None:
        self.repository_root = os.path.realpath(os.path.join(CWD, ".."))

    def run_ruff(self, fix: bool) -> tuple[int, str]:
        """
        Original Author: Josh Wilson (@person142)
        Source:
            https://github.com/scipy/scipy/blob/main/tools/lint_diff.py
        Unlike pycodestyle, ruff by itself is not capable of limiting
        its output to the given diff.
        """
        print("Running Ruff Check...")
        command = ["ruff", "check"]
        if fix:
            command.append("--fix")

        res = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            cwd=self.repository_root,
            encoding="utf-8",
        )
        return res.returncode, res.stdout

    def run_lint(self, fix: bool) -> None:

        # Ruff Linter
        retcode, ruff_errors = self.run_ruff(fix)
        ruff_errors and print(ruff_errors)

        if retcode:
            sys.exit(retcode)

        # C API Borrowed-ref Linter
        retcode, c_API_errors = self.run_check_c_api()
        c_API_errors and print(c_API_errors)

        sys.exit(retcode)

    def run_check_c_api(self) -> tuple[int, str]:
        # Running borrowed ref checker
        print("Running C API borrow-reference linter...")
        borrowed_ref_script = os.path.join(self.repository_root, "tools", "ci",
                                           "check_c_api_usage.sh")
        borrowed_res = subprocess.run(
            ["bash", borrowed_ref_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
        )

        # Exit with non-zero if C API Check fails
        return borrowed_res.returncode, borrowed_res.stdout


if __name__ == "__main__":
    parser = ArgumentParser()
    args = parser.parse_args()

    DiffLinter().run_lint(fix=False)
