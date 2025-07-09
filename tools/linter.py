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
        retcode, errors = self.run_ruff(fix)

        errors and print(errors)

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

        print(borrowed_res.stdout)

        # Exit with non-zero if either Ruff or C API check fails
        final_code = retcode or borrowed_res.returncode
        sys.exit(final_code)


if __name__ == "__main__":
    parser = ArgumentParser()
    args = parser.parse_args()

    DiffLinter().run_lint(fix=False)
