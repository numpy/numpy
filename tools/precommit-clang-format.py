#!/usr/bin/env python3
"""precommit-clang-format.py

This script wraps git-clang-format (installed via a python wheel in pre-commit)
and runs it with our desired arguments, returning desired exit codes.
git-clang-format only formats the changed blocks between two commits.

We need to provide the tool with the proper parameters to apply the diff if
possible (i.e. if it is comparing to HEAD), and if not, just display the diff
and exit error if it exists (when not pointing at HEAD, i.e. uneditable) This
script usage allows for comparing two diffs from arbritrary points in time
easily.

It can also be fake run without pre-commit: PRE_COMMIT_FROM_REF=someref
PRE_COMMIT_TO_REF=someref ./tools/precommit-clang-format.py file1.c file2.c
PRE_COMMIT_TO_REF can be omitted to use HEAD.

This script is known to work with clang-format==14.0.3 available on PyPi.
If their "no modified files to format" message or return code scheme changes,
this will need to be updated.
"""

import os
import subprocess
import sys
from typing import Iterable, NoReturn, Tuple

FORMAT_ARGS = ["--style", "file"]

# Environment variables set by pre-commit
PRE_COMMIT_TO_REF = os.environ.get("PRE_COMMIT_TO_REF")
PRE_COMMIT_FROM_REF = os.environ.get("PRE_COMMIT_FROM_REF")


class TC:
    """Terminal color escape sequences.

    Print starting with any member and end with "ENDC". All colors are
    the "light" variants.
    """

    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

    BLACK = "\033[90m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"

    YELLOW_BOLD = "\033[1;33m"
    RED_BOLD = "\033[1;31m"
    CYAN_BOLD = "\033[1;36m"
    GREEN_BOLD = "\033[1;32m"


def git_rev_parse(ref: str) -> str:
    """Get the output of rev-parse"""
    return subprocess.check_output(
        ("git", "rev-parse", "--short", ref), encoding=sys.stdout.encoding
    ).strip()


def tee_cmd(args: Iterable[str]) -> Tuple[str, int]:
    """Run a command with printing output, also return the output and
    return code.
    """
    output = ""

    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding=sys.stdout.encoding,
    )

    while proc.poll() is None:
        text = proc.stdout.readline()
        output += text
        sys.stdout.write(text)

    return (output.strip(), proc.returncode)


def compare_skip_formatting() -> NoReturn:
    """Print the diff and exit; do not apply changes."""
    run_cmd = (
        ["git-clang-format"]
        + FORMAT_ARGS
        + ["--diff", PRE_COMMIT_FROM_REF, PRE_COMMIT_TO_REF, "--"]
        + sys.argv
    )

    from_ref = git_rev_parse(PRE_COMMIT_FROM_REF)
    to_ref = git_rev_parse(PRE_COMMIT_TO_REF)

    print(
        f'\n${TC.CYAN}Comparing "{from_ref}" to "{to_ref}"{TC.ENDC}\n'
        f'{TC.YELLOW}Cannot safely apply changes; "to" ref "{to_ref}" is not HEAD.\n'
        f"I'll just show you the diff instead. Running...\n"
        f"> {run_cmd}{TC.ENDC}\n"
    )

    output, exitcode = tee_cmd(run_cmd)

    # Success! No diff
    if output == "no modified files to format":
        print(f"{TC.GREEN}Everything looks OK; no work for me here${TC.ENDC}")
        sys.exit(0)

    if exitcode == 0:
        # git-clang-format exits 0 even if a diff is created, so we need to
        # make sure this script fails
        print(
            f"{TC.RED}Changes between the two revisions are not properely"
            f" formatted, but I'm too\nscared to apply formatting to target"
            f" refs that are not HEAD. I quit.{TC.ENDC}"
        )
        sys.exit(1)

    # If we're here, git-clang-format exited nonzero which is some kind of error
    print(
        f"{TC.RED}Call the doctor! Something went wrong with git-clang-format${TC.ENDC}"
    )
    sys.exit(exitcode)


def main() -> NoReturn:
    print("Starting git-clang-format launcher")

    # Check if we have a TO_REF target and if so, enter this block
    if PRE_COMMIT_TO_REF is not None:
        # Get the hashes of the specified revs for easy comparison
        # Need a FROM_REF in this case so it's OK if the rev-parse fails
        to_ref = git_rev_parse(PRE_COMMIT_TO_REF)
        head_ref = git_rev_parse("HEAD")

        # If we are not comparing to HEAD, changes cannot be safely applied; all we
        # can do is print the diff and exit appropriately
        if to_ref != head_ref:
            # No return
            compare_skip_formatting()

    run_cmd = (
        ["git-clang-format"] + FORMAT_ARGS + [PRE_COMMIT_FROM_REF, "--"] + sys.argv
    )

    print(f"{TC.CYAN}Target ref is HEAD, running...\n>{run_cmd}{TC.ENDC}")

    output, exitcode = tee_cmd(run_cmd)

    if (
        output == "no modified files to format"
        or output == "clang-format did not modify any files"
    ):
        # If clang-format doesn't have any changes, no problems. Expect to exit 0
        print(f"{TC.GREEN}Everything looks OK; no work for me here{TC.ENDC}")
        sys.exit(exitcode)

    print(
        f"{TC.YELLOW}OK, I formatted that really nice for you."
        f" Retry your commit now.{TC.ENDC}"
    )
    # Need to exit as error if files were changed
    sys.exit(1)


if __name__ == "__main__":
    main()
