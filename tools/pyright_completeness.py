"""
Run PyRight's `--verifytypes` and check that its reported type completeness is above
a minimum threshold.

Requires `basedpyright` to be installed in the environment.

Example usage:

    spin run python tools/pyright_completeness.py --verifytypes numpy --ignoreexternal \
        --exclude-like '*.tests.*' '*.conftest.*'

We use `--ignoreexternal` to avoid "partially unknown" reports coming from the stdlib
`numbers` module, see https://github.com/microsoft/pyright/discussions/9911.
"""

import argparse
import fnmatch
import json
import subprocess
import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exclude-like",
        required=False,
        nargs="*",
        type=str,
        help="Exclude symbols whose names matches this glob pattern",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print detailed error diagnostics for symbols with unknown types",
    )
    args, unknownargs = parser.parse_known_args(argv)
    pyright_args = list(unknownargs)
    if "--outputjson" not in pyright_args:
        pyright_args.append("--outputjson")
    return run_pyright_with_coverage(pyright_args, args.exclude_like, args.verbose)


def print_verbose_errors(matched_symbols: list[dict]) -> None:
    """Print detailed diagnostics for symbols with unknown types."""
    unknown_symbols = [s for s in matched_symbols if not s["isTypeKnown"]]
    if not unknown_symbols:
        return

    print(f"\n{'=' * 60}")
    print(f"Symbols with unknown types: {len(unknown_symbols)}")
    print(f"{'=' * 60}\n")

    for symbol in unknown_symbols:
        print(f"[{symbol['category']}] {symbol['name']}")
        if symbol.get("diagnostics"):
            for diag in symbol["diagnostics"]:
                severity = diag.get("severity", "info").upper()
                message = diag.get("message", "").replace("\xa0", " ")
                file_path = diag.get("file", "")
                range_info = diag.get("range", {})
                start = range_info.get("start", {})
                line = start.get("line", 0) + 1  # Convert to 1-based
                col = start.get("character", 0) + 1

                print(f"  {severity}: {message}")
                if file_path:
                    print(f"    at {file_path}:{line}:{col}")
        print()


def run_pyright_with_coverage(
    pyright_args: list[str],
    exclude_like: Sequence[str],
    verbose: bool = False,
) -> int:
    result = subprocess.run(
        ["basedpyright", *pyright_args],
        capture_output=True,
        text=True,
    )

    try:
        data = json.loads(result.stdout)
    except json.decoder.JSONDecodeError:
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        return 1

    if exclude_like:
        symbols = data["typeCompleteness"]["symbols"]
        matched_symbols = [
            x
            for x in symbols
            if not any(fnmatch.fnmatch(x["name"], pattern) for pattern in exclude_like)
            and x["isExported"]
        ]
        covered = sum(x["isTypeKnown"] for x in matched_symbols) / len(matched_symbols)
        if verbose:
            print_verbose_errors(matched_symbols)
    else:
        covered = data["typeCompleteness"]["completenessScore"]
    sys.stderr.write(result.stderr)
    if covered < 1:
        sys.stdout.write(f"Coverage {covered:.1%} is below minimum required 100%\n")
        return 1
    sys.stdout.write("Coverage is at 100%\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
