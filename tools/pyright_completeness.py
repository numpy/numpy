"""
Run PyRight's `--verifytypes` and check that its reported type completeness is above
a minimum threshold.

Example usage:

    spin run python tools/pyright_completeness.py --verifytypes numpy --ignoreexternal \
        --fail-under 80 --exclude-like '*.tests.*' '*.conftest.*'

We use `--ignoreexternal` to avoid "partially unknown" reports coming from the stdlib
`numbers` module, see https://github.com/microsoft/pyright/discussions/9911.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import subprocess
import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fail-under",
        type=float,
        default=100.0,
        help="Fail if coverage is below this percentage",
    )
    parser.add_argument(
        "--exclude-like",
        required=False,
        nargs='*',
        type=str,
        help="Exclude symbols whose names matches this glob pattern",
    )
    args, unknownargs = parser.parse_known_args(argv)
    pyright_args = list(unknownargs)
    if "--outputjson" not in pyright_args:
        pyright_args.append("--outputjson")
    return run_pyright_with_coverage(pyright_args, args.fail_under, args.exclude_like)


def run_pyright_with_coverage(
    pyright_args: list[str],
    cov_fail_under: float,
    exclude_like: Sequence[str],
) -> int:
    result = subprocess.run(
        ["pyright", *pyright_args], capture_output=True, text=True
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
            x for x in symbols if not any(fnmatch.fnmatch(x["name"], pattern) for pattern in exclude_like)
            and x['isExported']
        ]
        cov_percent = (
            sum(x["isTypeKnown"] for x in matched_symbols) / len(matched_symbols) * 100
        )
    else:
        cov_percent = data["typeCompleteness"]["completenessScore"] * 100

    sys.stderr.write(result.stderr)
    if cov_percent < cov_fail_under:
        sys.stdout.write(
            f"Coverage {cov_percent:.1f}% is below minimum required "
            f"{cov_fail_under:.1f}%\n"
        )
        return 1
    sys.stdout.write(
        f"Coverage {cov_percent:.1f}% is at or above minimum required "
        f"{cov_fail_under:.1f}%\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
