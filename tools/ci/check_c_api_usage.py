#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from pathlib import Path

"""
Borrow-ref C API linter (Python version).

- Recursively scans source files under --root (default: numpy)
- Matches suspicious CPython C-API calls as whole identifiers
- Skips:
  - lines with '// noqa: borrowed-ref OK' or
    '// noqa: borrowed-ref - manual fix needed'
  - line comments (// ...)
  - block comments (/* ... */), even when they span lines
- Prints findings and exits 1 if any issues found, else 0
"""

# List of suspicious function calls:
SUSPICIOUS_FUNCS: tuple[str, ...] = (
    "PyList_GetItem",
    "PyDict_GetItem",
    "PyDict_GetItemWithError",
    "PyDict_GetItemString",
    "PyDict_SetDefault",
    "PyDict_Next",
    "PyWeakref_GetObject",
    "PyWeakref_GET_OBJECT",
    "PyList_GET_ITEM",
    "_PyDict_GetItemStringWithError",
    "PySequence_Fast"
)

# Match any function as a standalone C identifier: (?<!\w)(NAME)(?!\w)
FUNC_RX = re.compile(r"(?<!\w)(?:"
                     + "|".join(map(re.escape, SUSPICIOUS_FUNCS))
                     + r")(?!\w)")

NOQA_OK = "noqa: borrowed-ref OK"
NOQA_MANUAL = "noqa: borrowed-ref - manual fix needed"

DEFAULT_EXTS = {".c", ".h", ".c.src", ".cpp"}
DEFAULT_EXCLUDES = {"pythoncapi-compat"}

def strip_comments(line: str, in_block: bool) -> tuple[str, bool]:
    """
    Return (code_without_comments, updated_in_block).
    Removes // line comments and /* ... */ block comments (non-nesting, C-style).
    """
    i = 0
    out_parts: list[str] = []
    n = len(line)

    while i < n:
        if in_block:
            end = line.find("*/", i)
            if end == -1:
                # Entire remainder is inside a block comment.
                return ("".join(out_parts), True)
            i = end + 2
            in_block = False
            continue

        # Not in block: look for next // or /* from current i
        sl = line.find("//", i)
        bl = line.find("/*", i)

        if sl != -1 and (bl == -1 or sl < bl):
            # Line comment starts first: take code up to '//' and stop
            out_parts.append(line[i:sl])
            return ("".join(out_parts), in_block)

        if bl != -1:
            # Block comment starts: take code up to '/*', then enter block
            out_parts.append(line[i:bl])
            i = bl + 2
            in_block = True
            continue

        # No more comments
        out_parts.append(line[i:])
        break

    return ("".join(out_parts), in_block)

def iter_source_files(root: Path, exts: set[str], excludes: set[str]) -> list[Path]:
    """
    Return a list of source files under 'root', where filenames end with any of the
    extensions in 'exts' (e.g., '.c.src', '.c', '.h').
    Excludes directories whose names are in 'excludes'.
    """
    results: list[Path] = []

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune excluded directories
        dirnames[:] = [d for d in dirnames if d not in excludes]
        for fn in filenames:
            # endswith handles mult-suffice patterns, e.g., .c.src
            if any(fn.endswith(ext) for ext in exts):
                results.append(Path(dirpath) / fn)
    return results


def scan_file(path: Path) -> list[tuple[str, int, str, str]]:
    """
    Scan a single file.
    Returns list of (func_name, line_number, path_str, raw_line_str).
    """
    hits: list[tuple[str, int, str, str]] = []
    in_block = False

    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for lineno, raw in enumerate(f, 1):
                # Skip if approved by noqa markers
                if NOQA_OK in raw or NOQA_MANUAL in raw:
                    continue

                # Remove comments; if nothing remains, skip
                code, in_block = strip_comments(raw.rstrip("\n"), in_block)
                if not code.strip():
                    continue

                # Find all suspicious calls in non-comment code
                for m in FUNC_RX.finditer(code):
                    func = m.group(0)
                    hits.append((func, lineno, str(path), raw.rstrip("\n")))
    except FileNotFoundError:
        # File may have disappeared; ignore gracefully
        pass
    return hits


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Borrow-ref C API linter (Python).")
    ap.add_argument(
        "--root",
        default="numpy",
        type=str,
        help="Root directory to scan (default: numpy)"
        )
    ap.add_argument(
        "--ext",
        action="append",
        default=None,
        help="File extension(s) to include (repeatable). Defaults to .c,.h,.c.src,.cpp",
    )
    ap.add_argument(
        "--exclude",
        action="append",
        default=None,
        help="Directory name(s) to exclude (repeatable). Default: pythoncapi-compat",
    )
    args = ap.parse_args(argv)

    root = Path(args.root)
    if not root.exists():
        print(f"error: root '{root}' does not exist", file=sys.stderr)
        return 2

    exts = set(args.ext) if args.ext else set(DEFAULT_EXTS)
    excludes = set(args.exclude) if args.exclude else set(DEFAULT_EXCLUDES)

    files = list(iter_source_files(root, exts, excludes))
    print(f"Scanning {len(files)} C/C++ source files...")

    # Output file (mirrors your shell behavior)
    tmpdir = Path(".tmp")
    tmpdir.mkdir(exist_ok=True)
    fd, outpath = tempfile.mkstemp(
        prefix="c_api_usage_report.",
        suffix=".txt",
        dir=tmpdir
        )
    os.close(fd)

    findings = 0
    with open(outpath, "w", encoding="utf-8", errors="ignore") as out:
        out.write("Running Suspicious C API usage report workflow...\n\n")
        for p in files:
            for func, lineno, pstr, raw in scan_file(p):
                findings += 1
                out.write(f"Found suspicious call to {func} in file: {pstr}\n")
                out.write(f" -> {pstr}:{lineno}:{raw}\n")
                out.write("Recommendation:\n")
                out.write(
                    "If this use is intentional and safe, add "
                    "'// noqa: borrowed-ref OK' on the same line "
                    "to silence this warning.\n"
                )
                out.write(
                    "Otherwise, consider replacing the call "
                    "with a thread-safe API function.\n\n")

        if findings == 0:
            out.write("C API borrow-ref linter found no issues.\n")

    # Echo report and set exit status
    with open(outpath, "r", encoding="utf-8", errors="ignore") as f:
        sys.stdout.write(f.read())

    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
