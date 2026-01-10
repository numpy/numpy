#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from re import Pattern

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

def build_func_rx(funcs: tuple[str, ...]) -> Pattern[str]:
    return re.compile(r"(?<!\w)(?:" + "|".join(map(re.escape, funcs)) + r")(?!\w)")

def scan_file(
        path: Path,
        func_rx: Pattern[str],
        noqa_markers: tuple[str, ...]
        ) -> list[tuple[str, int, str, str]]:
    """
    Scan a single file.
    Returns list of (func_name, line_number, path_str, raw_line_str).
    """
    hits: list[tuple[str, int, str, str]] = []
    in_block = False
    noqa_set = set(noqa_markers)

    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for lineno, raw in enumerate(f, 1):
                # Skip if approved by noqa markers
                if any(mark in raw for mark in noqa_set):
                    continue

                # Remove comments; if nothing remains, skip
                code, in_block = strip_comments(raw.rstrip("\n"), in_block)
                if not code.strip():
                    continue

                # Find all suspicious calls in non-comment code
                for m in func_rx.finditer(code):
                    hits.append((m.group(0), lineno, str(path), raw.rstrip("\n")))
    except FileNotFoundError:
        # File may have disappeared; ignore gracefully
        pass
    return hits


def main(argv: list[str] | None = None) -> int:
    # List of suspicious function calls:
    suspicious_funcs: tuple[str, ...] = (
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
    func_rx = build_func_rx(suspicious_funcs)
    noqa_markers = (
        "noqa: borrowed-ref OK",
        "noqa: borrowed-ref - manual fix needed"
        )
    default_exts = {".c", ".h", ".c.src", ".cpp"}
    default_excludes = {"pythoncapi-compat"}

    ap = argparse.ArgumentParser(description="Borrow-ref C API linter (Python).")
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress normal output; exit status alone indicates result (useful\
              for CI).",
    )
    ap.add_argument(
        "-j", "--jobs",
        type=int,
        default=0,
        help="Number of worker threads (0=auto, 1=sequential).",
    )
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
        help=f"File extension(s) to include (repeatable). Defaults to {default_exts}",
    )
    ap.add_argument(
        "--exclude",
        action="append",
        default=None,
        help=f"Directory name(s) to exclude (repeatable). Default: {default_excludes}",
    )
    args = ap.parse_args(argv)

    if args.ext:
        exts = {e if e.startswith(".") else f".{e}" for e in args.ext}
    else:
        exts = set(default_exts)
    excludes = set(args.exclude) if args.exclude else set(default_excludes)

    root = Path(args.root)
    if not root.exists():
        print(f"error: root '{root}' does not exist", file=sys.stderr)
        return 2

    files = sorted(iter_source_files(root, exts, excludes), key=str)

    # Determine concurrency: auto picks a reasonable cap for I/O-bound work
    if args.jobs is None or args.jobs <= 0:
        max_workers = min(32, (os.cpu_count() or 1) * 5)
    else:
        max_workers = max(1, args.jobs)
    print(f'Scanning {len(files)} C/C++ source files...\n')

    # Output file (mirrors your shell behavior)
    tmpdir = Path(".tmp")
    tmpdir.mkdir(exist_ok=True)

    findings = 0

    # Run the scanning in parallel; only the main thread writes the report
    all_hits: list[tuple[str, int, str, str]] = []
    if max_workers == 1:
        for p in files:
            all_hits.extend(scan_file(p, func_rx, noqa_markers))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            fut_to_file = {ex.submit(scan_file, p, func_rx, noqa_markers):
                           p for p in files}
            for fut in as_completed(fut_to_file):
                try:
                    all_hits.extend(fut.result())
                except Exception as e:
                    print(f'Failed to scan {fut_to_file[fut]}: {e}')

    # Sort for deterministic output: by path, then line number
    all_hits.sort(key=lambda t: (t[2], t[1]))

    # There no hits, linter passed
    if not all_hits:
        if not args.quiet:
            print("All checks passed! C API borrow-ref linter found no issues.\n")
        return 0

    # There are some linter failures: create a log file
    with tempfile.NamedTemporaryFile(
        prefix="c_api_usage_report.",
        suffix=".txt",
        dir=tmpdir,
        mode="w+",
        encoding="utf-8",
        delete=False,
        ) as out:
        report_path = Path(out.name)
        out.write("Running Suspicious C API usage report workflow...\n\n")
        for func, lineo, pstr, raw in all_hits:
            findings += 1
            out.write(f"Found suspicious call to {func} in file: {pstr}\n")
            out.write(f" -> {pstr}:{lineo}a:{raw}\n")
            out.write("Recommendation:\n")
            out.write(
                "If this use is intentional and safe, add "
                "'// noqa: borrowed-ref OK' on the same line "
                "to silence this warning.\n"
            )
            out.write(
                "Otherwise, consider replacing the call "
                "with a thread-safe API function.\n\n"
            )

        out.flush()
        if not args.quiet:
            out.seek(0)
            sys.stdout.write(out.read())
            print(f"Report written to: {report_path}\n\n\
C API borrow-ref linter FAILED.")

    return 1


if __name__ == "__main__":

    sys.exit(main())
