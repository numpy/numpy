"""
Finds python and rst files containing::

    ..versionadded:: master

        Some not yet released feature

And rewrites them to something like::

    ..versionadded:: 2.0.0

        Some not yet released feature

The motivation is to allow PRs to be agnostic of the version they are being
merged into, rather than having to update them each time they miss a release.
"""
import os
import re
import itertools

import numpy as np
from refguide_check import iter_included_files

# the marker to look for in place of a version
PLACEHOLDER = 'master'


def match_line_column(m):
    """ Get the (line, column) corresponding to a regex Match object """
    i = m.start()
    line_start = 0
    line_no = 1
    while True:
        line_end = m.string.find('\n', line_start, i)
        if line_end == -1:
            break
        line_no += 1
        line_start = line_end + 1
    return line_no, i - line_start


def major_version(v_str):
    """ Accept x.y as well as the more common x.y.z """
    try:
        return np.lib.NumpyVersion(v_str)
    except ValueError:
        try:
            return np.lib.NumpyVersion(v_str + '.0')
        except:
            pass
        raise


def replace_rst_directive(fname, version):
    """ Replace version rst directives in `fname`

    The directives that use `PLACEHOLDER` will be changed to use `version`,
    with the modified files written in place
    """
    print(f"processing {fname}")
    with open(fname, 'r', encoding='utf8') as f:
        f_contents = f.read()

    errors = []
    replaced = []

    def handle_version(m):
        v_str = m.group(2)
        try:
            v = major_version(v_str)
        except ValueError as e:
            if v_str == PLACEHOLDER:
                replaced.append(m)
                v_str = version
            else:
                errors.append((m, e))
        else:
            if v.is_devversion:
                errors.append((m, ValueError("Expected non-dev version")))
        return m.group(1) + v_str + m.group(3)

    f_contents = re.sub(
        r'(\.\.\s+(?:deprecated|version(?:added|changed|removed))\s*::\s*)([^\n]+?)(\s*\n)',
        handle_version,
        f_contents
    )
    print(f"  replaced {len(replaced)} versions")

    if replaced:
        with open(fname, 'w', encoding='utf8') as f:
            f.write(f_contents)

    return replaced, errors


def main():
    all_errors = []
    base_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')

    for fname in itertools.chain(
        iter_included_files(os.path.join(base_dir, 'numpy'), suffixes=('.py')),
        iter_included_files(os.path.join(base_dir, 'doc'), suffixes=('.rst'))
    ):
        replaced, errors = replace_rst_directive(fname, version=np.__version__)
        if errors:
            all_errors.append((fname, errors))

    # todo: replace C comments too

    for fname, errors in all_errors:
        print("In file {}:".format(fname))
        for m, error in errors:
            l, c = match_line_column(m)
            print(f"  line {l}:{c}: {error!r}")

    if all_errors:
        raise SystemExit("Some errors occurred")

if __name__ == "__main__":
    main()
