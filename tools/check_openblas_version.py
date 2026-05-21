"""
Checks related to the OpenBLAS version used in CI.

Options:
1. Check that the BLAS used at build time is (a) scipy-openblas, and (b) its version is
   higher than a given minimum version. Note: this method only seems to give
   the first 3 version components, so 0.3.30.0.7 gets translated to 0.3.30 when reading
   it back out from `scipy.show_config()`.
2. Check requirements files in the main numpy repo and compare with the numpy-release
   repo. Goal is to ensure that `numpy-release` is not behind.

Both of these checks are primarily useful in a CI job.

Examples:

    # Requires install numpy
    $ python check_openblas_version.py --min-version 0.3.30

    # Only needs the requirements files
    $ python check_openblas_version.py --req-files \
            ../numpy-release/requirements/openblas_requirements.txt
"""

import argparse
import os.path
import pprint


def check_built_version(min_version):
    import numpy
    deps = numpy.show_config('dicts')['Build Dependencies']
    assert "blas" in deps
    print("Build Dependencies: blas")
    pprint.pprint(deps["blas"])
    assert deps["blas"]["version"].split(".") >= min_version.split(".")
    assert deps["blas"]["name"] == "scipy-openblas"


def check_requirements_files(reqfile):
    if not os.path.exists(reqfile):
        print(f"Path does not exist: {reqfile}")

    def get_version(line):
        req = line.split(";")[0].split("==")[1].split(".")[:5]
        return tuple(int(s) for s in req)

    def parse_reqs(reqfile):
        with open(reqfile) as f:
            lines = f.readlines()

        v32 = None
        v64 = None
        for line in lines:
            if "scipy-openblas32" in line:
                v32 = get_version(line)
            if "scipy-openblas64" in line:
                v64 = get_version(line)
        if v32 is None or v64 is None:
            raise AssertionError("Expected `scipy-openblas32` and "
                                 "`scipy-openblas64` in `ci_requirements.txt`, "
                                 f"got:\n  {'  '.join(lines)}")
        return v32, v64

    this_dir = os.path.abspath(os.path.dirname(__file__))
    reqfile_thisrepo = os.path.join(this_dir, '..', 'requirements',
                                    'ci_requirements.txt')

    v32_thisrepo, v64_thisrepo = parse_reqs(reqfile_thisrepo)
    v32_rel, v64_rel = parse_reqs(reqfile)

    def compare_versions(v_rel, v_thisrepo, bits):
        if not v_rel >= v_thisrepo:
            raise AssertionError(f"`numpy-release` version of scipy-openblas{bits} "
                                 f"{v_rel} is behind this repo: {v_thisrepo}")

    compare_versions(v64_rel, v64_thisrepo, "64")
    compare_versions(v32_rel, v32_thisrepo, "32")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--req-files",
        type=str,
        help="Path to the requirements file to compare with the one in this repo"
    )
    parser.add_argument(
        "--min-version",
        type=str,
        help="The minimum version that should have been used at build time for "
             "installed `numpy` package"
    )
    args = parser.parse_args()

    if args.min_version is None and args.req_files is None:
        raise ValueError("One of `--req-files` or `--min-version` needs to be "
                         "specified")

    if args.min_version:
        check_built_version(args.min_version)

    if args.req_files:
        check_requirements_files(args.req_files)


if __name__ == '__main__':
    main()
