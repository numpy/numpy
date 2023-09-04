#!/usr/bin/env python3
import os
import sys
import argparse


class MismatchCAPIError(ValueError):
    pass


def get_api_versions(apiversion):
    """
    Return current C API checksum and the recorded checksum.

    Return current C API checksum and the recorded checksum for the given
    version of the C API version.

    """
    # Compute the hash of the current API as defined in the .txt files in
    # code_generators
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    try:
        m = __import__('genapi')
        numpy_api = __import__('numpy_api')
        curapi_hash = m.fullapi_hash(numpy_api.full_api)
        apis_hash = m.get_versions_hash()
    finally:
        del sys.path[0]

    return curapi_hash, apis_hash[apiversion]


def check_api_version(apiversion):
    """Emits a MismatchCAPIWarning if the C API version needs updating."""
    curapi_hash, api_hash = get_api_versions(apiversion)

    # If different hash, it means that the api .txt files in
    # codegen_dir have been updated without the API version being
    # updated. Any modification in those .txt files should be reflected
    # in the api and eventually abi versions.
    # To compute the checksum of the current API, use numpy/core/cversions.py
    if not curapi_hash == api_hash:
        msg = ("API mismatch detected, the C API version "
               "numbers have to be updated. Current C api version is "
               f"{apiversion}, with checksum {curapi_hash}, but recorded "
               f"checksum in core/codegen_dir/cversions.txt is {api_hash}. If "
               "functions were added in the C API, you have to update "
               f"C_API_VERSION in {__file__}."
               )
        raise MismatchCAPIError(msg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api-version",
        type=str,
        help="C API version to verify (as a hex string)"
    )
    args = parser.parse_args()

    check_api_version(int(args.api_version, base=16))


if __name__ == "__main__":
    main()
