#!/usr/bin/env python3
"""
Script to download NumPy wheels from the Anaconda staging area.

Usage::

    $ ./tools/download-wheels.py <version> -w <optional-wheelhouse>

The default wheelhouse is ``release/installers``.

Dependencies
------------

- beautifulsoup4
- urllib3

Examples
--------

While in the repository root::

    $ python tools/download-wheels.py 1.19.0
    $ python tools/download-wheels.py 1.19.0 -w ~/wheelhouse

"""
import argparse
import os
import re
import shutil

import urllib3
from bs4 import BeautifulSoup

__version__ = "0.2"

# Edit these for other projects.

# The first URL is used to get the file names as it avoids the need for paging
# when the number of files exceeds the page length. Note that files/page is not
# stable and can change when the page layout changes. The second URL is used to
# retrieve the files themselves. This workaround is copied from SciPy.
NAMES_URL = "https://pypi.anaconda.org/multibuild-wheels-staging/simple/numpy/"
FILES_URL = "https://anaconda.org/multibuild-wheels-staging/numpy"

# Name prefix of the files to download.
PREFIX = "numpy"

# Name endings of the files to download.
WHL = r"-.*\.whl$"
ZIP = r"\.zip$"
GZIP = r"\.tar\.gz$"
SUFFIX = rf"({WHL}|{GZIP}|{ZIP})"


def get_wheel_names(version):
    """ Get wheel names from Anaconda HTML directory.

    This looks in the Anaconda multibuild-wheels-staging page and
    parses the HTML to get all the wheel names for a release version.

    Parameters
    ----------
    version : str
        The release version. For instance, "1.18.3".

    """
    http = urllib3.PoolManager(cert_reqs="CERT_REQUIRED")
    tmpl = re.compile(rf"^.*{PREFIX}-{version}{SUFFIX}")
    index_url = f"{NAMES_URL}"
    index_html = http.request('GET', index_url)
    soup = BeautifulSoup(index_html.data, 'html.parser')
    return sorted(soup.find_all(string=tmpl))


def download_wheels(version, wheelhouse, test=False):
    """Download release wheels.

    The release wheels for the given NumPy version are downloaded
    into the given directory.

    Parameters
    ----------
    version : str
        The release version. For instance, "1.18.3".
    wheelhouse : str
        Directory in which to download the wheels.

    """
    http = urllib3.PoolManager(cert_reqs="CERT_REQUIRED")
    wheel_names = get_wheel_names(version)

    for i, wheel_name in enumerate(wheel_names):
        wheel_url = f"{FILES_URL}/{version}/download/{wheel_name}"
        wheel_path = os.path.join(wheelhouse, wheel_name)
        with open(wheel_path, "wb") as f:
            with http.request("GET", wheel_url, preload_content=False,) as r:
                info = r.info()
                length = int(info.get('Content-Length', '0'))
                if length == 0:
                    length = 'unknown size'
                else:
                    length = f"{(length / 1024 / 1024):.2f}MB"
                print(f"{i + 1:<4}{wheel_name} {length}")
                if not test:
                    shutil.copyfileobj(r, f)
    print(f"\nTotal files downloaded: {len(wheel_names)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "version",
        help="NumPy version to download.")
    parser.add_argument(
        "-w", "--wheelhouse",
        default=os.path.join(os.getcwd(), "release", "installers"),
        help="Directory in which to store downloaded wheels\n"
             "[defaults to <cwd>/release/installers]")
    parser.add_argument(
        "-t", "--test",
        action='store_true',
        help="only list available wheels, do not download")

    args = parser.parse_args()

    wheelhouse = os.path.expanduser(args.wheelhouse)
    if not os.path.isdir(wheelhouse):
        raise RuntimeError(
            f"{wheelhouse} wheelhouse directory is not present."
            " Perhaps you need to use the '-w' flag to specify one.")

    download_wheels(args.version, wheelhouse, test=args.test)
