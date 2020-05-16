#!/usr/bin/env python
"""
Download NumPy wheels from Anaconda staging area.

"""
import sys
import os
import re
import shutil
import argparse

import urllib3
from bs4 import BeautifulSoup

__version__ = '0.1'

ANACONDA_INDEX = 'https://anaconda.org/multibuild-wheels-staging/numpy/files'
ANACONDA_FILES = 'https://anaconda.org/multibuild-wheels-staging/numpy/simple'


def get_wheel_names(version):
    """ Get wheel names from Anaconda HTML directory.

    This looks in the Anaconda multibuild-wheels-staging page and
    parses the HTML to get all the wheel names for a release version.

    Parameters
    ----------
    version : str
        The release version. For instance, "1.18.3".

    """
    tmpl = re.compile('^.*numpy-' + version + '.*\.whl$')
    http = urllib3.PoolManager(cert_reqs='CERT_REQUIRED')
    indx = http.request('GET', ANACONDA_INDEX)
    soup = BeautifulSoup(indx.data, 'html.parser')
    return soup.findAll(text=tmpl)


def download_wheels(version, wheelhouse):
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
    wheel_names = get_wheel_names(version[0])
    http = urllib3.PoolManager(cert_reqs='CERT_REQUIRED')
    for wheel_name in wheel_names:
        wheel_url = os.path.join(ANACONDA_FILES, wheel_name)
        wheel_path = os.path.join(wheelhouse, wheel_name)
        with open(wheel_path, 'wb') as f:
            with http.request('GET', wheel_url, preload_content=False,) as r:
                print(f"Downloading {wheel_name}")
                shutil.copyfileobj(r, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "version",
         help="NumPy version to download.")
    parser.add_argument(
        "-w", "--wheelhouse",
        default=os.path.join(os.getcwd(), "release", "installers"),
        help="Directory in which to store downloaded wheels\n"
             "[defaults to <cwd>/release/installers]")

    args = parser.parse_args()

    wheelhouse = os.path.expanduser(args.wheelhouse)
    download_wheels(args.version, wheelhouse)
