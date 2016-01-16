#!/bin/bash


# Set the version info based on python version.
VERSION="$(python setup.py --version)"

# Add the git hash if it is missing.
if [[ "${VERSION}" == *+ ]];
then
    VERSION="${VERSION}$(git rev-parse --short HEAD)"
fi

# Set the version of the package.
echo $VERSION > __conda_version__.txt

# Simply install.
$PYTHON setup.py install
