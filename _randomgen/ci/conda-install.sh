#!/usr/bin/env bash

if [[ ${TRAVIS_OS_NAME} == "osx" ]]; then wget https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda3.sh; fi
if [[ ${TRAVIS_OS_NAME} == "linux" ]]; then wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda3.sh; fi
chmod +x miniconda3.sh
./miniconda3.sh -b
export PATH=${HOME}/miniconda3/bin:$PATH
conda config --set always_yes true
conda update --all --quiet
conda create -n randomgen-test ${PKGS} pip --quiet
source activate randomgen-test

PKGS="python=${PYTHON} matplotlib numpy"
if [[ -n ${NUMPY} ]]; then PKGS="${PKGS}=${NUMPY}"; fi
PKGS="${PKGS} Cython";
if [[ -n ${CYTHON} ]]; then PKGS="${PKGS}=${CYTHON}"; fi
PKGS="${PKGS} pandas";
if [[ -n ${PANDAS} ]]; then PKGS="${PKGS}=${PANDAS}"; fi
echo conda create -n randomgen-test ${PKGS} pytest setuptools nose --quiet
conda create -n randomgen-test ${PKGS} pytest setuptools nose --quiet
