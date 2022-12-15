#!/bin/bash

set -e

conda env create -f environment.yml
git submodule update --init
