#!/bin/bash

set -e

conda env create -f environment.yml
conda init --all

git submodule update --init
