#!/bin/bash

set -e

curl micro.mamba.pm/install.sh | bash

conda init --all
micromamba shell init -s bash
micromamba env create -f environment.yml --yes
# Note that `micromamba activate numpy-dev` doesn't work, it must be run by the
# user (same applies to `conda activate`)

git submodule update --init
