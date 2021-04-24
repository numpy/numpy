#
# Dockerfile for NumPy development
#
# Usage:
# -------
#
# To make a local build of the container, from the 'Docker-dev' directory:
# docker build  --rm -f "Dockerfile" -t <build-tag> "."
#
# To use the container use the following command. It assumes that you are in
# the root folder of the NumPy git repository, making it available as
# /home/numpy in the container. Whatever changes you make to that directory
# are visible in the host and container.
# The docker image is retrieved from the NumPy dockerhub repository
#
# docker run --rm -it -v $(pwd):/home/numpy numpy/numpy-dev:<image-tag>
#
# By default the container will activate the conda environment numpy-dev
# which contains all the dependencies needed for NumPy development
#
# To build NumPy run: python setup.py build_ext --inplace
#
# To run the tests use: python runtests.py
#
# This image is based on: Ubuntu 20.04 (focal)
# https://hub.docker.com/_/ubuntu/?tab=tags&name=focal
# OS/ARCH: linux/amd64
FROM gitpod/workspace-base:latest

ARG MAMBAFORGE_VERSION="4.10.0-0"
ARG CONDA_ENV=numpy-dev


# ---- Configure environment ----
ENV CONDA_DIR=/home/gitpod/mambaforge3 \
    SHELL=/bin/bash
ENV PATH=${CONDA_DIR}/bin:$PATH \
    WORKSPACE=/workspace/numpy


# -----------------------------------------------------------------------------
# ---- Creating as root - note: make sure to change to gitpod in the end ----
USER root

# hadolint ignore=DL3008
RUN apt-get update && \
    apt-get install -yq --no-install-recommends \
    ca-certificates \
    dirmngr \
    dvisvgm \
    gnupg \
    gpg-agent \
    texlive-latex-extra \
    vim && \
    # this needs to be done after installing dirmngr
    apt-key adv --keyserver keyserver.ubuntu.com --recv-key C99B11DEB97541F0 && \
    apt-add-repository https://cli.github.com/packages && \
    apt-get install -yq --no-install-recommends \
    gh && \
    locale-gen en_US.UTF-8 && \
    apt-get clean && \
    rm -rf /var/cache/apt/* &&\
    rm -rf /var/lib/apt/lists/* &&\
    rm -rf /tmp/*

# Allows this Dockerfile to activate conda environments
SHELL ["/bin/bash", "--login", "-o", "pipefail", "-c"]

# -----------------------------------------------------------------------------
# ---- Installing mamba  ----
RUN wget -q -O mambaforge3.sh \
    "https://github.com/conda-forge/miniforge/releases/download/$MAMBAFORGE_VERSION/Mambaforge-$MAMBAFORGE_VERSION-Linux-x86_64.sh" && \
    bash mambaforge3.sh -p ${CONDA_DIR} -b && \
    rm mambaforge3.sh

# -----------------------------------------------------------------------------
# ---- Copy needed files ----
# basic workspace configurations
COPY ./tools/gitpod/workspace_config /usr/local/bin/workspace_config

RUN chmod a+rx /usr/local/bin/workspace_config && \
    workspace_config

# Copy conda environment file into the container - this needs to exists inside 
# the container to create a conda environment from it
COPY environment.yml /tmp/environment.yml

# -----------------------------------------------------------------------------
# ---- Create conda environment ----
# Install NumPy dependencies
RUN mamba env create -f /tmp/environment.yml && \
    conda activate ${CONDA_ENV} && \
    mamba install ccache -y && \
    # needed for docs rendering later on
    python -m pip install --no-cache-dir sphinx-autobuild && \
    conda clean --all -f -y && \
    rm -rf /tmp/*

# -----------------------------------------------------------------------------
# Always make sure we are not root
USER gitpod