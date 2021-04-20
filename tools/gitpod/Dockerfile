# 
# Dockerfile for NumPy development 
# 
# Usage: 
# -------
# 
# To make a local build of the container, from the root directory directory:
# docker build  --rm -f "./tools/gitpod/Dockerfile" -t <build-tag> "."
# 
# To use the container use the following command. It assumes that you are in
# the root folder of the NumPy git repository, making it available as
# /home/NumPy in the container. Whatever changes you make to that directory
# are visible in the host and container.
# The docker image is retrieved from the NumPy dockerhub repository
#
# docker run --rm -it -v $(pwd):/workspace/numpy NumPy:<image-tag>
# 
# By default the container will activate the conda environment numpy-dev
# which contains all the dependencies needed for NumPy development
# 
# To build NumPy run: python setup.py build_ext --inplace
# Followed by conda develop . if you need to build the docs
#
# To run the tests use: python runtests.py
# 
# This image is based on: Ubuntu 20.04 (focal)
# https://hub.docker.com/_/ubuntu/?tab=tags&name=focal
# OS/ARCH: linux/amd64
ARG ROOT_CONTAINER=gitpod/workspace-base:latest
ARG BASE_CONTAINER=${ROOT_CONTAINER}

# hadolint ignore=DL3006
FROM ${BASE_CONTAINER}

# -----------------------------------------------------------------------------
# ---- Mambaforge installer ----
# Default values can be overridden at build time
# (ARGS are in lower case to distinguish them from ENV)
ARG mambaforge_version="4.10.0-0"

ENV CONDA_DIR=/home/gitpod/mambaforge3
ENV PATH=${CONDA_DIR}/bin:$PATH \
    WORKSPACE=/workspace/numpy \
    CONDA_ENV=numpy-dev \
    GP_USER=gitpod

USER root

# Change default shell - this avoids issues with Conda later
SHELL ["/bin/bash", "--login", "-o", "pipefail", "-c"]


# hadolint ignore=DL3008
RUN apt-get update && \ 
    apt-get install -yq --no-install-recommends \
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

# Install mambaforge3
RUN wget -q -O mambaforge3.sh \
    https://github.com/conda-forge/miniforge/releases/download/${mambaforge_version}/Mambaforge-${mambaforge_version}-Linux-x86_64.sh && \
    bash mambaforge3.sh -p ${CONDA_DIR} -b && \
    rm mambaforge3.sh

# -----------------------------------------------------------------------------
# ---- Copy needed files ----
# Copy multiple scripts - fix directory permissions and 
# basic workspace configurations
COPY ./tools/gitpod/workspace_config /usr/local/bin/workspace_config
COPY environment.yml /tmp/environment.yml

RUN chmod a+rx /usr/local/bin/workspace_config 

RUN mamba env create -f /tmp/environment.yml -n $CONDA_ENV && \
    conda activate ${CONDA_ENV} && \
    mamba install ccache -y && \
    python -m pip install --no-cache-dir sphinx-autobuild && \
    conda clean --all -f -y && \ 
    rm -rf /tmp/* && \
    workspace_config

# -----------------------------------------------------------------------------
USER ${GP_USER}

WORKDIR ${WORKSPACE}